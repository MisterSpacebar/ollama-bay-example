"""Regression tests for the real-file validator, using explicit synthetic fixtures.

These tests do not replace running scripts.validate_research_outputs against
actual locally generated trustworthy_outputs files.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from backend import main as backend_main
from scripts.validate_research_outputs import main, validate


def make_gam(root: Path):
    folder = root / "environmental_gam"
    folder.mkdir(exist_ok=True)
    pd.DataFrame([{
        "outcome": "odo_min", "threshold_mgL": None, "model": "environmental_gam",
        "n_paired": 50, "skill_vs_baseline": -0.05,
    }]).to_csv(folder / "paired_scores.csv", index=False)


def make_warnings(root: Path, monotonic=True):
    folder = root / "transition_warning"
    folder.mkdir(exist_ok=True)
    rows = []
    for model in ("markov", "do_history", "environmental"):
        for h in (1, 3, 7):
            probability = (0.10, 0.15, 0.20)[(1, 3, 7).index(h)]
            if not monotonic and model == "markov" and h == 7:
                probability = 0.05
            rows.append({
                "node": "S1", "issue_date": "2025-07-15", "horizon_days": h,
                "origin_state": "Normal", "model": model, "label": 0,
                "probability": probability, "provenance": "observed_only",
            })
    pd.DataFrame(rows).to_csv(folder / "hindcast_predictions.csv", index=False)


class RealOutputValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_missing_root_is_not_a_success(self):
        result = validate(self.root / "not_here")
        self.assertEqual(result["status"], "blocked_missing_root")

    def test_empty_root_is_blocked(self):
        result = validate(self.root)
        self.assertEqual(result["status"], "blocked_no_generated_artifacts")

    def test_csv_row_hash_and_count_parity(self):
        make_gam(self.root)
        result = validate(self.root)
        self.assertEqual(result["status"], "pass_with_unverified_lineage")
        self.assertEqual(result["available_artifacts"], 1)
        self.assertEqual(result["artifacts"]["gam_scores"]["status"], "pass")
        self.assertTrue(result["artifacts"]["gam_scores"]["source_matches_adapter"])

    def test_invalid_provenance_fails_validation(self):
        make_warnings(self.root)
        path = self.root / "transition_warning/hindcast_predictions.csv"
        df = pd.read_csv(path)
        df.loc[0, "provenance"] = "gnn_imputed"
        df.to_csv(path, index=False)
        result = validate(self.root)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["artifacts"]["warning_hindcasts"]["status"], "fail")

    def test_nonmonotone_warning_risk_is_detected(self):
        make_warnings(self.root, monotonic=False)
        result = validate(self.root)
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["hindcast_logic"]["status"], "checked")
        self.assertTrue(any("Risk decreases" in v["message"] for v in result["issues"]))

    def test_warning_metadata_count_and_input_hash(self):
        make_warnings(self.root)
        folder = self.root / "transition_warning"
        (folder / "warning_metadata.json").write_text(json.dumps({
            "provenance": "observed_only",
            "n_hindcast_cases": 9,
            "daily_sha256": "0" * 64,
            "states_sha256": "0" * 64,
            "transitions_sha256": "0" * 64,
        }), encoding="utf-8")
        # 9 predictions / 3 matched models = 3 issue-day/horizon cases.
        result = validate(self.root)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(any("n_hindcast_cases" in v["message"] for v in result["issues"]))

    def test_running_api_parity_via_inprocess_http_client(self):
        make_warnings(self.root)
        app_client = TestClient(backend_main.app)
        with patch.dict(os.environ, {"BAY_RESEARCH_OUTPUT_DIR": str(self.root)}):
            original_get, original_post = __import__("requests").get, __import__("requests").post
            def local_get(url, **kwargs):
                self.assertTrue(url.startswith("http://127.0.0.1:8000/"))
                return app_client.get(url.split("8000", 1)[1])
            def local_post(url, **kwargs):
                self.assertTrue(url.startswith("http://127.0.0.1:8000/"))
                return app_client.post(url.split("8000", 1)[1], json=kwargs["json"])
            with patch("requests.get", side_effect=local_get):
                with patch("requests.post", side_effect=local_post):
                    result = validate(self.root, api_url="http://127.0.0.1:8000")
        self.assertNotEqual(result["status"], "fail")
        self.assertEqual(result["api"]["artifact_results"]["warning_hindcasts"], "pass")

    def test_cli_json_report(self):
        make_gam(self.root)
        dest = self.root / "result.json"
        with patch("sys.stdout"):
            status = main(["--root", str(self.root), "--report", str(dest)])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(dest.read_text())["available_artifacts"], 1)


if __name__ == "__main__":
    unittest.main()
