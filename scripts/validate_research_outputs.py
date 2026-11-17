"""Validate *local real exports* and (optionally) their running FastAPI adapter.

Run from the ollama-bay-example repository root:
  python -m scripts.validate_research_outputs --root "C:/.../trustworthy_outputs"
  python -m scripts.validate_research_outputs --api-url http://127.0.0.1:8000

This script never generates model outputs, reads only the fixed research artifact
allowlist, and does not send private CSVs to remote services. API comparison is
opt-in and intended for a local backend pointed at the SAME research root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from backend.research import (
    ARTIFACTS, DATE_COLUMNS, MAX_RESULT_ROWS, _artifact_path,
    configured_root, query_research, research_catalog,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(obj: Any) -> Any:
    """Canonicalize JSON-native values without silently rounding probabilities."""
    if isinstance(obj, dict):
        return {key: _normalize(value) for key, value in sorted(obj.items())}
    if isinstance(obj, list):
        return [_normalize(item) for item in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _source_records(frame: pd.DataFrame, limit: int) -> list[dict]:
    for col in DATE_COLUMNS:
        if col in frame and len(frame):
            frame = frame.sort_values(col, kind="stable")
            break
    return json.loads(frame.head(limit).to_json(orient="records", date_format="iso"))


def _issue(issues: list[dict], level: str, scope: str, message: str) -> None:
    issues.append({"level": level, "scope": scope, "message": message})


def _metadata_check(root: Path, issues: list[dict]) -> dict:
    """Check upstream output metadata and any source hashes available on disk.

    Hashes for missing upstream inputs cannot be verified. Record this as an
    explicit warning, rather than claiming research lineage is confirmed.
    """
    plans = {
        "environmental_gam/environmental_gam_metadata.json": {
            "daily_sha256": "daily_targets.csv",
            "risk_sha256": "episode_dynamics/recovery_risk_days.csv",
        },
        "transition_warning/warning_metadata.json": {
            "daily_sha256": "daily_targets.csv",
            "states_sha256": "episode_dynamics/station_day_states.csv",
            "transitions_sha256": "episode_dynamics/observed_transitions.csv",
        },
        "episode_dynamics/transition_metadata.json": {
            "regimes_sha256": "onset/regime_days.csv",
            "episodes_sha256": "episode_dynamics/episode_table.csv",
        },
        "episode_dynamics/survival_metadata.json": {},
        "transition_warning/dashboard/dashboard_metadata.json": {
            "predictions_sha256": "transition_warning/hindcast_predictions.csv",
            "windows_sha256": "transition_warning/warning_windows.csv",
            "daily_sha256": "daily_targets.csv",
            "states_sha256": "episode_dynamics/station_day_states.csv",
            "transitions_sha256": "episode_dynamics/observed_transitions.csv",
        },
    }
    findings = {}
    for meta_rel, checks in plans.items():
        path = (root / meta_rel).resolve()
        if not path.is_relative_to(root.resolve()):
            _issue(issues, "error", meta_rel, "Metadata path escapes configured root")
            continue
        if not path.exists():
            findings[meta_rel] = {"status": "missing"}
            _issue(issues, "warning", meta_rel, "Metadata missing; output lineage not fully verifiable")
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("Metadata must be a JSON object")
            if meta.get("provenance") != "observed_only":
                _issue(issues, "error", meta_rel, "Metadata does not establish observed_only provenance")
            matched = 0
            missing = 0
            for hash_key, relative_input in checks.items():
                claimed = meta.get(hash_key)
                source = (root / relative_input).resolve()
                if not source.is_relative_to(root.resolve()):
                    _issue(issues, "error", meta_rel, f"Source escapes root: {relative_input}")
                    continue
                if claimed is None:
                    _issue(issues, "warning", meta_rel, f"Missing metadata hash: {hash_key}")
                    continue
                if not source.is_file():
                    missing += 1
                    _issue(issues, "warning", meta_rel, f"Cannot verify {hash_key}: source file absent")
                elif _hash(source).lower() != str(claimed).lower():
                    _issue(issues, "error", meta_rel, f"Upstream source hash mismatch: {hash_key}")
                else:
                    matched += 1

            if meta_rel.endswith("survival_metadata.json"):
                inputs = meta.get("inputs", {})
                for field in ("episodes", "states", "transitions"):
                    spec = inputs.get(field, {})
                    digest = spec.get("sha256")
                    source_name = {"episodes": "episode_table.csv",
                                   "states": "station_day_states.csv",
                                   "transitions": "observed_transitions.csv"}[field]
                    path_src = (root / "episode_dynamics" / source_name).resolve()
                    if digest and path_src.is_file():
                        if digest != _hash(path_src):
                            _issue(issues, "error", meta_rel, f"Upstream source hash mismatch: {field}")
                        else:
                            matched += 1
                    else:
                        missing += 1
                        _issue(issues, "warning", meta_rel, f"Cannot verify source: {field}")

            if meta_rel.endswith("warning_metadata.json"):
                scored = root / "transition_warning/hindcast_predictions.csv"
                if scored.exists() and "n_hindcast_cases" in meta:
                    count = len(pd.read_csv(scored))
                    if count % 3 or count // 3 != int(meta["n_hindcast_cases"]):
                        _issue(issues, "error", meta_rel,
                               "Metadata n_hindcast_cases disagrees with three-model hindcast rows")
            if meta_rel.endswith("environmental_gam_metadata.json"):
                preds = root / "environmental_gam/time_block_predictions.csv"
                if preds.exists() and "n_out_of_sample_predictions" in meta:
                    if len(pd.read_csv(preds)) != int(meta["n_out_of_sample_predictions"]):
                        _issue(issues, "error", meta_rel,
                               "Metadata n_out_of_sample_predictions disagrees with CSV")
            findings[meta_rel] = {
                "status": "parsed", "source_hashes_verified": matched,
                "source_hashes_unavailable": missing, "sha256": _hash(path),
            }
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            findings[meta_rel] = {"status": "invalid"}
            _issue(issues, "error", meta_rel, f"Metadata invalid: {exc}")
    return findings


def _check_hindcast_logic(root: Path, issues: list[dict]) -> dict:
    """Independent event monotonicity and paired-label checks on all scored rows."""
    path = root / "transition_warning/hindcast_predictions.csv"
    if not path.is_file():
        return {"status": "not_available"}
    try:
        df = pd.read_csv(path)
        if df.empty:
            return {"status": "empty", "cases": 0}
        p = pd.to_numeric(df["probability"], errors="raise")
        labels = pd.to_numeric(df["label"], errors="raise")
        if not p.between(0, 1).all() or not labels.isin([0, 1]).all():
            _issue(issues, "error", "warning_hindcasts", "Invalid probability/label values")
        if not df["provenance"].eq("observed_only").all():
            _issue(issues, "error", "warning_hindcasts", "Non-observed hindcast provenance")
        groups = df.groupby(["node", "issue_date", "model"], dropna=False)
        for key, group in groups:
            group = group.sort_values("horizon_days")
            if (group["probability"].diff().dropna() < -1e-9).any():
                _issue(issues, "error", "warning_hindcasts",
                       f"Risk decreases as horizon increases for station/date/model {key}")
                break
        return {"status": "checked", "rows": len(df),
                "paired_case_groups": df.groupby(["node", "issue_date", "horizon_days"]).ngroups,
                "distinct_stations": int(df["node"].nunique()),
                "distinct_dates": int(df["issue_date"].nunique())}
    except (KeyError, OSError, ValueError, TypeError) as exc:
        _issue(issues, "error", "warning_hindcasts", f"Logic check failed: {exc}")
        return {"status": "invalid"}


def validate(root: Path | None, api_url: str | None = None) -> dict:
    issues: list[dict] = []
    if root is None:
        return {"status": "blocked_missing_root", "available_artifacts": 0,
                "issues": [{"level": "error", "scope": "configuration",
                            "message": "Supply --root or BAY_RESEARCH_OUTPUT_DIR"}]}
    root = root.expanduser().resolve()
    if not root.is_dir():
        return {"status": "blocked_missing_root", "available_artifacts": 0,
                "issues": [{"level": "error", "scope": "configuration",
                            "message": "Configured research output directory does not exist"}]}

    catalog = research_catalog(root)
    report = {"status": "pending", "read_only": True,
              "available_artifacts": len(catalog["available"]),
              "missing_artifacts": [x["artifact"] for x in catalog["missing"]],
              "artifacts": {}, "issues": issues, "api": "not_requested"}
    if not catalog["available"]:
        report["status"] = "blocked_no_generated_artifacts"
        _issue(issues, "error", "inputs", "No generated allowlisted CSV exports available")
        return report
    for item in catalog["available"]:
        kind = item["artifact"]
        rel = ARTIFACTS[kind][0]
        try:
            path = _artifact_path(root, kind)
            original = pd.read_csv(path, low_memory=False)
            queried = query_research(kind, limit=MAX_RESULT_ROWS, root=root)
            independent = _source_records(original, MAX_RESULT_ROWS)
            same_rows = _normalize(independent) == _normalize(queried["rows"])
            same_hash = _hash(path) == queried["file_sha256"]
            same_n = len(original) == queried["n_matching"]
            report["artifacts"][kind] = {
                "status": "pass" if same_rows and same_hash and same_n else "fail",
                "rows": len(original),
                "sha256": queried["file_sha256"],
                "preview_rows": len(queried["rows"]),
                "source_matches_adapter": bool(same_rows),
                "row_count_matches": bool(same_n),
                "sha256_matches": bool(same_hash),
            }
            if not same_rows or not same_hash or not same_n:
                _issue(issues, "error", kind, "Adapter differs from independently read CSV")
        except (ValueError, TypeError, OSError, KeyError, pd.errors.ParserError) as exc:
            report["artifacts"][kind] = {"status": "fail"}
            _issue(issues, "error", kind, f"Invalid output or query: {exc}")

    report["metadata"] = _metadata_check(root, issues)
    report["hindcast_logic"] = _check_hindcast_logic(root, issues)

    if api_url:
        import requests
        base = api_url.rstrip("/")
        api_results = {}
        try:
            response = requests.get(f"{base}/api/research/catalog", timeout=10)
            response.raise_for_status()
            remote = response.json()
            available_remote = {x["artifact"] for x in remote["available"]}
            available_local = {x["artifact"] for x in catalog["available"]}
            if available_remote != available_local:
                _issue(issues, "error", "api", "API and local catalog disagree on available artifacts")
            for kind, local in report["artifacts"].items():
                if local["status"] != "pass":
                    continue
                res = requests.post(f"{base}/api/research/query",
                                    json={"artifact": kind, "limit": MAX_RESULT_ROWS}, timeout=45)
                res.raise_for_status()
                result = res.json()
                same = (
                    result.get("file_sha256") == local["sha256"]
                    and result.get("n_matching") == local["rows"]
                    and _normalize(result.get("rows")) ==
                    _normalize(query_research(kind, limit=MAX_RESULT_ROWS, root=root)["rows"])
                )
                api_results[kind] = "pass" if same else "fail"
                if not same:
                    _issue(issues, "error", "api/" + kind, "API response differs from local export")
            report["api"] = {"status": "checked", "artifact_results": api_results}
        except (requests.RequestException, ValueError, KeyError) as exc:
            report["api"] = {"status": "failed"}
            _issue(issues, "error", "api", f"API parity check failed: {exc}")

    n_errors = sum(x["level"] == "error" for x in issues)
    n_warnings = sum(x["level"] == "warning" for x in issues)
    report["n_errors"] = n_errors
    report["n_warnings"] = n_warnings
    report["status"] = "fail" if n_errors else (
        "pass_with_unverified_lineage" if n_warnings else "pass"
    )
    report["interpretation"] = (
        "Passing this test establishes file/schema/API agreement for the selected local "
        "exports, not scientific model validity, operational calibration, a running Ollama "
        "model, or a browser-tested Streamlit UI. Missing source inputs mean upstream "
        "hash lineage was not independently verified."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None,
                        help="Root of trustworthy_outputs; default BAY_RESEARCH_OUTPUT_DIR")
    parser.add_argument("--api-url", default=None,
                        help="Optional running FastAPI base URL; default skips network calls")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional local JSON report file; don't commit source-specific reports")
    args = parser.parse_args(argv)
    root = args.root if args.root else configured_root()
    result = validate(root, args.api_url)
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    return 0 if result["status"] in ("pass", "pass_with_unverified_lineage") else 2


if __name__ == "__main__":
    sys.exit(main())
