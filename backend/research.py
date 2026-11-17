"""Read-only, schema-aware adapters for locally generated bay research outputs.

This module NEVER refits models, interpolates measurements, or reads arbitrary
file paths from model/user input. Output files are generated separately by
multinode-bay-imputation/trustworthy and are retrospective research artifacts.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

# (relative path, minimum schema, meaning)
ARTIFACTS = {
    "gam_scores": (
        "environmental_gam/paired_scores.csv",
        ("outcome", "threshold_mgL", "model", "n_paired", "skill_vs_baseline"),
        "Held-out monthly paired GAM vs baseline scores; interpret with sample size and event counts.",
    ),
    "gam_effects": (
        "environmental_gam/gam_smooth_effects.csv",
        ("outcome", "threshold_mgL", "predictor", "grid_value",
         "difference_from_reference", "provenance"),
        "Conditional same-day smooth associations, not causal effects or future forecasts.",
    ),
    "episode_summary": (
        "episode_dynamics/station_episode_summary.csv",
        ("node", "threshold_mgL", "n_observed_runs", "n_identifiable_onsets"),
        "Descriptive observed daily low-oxygen runs; unknown boundaries are censored.",
    ),
    "episode_runs": (
        "episode_dynamics/episode_table.csv",
        ("node", "threshold_mgL", "start_date_local", "end_date_local",
         "censored_left", "censored_right", "provenance"),
        "Observed contiguous eligible low-oxygen calendar days, not continuous hypoxia hours.",
    ),
    "transitions": (
        "episode_dynamics/transition_matrix_by_station.csv",
        ("node", "from_state", "to_state", "n_transitions", "row_total", "probability"),
        "Observed adjacent-day transition counts; zero-denominator probabilities are undefined.",
    ),
    "markov_summary": (
        "episode_dynamics/markov_monte_carlo_summary.csv",
        ("block_scheme", "metric", "point_estimate", "ci_lower", "ci_upper"),
        "Exploratory stationary Markov uncertainty; not calibrated live warnings.",
    ),
    "warning_scores": (
        "transition_warning/hindcast_scores.csv",
        ("horizon_days", "model", "n_paired", "n_events", "brier"),
        "Retrospective monthly chronological hindcast scores; compare matched cases.",
    ),
    "warning_hindcasts": (
        "transition_warning/hindcast_predictions.csv",
        ("node", "issue_date", "horizon_days", "origin_state", "model",
         "label", "probability", "provenance"),
        "Out-of-sample retrospective end-of-day severe-onset probabilities and observed future labels.",
    ),
    "warning_snapshot": (
        "transition_warning/dashboard/latest_warning_snapshot.csv",
        ("node", "issue_date", "current_state", "status"),
        "Latest historical file observation per station, NOT a present-day/live forecast.",
    ),
}
MAX_BYTES = 32 * 1024 * 1024
MAX_RESULT_ROWS = 60
DATE_COLUMNS = ("date", "start_date_local", "issue_date")
HORIZONS = (1, 3, 7)
WARNING_MODELS = ("markov", "do_history", "environmental")


def configured_root() -> Path | None:
    value = os.getenv("BAY_RESEARCH_OUTPUT_DIR", "").strip()
    return Path(value).expanduser().resolve() if value else None


def _artifact_path(root: Path, kind: str) -> Path:
    if kind not in ARTIFACTS:
        raise ValueError(f"Unknown research artifact {kind!r}. Choose from: {list(ARTIFACTS)}")
    candidate = (root / ARTIFACTS[kind][0]).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Research path escapes the configured root (symlink rejected)")
    return candidate


def research_catalog(root: Path | None = None) -> dict:
    root = root if root is not None else configured_root()
    configured = root is not None and root.is_dir()
    available, missing = [], []
    for kind, (relative_path, _, description) in ARTIFACTS.items():
        item = {"artifact": kind, "relative_path": relative_path, "description": description}
        if configured:
            try:
                path = _artifact_path(root, kind)
                present = path.is_file() and path.stat().st_size <= MAX_BYTES
            except (OSError, ValueError):
                present = False
        else:
            present = False
        (available if present else missing).append(item)
    return {
        "enabled": bool(configured),
        "read_only": True,
        "available": available,
        "missing": missing,
        "interpretation": (
            "Only locally generated retrospective research outputs are listed. "
            "An available file is not evidence of complete validation, a fresh live feed, "
            "or agreement with currently uploaded datasets. Missing files are not zero results."
        ),
    }


def _validate(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    required = ARTIFACTS[kind][1]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{kind} missing required upstream columns: {sorted(missing)}")
    if "provenance" in frame:
        if frame["provenance"].isna().any() or not frame["provenance"].eq("observed_only").all():
            raise ValueError(f"{kind} contains non-observed or unknown provenance")
    if "node" in frame and frame["node"].isna().any():
        raise ValueError(f"{kind} contains missing station identities")
    if "probability" in frame:
        values = pd.to_numeric(frame["probability"], errors="coerce")
        if (values.dropna().lt(0) | values.dropna().gt(1)).any():
            raise ValueError(f"{kind} contains probabilities outside [0, 1]")
        if kind == "warning_hindcasts" and values.isna().any():
            raise ValueError("Scored hindcasts require finite probabilities")
    if "horizon_days" in frame:
        horizons = pd.to_numeric(frame["horizon_days"], errors="coerce")
        if not horizons.isin(HORIZONS).all():
            raise ValueError(f"{kind} contains unsupported warning horizons")
        frame = frame.copy()
        frame["horizon_days"] = horizons.astype(int)
    if kind == "warning_hindcasts":
        label = pd.to_numeric(frame["label"], errors="coerce")
        if not label.isin((0, 1)).all():
            raise ValueError("Scored hindcasts require observed binary future labels")
        if not frame["model"].isin(WARNING_MODELS).all():
            raise ValueError("Unknown hindcast prediction model")
        if not frame["origin_state"].isin(("Normal", "Moderate")).all():
            raise ValueError("Already-Severe/unknown origins cannot be used as onset warnings")
        if frame.duplicated(["node", "issue_date", "horizon_days", "model"]).any():
            raise ValueError("Duplicate hindcast prediction keys")
        groups = frame.groupby(["node", "issue_date", "horizon_days"], dropna=False)
        if not groups["model"].nunique().eq(len(WARNING_MODELS)).all():
            raise ValueError("Hindcast probabilities must contain all three matched models")
        if groups["label"].nunique().gt(1).any():
            raise ValueError("Hindcast models disagree on the held-out future outcome")
    if "n_paired" in frame:
        n = pd.to_numeric(frame["n_paired"], errors="coerce")
        if n.isna().any() or n.lt(0).any():
            raise ValueError("Paired evaluation count is invalid")
    return frame


def query_research(
    artifact: str,
    station: str | None = None,
    threshold_mg_l: float | None = None,
    horizon_days: int | None = None,
    model: str | None = None,
    issue_date: str | None = None,
    limit: int = 20,
    root: Path | None = None,
) -> dict:
    """Read a fixed upstream output table, filter it and disclose its provenance.

    Client/model cannot select a pathname or request access to input sensor CSVs.
    Filtering is exact, not station-name fuzzy matching or bay-wide averaging.
    """
    if not isinstance(limit, int) or not 1 <= limit <= MAX_RESULT_ROWS:
        raise ValueError(f"limit must be between 1 and {MAX_RESULT_ROWS}")
    root = root if root is not None else configured_root()
    if root is None or not root.is_dir():
        raise ValueError("Research outputs unavailable; configure BAY_RESEARCH_OUTPUT_DIR")
    path = _artifact_path(root, artifact)
    if not path.is_file():
        raise ValueError(f"Research output {artifact!r} is missing: {ARTIFACTS[artifact][0]}")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Research output {artifact!r} is too large for interactive exploration")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"Cannot parse research output {artifact!r}: {exc}") from exc
    frame = _validate(frame, artifact)

    filters = {}
    for key, wanted, column in (
        ("station", station, "node"),
        ("threshold_mg_l", threshold_mg_l, "threshold_mgL"),
        ("horizon_days", horizon_days, "horizon_days"),
        ("model", model, "model"),
        ("issue_date", issue_date, "issue_date"),
    ):
        if wanted is None:
            continue
        if column not in frame:
            raise ValueError(f"Filter {key} is unsupported by {artifact!r}")
        if key == "threshold_mg_l":
            wanted = float(wanted)
            if wanted not in (2.0, 4.0):
                raise ValueError("Threshold must be 2 or 4 mg/L")
            mask = pd.to_numeric(frame[column], errors="coerce").eq(wanted)
        elif key == "horizon_days":
            if wanted not in HORIZONS:
                raise ValueError("Horizon must be 1, 3 or 7 days")
            mask = frame[column].eq(wanted)
        elif key == "issue_date":
            try:
                wanted = pd.Timestamp(wanted)
                if wanted.tzinfo is not None or wanted.strftime("%Y-%m-%d") != issue_date:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError("issue_date must be a local calendar date YYYY-MM-DD")
            mask = frame[column].astype(str).eq(issue_date)
        else:
            mask = frame[column].astype(str).eq(str(wanted))
        frame = frame.loc[mask]
        filters[key] = str(wanted)

    # Maintain deterministic date order before limiting the display.
    for date_col in DATE_COLUMNS:
        if date_col in frame and len(frame):
            frame = frame.sort_values(date_col, kind="stable")
            break
    total = len(frame)
    displayed = frame.head(limit)
    records = [] if displayed.empty else __import__("json").loads(
        displayed.to_json(orient="records", date_format="iso")
    )
    return {
        "artifact": artifact,
        "relative_path": ARTIFACTS[artifact][0],
        "description": ARTIFACTS[artifact][2],
        "status": "ok" if total else "no_matching_rows",
        "filters": filters,
        "n_matching": total,
        "n_returned": len(records),
        "truncated": total > len(records),
        "rows": records,
        "file_sha256": digest,
        "validation": "schema_and_provenance_checked; upstream_input_hashes_not_verified_here",
        "forecast_type": (
            "retrospective_out_of_sample_hindcast"
            if artifact == "warning_hindcasts" else "historical_research_output"
        ),
        "interpretation": (
            "Research-only, not live or automatically updated. Missing/unevaluated cases "
            "are not negative events. GAM smooths are observational associations. "
            "Brier and skill metrics require paired sample sizes and calibration review. "
            "Station results must not be averaged as a bay-wide forecast."
        ),
    }
