"""Lightweight data-cleaning helpers, adapted from the physical-bounds
quarantine concept in https://github.com/MisterSpacebar/multinode-bay-imputation
(preprocess.py). This does NOT replicate that project's multi-node GNN
imputation/resampling pipeline -- just the sanity-check + optional fill step,
applied generically to any loaded dataframe.
"""

import re

import pandas as pd

# Column-name keywords (case-insensitive) mapped to plausible physical ranges.
# Matched against a normalized column name so both "Dissolved Oxygen (mg/L)"
# and "odo_mgL" style columns are caught.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "temp": (-5.0, 45.0),  # water/air temperature, deg C
    "sal": (0.0, 45.0),  # salinity, ppt
    "odo_mg": (0.0, 20.0),  # dissolved oxygen, mg/L
    "odo_pct": (0.0, 200.0),  # oxygen saturation, %
    "do_mg": (0.0, 20.0),
    "do_per": (0.0, 200.0),
    "depth": (0.0, 100.0),  # m
    "pressure": (0.0, 200.0),  # psia
    "spec_cond": (0.0, 100000.0),  # uS/cm
    "turbidity": (0.0, 10000.0),  # FNU/NTU
    "ph": (0.0, 14.0),
    "chlorophyll": (0.0, 1000.0),  # ug/L
    "chl_": (0.0, 1000.0),
}


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def _bounds_for_column(name: str) -> tuple[float, float] | None:
    normalized = _normalize(name)
    # "ph" must be its own word (e.g. "pH"), not a substring match inside
    # unrelated words like "chlorophyll" or "phosphate".
    tokens = re.findall(r"[a-z]+", name.lower())
    for keyword, bounds in PHYSICAL_BOUNDS.items():
        if keyword == "ph":
            if "ph" in tokens:
                return bounds
            continue
        if _normalize(keyword) in normalized:
            return bounds
    return None


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Quarantine physically-impossible numeric readings (set to NaN).

    Returns the cleaned dataframe (copy) and a report of what changed per column.
    """
    cleaned = df.copy()
    report: list[dict] = []

    for col in cleaned.columns:
        bounds = _bounds_for_column(col)
        if bounds is None:
            continue
        numeric = pd.to_numeric(cleaned[col], errors="coerce")
        lower, upper = bounds
        out_of_bounds = numeric.notna() & ((numeric < lower) | (numeric > upper))
        n_bad = int(out_of_bounds.sum())
        n_missing_before = int(numeric.isna().sum())
        if n_bad:
            numeric.loc[out_of_bounds] = pd.NA
        cleaned[col] = numeric
        if n_bad:
            report.append(
                {
                    "column": col,
                    "bounds": [lower, upper],
                    "quarantined": n_bad,
                    "missing_before": n_missing_before,
                }
            )

    return cleaned, report
