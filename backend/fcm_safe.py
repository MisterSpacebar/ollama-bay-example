"""Time- and station-safe adapter for the existing lightweight FCM.

FCM weights are exploratory lagged *associations*, never causal discovery or
validated early-warning probabilities. The base FCM still fills covariates
retrospectively, so its scenario trajectories are not operational forecasts.
"""
from __future__ import annotations

import pandas as pd

from backend.fcm import fcm
from backend.trustworthy_bay import STATION_COLUMNS, TIME_COLUMNS


def safe_fcm(
    df: pd.DataFrame,
    columns: list[str],
    forcing: list[str] | None = None,
    lag: int = 1,
    alpha: float = 2.0,
    clamp: dict[str, float] | None = None,
    steps_ahead: int = 20,
    timestamp_col: str | None = None,
    station_col: str | None = None,
) -> dict:
    """Fit only a single node with a unique, regular, ordered time axis.

    A single row lag is not a fixed elapsed time if sampling intervals vary.
    Never learn spurious transitions between concatenated stations or epochs.
    """
    if timestamp_col is None:
        timestamp_col = next((col for col in TIME_COLUMNS if col in df), None)
    if not timestamp_col or timestamp_col not in df:
        raise ValueError("FCM requires timestamp_col containing observed chronological timestamps")
    if station_col is None:
        station_col = next((col for col in STATION_COLUMNS if col in df), None)
    if station_col is not None:
        if station_col not in df:
            raise ValueError(f"Unknown station_col: {station_col!r}")
        if df[station_col].isna().any() or df[station_col].nunique() != 1:
            raise ValueError("FCM requires one station/source at a time; filter nodes before fitting")
    # An explicit node name can be reused across 2025/2026/platform exports.
    # Inspect ALL available identity columns, not just the selected one.
    for other in STATION_COLUMNS:
        if other in df and (df[other].isna().any() or df[other].nunique() != 1):
            raise ValueError(f"FCM contains multiple station/source identities in {other!r}; filter first")

    if pd.api.types.is_numeric_dtype(df[timestamp_col]):
        raise ValueError("FCM requires date/time timestamps, not numeric row IDs or years")
    time = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    if time.isna().any() or time.duplicated().any():
        raise ValueError("FCM requires valid, unique timestamps")
    order = time.argsort(kind="stable")
    time_sorted = time.iloc[order]
    elapsed = time_sorted.diff().dropna()
    if elapsed.empty or (elapsed <= pd.Timedelta(0)).any() or elapsed.nunique() != 1:
        raise ValueError("FCM requires fixed regular sampling; aggregate/resample explicitly first")
    if lag < 1:
        raise ValueError("lag must be at least 1")

    result = fcm(
        df.iloc[order].reset_index(drop=True), columns=columns, forcing=forcing,
        lag=lag, alpha=alpha, clamp=clamp, steps_ahead=steps_ahead,
    )
    result["timestamp_col"] = timestamp_col
    result["station_col"] = station_col
    result["sample_interval_minutes"] = elapsed.iloc[0].total_seconds() / 60
    result["interpretation"] = (
        "Exploratory, noncausal lagged associations from one regularly sampled station. "
        "The legacy output field top_causal_edges is retained for compatibility "
        "but its values do not demonstrate causal effects. Retrospective filling "
        "and fitted normalization mean scenario trajectories are not validated forecasts."
    )
    return result
