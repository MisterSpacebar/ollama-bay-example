"""Observed-only, coverage-qualified dissolved-oxygen summaries.

Adapted from multinode-bay-imputation/trustworthy: quality.py, regimes.py,
and episode_dynamics.py. This module does not train a forecast, interpolate
oxygen measurements, or infer why sensors were offline.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

DO_BOUNDS_MG_L = (0.0, 20.0)
THRESHOLDS_MG_L = (2.0, 4.0)
STATION_COLUMNS = ("node", "station_id", "station", "source_file")
TIME_COLUMNS = ("datetime", "timestamp", "date_time", "time", "date", "Date")
DO_COLUMNS = ("odo_mgL", "exo.odo_mgL", "DO (mg/L)", "Dissolved Oxygen (mg/L)", "do_mgL")


def _resolve(df: pd.DataFrame, name: str | None, candidates: tuple[str, ...], kind: str) -> str | None:
    if name is not None:
        if name not in df:
            raise ValueError(f"Unknown {kind} column {name!r}. Available: {list(df.columns)}")
        return name
    return next((c for c in candidates if c in df), None)


def _observed_flags(series: pd.Series) -> pd.Series:
    # A Python bool('False') is True. Never use astype(bool) on CSV string flags.
    mapped = series.astype("string").str.strip().str.lower().map(
        {"1": True, "0": False, "true": True, "false": False}
    )
    if mapped.isna().any():
        raise ValueError("Observation flags must be explicit 0/1/true/false, without nulls")
    return mapped.astype(bool)


def _expected_slots(day: date, timezone: str, interval_minutes: int) -> int:
    midnight = pd.Timestamp(day).tz_localize(timezone)
    tomorrow = midnight + pd.DateOffset(days=1)
    return int((tomorrow - midnight) / pd.Timedelta(minutes=interval_minutes))


def _episodes(daily: pd.DataFrame, threshold: float, station: str) -> list[dict]:
    records = []
    low = daily["eligible"] & daily[f"hours_lt{int(threshold)}"].gt(0)
    values = low.to_numpy(dtype=bool)
    for i in range(len(values)):
        if not values[i] or (i > 0 and values[i - 1]):
            continue
        end = i
        while end + 1 < len(values) and values[end + 1]:
            end += 1
        left_identifiable = i > 0 and bool(daily.iloc[i - 1]["eligible"])
        right_recovered = end + 1 < len(values) and bool(daily.iloc[end + 1]["eligible"])
        segment = daily.iloc[i : end + 1]
        records.append({
            "station": station,
            "threshold_mg_l": threshold,
            "start_date": str(segment.iloc[0]["date"]),
            "end_date": str(segment.iloc[-1]["date"]),
            "calendar_days": len(segment),
            "observed_hours_lower_bound": round(float(segment[f"hours_lt{int(threshold)}"].sum()), 4),
            "minimum_do_mg_l": round(float(segment["do_min_mg_l"].min()), 4),
            "identifiable_onset": left_identifiable,
            "observed_recovery": right_recovered,
            "censored_left": not left_identifiable,
            "censored_right": not right_recovered,
        })
    return records


def hypoxia_audit(
    df: pd.DataFrame,
    timestamp_col: str | None = None,
    do_col: str | None = None,
    station_col: str | None = None,
    observed_col: str | None = None,
    interval_minutes: int = 5,
    min_coverage: float = 0.95,
    timezone: str = "America/New_York",
    station: str | None = None,
) -> dict:
    """Audit complete local calendar days and descriptive daily hypoxia runs.

    When an *_observed flag accompanies a reconstructed DO value, ONLY flagged
    real observations enter this calculation. Unflagged raw uploads are usable
    but their measurement provenance is explicitly unverified.
    """
    if not isinstance(interval_minutes, int) or interval_minutes <= 0 or 1440 % interval_minutes:
        raise ValueError("interval_minutes must be a positive divisor of 1440")
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if df.empty:
        raise ValueError("Cannot audit an empty dataset")

    timestamp_col = _resolve(df, timestamp_col, TIME_COLUMNS, "timestamp")
    do_col = _resolve(df, do_col, DO_COLUMNS, "dissolved oxygen")
    if not timestamp_col or not do_col:
        raise ValueError("Identify timestamp_col and do_col (dissolved oxygen in mg/L)")
    explicit_station_col = station_col is not None
    station_col = _resolve(df, station_col, STATION_COLUMNS, "station")
    separate_sources = (
        not explicit_station_col and station_col not in (None, "source_file")
        and "source_file" in df and df["source_file"].nunique(dropna=False) > 1
    )
    auto_observed = (f"{do_col}_observed", "odo_mgL_observed")
    observed_col = _resolve(df, observed_col, auto_observed, "observation flag")
    if observed_col is None and ("imputed" in do_col.lower() or any(c.endswith("_observed") for c in df)):
        raise ValueError("Reconstructed/flagged data require the matching observed_col for dissolved oxygen")

    timestamps = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Missing or invalid timestamps; repair upstream, do not infer observation dates")
    try:
        local_dates = timestamps.dt.tz_convert(timezone).dt.date
    except Exception as exc:
        raise ValueError(f"Invalid timezone {timezone!r}: {exc}") from exc
    numeric = pd.to_numeric(df[do_col], errors="coerce").astype(float)
    observed = _observed_flags(df[observed_col]) if observed_col else numeric.notna()
    valid = observed & numeric.between(*DO_BOUNDS_MG_L) & np.isfinite(numeric)

    identity = df[station_col].astype("string") if station_col else "single_dataset"
    if separate_sources:
        # A node label reused by different exports is not proof of one continuous sensor.
        identity = df["source_file"].astype("string") + "::" + identity
    working = pd.DataFrame({
        "timestamp": timestamps, "local_date": local_dates,
        "station": identity,
        "do": numeric, "observed": observed, "valid": valid,
    })
    if working["station"].isna().any():
        raise ValueError("Station identity cannot be missing; do not combine unknown nodes")
    if station is not None:
        working = working.loc[working["station"] == str(station)]
        if working.empty:
            raise ValueError(f"Station {station!r} not found")
    all_daily: list[pd.DataFrame] = []
    summaries: list[dict] = []
    all_episodes: list[dict] = []

    for node, group in working.groupby("station", sort=True):
        node = str(node)
        group = group.sort_values("timestamp").copy()
        # Two measurements in one scheduled slot would artificially inflate coverage.
        slots = group["timestamp"].dt.floor(f"{interval_minutes}min")
        if slots.duplicated().any():
            raise ValueError(f"{node}: duplicate time slots at {interval_minutes}-minute resolution")
        by_date = {day: sub for day, sub in group.groupby("local_date", sort=True)}
        dates = pd.date_range(min(by_date), max(by_date), freq="D")
        rows = []
        for day_ts in dates:
            day = day_ts.date()
            sub = by_date.get(day)
            expected = _expected_slots(day, timezone, interval_minutes)
            good = sub.loc[sub["valid"], "do"] if sub is not None else pd.Series(dtype=float)
            coverage = len(good) / expected
            if coverage > 1:
                raise ValueError(f"{node} {day}: more valid readings than expected slots")
            eligible = bool(coverage >= min_coverage)
            rows.append({
                "station": node,
                "date": day.isoformat(),
                "eligible": eligible,
                "coverage": round(coverage, 6),
                "valid_slots": int(len(good)),
                "expected_slots": expected,
                "invalid_observed_slots": int((sub["observed"] & ~sub["valid"]).sum()) if sub is not None else 0,
                "do_min_mg_l": float(good.min()) if eligible else np.nan,
                "hours_lt2": float((good < 2.0).sum() * interval_minutes / 60) if eligible else np.nan,
                "hours_lt4": float((good < 4.0).sum() * interval_minutes / 60) if eligible else np.nan,
            })
        daily = pd.DataFrame(rows)
        all_daily.append(daily)
        for threshold in THRESHOLDS_MG_L:
            events = _episodes(daily, threshold, node)
            all_episodes.extend(events)
            observed_low_days = int((daily[f"hours_lt{int(threshold)}"] > 0).sum())
            summaries.append({
                "station": node,
                "threshold_mg_l": threshold,
                "date_start": daily.iloc[0]["date"],
                "date_end": daily.iloc[-1]["date"],
                "eligible_days": int(daily["eligible"].sum()),
                "unknown_days": int((~daily["eligible"]).sum()),
                "observed_low_do_days": observed_low_days,
                "observed_runs": len(events),
                "identifiable_onsets": sum(e["identifiable_onset"] for e in events),
                "observed_recoveries": sum(e["observed_recovery"] for e in events),
                "hours_below_lower_bound": round(float(daily[f"hours_lt{int(threshold)}"].sum()), 4),
                "invalid_observed_slots": int(daily["invalid_observed_slots"].sum()),
            })

    daily_all = pd.concat(all_daily, ignore_index=True).sort_values(["date", "station"])
    last_days = daily_all.tail(20).replace({np.nan: None}).to_dict("records")
    all_episodes.sort(key=lambda event: (event["start_date"], event["station"], event["threshold_mg_l"]))
    return {
        "operation": "hypoxia_audit",
        "status": "retrospective_descriptive_only_not_forecast",
        "provenance": "observed_flag_filtered" if observed_col else "unflagged_assumed_raw_unverified",
        "timestamp_col": timestamp_col,
        "do_col": do_col,
        "observed_col": observed_col,
        "station_col": "source_file+" + station_col if separate_sources else station_col,
        "timezone": timezone,
        "interval_minutes": interval_minutes,
        "min_coverage": min_coverage,
        "stations_analyzed": int(working["station"].nunique()),
        "n_daily_rows": len(daily_all),
        "station_summaries": summaries[:50],
        "summaries_truncated": len(summaries) > 50,
        "episodes": all_episodes[-25:],
        "episodes_truncated": len(all_episodes) > 25,
        "latest_daily_records": last_days,
        "interpretation": (
            "Days below 95% coverage (or configured threshold) are Unknown, never normal. "
            "Hypoxia means any measured value STRICTLY below 2/4 mg/L on an eligible day. "
            "Hours are observed lower bounds, not imputed exposure. Missing boundaries "
            "censor onset/recovery. Suspect zero oxygen is retained if in physical range. "
            "Timezone-based calendar days may contain 23/24/25 hours. "
            "Sensor provenance and station continuity must be independently verified."
        ),
    }
