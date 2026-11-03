"""Generic Fuzzy Cognitive Map (FCM), adapted from
https://github.com/MisterSpacebar/multinode-bay-imputation (fcm.py).

The original project's full spatio-temporal GNN (model.py) needs PyTorch and a
multi-stage training pipeline, so it isn't ported here. The FCM piece is a much
lighter weight (closed-form Ridge regression, pure numpy/pandas) causal model
that's well suited to interactive "what drives what" discussion and what-if
simulation, so this reimplements it generically over whatever numeric columns
are in the currently loaded dataset (rather than a fixed, curated concept list).

Learns a directed, weighted causal graph between chosen columns (`concepts`) via
a 1-step-lag Ridge regression, then can simulate forward from a baseline state,
optionally clamping some concepts to scenario values (e.g. "what if rainfall
were high") and observing how the rest evolve.

Caveat vs. the original: this operates on row order, not true elapsed time, so
it assumes the loaded data is already in roughly chronological order (true for
our concatenated station/month files) rather than resampling onto a fixed grid.
"""

import numpy as np
import pandas as pd

MAX_TRAJECTORY_ROWS = 60


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form Ridge: w = (X'X + alpha*I)^-1 X'y (no intercept)."""
    xtx = x.T @ x
    return np.linalg.solve(xtx + alpha * np.eye(x.shape[1]), x.T @ y)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def fcm(
    df: pd.DataFrame,
    columns: list[str],
    forcing: list[str] | None = None,
    lag: int = 1,
    alpha: float = 2.0,
    clamp: dict[str, float] | None = None,
    steps_ahead: int = 20,
) -> dict:
    """Fit an FCM over `columns` and, if `clamp` is given, simulate a scenario.

    `forcing` concepts (e.g. rainfall) are treated as exogenous -- no incoming
    edges -- and held at their baseline unless overridden in `clamp`.
    """
    forcing = forcing or []
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {missing}. Available: {list(df.columns)}")
    bad_forcing = [c for c in forcing if c not in columns]
    if bad_forcing:
        raise ValueError(f"`forcing` must be a subset of `columns`: {bad_forcing}")

    ordered = forcing + [c for c in columns if c not in forcing]
    n_forcing = len(forcing)
    n_concepts = len(ordered)
    if n_concepts < 2:
        raise ValueError("Need at least 2 columns to build an FCM")

    arr = df[ordered].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    n_rows = arr.shape[0]
    if n_rows < lag + 5:
        raise ValueError(f"Not enough rows ({n_rows}) for lag={lag}; need at least {lag + 5}")

    col_min = np.nanmin(arr, axis=0)
    col_max = np.nanmax(arr, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0
    arr_norm = (arr - col_min) / col_range

    col_means = np.nanmean(arr_norm, axis=0)
    col_means[np.isnan(col_means)] = 0.5
    nan_mask = np.isnan(arr_norm)
    arr_filled = arr_norm.copy()
    arr_filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    x_lag = arr_filled[:-lag]
    y_lag = arr_filled[lag:]

    w = np.zeros((n_concepts, n_concepts))
    for j in range(n_forcing, n_concepts):
        if nan_mask[lag:, j].mean() > 0.6:
            continue
        w[:, j] = _ridge(x_lag, y_lag[:, j], alpha)
    w[:, :n_forcing] = 0.0

    abs_max = np.abs(w).max()
    if abs_max > 0:
        w = w / abs_max

    edges = [
        {"from": ordered[i], "to": ordered[j], "weight": round(float(w[i, j]), 4)}
        for i in range(n_concepts)
        for j in range(n_concepts)
        if i != j and abs(w[i, j]) > 0.05
    ]
    edges.sort(key=lambda e: -abs(e["weight"]))

    result = {
        "operation": "fcm",
        "concepts": ordered,
        "forcing": forcing,
        "lag": lag,
        "n_rows_used": n_rows,
        "top_causal_edges": edges[:15],
    }

    if not clamp:
        return result

    baseline = arr_filled.mean(axis=0)
    clamped_idx: list[int] = []
    clamped_val: list[float] = []
    for name, value in clamp.items():
        if name not in ordered:
            raise ValueError(f"Unknown concept to clamp: {name!r}. Available: {ordered}")
        idx = ordered.index(name)
        clamped_idx.append(idx)
        clamped_val.append((value - col_min[idx]) / col_range[idx])
    for name in forcing:
        idx = ordered.index(name)
        if idx not in clamped_idx:
            clamped_idx.append(idx)
            clamped_val.append(baseline[idx])

    idx_arr = np.array(clamped_idx, dtype=int)
    val_arr = np.array(clamped_val, dtype=np.float64)

    a = baseline.copy()
    trajectory = [a.copy()]
    n_steps = min(steps_ahead, MAX_TRAJECTORY_ROWS)
    for _ in range(n_steps):
        a_new = _sigmoid(w.T @ a)
        if len(idx_arr):
            a_new[idx_arr] = val_arr
        a = a_new
        trajectory.append(a.copy())
    traj_real = np.stack(trajectory) * col_range + col_min

    result["clamped"] = clamp
    result["steps_simulated"] = n_steps
    result["final_values"] = {c: round(float(traj_real[-1, i]), 4) for i, c in enumerate(ordered)}
    result["trajectory_csv"] = pd.DataFrame(traj_real, columns=ordered).round(4).to_csv(index_label="step")
    return result
