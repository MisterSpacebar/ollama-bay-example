"""Built-in statistical "models" the chat tool can call for grounded datapoints:
imputation, trend/forecast, anomaly detection, correlation, and group summaries.
Pure pandas/numpy -- no extra ML dependencies.
"""

import inspect

import numpy as np
import pandas as pd

from backend.fcm import fcm as _fcm


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"Unknown column: {column!r}. Available: {list(df.columns)}")
    return pd.to_numeric(df[column], errors="coerce")


def impute(df: pd.DataFrame, column: str) -> dict:
    """Linear interpolation over row order; reports how many values were filled."""
    series = _numeric_column(df, column)
    missing_before = int(series.isna().sum())
    filled = series.interpolate(method="linear", limit_direction="both")
    missing_after = int(filled.isna().sum())
    return {
        "operation": "impute",
        "column": column,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "imputed_count": missing_before - missing_after,
        "imputed_mean": float(filled.mean()) if filled.notna().any() else None,
    }


def trend_forecast(df: pd.DataFrame, column: str, steps_ahead: int = 5) -> dict:
    """Simple linear trend over row order, extrapolated `steps_ahead` points forward."""
    series = _numeric_column(df, column).dropna()
    if len(series) < 2:
        raise ValueError(f"Not enough numeric values in {column!r} to fit a trend")
    x = np.arange(len(series), dtype=np.float64)
    y = series.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * x
    residuals = y - predicted
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    future_x = np.arange(len(series), len(series) + steps_ahead, dtype=np.float64)
    forecast = (intercept + slope * future_x).tolist()
    return {
        "operation": "trend_forecast",
        "column": column,
        "n_points": len(series),
        "slope_per_row": float(slope),
        "intercept": float(intercept),
        "r_squared": round(r_squared, 4),
        "forecast_next_values": [round(v, 4) for v in forecast],
    }


def anomalies(df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> dict:
    """Flags values whose z-score exceeds `z_threshold`."""
    series = _numeric_column(df, column)
    valid = series.dropna()
    if valid.empty or valid.std() == 0:
        return {"operation": "anomalies", "column": column, "count": 0, "examples": []}
    z = (valid - valid.mean()) / valid.std()
    flagged = valid[z.abs() > z_threshold]
    return {
        "operation": "anomalies",
        "column": column,
        "z_threshold": z_threshold,
        "count": int(flagged.shape[0]),
        "examples": [round(v, 4) for v in flagged.head(10).tolist()],
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
    }


def correlation(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Pearson correlation matrix across numeric columns (optionally restricted to `columns`,
    e.g. to get a single pairwise correlation, pass exactly the two column names you want)."""
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Unknown column(s): {missing}. Available: {list(df.columns)}")
        numeric_df = df[columns].apply(pd.to_numeric, errors="coerce")
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns for correlation")
    corr = numeric_df.corr(numeric_only=True).round(4)
    return {"operation": "correlation", "columns": list(numeric_df.columns), "matrix_csv": corr.to_csv()}


def group_summary(df: pd.DataFrame, column: str, group_by: str) -> dict:
    """Mean/std/min/max/count of `column`, grouped by `group_by`."""
    if group_by not in df.columns:
        raise ValueError(f"Unknown group_by column: {group_by!r}")
    series = _numeric_column(df, column)
    grouped = series.groupby(df[group_by]).agg(["mean", "std", "min", "max", "count"]).round(4)
    return {"operation": "group_summary", "column": column, "group_by": group_by, "table_csv": grouped.to_csv()}


def train_predict(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    new_rows: list[dict] | None = None,
) -> dict:
    """Fit an ordinary-least-squares multiple linear regression (`target` ~ `features`)
    on the dataset, report train/holdout fit quality, and optionally predict `new_rows`
    (list of {feature: value} dicts). This is a basic, transparent model -- not a neural
    net -- intended for quick "what's the relationship / rough prediction" questions.
    """
    missing = [c for c in [*features, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Unknown column(s): {missing}. Available: {list(df.columns)}")
    if len(features) == 0:
        raise ValueError("Need at least one feature column")

    numeric = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(numeric)
    min_rows = len(features) + 2
    if n < min_rows:
        raise ValueError(f"Only {n} complete rows after dropping missing values; need at least {min_rows}")

    X = numeric[features].to_numpy(dtype=np.float64)
    y = numeric[target].to_numpy(dtype=np.float64)
    X_design = np.column_stack([np.ones(n), X])

    # Deterministic holdout: last 20% of rows (by current order) as a test set.
    n_test = max(1, int(n * 0.2)) if n >= min_rows + 5 else 0
    n_train = n - n_test
    coeffs, *_ = np.linalg.lstsq(X_design[:n_train], y[:n_train], rcond=None)

    def _r2_mae(idx_slice: slice) -> tuple[float, float]:
        y_true = y[idx_slice]
        y_pred = X_design[idx_slice] @ coeffs
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(y_true - y_pred)))
        return round(r2, 4), round(mae, 4)

    train_r2, train_mae = _r2_mae(slice(0, n_train))
    test_r2, test_mae = (None, None) if n_test == 0 else _r2_mae(slice(n_train, n))

    intercept, *coef_values = coeffs.tolist()
    coefficients = dict(zip(features, coef_values))

    result = {
        "operation": "train_predict",
        "features": features,
        "target": target,
        "n_rows_used": n,
        "n_train": n_train,
        "n_test": n_test,
        "intercept": round(intercept, 6),
        "coefficients": {k: round(v, 6) for k, v in coefficients.items()},
        "train_r_squared": train_r2,
        "train_mae": train_mae,
        "test_r_squared": test_r2,
        "test_mae": test_mae,
    }

    if new_rows:
        predictions = []
        for row in new_rows:
            missing_feat = [f for f in features if f not in row]
            if missing_feat:
                raise ValueError(f"new_rows entry missing feature(s): {missing_feat}")
            x_row = np.array([1.0] + [float(row[f]) for f in features])
            predictions.append(round(float(x_row @ coeffs), 6))
        result["predictions"] = predictions

    return result


OPERATIONS = {
    "impute": impute,
    "trend_forecast": trend_forecast,
    "anomalies": anomalies,
    "correlation": correlation,
    "group_summary": group_summary,
    "train_predict": train_predict,
    "fcm": _fcm,
}


def run_data_model(df: pd.DataFrame, operation: str, **kwargs) -> dict:
    func = OPERATIONS.get(operation)
    if func is None:
        raise ValueError(f"Unknown operation: {operation!r}. Available: {list(OPERATIONS)}")
    # Drop arguments this operation doesn't accept (e.g. a stray 'target' meant for
    # train_predict) instead of crashing -- small models sometimes mix up parameters.
    accepted = set(inspect.signature(func).parameters) - {"df"}
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    try:
        return func(df, **filtered)
    except TypeError as e:
        raise TypeError(f"{operation} call failed ({e}). Accepted arguments: {sorted(accepted)}") from e
