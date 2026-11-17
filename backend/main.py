import asyncio
import base64
import builtins as _builtins_module
import io
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from ollama import AsyncClient
from pydantic import BaseModel

from backend.processing import clean_dataframe
from backend.modeling import run_data_model
from backend.charts import make_chart, prepare_native_chart_data
from backend.fcm_safe import safe_fcm as fit_fcm
from backend.trustworthy_bay import hypoxia_audit
from backend.research import query_research, research_catalog

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "granite4.1:8b")
# Default context window is way smaller than what multi-dataset prompts + tool-call
# history actually need -- without raising this, older messages (even the original
# question) can silently fall out of context after a couple of tool rounds. The model
# supports up to 131072 tokens; 16384 is a moderate step up that still fits on modest GPUs.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))  # allow longer final answers
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))  # more focused/consistent analysis
OLLAMA_CHAT_OPTIONS = {
    "num_ctx": OLLAMA_NUM_CTX,
    "num_predict": OLLAMA_NUM_PREDICT,
    "temperature": OLLAMA_TEMPERATURE,
}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SAMPLE_ROWS = 20
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "water_data")).resolve()
DATA_EXTS = (".csv", ".xlsx", ".xlsm", ".json")

app = FastAPI(title="Ollama Data Chat")
client = AsyncClient(host=OLLAMA_HOST)

# In-memory stores: dataset_id -> DataFrame, and group_id -> {name: DataFrame} for
# "ask about everything" style questions spanning multiple loaded datasets.
datasets: dict[str, pd.DataFrame] = {}
dataset_groups: dict[str, dict[str, pd.DataFrame]] = {}
# Precomputed-once FCM summaries (causal graph), so chat doesn't need to refit it
# on every question -- keyed the same way as `datasets` / `dataset_groups` values.
dataset_fcm: dict[str, dict | None] = {}
group_fcm: dict[str, dict[str, dict | None]] = {}

_FCM_EXCLUDE_NAME_PATTERN = re.compile(r"lat|lon|^id$|_id$|^year$|^month$|^day$|^record_number$", re.IGNORECASE)
FCM_MAX_CONCEPTS = 12


def compute_fcm_summary(df: pd.DataFrame) -> dict | None:
    """Best-effort, auto-column FCM fit for use as free grounding context.
    Returns None (never raises) if the dataset isn't suitable (too few numeric
    columns, too few rows, degenerate data, etc.)."""
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if not _FCM_EXCLUDE_NAME_PATTERN.search(c)]
    numeric_cols = numeric_cols[:FCM_MAX_CONCEPTS]
    if len(numeric_cols) < 2:
        return None
    try:
        result = fit_fcm(df, columns=numeric_cols)
    except Exception:
        return None
    return {"concepts": result["concepts"], "top_causal_edges": result["top_causal_edges"], "interpretation": result["interpretation"]}


def fcm_summary_to_text(summary: dict | None) -> str:
    if not summary or not summary["top_causal_edges"]:
        return ""
    lines = [
        f"- {e['from']} -> {e['to']} (lagged association {e['weight']:+.2f})"
        for e in summary["top_causal_edges"][:8]
    ]
    return "Exploratory, noncausal lagged associations (single station, regular time axis):\n" + "\n".join(lines)


CSV_ENCODINGS = ("utf-8", "cp1252", "latin-1")


def _read_csv_any_encoding(source) -> pd.DataFrame:
    """Try a few common encodings since CSV exports (e.g. from Excel) are often not UTF-8."""
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e
    raise last_error


def read_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(raw)
    if ext == ".csv":
        return _read_csv_any_encoding(buf)
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(buf)
    if ext == ".json":
        return pd.read_json(buf)
    raise HTTPException(400, "Unsupported file type. Use .csv, .xlsx, .xlsm or .json")


def read_dataframe_path(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return _read_csv_any_encoding(path)
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    raise HTTPException(400, "Unsupported file type. Use .csv, .xlsx, .xlsm or .json")


def resolve_in_data_dir(rel_path: str) -> Path:
    """Resolve a path relative to DATA_DIR, rejecting any escape via '..' or symlinks."""
    candidate = (DATA_DIR / rel_path).resolve()
    if candidate != DATA_DIR and DATA_DIR not in candidate.parents:
        raise HTTPException(400, "Invalid path")
    if not candidate.exists():
        raise HTTPException(404, "Path not found")
    return candidate


def load_group_dataframe(target: Path) -> pd.DataFrame:
    """Load a single data file, or concatenate every data file in a directory."""
    if target.is_dir():
        files = sorted(p for p in target.iterdir() if p.suffix.lower() in DATA_EXTS)
        if not files:
            raise HTTPException(400, "No data files in that directory")
        frames = []
        for f in files:
            try:
                frame = read_dataframe_path(f)
            except Exception as e:
                raise HTTPException(400, f"Could not parse {f.name}: {e}")
            frame["source_file"] = f.name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False)
    try:
        return read_dataframe_path(target)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")


# --- pandas tool-calling: lets the model run real code for exact answers ---
TOOL_TIMEOUT_SECONDS = 5
MAX_TOOL_ROUNDS = 6
# Seconds of silence before /api/chat sends an SSE keep-alive comment while the model is busy.
CHAT_HEARTBEAT_SECONDS = float(os.getenv("CHAT_HEARTBEAT_SECONDS", "10"))
_pandas_executor = ThreadPoolExecutor(max_workers=2)
# Best-effort denylist; exec() with a small builtins allowlist is the real boundary but
# isn't a watertight sandbox, so this only runs against the user's own local data/model.
# Imports aren't blocked at the text level anymore (the model reflexively writes
# 'import pandas as pd'/combined imports out of habit, in all sorts of phrasings that are
# hard to pattern-match) -- instead a restricted `__import__` (see `_safe_import` below)
# actually enforces a small safe-module allowlist, so any import attempt either succeeds
# harmlessly or fails with a clear ImportError.
_UNSAFE_CODE_PATTERN = re.compile(r"__|open\s*\(|exec\s*\(|eval\s*\(|os\.|sys\.|subprocess")
# Safe, side-effect-free builtins only -- no open/eval/exec/compile/getattr.
_SAFE_BUILTIN_NAMES = (
    "len", "range", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "min", "max", "sum", "abs", "round", "all", "any",
    "list", "dict", "set", "tuple", "str", "int", "float", "bool",
)
_SAFE_BUILTINS = {name: getattr(_builtins_module, name) for name in _SAFE_BUILTIN_NAMES}

# A small allowlist of safe, read-only stdlib/third-party modules the model is allowed to
# import in sandboxed code (pandas/numpy are already provided as `pd`/`np` anyway, but the
# model often writes `import pandas as pd` out of habit -- letting that actually succeed is
# simpler and more robust than trying to pattern-match every phrasing of a harmless import).
_ALLOWED_IMPORT_MODULES = {
    "pandas", "numpy", "math", "re", "statistics", "itertools", "collections", "datetime", "io",
}


def _safe_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _ALLOWED_IMPORT_MODULES:
        raise ImportError(
            f"Importing {name!r} isn't allowed here. Only these modules are permitted: "
            f"{sorted(_ALLOWED_IMPORT_MODULES)} -- pandas/numpy are already available as "
            "`pd`/`np` without even needing to import them."
        )
    return _builtins_module.__import__(name, *args, **kwargs)


_SAFE_BUILTINS["__import__"] = _safe_import

PANDAS_TOOL = {
    "type": "function",
    "function": {
        "name": "run_pandas_code",
        "description": (
            "Run pandas code against the full dataset(s) to compute exact answers (sums, "
            "counts, averages, filters, groupby, min/max, date ranges, etc.) instead of "
            "guessing from the sample. If only one dataset is loaded, it's available as "
            "`df`. If MULTIPLE datasets are loaded (the common case), `df` is None -- you "
            "MUST use `dfs['name']` instead (see the exact dataset names listed above), e.g. "
            "`dfs['water_stations_2026/north_bay_village_north']`. Assign the final answer "
            "to a variable named `result`. "
            "`pd`, `np`, `df`, and `dfs` are ALREADY loaded in-memory for you with the real "
            "data -- no need to `import pandas`/`import numpy` (harmless if you do, but "
            "pointless) or `pd.read_csv(...)`/`open(...)` any file (there IS no file to "
            "read -- that will error). NEVER type out fake/sample/placeholder rows yourself "
            "(e.g. a hand-written CSV string via `io.StringIO`) to work around missing data "
            "-- always use the real `df`/`dfs[name]` that's already loaded; if a column you "
            "need isn't there, that's a real error to report, not something to fake your way "
            "around. "
            "`result` should be a reduced/aggregated answer (a number, short dict, small "
            "table, etc.) -- NOT the raw dataframe or a big slice of raw rows; if you just "
            "want to inspect something, assign a small reduction (e.g. `.head()`, "
            "`.columns.tolist()`, `.describe()`, `.groupby(...).mean()`) rather than the "
            "whole dataset, since large raw dumps don't help answer the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code using `df`/`dfs` and `pd` that sets a `result` variable.",
                }
            },
            "required": ["code"],
        },
    },
}


def _strip_comment_lines(code: str) -> str:
    """Drop full-line '#' comments before the denylist check, so a harmless
    '# import pandas for aggregation' style comment doesn't false-positive."""
    return "\n".join(line for line in code.splitlines() if not line.strip().startswith("#"))


def _execute_pandas_code(dfs: dict[str, pd.DataFrame], code: str) -> tuple[str, dict | None]:
    if _UNSAFE_CODE_PATTERN.search(_strip_comment_lines(code)):
        return (
            "Error: this code uses a file/dunder/system construct that isn't allowed here "
            "-- but you almost certainly don't need one. `pd` (pandas), `np` (numpy), `df` "
            "(if one dataset) and `dfs` (if multiple) are ALREADY loaded with the real data "
            "-- there's no need to `pd.read_csv(...)`/`open(...)` any file (there's no file "
            "to read); just reference `pd`/`np`/`df`/`dfs[name]` directly. Remove any "
            "`open(`/`exec(`/`eval(` lines and retry with only plain pandas/numpy "
            "operations on the existing `df`/`dfs` variable.",
            None,
        )
    single_df = next(iter(dfs.values())) if len(dfs) == 1 else None
    sandbox_globals = {"__builtins__": _SAFE_BUILTINS, "pd": pd, "np": np, "df": single_df, "dfs": dfs}
    sandbox_locals: dict = {}
    try:
        exec(code, sandbox_globals, sandbox_locals)
    except FileNotFoundError as e:
        names = ", ".join(repr(n) for n in dfs.keys())
        return (
            f"Error executing code: {e} -- there's no file to read; the real data is "
            f"already loaded in memory as `df`/`dfs[name]`, not on disk. Available dataset "
            f"names: {names}. Retry using `df`/`dfs[...]` directly instead of `pd.read_csv`/`open`.",
            None,
        )
    except Exception as e:
        if single_df is None and "NoneType" in str(e):
            names = ", ".join(repr(n) for n in dfs.keys())
            return (
                f"Error executing code: {e} -- this is because multiple datasets are "
                f"loaded, so `df` is None. Use `dfs[name]` instead, e.g. "
                f"dfs[{next(iter(dfs.keys()))!r}]. Available dataset names: {names}. "
                "Retry with the correct `dfs[...]` lookup.",
                None,
            )
        if "no longer supported for offsets" in str(e):
            return (
                f"Error executing code: {e} -- this pandas version renamed some resample/"
                "offset aliases. Use 'ME' instead of 'M' (month-end), 'YE' instead of 'Y' "
                "(year-end), and 'h'/'min'/'s' (lowercase) instead of 'H'/'T'/'S'. Retry "
                "with the updated alias.",
                None,
            )
        return (f"Error executing code: {e}", None)
    result = sandbox_locals.get("result", sandbox_globals.get("result"))
    if result is None:
        return ("Error: code did not assign a `result` variable", None)
    return (_summarize_tool_result(result), _dataframe_to_display(result))


_MAX_RESULT_ROWS = 20
_MAX_RESULT_CHARS = 1500
_MAX_DISPLAY_ROWS = 500


def _dataframe_to_display(result) -> dict | None:
    """Build a small JSON-records table for the frontend to render inline, completely
    separate from the (smaller/summarized) text that gets fed back to the model."""
    if isinstance(result, pd.Series):
        result = result.rename(result.name or "value").reset_index()
    if not isinstance(result, pd.DataFrame) or result.shape[0] == 0 or result.shape[1] == 0:
        return None
    shown = result.head(_MAX_DISPLAY_ROWS)
    return {
        "columns": [str(c) for c in shown.columns],
        "rows": json.loads(shown.to_json(orient="records", date_format="iso")),
        "total_rows": len(result),
        "truncated": len(result) > _MAX_DISPLAY_ROWS,
    }


def _tabular_from_model_result(result: dict) -> dict | None:
    """run_data_model operations sometimes embed a CSV blob (group_summary/correlation) or a
    list of row-dicts (fcm's top_causal_edges) -- surface the first one found as a table."""
    for key, value in result.items():
        if key.endswith("_csv") and isinstance(value, str) and value.strip():
            try:
                return _dataframe_to_display(pd.read_csv(io.StringIO(value)))
            except Exception:
                continue
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            try:
                return _dataframe_to_display(pd.DataFrame(value))
            except Exception:
                continue
    return None


def _summarize_tool_result(result) -> str:
    """Render a tool result compactly so a big raw dataframe/series can't flood the model's
    context -- if the result is just raw unreduced rows, nudge it to aggregate instead."""
    if isinstance(result, (pd.DataFrame, pd.Series)):
        n_rows = len(result)
        truncated_note = ""
        shown = result
        if n_rows > _MAX_RESULT_ROWS:
            shown = result.head(_MAX_RESULT_ROWS)
            truncated_note = (
                f"\n(Showing first {_MAX_RESULT_ROWS} of {n_rows} rows -- if you need "
                "overall numbers rather than individual rows, aggregate first, e.g. "
                "`.mean()`, `.sum()`, `.groupby(...).agg(...)`, or `.describe()`, and "
                "assign THAT to `result` instead of the raw rows.)"
            )
        text = shown.to_csv()
        return (text[:_MAX_RESULT_CHARS] + truncated_note)[:_MAX_RESULT_CHARS + 400]
    text = str(result)
    return text[:_MAX_RESULT_CHARS]


async def run_pandas_tool(dfs: dict[str, pd.DataFrame], code: str) -> tuple[str, dict | None]:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_pandas_executor, _execute_pandas_code, dfs, code),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return (f"Error: code execution timed out ({TOOL_TIMEOUT_SECONDS}s limit)", None)


# --- built-in statistical models: grounded datapoints for imputation/trend/anomalies ---
MODEL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_data_model",
        "description": (
            "Run a built-in statistical model over the FULL data of ONE dataset to get "
            "grounded datapoints instead of guessing. This only works within a SINGLE "
            "dataset at a time -- to combine/compare columns across two different loaded "
            "datasets, use `run_pandas_code` with the `dfs` dict instead (e.g. merge/align "
            "them yourself, then call `.corr()`). Operations: "
            "'impute' (linear-interpolate missing values in `column`, reports how many filled), "
            "'trend_forecast' (linear trend over the full timeline of `column`, plus next "
            "`steps_ahead` forecast values -- THIS is the one to reach for when a question is "
            "about whether something changed, improved, or worsened over time, e.g. 'has "
            "dissolved oxygen gotten better since 2020' or 'is X on an upward/downward "
            "trend' -- the sign/size of the slope is your actual evidence), "
            "'anomalies' (z-score outliers in `column` above `z_threshold`), "
            "'correlation' (Pearson correlation matrix; restrict to specific columns with "
            "`columns`, e.g. a 2-item list for a single pairwise correlation -- otherwise "
            "uses all numeric columns in the dataset; NOTE this only measures how two "
            "variables move together across all rows, it says nothing about change over "
            "time -- don't use it to answer 'has this improved/changed over time' questions, "
            "use 'trend_forecast' or a before/after `group_summary`/`run_pandas_code` "
            "comparison for those instead), "
            "'group_summary' (mean/std/min/max/count of `column` grouped by `group_by` -- "
            "useful for a direct before/after comparison, e.g. group by a "
            "year-bucket/period column to see how the average changed across periods), "
            "'train_predict' (fit a linear regression of `target` on `features`, report "
            "fit quality, and optionally predict for `new_rows` -- a list of "
            "{feature_name: value} objects), "
            "'fcm' (Fuzzy Cognitive Map: fits a directed, weighted lagged-association graph between "
            "`columns`, e.g. temperature/salinity/dissolved-oxygen, so you can discuss what "
            "drives what in `top_causal_edges`. Pass `forcing` for exogenous drivers like "
            "rainfall that only have outgoing edges. Pass `clamp` -- a {column: value} dict "
            "of scenario values -- to additionally simulate `steps_ahead` steps forward and "
            "see the resulting `final_values` for an exploratory what-if discussion; "
            "this is NOT a causal estimate or validated forecast). "
            "'hypoxia_audit' (coverage-qualified, observed-only DO episodes at 2 and "
            "4 mg/L; specify timestamp_col and do_col in mg/L, and optionally "
            "station_col, observed_col, interval_minutes, min_coverage, timezone, station)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "impute", "trend_forecast", "anomalies", "correlation",
                        "group_summary", "train_predict", "fcm", "hypoxia_audit",
                    ],
                },
                "column": {"type": "string", "description": "Target numeric column."},
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of numeric columns (correlation: to correlate; fcm: the concepts).",
                },
                "forcing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exogenous concepts with no incoming edges, e.g. rainfall (fcm only).",
                },
                "clamp": {
                    "type": "object",
                    "description": "{column: scenario_value} to simulate a what-if forward (fcm only).",
                },
                "group_by": {"type": "string", "description": "Categorical column (group_summary only)."},
                "steps_ahead": {"type": "integer", "description": "Forecast horizon (trend_forecast, fcm)."},
                "z_threshold": {"type": "number", "description": "Outlier threshold (anomalies only)."},
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Feature column names (train_predict only).",
                },
                "target": {"type": "string", "description": "Column to predict (train_predict only)."},
                "new_rows": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of {feature_name: value} objects to predict (train_predict only).",
                },
                "timestamp_col": {"type": "string", "description": "Time column for fcm/hypoxia_audit."},
                "do_col": {"type": "string", "description": "Dissolved oxygen concentration column in mg/L (hypoxia_audit)."},
                "station_col": {"type": "string", "description": "Station/source identity column (fcm/hypoxia_audit)."},
                "observed_col": {"type": "string", "description": "Boolean observation provenance column (hypoxia_audit)."},
                "interval_minutes": {"type": "integer", "description": "Sampling interval in minutes (hypoxia_audit)."},
                "min_coverage": {"type": "number", "description": "Required valid coverage per day, default 0.95."},
                "timezone": {"type": "string", "description": "Timezone for local daily labels (hypoxia_audit)."},
                "station": {"type": "string", "description": "Optional station to inspect (hypoxia_audit)."},
                "dataset": {
                    "type": "string",
                    "description": "Which dataset name to use, only needed when multiple datasets are loaded.",
                },
            },
            "required": ["operation"],
        },
    },
}


def _select_df(dfs: dict[str, pd.DataFrame], dataset_name: str | None) -> pd.DataFrame:
    # With only one dataset loaded there's no ambiguity, so ignore a wrong/hallucinated name.
    if len(dfs) == 1:
        return next(iter(dfs.values()))
    if dataset_name:
        if dataset_name not in dfs:
            raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {list(dfs)}")
        return dfs[dataset_name]
    raise ValueError(f"Multiple datasets loaded; specify `dataset` from: {list(dfs)}")


def _collect_requested_columns(kwargs: dict) -> list[str]:
    """Pull out every string that looks like it's meant to be a column name, across
    whichever operation-specific kwargs were passed (column, columns, group_by, etc.)."""
    names: list[str] = []
    for key in ("column", "group_by", "target"):
        value = kwargs.get(key)
        if isinstance(value, str):
            names.append(value)
    for key in ("columns", "features", "forcing"):
        value = kwargs.get(key)
        if isinstance(value, list):
            names.extend(v for v in value if isinstance(v, str))
    clamp = kwargs.get("clamp")
    if isinstance(clamp, dict):
        names.extend(clamp.keys())
    for row in kwargs.get("new_rows") or []:
        if isinstance(row, dict):
            names.extend(row.keys())
    return names


def _cross_dataset_column_hint(dfs: dict[str, pd.DataFrame], dataset_name: str | None, df: pd.DataFrame, kwargs: dict) -> str | None:
    """If the model mixed up real column names from a DIFFERENT loaded dataset (a common
    mistake with 'Load ALL'), say exactly which dataset they belong to instead of a plain
    'unknown column' error -- this is usually not hallucination, just wrong `dataset`."""
    if len(dfs) < 2:
        return None
    missing = [c for c in _collect_requested_columns(kwargs) if c not in df.columns]
    if not missing:
        return None
    hints = {
        name: found
        for name, other_df in dfs.items()
        if name != dataset_name and (found := [c for c in missing if c in other_df.columns])
    }
    if not hints:
        return None
    hint_text = "; ".join(f"{cols} belong to dataset {name!r}" for name, cols in hints.items())
    return f"Error: column(s) {missing} not found in dataset {dataset_name!r}. {hint_text}. Retry with that `dataset` argument instead."


def _execute_data_model(dfs: dict[str, pd.DataFrame], kwargs: dict) -> tuple[str, dict | None]:
    operation = kwargs.pop("operation", None)
    dataset_name = kwargs.pop("dataset", None)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        df = _select_df(dfs, dataset_name)
        hint = _cross_dataset_column_hint(dfs, dataset_name, df, kwargs)
        if hint:
            return (hint, None)
        result = run_data_model(df, operation, **kwargs)
    except Exception as e:
        return (f"Error running model: {e}", None)
    text = json.dumps(result)[:6000]
    display = _tabular_from_model_result(result) if isinstance(result, dict) else None
    return (text, display)


async def run_model_tool(dfs: dict[str, pd.DataFrame], kwargs: dict) -> tuple[str, dict | None]:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_pandas_executor, _execute_data_model, dfs, kwargs),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return (f"Error: model execution timed out ({TOOL_TIMEOUT_SECONDS}s limit)", None)


CHART_TOOL = {
    "type": "function",
    "function": {
        "name": "make_chart",
        "description": (
            "Render an actual interactive chart from ONE dataset and show it directly to "
            "the user in the chat (hover for exact values, zoom/pan) -- reach for this "
            "whenever a picture would genuinely help (e.g. 'show me the trend', 'plot X "
            "over time', 'what does the distribution look like', 'chart X vs Y'), not just "
            "for a numeric answer you'd get from run_pandas_code/run_data_model. "
            "'line': y column(s) plotted over x (or row order if no x) -- good for trends "
            "over time, optionally split into one line per `group_by` value; "
            "'scatter': y[0] vs x -- good for the relationship between two variables "
            "(requires x), optionally colored by `group_by`; "
            "'bar': mean of y[0] grouped by x (or a single bar if no x), optionally split "
            "by `group_by`; "
            "'hist': distribution of y[0] (this one is a static image, not interactive). "
            "The chart is shown to the user automatically -- you don't need to describe "
            "every value back to them, just briefly say what it shows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": ["line", "scatter", "bar", "hist"]},
                "y": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column(s) to plot on the y-axis (just one for scatter/hist).",
                },
                "x": {"type": "string", "description": "Optional x-axis column (required for scatter)."},
                "group_by": {"type": "string", "description": "Optional categorical column to split/color line, scatter, or bar charts by."},
                "dataset": {
                    "type": "string",
                    "description": "Which dataset name to use, only needed when multiple datasets are loaded.",
                },
            },
            "required": ["chart_type", "y"],
        },
    },
}


def _execute_make_chart(dfs: dict[str, pd.DataFrame], kwargs: dict) -> tuple[str, dict | None]:
    dataset_name = kwargs.get("dataset")
    chart_type = kwargs.get("chart_type")
    y = kwargs.get("y") or []
    if isinstance(y, str):  # model sometimes passes a bare string instead of a 1-item list
        y = [y]
    x = kwargs.get("x")
    group_by = kwargs.get("group_by")
    try:
        df = _select_df(dfs, dataset_name)
        fake_kwargs = {"columns": [c for c in [*y, x] if c], "group_by": group_by}
        hint = _cross_dataset_column_hint(dfs, dataset_name, df, fake_kwargs)
        if hint:
            return (hint, None)
        if chart_type == "hist":
            png_bytes = make_chart(df, chart_type, y, x, group_by)
        else:
            plot_df = prepare_native_chart_data(df, chart_type, y, x, group_by)
    except KeyError as e:
        return (
            f"Error generating chart: column {e} not found. `y` must be a list of exact "
            f"column name(s) from the dataset, e.g. y=[{list(df.columns)[0]!r}], not a "
            "single bare string or a mistyped name.",
            None,
        )
    except Exception as e:
        return (f"Error generating chart: {e}", None)
    caption = f"{chart_type} chart of {', '.join(y)}"
    if x:
        caption += f" vs {x}"
    if group_by:
        caption += f", grouped by {group_by}"
    text = f"Chart generated and already shown to the user: {caption}."
    if chart_type == "hist":
        image_b64 = base64.b64encode(png_bytes).decode("ascii")
        return (text, {"image_base64": image_b64, "caption": caption})
    rows = json.loads(plot_df.to_json(orient="records", date_format="iso"))
    return (
        text,
        {
            "chart_type": chart_type,
            "x": x,
            "y": y,
            "group_by": group_by,
            "rows": rows,
            "caption": caption,
        },
    )


async def run_chart_tool(dfs: dict[str, pd.DataFrame], kwargs: dict) -> tuple[str, dict | None]:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_pandas_executor, _execute_make_chart, dfs, kwargs),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return (f"Error: chart generation timed out ({TOOL_TIMEOUT_SECONDS}s limit)", None)


MAP_TOOL = {
    "type": "function",
    "function": {
        "name": "make_map",
        "description": (
            "Plot real geographic locations (station positions, sample sites, sensor "
            "coordinates, etc.) on an actual interactive map with real coastlines/streets -- "
            "reach for this instead of make_chart whenever the question is literally about "
            "WHERE something is located (e.g. 'show me where the stations are', 'map the "
            "sensor locations', 'where was this sample taken'), not just an abstract x/y "
            "relationship. Needs latitude and longitude column names from the dataset (look "
            "for names like 'latitude'/'longitude', 'lat'/'lon', 'gps.lat'/'gps.lon'). If "
            "the dataset has a station/platform identifier column (e.g. 'platform_id', "
            "'Station ID', 'Station name'), ALWAYS pass it as `label_col` -- this is how "
            "distinct real stations get told apart from GPS noise; without it, many rows "
            "from the SAME stationary sensor (with tiny meter-scale GPS jitter) could look "
            "like several different locations. The map is shown to the user automatically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "lat_col": {"type": "string", "description": "Exact latitude column name."},
                "lon_col": {"type": "string", "description": "Exact longitude column name."},
                "label_col": {
                    "type": "string",
                    "description": "Optional column to label each point with (e.g. station/platform name).",
                },
                "dataset": {
                    "type": "string",
                    "description": "Which dataset name to use, only needed when multiple datasets are loaded.",
                },
            },
            "required": ["lat_col", "lon_col"],
        },
    },
}

_MAX_MAP_POINTS = 500


def _execute_make_map(dfs: dict[str, pd.DataFrame], kwargs: dict) -> tuple[str, dict | None]:
    dataset_name = kwargs.get("dataset")
    lat_col = kwargs.get("lat_col")
    lon_col = kwargs.get("lon_col")
    label_col = kwargs.get("label_col")
    try:
        df = _select_df(dfs, dataset_name)
        fake_kwargs = {"columns": [c for c in [lat_col, lon_col, label_col] if c]}
        hint = _cross_dataset_column_hint(dfs, dataset_name, df, fake_kwargs)
        if hint:
            return (hint, None)
        cols = [lat_col, lon_col] + ([label_col] if label_col else [])
        points = df[cols].copy()
        points[lat_col] = pd.to_numeric(points[lat_col], errors="coerce")
        points[lon_col] = pd.to_numeric(points[lon_col], errors="coerce")
        points = points.dropna(subset=[lat_col, lon_col])
        # Drop physically-impossible coordinates (sensor glitches/placeholder values, e.g.
        # a longitude of 238 instead of a real -180..180 value) -- a single bad point blows
        # out the map's auto-zoom bounding box and makes the whole map useless, regardless of
        # whether general dataset cleaning was requested at load time.
        n_before_range_filter = len(points)
        points = points[points[lat_col].between(-90, 90) & points[lon_col].between(-180, 180)]
        # A corrupted value can still fall inside a technically-valid lat/lon range (e.g. a
        # sign flip or digit swap landing on another continent) -- real station locations
        # cluster together, so also drop anything far (>5 degrees, ~550km) from the median
        # of the remaining points, which legitimate same-region stations won't exceed.
        if len(points) > 0:
            median_lat, median_lon = points[lat_col].median(), points[lon_col].median()
            points = points[
                (points[lat_col] - median_lat).abs().le(5) & (points[lon_col] - median_lon).abs().le(5)
            ]
        n_dropped_invalid = n_before_range_filter - len(points)
        # Stationary platforms log many readings with tiny GPS jitter (meter-scale noise)
        # rather than truly distinct locations -- collapse per real site instead of per raw
        # reading. If a label (e.g. platform/station ID) is given, that's the real identity:
        # group by it and take the median position. Otherwise fall back to coarse rounding
        # (~110m) so natural GPS noise merges into one point instead of looking like many.
        if label_col and label_col in points.columns:
            points = points.groupby(label_col, as_index=False)[[lat_col, lon_col]].median()
        else:
            points["_lat_r"] = points[lat_col].round(3)
            points["_lon_r"] = points[lon_col].round(3)
            points = points.drop_duplicates(subset=["_lat_r", "_lon_r"]).drop(columns=["_lat_r", "_lon_r"])
    except KeyError as e:
        return (
            f"Error generating map: column {e} not found. Check the exact lat/lon column "
            f"names for this dataset (available columns: {list(df.columns)}).",
            None,
        )
    except Exception as e:
        return (f"Error generating map: {e}", None)
    if points.empty:
        return ("Error generating map: no valid (non-missing, numeric) lat/lon rows found.", None)
    total_points = len(points)
    truncated = total_points > _MAX_MAP_POINTS
    if truncated:
        points = points.head(_MAX_MAP_POINTS)
    rows = json.loads(points.to_json(orient="records"))
    text = f"Map generated and already shown to the user: {total_points} unique location(s) plotted."
    if n_dropped_invalid:
        text += f" ({n_dropped_invalid} row(s) had out-of-range/invalid coordinates and were excluded.)"
    display = {
        "lat_col": lat_col,
        "lon_col": lon_col,
        "label_col": label_col,
        "rows": rows,
        "total_points": total_points,
        "truncated": truncated,
    }
    return (text, display)


async def run_map_tool(dfs: dict[str, pd.DataFrame], kwargs: dict) -> tuple[str, dict | None]:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_pandas_executor, _execute_make_map, dfs, kwargs),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return (f"Error: map generation timed out ({TOOL_TIMEOUT_SECONDS}s limit)", None)


def to_records(df: pd.DataFrame) -> list[dict]:
    # to_json handles NaN/NaT/numpy types safely
    return json.loads(df.to_json(orient="records", date_format="iso"))


def build_context(dfs: dict[str, pd.DataFrame], fcm_summaries: dict[str, dict | None] | None = None) -> str:
    fcm_summaries = fcm_summaries or {}
    if len(dfs) == 1:
        name, df = next(iter(dfs.items()))
        schema = "\n".join(f"- {c}: {t}" for c, t in df.dtypes.astype(str).items())
        stats = df.describe(include="all").transpose().to_csv()
        sample = df.head(SAMPLE_ROWS).to_csv(index=False)
        fcm_text = fcm_summary_to_text(fcm_summaries.get(name))
        return (
            f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
            f"Columns:\n{schema}\n\n"
            f"Summary statistics (CSV):\n{stats}\n"
            f"First {min(SAMPLE_ROWS, len(df))} rows (CSV):\n{sample}"
            + (f"\n\n{fcm_text}" if fcm_text else "")
        )

    names = list(dfs.keys())
    column_to_datasets: dict[str, list[str]] = {}
    for name, df in dfs.items():
        for c in df.columns:
            column_to_datasets.setdefault(c, []).append(name)
    index_lines = "\n".join(
        f"- {col!r} -> {', '.join(owners)!r}" if len(owners) == 1 else f"- {col!r} -> {owners}"
        for col, owners in sorted(column_to_datasets.items())
    )
    parts = [
        f"{len(dfs)} datasets are loaded. Each tool can operate on the FULL data for any of them.\n"
        f"Available dataset names (use EXACTLY these strings, nothing else): {names}\n\n"
        "COLUMN -> DATASET INDEX (look a column up here FIRST instead of guessing/searching "
        f"across datasets):\n{index_lines}"
    ]
    for name, df in dfs.items():
        fcm_text = fcm_summary_to_text(fcm_summaries.get(name))
        parts.append(
            f"\n### Dataset '{name}' -- {len(df)} rows\n"
            f"Columns: {', '.join(df.columns)}\n"
            f"First 3 rows (CSV):\n{df.head(3).to_csv(index=False)}"
            + (f"\n{fcm_text}" if fcm_text else "")
        )
    return "\n".join(parts)


@app.get("/api/health")
async def health():
    try:
        await client.show(MODEL)
        return {"ollama": "ok", "model": MODEL}
    except Exception as e:
        raise HTTPException(503, f"Ollama/model '{MODEL}' unavailable: {e}")


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    clean: bool = Form(False),
):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        df = read_dataframe(file.filename or "", raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")

    cleaning_report: list[dict] = []
    if clean:
        df, cleaning_report = clean_dataframe(df)

    dataset_id = uuid.uuid4().hex
    datasets[dataset_id] = df
    fcm_summary = compute_fcm_summary(df)
    dataset_fcm[dataset_id] = fcm_summary
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
        "preview": to_records(df.head(50)),
        "cleaning_report": cleaning_report,
        "fcm_summary": fcm_summary,
    }


@app.get("/api/local-datasets")
async def local_datasets():
    """List loadable groups (directories containing data files) under DATA_DIR."""
    if not DATA_DIR.exists():
        return {"data_dir": str(DATA_DIR), "groups": []}

    groups = []
    for dirpath, _dirnames, filenames in os.walk(DATA_DIR):
        data_files = sorted(f for f in filenames if Path(f).suffix.lower() in DATA_EXTS)
        if not data_files:
            continue
        rel_dir = Path(dirpath).relative_to(DATA_DIR)
        groups.append(
            {
                "path": "." if rel_dir == Path(".") else rel_dir.as_posix(),
                "files": data_files,
            }
        )
    groups.sort(key=lambda g: g["path"])
    return {"data_dir": str(DATA_DIR), "groups": groups}


class LoadLocalRequest(BaseModel):
    path: str  # relative to DATA_DIR; a directory concatenates all data files within it
    clean: bool = False  # quarantine physically-impossible sensor readings (-> NaN)


@app.post("/api/local-datasets/load")
async def load_local_dataset(req: LoadLocalRequest):
    target = resolve_in_data_dir(req.path)
    df = load_group_dataframe(target)
    label = target.relative_to(DATA_DIR).as_posix() or target.name

    cleaning_report: list[dict] = []
    if req.clean:
        df, cleaning_report = clean_dataframe(df)

    dataset_id = uuid.uuid4().hex
    datasets[dataset_id] = df
    fcm_summary = compute_fcm_summary(df)
    dataset_fcm[dataset_id] = fcm_summary
    return {
        "dataset_id": dataset_id,
        "filename": label,
        "rows": len(df),
        "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
        "preview": to_records(df.head(50)),
        "cleaning_report": cleaning_report,
        "fcm_summary": fcm_summary,
    }


class LoadAllRequest(BaseModel):
    clean: bool = False


@app.post("/api/local-datasets/load-all")
async def load_all_local_datasets(req: LoadAllRequest):
    """Load every folder/file under DATA_DIR at once, for questions spanning the whole corpus."""
    listing = await local_datasets()
    groups = listing["groups"]
    if not groups:
        raise HTTPException(400, "No data files found under DATA_DIR")

    dfs: dict[str, pd.DataFrame] = {}
    fcm_summaries: dict[str, dict | None] = {}
    summaries = []
    for g in groups:
        target = resolve_in_data_dir(g["path"])
        df = load_group_dataframe(target)
        if req.clean:
            df, _ = clean_dataframe(df)
        dfs[g["path"]] = df
        fcm_summary = compute_fcm_summary(df)
        fcm_summaries[g["path"]] = fcm_summary
        # Also register individually so Chart/Train-predict (single-dataset endpoints) can target it.
        individual_id = uuid.uuid4().hex
        datasets[individual_id] = df
        dataset_fcm[individual_id] = fcm_summary
        summaries.append(
            {
                "name": g["path"],
                "dataset_id": individual_id,
                "rows": len(df),
                "columns": list(df.columns),
                "fcm_summary": fcm_summary,
            }
        )

    group_id = uuid.uuid4().hex
    dataset_groups[group_id] = dfs
    group_fcm[group_id] = fcm_summaries
    return {"dataset_group_id": group_id, "datasets": summaries}


RESEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "query_research_output",
        "description": (
            "Read independently computed research results from the configured upstream "
            "trustworthy_outputs directory; NEVER treat these as live readings. "
            "Available artifacts: gam_scores (paired validation), gam_effects "
            "(conditional noncausal smooths), episode_summary, episode_runs, "
            "transitions, markov_summary, warning_scores, warning_hindcasts "
            "(monthly out-of-sample 1/3/7-day severe-onset risks), warning_snapshot "
            "(latest historical records, NOT current conditions). Use the actual "
            "file evidence and cite artifact/filters; missing rows are not zeros. "
            "Do not confuse a hindcast future label with an issue-day predictor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact": {"type": "string", "enum": [
                    "gam_scores", "gam_effects", "episode_summary", "episode_runs",
                    "transitions", "markov_summary", "warning_scores",
                    "warning_hindcasts", "warning_snapshot"]},
                "station": {"type": "string"},
                "threshold_mg_l": {"type": "number"},
                "horizon_days": {"type": "integer"},
                "model": {"type": "string"},
                "issue_date": {"type": "string", "description": "YYYY-MM-DD"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["artifact"],
        },
    },
}


def execute_research_tool(kwargs: dict) -> tuple[str, dict | None]:
    try:
        result = query_research(**kwargs)
        display = {"rows": result["rows"], "total_rows": result["n_matching"],
                   "truncated": result["truncated"]} if result["rows"] else None
        compact = {key: value for key, value in result.items() if key != "rows"}
        compact["rows"] = result["rows"][:12]
        return json.dumps(compact, default=str)[:14000], display
    except (ValueError, TypeError, OSError) as exc:
        return f"Error querying research output: {exc}", None


class ChatRequest(BaseModel):
    dataset_id: str | None = None
    dataset_group_id: str | None = None
    question: str
    research_only: bool = False


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if req.research_only:
        if not research_catalog()["enabled"]:
            raise HTTPException(400, "Research outputs not configured")
        dfs = {}
        fcm_summaries = {}
    elif req.dataset_group_id:
        dfs = dataset_groups.get(req.dataset_group_id)
        if dfs is None:
            raise HTTPException(404, "Unknown dataset_group_id; load all datasets first")
        fcm_summaries = group_fcm.get(req.dataset_group_id, {})
    elif req.dataset_id:
        df = datasets.get(req.dataset_id)
        if df is None:
            raise HTTPException(404, "Unknown dataset_id; upload a file first")
        dfs = {"data": df}
        fcm_summaries = {"data": dataset_fcm.get(req.dataset_id)}
    else:
        raise HTTPException(400, "Provide dataset_id or dataset_group_id")

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You're a knowledgeable data analyst in an easygoing conversation with "
                "someone exploring this dataset. Talk like a thoughtful colleague, not a "
                "report generator: reason out loud, bring in your broader scientific and "
                "domain knowledge, make connections, and don't feel boxed into just "
                "reciting numbers back at people.\n\n"
                "The main thing to stay grounded on: this dataset is numeric sensor/"
                "weather/lab-sample readings, so it generally won't tell you the specific "
                "identity behind a place, agency, program, or named event -- it's easy to "
                "default to a familiar example (e.g. assuming 'water quality' means "
                "Chesapeake Bay) when really you don't know that from the data in front of "
                "you. So when you're about to state something as a specific fact about "
                "*this* dataset/place (a name, an agency, a specific program or incident), "
                "take a beat and check: is that actually in the data (station names, file/"
                "folder names, column values), or is it your best guess? If it's in the "
                "data, go ahead and use it directly. If it's a guess, that's fine too --just "
                "say so naturally rather than presenting it as established fact.\n\n"
                "Otherwise, just answer the way a knowledgeable person naturally would. "
                "For interpretive or conceptual questions (causes, mechanisms, what some "
                "intervention would generally do), lead with your general reasoning -- "
                "explain how aeration tends to affect dissolved oxygen, say, while making "
                "it clear that's general understanding rather than something this specific "
                "dataset proves. If the data has something relevant to add -- a trend, "
                "correlation, anomaly -- weave it in as supporting context; if nothing in "
                "the data is relevant, it's totally fine to just reason it through without "
                "forcing in an unrelated stat for the sake of having 'proof'.\n\n"
                "Whatever you end up pulling from a tool, always circle back and actually "
                "answer the question the person asked, in plain language -- don't let the "
                "reply just become a dump of whatever table or numbers a tool happened to "
                "return. If someone asks something like 'since the municipality started "
                "doing X in 2020, how much has it helped Y', they want your actual verdict "
                "(helped a lot / a little / not really / unclear), reasoned from whatever "
                "evidence you've got -- not a raw correlation matrix with no connection back "
                "to the question. For a question framed around a change over time (did "
                "something improve after a point, has a trend shifted), reach for an "
                "analysis that actually looks across time -- a trend/forecast, or comparing "
                "stats before vs. after the relevant date/year (e.g. filter and compare "
                "averages, or groupby year) -- a single whole-period correlation usually "
                "won't tell you whether things changed. Concretely: for a question like "
                "'has dissolved oxygen improved since the 2020 incident', call "
                "`run_data_model` with operation='trend_forecast' on the dissolved-oxygen "
                "column (or group_summary grouped by year) -- NOT operation='correlation', "
                "which only tells you how two variables move together, not whether either "
                "one got better or worse over time. And keep the same honesty from "
                "above about the specific intervention itself: sensor data generally won't "
                "confirm that a particular municipal program/action actually happened on a "
                "particular date, so be upfront that the intervention itself is taken from "
                "the question/outside knowledge, while whatever trend you found in the "
                "data is what you can actually point to as (or not as) corroborating it.\n\n"
                "One more thing worth double-checking: sensor/station data often has "
                "terse column names (e.g. 'exo.odo_mgL', 'sal_ppt') that encode both a "
                "parameter and its unit -- the unit suffix is a strong, reliable clue (mg/L "
                "reads as a concentration, ppt as parts-per-thousand/salinity, C as "
                "temperature, etc.), so lean on that rather than pattern-matching the "
                "abbreviation to the first similar-sounding word that comes to mind (e.g. "
                "'odo' is the standard abbreviation for optical dissolved oxygen in water- "
                "quality sondes, not 'odor'). If a column's meaning still isn't clear from "
                "its name/unit, it's fine to say you're inferring it rather than stating it "
                "with false confidence.\n\n"
                "You also have three tools you can reach for whenever they'd actually help "
                "answer the question -- there's no fixed rule about when to use them, use "
                "your judgment. `run_pandas_code` runs real pandas code against the full "
                "dataset, handy for exact numbers: sums, counts, averages, filters, "
                "groupby, min/max, date ranges, whatever the question calls for. "
                "`run_data_model` wraps a handful of built-in statistical models: imputing "
                "missing values, trend/forecast, anomaly/outlier detection, correlation, "
                "group summaries, a simple prediction model (train_predict), and a Fuzzy "
                "Cognitive Map (fcm) that learns a causal graph and can simulate what-if "
                "scenarios. Each dataset below already comes with a precomputed 'causal "
                "associations (FCM) from regularly sampled single stations, so for "
                "'what drives X'-type questions it's usually quickest to just read that "
                "instead of re-running the fcm tool -- save the live fcm call for a "
                "specific what-if/clamp scenario that summary doesn't already cover. "
                "`make_chart` renders an actual interactive chart (line/scatter/bar, plus a "
                "static image for hist) that's shown directly to the user -- reach for it "
                "whenever someone asks to 'see'/'show'/'plot'/'chart' something, or a trend/"
                "distribution/relationship would genuinely be clearer as a picture than a "
                "sentence; it appears automatically in the chat, so you don't need to "
                "transcribe every value from it, just briefly describe what it shows as "
                "part of your answer. "
                "`make_map` is specifically for WHERE-type geographic questions (station "
                "locations, sample sites) -- it plots real lat/lon coordinates on an actual "
                "interactive map, which is far more useful than a plain x/y scatter when the "
                "question is literally about location; use `make_chart` (scatter) instead "
                "for a non-geographic x/y relationship. "
                "If someone asks for 'EDA'/'exploratory data analysis'/'a general look at' a "
                "column or dataset, that's a request for SEVERAL complementary views, not "
                "just one chart repeated -- a reasonable default EDA pass is: a trend/line "
                "chart over time (if there's a time/date-like column), a histogram of the "
                "distribution, and maybe a scatter/correlation view against another related "
                "column, calling `make_chart` more than once if it genuinely adds a new "
                "angle rather than reproducing the same chart you (or an earlier turn in "
                "this conversation) already made. Concretely: if asked 'chart out the EDA "
                "for rainfall', don't stop after one `make_chart(line, y=['rain_in'], "
                "x='year')` call -- follow it with `make_chart(hist, y=['rain_in'])` for the "
                "distribution, and if there's a clearly related column (e.g. temperature), "
                "`make_chart(scatter, y=['rain_in'], x='temp_min')` too, THEN write one "
                "answer summarizing all of them together. Only stop at a single chart if the "
                "question asked for one specific thing (e.g. 'just the trend').\n\n"
                "Important: `make_chart`/`make_map` display the actual visual to the user "
                "automatically and separately from your text reply -- NEVER write your own "
                "markdown image tag or a `data:image/...;base64,...` string in your answer; "
                "there is no real image data available to you to embed that way, so doing it "
                "would just be a fabricated, broken placeholder. Just describe what the "
                "already-shown chart/map depicts in plain text.\n\n"
                "A couple of practical notes on using the tools smoothly: if a tool call "
                "comes back complaining about an unknown column or dataset name, don't just "
                "retry with another guess -- it usually lists the actual available names "
                "right there in the error, so use those. And since multiple datasets can be "
                "loaded at once with genuinely different column names for similar things "
                "(say, 'Temperature (C)' in one and 'exo.temp_C' in another), double-check "
                "the 'Columns:' list for whichever dataset you're pulling from and make "
                "sure the `dataset` argument you pass actually matches it. When in doubt, a "
                "quick `result = list(df.columns)` (or `dfs[name].columns`) via "
                "`run_pandas_code` will clear things up before you commit to column names. "
                "You've got a limited number of tool calls per question, so once a call "
                "comes back with a real, on-topic number or result, that's usually your cue "
                "to stop digging and write the actual answer -- don't keep calling more "
                "tools 'just to double check' once you already have something solid to work "
                "with; a confident answer from one good result beats running out of tool "
                "calls partway through a second or third exploratory detour.\n\n"
                "One hard rule that matters a lot: NEVER present a specific number -- a "
                "correlation, slope, percentage, p-value, whatever -- as something the data "
                "showed unless it actually came back from a tool call in this conversation. "
                "If every tool call you tried ended in an error (no real result ever came "
                "back), say so plainly instead of inventing plausible-sounding numbers to "
                "fill the gap -- a made-up statistic dressed up as a real finding is worse "
                "than admitting the analysis didn't go through. It's always fine to reason "
                "qualitatively from general domain knowledge instead (e.g. 'rainfall "
                "typically lowers salinity and can affect dissolved oxygen') as long as "
                "you're clear that's general reasoning, not a number this dataset produced.\n\n"
                "BAY SENSOR SCIENTIFIC SAFEGUARDS: Never treat GNN-imputed values as observed "
                "low oxygen. For 2/4 mg/L episodes use run_data_model operation='hypoxia_audit' "
                "with an observed flag when the file has one. Unknown/low-coverage days are "
                "not healthy days; missing data censor onsets and recoveries. Keep stations "
                "and independent instrument sources distinct. All episodes here are "
                "retrospective descriptions, not early-warning forecasts. FCM weights are "
                "lagged associations, not causal effects, and its trajectories are not "
                "validated predictions. Do not promise that an FCM exists for mixed or "
                "irregularly sampled datasets.\n\n"
                "RESEARCH ARTIFACTS: A separate read-only scientific output catalog may "
                "be configured. For questions on held-out GAM scores, observed episode "
                "transitions, recovery, Markov, 1/3/7-day hindcasts, use "
                "query_research_output; never invent results or describe retrospective "
                "research as a live forecast. GAM curves are conditional associations; "
                "never infer causal effects. Hindcast labels are historical future "
                "outcomes and must never be treated as issue-date inputs. "
                "Use one station/date/horizon/model filter when relevant; report "
                "missing files, sparse samples and unmatched dates explicitly.\\n"
                + f"Research output catalog: {json.dumps(research_catalog())[:5500]}\\n\\n"
                + build_context(dfs, fcm_summaries)
            ),
        },
        {"role": "user", "content": req.question},
    ]

    stream_events: list[dict] = []

    async def pipeline(emit) -> str:
        """Run the tool loop, emitting status/tool events as they happen; return the final text."""
        response = None
        any_tool_succeeded = False
        for round_no in range(1, MAX_TOOL_ROUNDS + 1):
            await emit({"status": f"Asking the model (round {round_no} of up to {MAX_TOOL_ROUNDS})"})
            response = await client.chat(
                model=MODEL,
                messages=messages,
                tools=([RESEARCH_TOOL] if req.research_only else [PANDAS_TOOL, MODEL_TOOL, CHART_TOOL, MAP_TOOL] + ([RESEARCH_TOOL] if research_catalog()["enabled"] else [])),
                options=OLLAMA_CHAT_OPTIONS,
            )
            tool_calls = response.message.tool_calls
            if not tool_calls:
                break
            messages.append({"role": "assistant", "content": response.message.content or "", "tool_calls": tool_calls})
            for call in tool_calls:
                name = call.function.name
                await emit({"status": f"Running {name}"})
                if name == "query_research_output":
                    output, display = execute_research_tool(dict(call.function.arguments))
                elif name == "run_data_model":
                    output, display = await run_model_tool(dfs, dict(call.function.arguments))
                elif name == "make_chart":
                    output, display = await run_chart_tool(dfs, dict(call.function.arguments))
                elif name == "make_map":
                    output, display = await run_map_tool(dfs, dict(call.function.arguments))
                else:
                    code = call.function.arguments.get("code", "")
                    output, display = await run_pandas_tool(dfs, code)
                if not output.startswith("Error"):
                    any_tool_succeeded = True
                events = [{"tool": f"ran {name} -> {output[:200]}"}]
                if display is not None:
                    if "image_base64" in display:
                        events.append({"chart": display})
                    elif "lat_col" in display:
                        events.append({"map": display})
                    elif "chart_type" in display:
                        events.append({"chart_native": display})
                    else:
                        events.append({"table": display})
                for event in events:
                    stream_events.append(event)
                    await emit(event)
                messages.append({"role": "tool", "tool_name": name, "content": output})
        else:
            # Exhausted rounds while the model kept calling tools; strip the tools param so it
            # MUST answer in plain text this round, and explicitly tell it to stop exploring and
            # synthesize from whatever's already in the conversation instead of continuing to dig.
            if any_tool_succeeded:
                wrap_up_instruction = (
                    "You've used up your tool-call budget. Don't call any more tools -- "
                    "answer the original question now, in plain language, using whatever "
                    "results you already got back above (there's likely at least one useful "
                    "number in there, e.g. a trend/forecast result)."
                )
            else:
                wrap_up_instruction = (
                    "You've used up your tool-call budget and EVERY tool call above resulted in "
                    "an error -- none of them produced any real data or numbers. This means you "
                    "have NO grounded result to report. Do NOT invent, estimate, or make up any "
                    "specific numbers, correlations, or statistics as if they came from the data "
                    "-- that would be fabrication. Instead, tell the user plainly that the "
                    "analysis couldn't be completed (briefly mention why, e.g. the code kept "
                    "getting blocked), and suggest a more specific/narrower way to ask so it's "
                    "more likely to succeed next time. You may still share general domain "
                    "reasoning (e.g. how rainfall generally affects dissolved oxygen) as long as "
                    "you're clear that's general knowledge, not something this data confirmed."
                )
            messages.append({"role": "user", "content": wrap_up_instruction})
            await emit({"status": "Writing the final answer"})
            response = await client.chat(model=MODEL, messages=messages, options=OLLAMA_CHAT_OPTIONS)

        final_text = response.message.content or ""

        any_tool_attempted = any("tool" in e for e in stream_events)
        if any_tool_attempted and not any_tool_succeeded:
            # Every tool call failed (no grounded result ever came back) -- regardless of how we
            # got here (voluntary stop or exhausted budget), the model's free-form answer at this
            # point is suspect: double-check it isn't presenting fabricated numbers as real
            # findings, by forcing one more explicit, instructed rewrite rather than trusting it.
            messages.append({"role": "assistant", "content": final_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Wait -- every single tool call you made above ended in an error, so "
                        "there is NO real data or computed number backing up your last answer. "
                        "Rewrite your answer: do not state any specific number, correlation, "
                        "percentage, or statistic as something the data showed, since none of "
                        "that was actually computed. Say plainly that the analysis couldn't be "
                        "completed and briefly why, then (optionally) share general domain "
                        "reasoning clearly labeled as such, not as a data-backed finding."
                    ),
                }
            )
            await emit({"status": "Checking the answer against failed tool calls"})
            response = await client.chat(model=MODEL, messages=messages, options=OLLAMA_CHAT_OPTIONS)
            final_text = response.message.content or final_text

        if not final_text.strip():
            # Some small models can return a genuinely empty completion here (rare, but seen in
            # practice) -- don't silently hand the user nothing after real tool work happened.
            final_text = (
                "I wasn't able to put together a clear final answer from the tool results above "
                "(the model returned an empty response) -- the tool calls and any numbers found "
                "are shown above though, so take a look at those, or try asking again, maybe "
                "phrased a bit more narrowly."
                if stream_events
                else "I wasn't able to generate a response -- please try asking again."
            )
        return final_text

    async def stream():
        # Events are streamed as the pipeline produces them, with a keep-alive comment while
        # the model is busy, so the client never waits silently past its read timeout. If the
        # client disconnects, the pipeline task is cancelled, which aborts the Ollama request
        # instead of leaving it to occupy Ollama's single slot.
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(event: dict) -> None:
            await queue.put(event)

        async def runner() -> str:
            try:
                return await pipeline(emit)
            finally:
                await queue.put(None)

        task = asyncio.create_task(runner())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=CHAT_HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event)}\n\n"
            try:
                final_text = await task
                # Full text is already generated; chunk it to keep the streaming UX.
                for i in range(0, len(final_text), 40):
                    yield f"data: {json.dumps({'token': final_text[i:i + 40]})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/research/catalog")
async def research_catalog_endpoint():
    return research_catalog()


class ResearchQueryRequest(BaseModel):
    artifact: str
    station: str | None = None
    threshold_mg_l: float | None = None
    horizon_days: int | None = None
    model: str | None = None
    issue_date: str | None = None
    limit: int = 20


@app.post("/api/research/query")
async def research_query_endpoint(req: ResearchQueryRequest):
    try:
        return query_research(**req.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class HypoxiaAuditRequest(BaseModel):
    dataset_id: str
    timestamp_col: str | None = None
    do_col: str | None = None
    station_col: str | None = None
    observed_col: str | None = None
    interval_minutes: int = 5
    min_coverage: float = 0.95
    timezone: str = "America/New_York"
    station: str | None = None


@app.post("/api/hypoxia-audit")
async def hypoxia_audit_endpoint(req: HypoxiaAuditRequest):
    df = datasets.get(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Unknown dataset_id; load data first")
    try:
        return hypoxia_audit(
            df, timestamp_col=req.timestamp_col, do_col=req.do_col,
            station_col=req.station_col, observed_col=req.observed_col,
            interval_minutes=req.interval_minutes, min_coverage=req.min_coverage,
            timezone=req.timezone, station=req.station,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class ChartRequest(BaseModel):
    dataset_id: str
    chart_type: str  # line, scatter, bar, hist
    y: list[str]
    x: str | None = None
    group_by: str | None = None


@app.post("/api/chart")
async def chart_endpoint(req: ChartRequest):
    df = datasets.get(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Unknown dataset_id; load a dataset first")
    try:
        png = make_chart(df, req.chart_type, req.y, req.x, req.group_by)
    except Exception as e:
        raise HTTPException(400, f"Could not build chart: {e}")
    return Response(content=png, media_type="image/png")


class TrainPredictRequest(BaseModel):
    dataset_id: str
    features: list[str]
    target: str
    new_rows: list[dict] | None = None


@app.post("/api/train-predict")
async def train_predict_endpoint(req: TrainPredictRequest):
    df = datasets.get(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Unknown dataset_id; load a dataset first")
    try:
        result = run_data_model(
            df, "train_predict", features=req.features, target=req.target, new_rows=req.new_rows
        )
    except Exception as e:
        raise HTTPException(400, str(e))
    return result


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")
