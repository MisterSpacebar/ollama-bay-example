import asyncio
import io
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from ollama import AsyncClient
from pydantic import BaseModel

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "granite4.1:3b")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SAMPLE_ROWS = 20
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "water_data")).resolve()
DATA_EXTS = (".csv", ".xlsx", ".xlsm", ".json")

app = FastAPI(title="Ollama Data Chat")
client = AsyncClient(host=OLLAMA_HOST)

# In-memory store: dataset_id -> DataFrame
datasets: dict[str, pd.DataFrame] = {}


def read_dataframe(filename: str, raw: bytes) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(raw)
    if ext == ".csv":
        return pd.read_csv(buf)
    if ext in (".xlsx", ".xlsm"):
        return pd.read_excel(buf)
    if ext == ".json":
        return pd.read_json(buf)
    raise HTTPException(400, "Unsupported file type. Use .csv, .xlsx, .xlsm or .json")


def read_dataframe_path(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
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


# --- pandas tool-calling: lets the model run real code for exact answers ---
TOOL_TIMEOUT_SECONDS = 5
MAX_TOOL_ROUNDS = 4
_pandas_executor = ThreadPoolExecutor(max_workers=2)
# Best-effort denylist; exec() with empty __builtins__ is the real boundary but
# isn't a watertight sandbox, so this only runs against the user's own local data/model.
_UNSAFE_CODE_PATTERN = re.compile(r"__|import\s|open\s*\(|exec\s*\(|eval\s*\(|os\.|sys\.|subprocess")

PANDAS_TOOL = {
    "type": "function",
    "function": {
        "name": "run_pandas_code",
        "description": (
            "Run pandas code against the full dataset to compute exact answers (sums, "
            "counts, averages, filters, groupby, min/max, date ranges, etc.) instead of "
            "guessing from the sample. The dataframe is available as `df`. Assign the "
            "final answer to a variable named `result`. Only `pd` and `df` are available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code using `df` and `pd` that sets a `result` variable.",
                }
            },
            "required": ["code"],
        },
    },
}


def _execute_pandas_code(df: pd.DataFrame, code: str) -> str:
    if _UNSAFE_CODE_PATTERN.search(code):
        return "Error: code contains disallowed constructs (imports/dunder/system access)"
    sandbox_globals = {"__builtins__": {}, "pd": pd, "df": df}
    sandbox_locals: dict = {}
    try:
        exec(code, sandbox_globals, sandbox_locals)
    except Exception as e:
        return f"Error executing code: {e}"
    result = sandbox_locals.get("result", sandbox_globals.get("result"))
    if result is None:
        return "Error: code did not assign a `result` variable"
    text = result.to_csv() if isinstance(result, (pd.DataFrame, pd.Series)) else str(result)
    return text[:4000]


async def run_pandas_tool(df: pd.DataFrame, code: str) -> str:
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_pandas_executor, _execute_pandas_code, df, code),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return f"Error: code execution timed out ({TOOL_TIMEOUT_SECONDS}s limit)"


def to_records(df: pd.DataFrame) -> list[dict]:
    # to_json handles NaN/NaT/numpy types safely
    return json.loads(df.to_json(orient="records", date_format="iso"))


def build_context(df: pd.DataFrame) -> str:
    schema = "\n".join(f"- {c}: {t}" for c, t in df.dtypes.astype(str).items())
    stats = df.describe(include="all").transpose().to_csv()
    sample = df.head(SAMPLE_ROWS).to_csv(index=False)
    return (
        f"Rows: {len(df)}, Columns: {len(df.columns)}\n\n"
        f"Columns:\n{schema}\n\n"
        f"Summary statistics (CSV):\n{stats}\n"
        f"First {min(SAMPLE_ROWS, len(df))} rows (CSV):\n{sample}"
    )


@app.get("/api/health")
async def health():
    try:
        await client.show(MODEL)
        return {"ollama": "ok", "model": MODEL}
    except Exception as e:
        raise HTTPException(503, f"Ollama/model '{MODEL}' unavailable: {e}")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 20 MB)")
    try:
        df = read_dataframe(file.filename or "", raw)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")

    dataset_id = uuid.uuid4().hex
    datasets[dataset_id] = df
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
        "preview": to_records(df.head(50)),
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


@app.post("/api/local-datasets/load")
async def load_local_dataset(req: LoadLocalRequest):
    target = resolve_in_data_dir(req.path)

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
        df = pd.concat(frames, ignore_index=True, sort=False)
        label = target.relative_to(DATA_DIR).as_posix() or target.name
    else:
        try:
            df = read_dataframe_path(target)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"Could not parse file: {e}")
        label = target.relative_to(DATA_DIR).as_posix()

    dataset_id = uuid.uuid4().hex
    datasets[dataset_id] = df
    return {
        "dataset_id": dataset_id,
        "filename": label,
        "rows": len(df),
        "columns": [{"name": c, "dtype": str(t)} for c, t in df.dtypes.items()],
        "preview": to_records(df.head(50)),
    }


class ChatRequest(BaseModel):
    dataset_id: str
    question: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    df = datasets.get(req.dataset_id)
    if df is None:
        raise HTTPException(404, "Unknown dataset_id; upload a file first")

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a data analyst. You have a `run_pandas_code` tool that runs real "
                "pandas code against the FULL dataset (not just the sample below). Use it "
                "whenever a question needs an exact number: sums, counts, averages, filters, "
                "groupby, min/max, date ranges, etc. Only rely on the sample/summary for "
                "qualitative or schema questions.\n\n" + build_context(df)
            ),
        },
        {"role": "user", "content": req.question},
    ]

    tool_log: list[str] = []
    response = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.chat(model=MODEL, messages=messages, tools=[PANDAS_TOOL])
        tool_calls = response.message.tool_calls
        if not tool_calls:
            break
        messages.append({"role": "assistant", "content": response.message.content or "", "tool_calls": tool_calls})
        for call in tool_calls:
            code = call.function.arguments.get("code", "")
            output = await run_pandas_tool(df, code)
            tool_log.append(f"ran pandas code -> {output[:200]}")
            messages.append({"role": "tool", "tool_name": call.function.name, "content": output})
    else:
        # Exhausted rounds while the model kept calling tools; force a final answer.
        response = await client.chat(model=MODEL, messages=messages)

    final_text = response.message.content or ""

    async def stream():
        try:
            for log in tool_log:
                yield f"data: {json.dumps({'tool': log})}\n\n"
            # Full text is already generated; chunk it to keep the streaming UX.
            for i in range(0, len(final_text), 40):
                yield f"data: {json.dumps({'token': final_text[i:i + 40]})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")
