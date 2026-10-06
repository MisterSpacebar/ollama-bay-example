# Ollama Data Chat

FastAPI backend that loads CSV/Excel/JSON files and lets you ask questions about them using a Granite model served by Ollama. A minimal frontend is served at `/`.

## Setup

```powershell
ollama pull granite4.1:3b
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://localhost:8000.

## Config (env vars)

- `OLLAMA_MODEL` (default `granite4.1:3b`)
- `OLLAMA_HOST` (default `http://localhost:11434`)

## API

- `GET /api/health` – checks Ollama and the model
- `POST /api/upload` – multipart `file`; returns `dataset_id`, columns, preview rows
- `POST /api/chat` – `{dataset_id, question}`; streams answer tokens as SSE
