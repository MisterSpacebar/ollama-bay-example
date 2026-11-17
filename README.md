# Ollama Data Chat

FastAPI backend that loads CSV/Excel/JSON files and lets you ask questions about them using a Granite model served by Ollama. A minimal frontend is served at `/`.

## Setup

```powershell
ollama pull granite4.1:8b
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open http://localhost:8000.

## Config (env vars)

- `OLLAMA_MODEL` (default `granite4.1:8b`; the smaller `granite4.1:3b` also works and uses less
  memory/VRAM, but struggles with multi-hop/multi-dataset reasoning -- use `granite4.1:3b` only
  if resources are constrained)
- `OLLAMA_HOST` (default `http://localhost:11434`)

## API

- `GET /api/health` – checks Ollama and the model
- `POST /api/upload` – multipart `file`; returns `dataset_id`, columns, preview rows
- `POST /api/chat` – `{dataset_id, question}`; streams answer tokens as SSE


## Trustworthy bay observation analysis

Adapted from the observed-only quality and episode analysis methods in
[the multi-node bay research project](https://github.com/MisterSpacebar/multinode-bay-imputation/tree/main/trustworthy).

This update adds a **Observed oxygen** tab in Streamlit, a
`POST /api/hypoxia-audit` endpoint, and the
`run_data_model(operation="hypoxia_audit")` chat operation.
Select a timestamp, dissolved oxygen in mg/L, and (when present) a station/source
identifier and an explicit `*_observed` provenance flag. It reports eligible
local calendar days (95% valid observations by default), observed events strictly
below 2 and 4 mg/L, identifiable onsets, observed recovery, censored boundaries,
and measured low-oxygen hours as *lower bounds*. A recorded zero remains valid
unless another QC rule invalidates it. Missing days are **unknown**, not healthy.

Example (replace the dataset ID after loading a CSV):

```json
{
  "dataset_id": "your-dataset-id",
  "timestamp_col": "datetime",
  "do_col": "odo_mgL",
  "station_col": "node",
  "observed_col": "odo_mgL_observed",
  "interval_minutes": 5,
  "min_coverage": 0.95,
  "timezone": "America/New_York"
}
```

The FCM adapter now requires a single station and a unique regularly sampled
timestamp axis. Its weights are **exploratory lagged associations, not causal
effects**; its scenario trajectories are not operational forecasts.
The underlying basic FCM still fills missing covariates retrospectively.
Data without explicit DO provenance flags are labeled unverified; they should
not be treated as observed-only if they contain GNN-reconstructed oxygen.
Avoid combining independent sensor deployments merely because their station
names or coordinates match. Naive datetime strings are interpreted as UTC.

**Scope:** These are retrospective diagnostics, not live hypoxia alerts.
The GNN, GAM, Markov onset warnings, and their validation remain in the
upstream scientific repository. Run synthetic regression tests using:

```bash
python -m unittest discover -s tests -v
python -m py_compile backend/*.py streamlit_app.py
```
