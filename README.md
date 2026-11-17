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


## Read-only upstream GAM, episode and Markov research outputs

The **Research outputs** Streamlit tab, `GET /api/research/catalog`,
`POST /api/research/query`, and Ollama `query_research_output` tool read
research tables produced by the upstream
[multinode-bay-imputation](https://github.com/MisterSpacebar/multinode-bay-imputation)
pipeline, without refitting models or confusing GNN reconstruction with
observed DO.

Set the environment variable **on the FastAPI server** to the actual
`trustworthy_outputs` directory created locally by that project:

```powershell
$env:BAY_RESEARCH_OUTPUT_DIR = "C:\\path\\to\\multinode-bay-imputation\\trustworthy_outputs"
uvicorn backend.main:app --reload
```

Do **not** point this variable at raw private sensor data or at the repository
root. The adapter is disabled when the directory is absent or unconfigured.
Its catalog lists missing generated artifacts rather than fabricating results.
If no datasets have been loaded, clients can use
`POST /api/chat` with `{"research_only": true, "question": "..."}`.
When a dataset is loaded, the chat agent can also query the research tool.

Supported exports include `environmental_gam/paired_scores.csv` and
`gam_smooth_effects.csv`, episode summary/run records, station-level
transition matrices, Markov uncertainty summaries,
`transition_warning/hindcast_scores.csv`,
`hindcast_predictions.csv`, and optional historical dashboard snapshots.

Example historical query:

```json
{
  "artifact": "warning_hindcasts",
  "station": "EXACT_STATION_ID",
  "horizon_days": 3,
  "issue_date": "2025-07-15",
  "model": "do_history"
}
```

The response returns the exact output file's SHA-256, selected records,
the number of matched rows and provenance/limitations. It never exposes
arbitrary paths to model tool calls. A valid file is **not** independently
verified against its original upstream input hashes; if files from different
analysis runs were combined, results may not be comparable.

**Scientific limitations:** The GAM severity smooth is a conditional
**same-day association**, not a forecast or intervention effect. The GAM
recovery model concerns observed active episodes, not a new-onset warning.
The Markov model is a stationary benchmark; its Monte Carlo paths are
hypothetical. Warning hindcasts predict first Severe onset in 1/3/7 days
from eligible non-Severe end-of-day origins; the reported outcome was
observed later and is included *only* for retrospective evaluation.
Neither historical snapshots nor hindcasts imply current-day live alerts.
Missing files, stations, and dates are not evidence of zero risk.

To validate the adapter and HTTP endpoints, in addition to earlier unit tests:

```bash
python -m unittest discover -s tests -v
python -m py_compile backend/*.py streamlit_app.py
```

The suite uses synthetic CSVs and mocks Ollama; it does not claim to validate
scientific performance against your private source data, nor to exercise a
real browser/Streamlit session.


## Development and test environments

Install the application dependencies and HTTP test client through the separate
[test requirements](requirements-test.txt):

```powershell
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m py_compile backend/*.py streamlit_app.py
```

The application itself only needs `requirements.txt`; `httpx` is used by
FastAPI/Starlette's `TestClient` in the integration tests. The direct runtime
`numpy` dependency is declared explicitly. CI runs the test dependency set
on Python 3.11, while development may use newer interpreters.

On Windows without symlink privileges, the test that creates a **real**
symlink is skipped (Developer Mode or elevated privileges enable it). A
separate mocked-resolution regression test always exercises the path-escape
rejection code without requiring symlink creation. A mocked check does not
replace the Linux CI real-symlink test.

Some newer Starlette/HTTP-client combinations may show deprecation notices.
Treat those as dependency-compatibility warnings to investigate before any
library migration, not as a reason to bypass the integration tests.


## Real `trustworthy_outputs` integration validation

The research browser now works **without uploading a separate sensor dataset**.
Start FastAPI with `BAY_RESEARCH_OUTPUT_DIR` set to the locally generated
`trustworthy_outputs` directory, then run Streamlit. The **Research outputs**
screen can browse results and ask Ollama questions using research-only mode.

From the `ollama-bay-example` repository on Windows PowerShell, point the
API to the **actual generated output directory** of the other repository:

```powershell
python -m pip install -r requirements-test.txt
$env:BAY_RESEARCH_OUTPUT_DIR = "C:\\path\\to\\multinode-bay-imputation\\trustworthy_outputs"
uvicorn backend.main:app --reload
```

From another PowerShell terminal, run the independent source-versus-adapter
validator. The `--api-url` option compares live HTTP responses with the
original output CSV rows, counts and SHA-256 hashes. It does **not** contact
Ollama or regenerate models:

```powershell
python -m scripts.validate_research_outputs --root "C:\\path\\to\\multinode-bay-imputation\\trustworthy_outputs" --api-url http://127.0.0.1:8000 --report validation_report.local.json
```

To use only local files when the API is offline, omit `--api-url`.
The validator reports available/missing supported outputs, checks upstream
observed-only provenance, tests station/model pairing and monotonic warning
horizons, verifies metadata-linked input hashes **where source files exist**,
and compares the actual CSV records to both the backend adapter and optional
running FastAPI endpoints. The command returns a nonzero exit code for missing
all supported outputs, invalid input, mismatched metadata/hashes, and API
disagreement. A pass with unavailable input hashes is labeled
`pass_with_unverified_lineage`, not a fully established scientific lineage.
Never commit a source-specific local validation report or private exports.

The upstream results are generated separately in `multinode-bay-imputation`:

```powershell
# Run these from the *multinode-bay-imputation* repository root.
python -m trustworthy.quality
python -m trustworthy.regimes
python -m trustworthy.episode_dynamics
python -m trustworthy.episode_transitions --regimes trustworthy_outputs/onset/regime_days.csv --out-dir trustworthy_outputs/episode_dynamics
python -m trustworthy.episode_survival
python -m trustworthy.environmental_gam
python -m trustworthy.transition_warning
python -m trustworthy.warning_dashboard
```

These commands assume upstream requirements and observed-only inputs are
available; consult the upstream phase documents for exact options and ordering.
Do not replace missing observed-only source inputs with GNN reconstructions.

Finally, to browse the results:

```powershell
streamlit run streamlit_app.py
```

Open the **Research outputs** screen, query a historical station/date/model,
and use the separate research-only chat input. This is a usability check;
neither a successful validator nor an Ollama answer is a substitute for
independent scientific calibration. If a browser or Ollama server is not
available, note that those stages remain untested.
