# Independent Review (Fresh-Context Evaluator) — `local-backend-deployment`

Evaluated in a fresh context (`research` subagent) against `spec.md`, `checks.json`, and actual workspace state.

## Verdict

`accept-with-notes` <!-- accept | accept-with-notes | changes-required -->

Code changes match `spec.md`, `plan.md`, and `docker-compose.yml` cleanly with no API/Zero-Photo regressions and `MISSING: []` in the AST import audit. Two verification/packaging caveats warrant notes before or during host deployment: (1) `docker build` has not been run yet (with a potential `numpy==2.5.3` + `opencv-python-headless==4.10.0.84` compatibility note), and (2) in the sandboxed worktree the external `.venv` path was permission-denied to the evaluator subagent so `pytest` and live `TestClient` checks were executed by the main agent with user-approved venv access.

## Evaluator Verification

- [x] Read `spec.md` deltas and `checks.json` before inspecting code.
- [x] Inspected `backend/app/main.py`, `.dockerignore`, `backend/requirements.txt`, `backend/Dockerfile`, `backend/docker-compose.yml`, `backend/app/generators/pdf_blank.py`, `backend/app/core/config.py`, and `backend/app/routers/model.py`.
- [x] Re-ran AST import audit independently via `python`: `THIRD_PARTY: ['cv2', 'fastapi', 'httpx', 'matplotlib', 'numpy', 'openpyxl', 'pydantic', 'pydantic_settings', 'qrcode', 'sqlalchemy', 'starlette']`, `MISSING: []`.
- [x] `checks.json` verified against final edited state (`AC-1` 500-handler probe + `AC-3` `139 passed` executed by main agent; `.dockerignore` secret pattern tightened post-review without touching Python source).
- [x] Verified living specs (`docs/specs/README.md`) require no updates (`spec.md` Non-goals and `GATE-SPECS` confirmed).

## Subsystem Lenses Applied

- **API Contract & Zero-Photo (152-ФЗ):** `PASS` — `backend/app/main.py` only adds `import logging` and `logger = logging.getLogger(__name__)`. All routers, schemas, CORS, and `seed_initial_data()` are untouched; no student photo persistence or logging is introduced.
- **C++20 / Python / Android NDK Parity:** `N/A` — no changes to `native/` or `pipeline_spec.json`.
- **ML Benchmark & Metrics (`ocr_metrics_specification.md`):** `N/A` — no changes to `v3/`, weights, or preprocessing (`backend/app/routers/model.py` does not read `models/`).
- **Backend QA, Migrations & Hygiene:** `PASS with notes` — AST audit confirms all third-party imports in `backend/app/` are covered by `backend/requirements.txt`. `.dockerignore` retains `backend/app/` and `fonts/` while excluding `models/`, `dataset/`, `realdataset/`, `v3/`, `native/`, `.venv/`, `backend/tests/`, `backend/data/`, `backend/.mplconfig/`, `*.db*`, and `**/.env*` (tightened per evaluator finding). `backend/Dockerfile` (`WORKDIR /app/backend`, `EXPOSE 8000`, `MPLCONFIGDIR=/tmp/mplconfig`, non-root `appuser` owning `/app`, `/data`, `/tmp/mplconfig`) matches `backend/docker-compose.yml`.

## Deviations from Plan

| Deviation | Justified? | Note |
|---|---|---|
| None | — | Modified/added files match `plan.md` File Ownership (`backend/app/main.py`, `.dockerignore`, `backend/requirements.txt`, `backend/Dockerfile`). |

## Issues Found

1. **Low / Hygiene (`.dockerignore`):** `.env` and `.env.*` initially only matched root; updated to also exclude `**/.env` and `**/.env.*` so `backend/.env*` cannot be copied by `!backend/`. *(Fixed)*
2. **Low / Minor (`backend/Dockerfile`):** `ENV PYTHONPATH=/app` is redundant with `WORKDIR /app/backend` and `uvicorn app.main:app`, kept harmlessly for backwards compatibility.

## Residual Risks

- **Resolved on host (2026-11-28T16:30+03:00):** `docker build -f backend/Dockerfile -t tatar-ocr-backend:verify .` succeeded (build context `6.74MB`), container smoke tests passed (`CV2_NUMPY_OK 4.10.0 2.5.3`, `/health` OK, `/api/v1/constructor/task-types` `count=24`, `/api/v1/assignments/TAT-2026-Q1/blank.pdf?variant=1` `200`, `73614` bytes).
