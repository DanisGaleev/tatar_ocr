# AGENTS.md

## Project
- **Tatar OCR («Кара куян»)** — backend (FastAPI + SQLite), C++20 native OCR core (`tatar_ocr_core`), ML training/eval pipeline (`v3/`), and shared declarative spec (`pipeline_spec.json`).
- **Zero-Photo & 152-ФЗ:** worksheet camera frames stay on the teacher's device; backend ingests only structured JSON results (`backend/app/routers/submissions.py`). Textbook-photo extraction (`backend/app/routers/constructor.py`) is an explicit isolated exception and must not store raw student photos.
- **Sister repository:** `kara_kuyan_frontend` (consumes `native/` as a Git submodule and the FastAPI OpenAPI contract).

## Context Loading Order
1. `AGENTS.md` (this file)
2. `docs/ECL.md` (change lifecycle)
3. `docs/STATUS.md` (current state & Session Handoff)
4. Active change in `harness/changes/active/<slug>/` (`summary.md`, `spec.md`, `plan.md`, `tasks.md`, `checks.json`) if present
5. Only the task-specific docs/specs needed (`docs/ARCHITECTURE.md`, `docs/specs/README.md`, `docs/adr/`)

## Verified Commands (Windows PowerShell)
- **Backend unit & API tests:**
  `.\.venv\Scripts\python.exe -m pytest backend\tests -q`
- **Single backend test file (preferred during iteration):**
  `.\.venv\Scripts\python.exe -m pytest backend\tests\test_api.py -q`
- **Run FastAPI backend locally:**
  `.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --port 8000`
- **Build C++20 native core (`tatar_ocr_core` + `tatar_ocr_native`):**
  `cmake -S native -B native\build -DCMAKE_BUILD_TYPE=Release`
  `cmake --build native\build --config Release`
- **OCR 12-sheet benchmark (note `--model`, not `--onnx`):**
  `.\.venv\Scripts\python.exe v3\evaluate_blanks.py --model models\tatar_ocr_v4_handwritten39_int8.onnx`

## Hard Boundaries
- **Autopilot (no user memorization):** Drive the ECL/OpenSpec/checks.json workflow proactively from plain user requests ("add feature X", "fix bug Y"). Never expect the user to remember harness commands or file names; guide them with concise checkpoints ("Confirm plan?", "Ready to close/archive?").
- **No structured implementation without an active change** in `harness/changes/active/` (see `docs/ECL.md` for Small vs Structured Change).
- **One active change** at a time; at most one writer per file.
- **Default-FAIL verification:** every criterion in `checks.json` starts `"status": "FAIL"` and flips to `"PASS"` only with recorded `command` + `evidence`. Never claim tests/builds pass unless run after the final edit.
- **C++20 / Python / Android parity:** `native/` is compiled by Android NDK (`TATAR_OCR_BUILD_PYTHON=OFF`) in `kara_kuyan_frontend` and via `pybind11` here. Keep `pipeline_spec.json` as SSOT; do not diverge C++ and Python geometry/preprocessing logic.
- **Contract integrity:** any API change requires checking `api_specification.md` and `openapi_specification.yaml` (OpenSpec delta) without breaking `kara_kuyan_frontend` assumptions silently.
- **Update relevant ADR** in `docs/adr/` before altering architectural invariants.
- **Workspace hygiene:** never leave temporary `tmp_*` / `test_*` / `debug_*` scripts in the repo.

## Rule Evolution (Error-Driven)
- Keep this file concise (< 120 lines). Add a rule only when an agent repeats the same mistake twice; prune obsolete rules instead of piling up prose.
