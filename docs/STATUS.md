# Project Status & Session Handoff

## State Snapshot (updated 2026-11-27)

- **Repository baseline:** branch `form_recognition_server_upload` (from `1ac7df3`), native C++20 OCR core (`tatar_ocr_native`), ML model v4 INT8 (`models/tatar_ocr_v4_handwritten39_int8.onnx`, ~1.0 MB), FastAPI backend (`backend/app/`).
- **Verified Backend Suite:** `pytest backend\tests -q` → **147 passed** (139 existing + 8 new tests in `backend/tests/test_blank_recognition.py`, including real photo E2E tests on `samples/test_blank_phone_photo.jpg` and `benchmark/images/blank_01.jpg`, verified 2026-11-27).
- **Production OCR Model (`v4 INT8`):** `models/tatar_ocr_v4_handwritten39_int8.onnx` (`1,024,619` bytes). Recorded benchmark in `docs/PIPELINE.md` on 12 canonical blanks (`449` chars, `57` fields): `Acc_char = 96.88%`, `R_Tatar = 97.33%`, `Raw FEM = 77.19%`, `FFM (Lev<=1) = 98.25%`, `ZTSR = 41.67%`.
- **Agent Harness:** Unified ECL + OpenSpec + Anthropic Harness + ADR setup (`AGENTS.md`, `docs/ECL.md`, `harness/templates/change/`, `.agents/skills/tatar-ocr-delivery/`).

## Active Change

- *None* (`harness/changes/active/` is empty; `server-blank-recognition` archived to `harness/changes/archive/server-blank-recognition/`).

## Session Handoff (for next agent session)

- **Current focus:** Server-side blank photo recognition (`POST /api/v1/submissions/recognize` & `/recognize-base64`) implemented and verified.
- **Known discrepancy in docs to keep in mind:** `README.md` currently prints `--onnx` for `v3/evaluate_blanks.py`, whereas the actual CLI flag in `v3/evaluate_blanks.py` is `--model`.
- **Cross-repo open contract items (from `kara_kuyan_frontend` integration):**
  - Confirm `X-Teacher-UUID` multi-teacher isolation semantics and production auth roadmap before auth changes.
  - Keep `openapi_specification.yaml` and `api_specification.md` aligned whenever any router/schema in `backend/app/` changes.
  - Keep `native/` CMake build compatible with `TATAR_OCR_BUILD_PYTHON=OFF` (Android NDK) as well as `ON` (`pybind11`).

## Completed Structured Changes

| Change | Outcome | Evidence |
|---|---|---|
| *(harness bootstrap)* | Installed unified ECL + OpenSpec + Anthropic Harness + ADR scaffolding | `AGENTS.md`, `docs/ECL.md`, `backend/tests` (`139 passed`) |
| `server-blank-recognition` | Added C++20 bridge, transient in-RAM recognition service, `/submissions/recognize*` endpoints | `backend/tests/test_blank_recognition.py` (`8 passed`), full suite `147 passed` |
| `local-backend-deployment` | Fixed missing `logger` in `backend/app/main.py`; slimmed `backend/Dockerfile` via `backend/requirements.txt` & `.dockerignore` with non-root `appuser` | `backend/tests` (`139 passed`), `docker build` + container smoke (`CV2_NUMPY_OK 4.10.0 2.5.3`, `/health`, PDF `73614` B) |
