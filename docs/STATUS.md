# Project Status & Session Handoff

## State Snapshot (updated 2026-11-28)

- **Repository baseline:** commit `1ac7df3` (`main`), native C++20 OCR core (`tatar_ocr_core`), ML model v4 INT8 (`models/tatar_ocr_v4_handwritten39_int8.onnx`, ~1.0 MB), FastAPI backend (`backend/app/`).
- **Verified Backend Suite:** `C:\Users\galee\PycharmProjects\tatar_ocr\.venv\Scripts\python.exe -m pytest backend\tests -q` → **169 passed** (5 Starlette deprecation warnings on `HTTP_422_UNPROCESSABLE_ENTITY`, verified 2026-11-28).
- **Production OCR Model (`v4 INT8`):** `models/tatar_ocr_v4_handwritten39_int8.onnx` (`1,024,619` bytes). Recorded benchmark on 12 canonical blanks (`449` chars, `57` fields): `Acc_char = 96.88%`, `R_Tatar = 97.33%`, `FEM = 78.95% (45/57)`, `FFM (Lev<=1) = 96.49% (55/57)`. Russian mask (`language="ru"`, 33 classes) on 374 Russian chars: `365/374 (97.59%)`, 0 Tatar leaks.
- **Agent Harness:** Initialized unified ECL + OpenSpec + Anthropic Harness + ADR setup (`AGENTS.md`, `docs/ECL.md`, `harness/templates/change/`, `.agents/skills/tatar-ocr-delivery/`).

## Active Change

- *None* (`harness/changes/active/` is empty).

## Session Handoff (for next agent session)

- **Current focus:** Ready to accept the next feature or bugfix via `docs/ECL.md`.
- **Known discrepancy in docs to keep in mind:** `README.md` currently prints `--onnx` for `v3/evaluate_blanks.py`, whereas the actual CLI flag in `v3/evaluate_blanks.py` is `--model`.
- **Cross-repo open contract items (from `kara_kuyan_frontend` integration):**
  - Confirm `X-Teacher-UUID` multi-teacher isolation semantics and production auth roadmap before auth changes.
  - Keep `openapi_specification.yaml` and `api_specification.md` aligned whenever any router/schema in `backend/app/` changes.
  - Keep `native/` CMake build compatible with `TATAR_OCR_BUILD_PYTHON=OFF` (Android NDK) as well as `ON` (`pybind11`).

## Completed Structured Changes

| Change | Outcome | Evidence |
|---|---|---|
| *(harness bootstrap)* | Installed unified ECL + OpenSpec + Anthropic Harness + ADR scaffolding | `AGENTS.md`, `docs/ECL.md`, `backend/tests` (`139 passed`) |
| `russian-primary-tasks-and-ocr-mask` | Added 14 Russian primary-school (1–4) generators, all 4 addition modes with `tt`/`ru` isolation, and 33-letter Russian OCR mask | `169 passed`; C++20 `ON`/`OFF` builds pass; Russian mask `97.59%` (`0` Tatar leaks); `reviews/review.md` (`accept`) |
