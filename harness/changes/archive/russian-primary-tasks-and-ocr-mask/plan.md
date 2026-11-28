# Plan (ExecPlan) — Russian Primary School Tasks & OCR Alphabet Mask

## Approach

Build a clean parallel `backend/app/generators/russian/` subsystem mirroring `tatar_ocr`'s deterministic `BaseTaskGenerator` / `TaskDraft` architecture, add a backward-compatible `language` (`"tt"` default, `"ru"` for Russian) discriminator across models, schemas, routers, photo extractor, PDF renderer, and checker, and implement a 33-letter Russian logit/class mask in `pipeline_spec.json` + C++20 `NativeBlankScanner` + Python `BlankOCRScanner`.

Rejected alternatives:
- Training a separate 33-class ONNX model: unnecessary because `tatar_ocr_v4_handwritten39_int8.onnx` already outputs logits for all 33 Russian letters + 6 Tatar letters; masking logits at softmax time gives immediate zero-footprint support.
- Separate database tables for Russian tasks/tests: increases duplication across routers, analytics, and submissions; adding a `language` column with `"tt"` default preserves existing endpoints and analytics seamlessly.

## Milestones

1. **M1 — Environment check & Russian generators (`backend/app/generators/russian/`):** Verify Python/CMake tooling, implement Russian phonetics, curated primary-school lexicon, and 14 generators across 4 categories.
2. **M2 — Parallel backend & photo extractor integration:** Add `language` support to base classes, registry, checker, DB models + SQLite migration, Pydantic schemas, routers (`constructor`, `assignments`, `classes`, `model`), `pdf_blank.py`, `main.py` seeds, and `task_extractor`.
3. **M3 — OCR Russian 33-letter mask:** Update `pipeline_spec.json`, `native/include/tatar_ocr.hpp`, `native/src/tatar_ocr.cpp`, `native/src/py_module.cpp`, and `blank_pipeline.py`; build native C++20 core (`ON` and `OFF`).
4. **M4 — Comprehensive testing, living spec sync & independent evaluation:** Write and run unit, API, isolation, and OCR mask tests; sync `openapi_specification.yaml`, `api_specification.md`, `docs/PIPELINE.md`, `docs/STATUS.md`; run fresh-context evaluation.

## File Ownership

| File / Area | Writer | Notes |
|---|---|---|
| `harness/changes/active/russian-primary-tasks-and-ocr-mask/*` | main | ECL change artifacts |
| `backend/app/generators/russian/*` | main | 14 Russian primary-school generators + phonetics + lexicon |
| `backend/app/generators/{base,phonetics,checker,registry,pdf_blank}.py` | main | Parallel `tt`/`ru` support |
| `backend/app/{models,schemas,routers,core/database,main}.py` | main | `language` column/migration/API/seeds/manifest |
| `backend/app/services/task_extractor/*` | main | Russian OCR + YandexGPT + mock + validation |
| `pipeline_spec.json`, `native/*`, `blank_pipeline.py` | main | 33-letter Russian OCR mask |
| `backend/tests/test_russian_*.py`, `test_ocr_russian_mask.py` | main | Comprehensive test coverage |
| `api_specification.md`, `openapi_specification.yaml`, `docs/PIPELINE.md`, `docs/STATUS.md` | main | Living spec sync |
| `harness/changes/active/russian-primary-tasks-and-ocr-mask/reviews/review.md` | evaluator | Independent Fresh-Context review |

## Risks & Mitigations

- **Worktree missing `.venv` / `.native_deps`:** Check main repo `.venv` & `.native_deps` via `BypassSandbox: true`; fallback to `uv sync` if needed.
- **Ambiguous Russian gap tasks (homophones like `л_са` = лиса/леса):** Every lexicon item is reviewed for strict 1-to-1 uniqueness and verified by unit tests.
- **C++20 Android NDK parity:** Keep `tatar_ocr.hpp` / `tatar_ocr.cpp` free of `pybind11`/Python headers and build with `-DTATAR_OCR_BUILD_PYTHON=OFF`.

## Surprises & Discoveries

*(Updated during execution whenever codebase/tests/build reveal unexpected facts)*
- `2026-11-28`: In this Git worktree, `.venv` and `.native_deps` live in the main repo (`C:\Users\galee\PycharmProjects\tatar_ocr\`). Baseline `C:\Users\galee\PycharmProjects\tatar_ocr\.venv\Scripts\python.exe -m pytest backend\tests -q` passed **139 passed** (`BypassSandbox: true`). `.native_deps` contains `nlohmann_json`, `onnxruntime-win-x64-1.20.1`, and `opencv`.

## Decision Log

- `2026-11-28`: Default `language="tt"` everywhere on existing endpoints/models so all `kara_kuyan_frontend` Tatar calls and existing 139 tests remain 100% compatible.
