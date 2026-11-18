# Architecture Map (`tatar_ocr`)

Compact system map for quick context loading. Do not duplicate deep algorithmic specs here—follow links to the living specs.

## 1. Four Subsystems

| Subsystem | Location | Role | Consumed By |
|---|---|---|---|
| **Declarative Spec (SSOT)** | `pipeline_spec.json` | Canvas `2100×2970`, ArUco `DICT_4X4_50` IDs/anchors, thresholds, 39-letter Tatar alphabet, default ONNX path | `native/`, `blank_pipeline.py`, `kara_kuyan_frontend` |
| **Native C++20 OCR Core** | `native/` (`tatar_ocr::NativeBlankScanner`) | ArUco rectification, grid snapping, border suppression, batched `(N, 1, 64, 64)` ONNX Runtime inference, `pybind11` module `tatar_ocr_native` | Python (`blank_pipeline.py`), Android NDK submodule in `kara_kuyan_frontend` |
| **ML Training & Evaluation** | `v3/`, `models/`, `benchmark/` | `TatarOCRNetV3_MultiScale` training (`train_v4.py`), INT8 ONNX export, 12-blank canonical eval (`v3/evaluate_blanks.py`) | `models/*.onnx`, `docs/PIPELINE.md` |
| **FastAPI Backend** | `backend/app/` | Teacher auth (`X-Teacher-UUID`), classes/rosters, task bank & generators, modular test assembly + PDF blanks + offline bundle, Zero-Photo submission sync, analytics, Excel gradebook export, model manifest | `kara_kuyan_frontend` |

## 2. Backend Module Layout (`backend/app/`)

- `core/` — `config.py` (`Settings`, SQLite `DATABASE_URL`, Yandex OCR/GPT env vars), `database.py` (`init_db`, `AsyncSessionLocal`).
- `models/` — SQLAlchemy ORM (`teacher.py`, `school.py`, `task.py`, `test.py`).
- `schemas/` — Pydantic request/response contracts.
- `routers/` — `auth.py`, `classes.py`, `constructor.py`, `assignments.py`, `submissions.py`, `analytics.py`, `reports.py`, `model.py`.
- `generators/` — deterministic Tatar task generators + `phonetics.py` (`build_expected_cells`).
- `services/` — business logic, PDF blank generation, answer checking, textbook task extractor.

## 3. Cross-Repo Contract with `kara_kuyan_frontend`

- **Shared C++20 submodule:** `kara_kuyan_frontend` compiles `native/src/tatar_ocr.cpp` with `TATAR_OCR_BUILD_PYTHON=OFF` against OpenCV 5 + `onnxruntime-android`. Breaking changes to `NativeBlankScanner` signatures or `pipeline_spec.json` schema break the mobile build.
- **Shared HTTP API:** Mobile client targets `/api/v1/*` documented in `api_specification.md` and `openapi_specification.yaml`.
- **Zero-Photo boundary:** Student worksheet photos are never persisted on disk or transmitted to `backend/`. Only `constructor.py` textbook-photo task import handles an image upload transiently.
