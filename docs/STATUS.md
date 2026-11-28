# Project Status & Session Handoff

## State Snapshot (updated 2026-11-28)

- **Repository baseline:** merged `main` (`local_backend_deployment`, `russian_tasks_generator_backend`, `server-blank-recognition`) + `backend_business_logic_audit` (Stages 1–3), native C++20 OCR core (`tatar_ocr_core`), ML model v4 INT8 (`models/tatar_ocr_v4_handwritten39_int8.onnx`, ~1.0 MB, SHA-256 `656096ed97d25cfdd07390f292a7b9a9f838481adf126587e40dccae25c9f191`), FastAPI backend (`backend/app/`).
- **Verified Backend Suite:** `.\.venv\Scripts\python.exe -m pytest backend\tests -q` → verified after merge with `origin/main` (Stages 1–3 + Russian tasks + blank recognition + Docker).
- **Production OCR Model (`v4 INT8`):** `models/tatar_ocr_v4_handwritten39_int8.onnx` (`1,024,619` bytes). C++20 `evaluate_blanks.py` on 12 canonical blanks (`449` chars, `57` fields): `Acc_char = 96.88% (435/449)`, `R_Tatar = 97.33% (73/75)`, `FEM = 78.95% (45/57)`, `FFM (Lev<=1) = 96.49% (55/57)`. Russian mask (`language="ru"`, 33 classes) on 374 Russian chars: `365/374 (97.59%)`, 0 Tatar leaks.
- **Agent Harness:** Unified ECL + OpenSpec + Anthropic Harness + ADR setup (`AGENTS.md`, `docs/ECL.md`, `harness/templates/change/`, `.agents/skills/tatar-ocr-delivery/`).

## Active Change

- *None* (`harness/changes/active/` contains only `.gitkeep`; Stages 1–3 archived).

## Session Handoff (for next agent session)

- **Current focus:** All three backend audit stages (`backend-critical-correctness`, `backend-ocr-pdf-generators-contract`, `backend-auth-manifest-openapi-sync`) are complete and archived (`162 passed`, 0 warnings). Ready for git review/commit and Android NDK `.so` rebuild in `kara_kuyan_frontend`.
- **Known note for existing DBs:** Pre-existing `tatar_ocr.db` files on disk created before Stage 1 lack new `SubmissionModel`/`ClassAssignmentModel` FK and unique constraints (`Base.metadata.create_all` does not alter existing tables); recreate or migrate local dev `.db` if needed.
- **Known discrepancy in docs to keep in mind:** `README.md` currently prints `--onnx` for `v3/evaluate_blanks.py`, whereas the actual CLI flag in `v3/evaluate_blanks.py` is `--model`.
- **Cross-repo open contract items (from `kara_kuyan_frontend` integration):**
  - `X-Teacher-UUID` multi-teacher isolation semantics implemented cleanly in Stage 3 (`403` cross-teacher, backward-compatible when omitted).
  - Keep `openapi_specification.yaml` and `api_specification.md` aligned whenever any router/schema in `backend/app/` changes.
  - Keep `native/` CMake build compatible with `TATAR_OCR_BUILD_PYTHON=OFF` (Android NDK) as well as `ON` (`pybind11`).

## Completed Structured Changes

| Change | Outcome | Evidence |
|---|---|---|
| *(harness bootstrap)* | Installed unified ECL + OpenSpec + Anthropic Harness + ADR scaffolding | `AGENTS.md`, `docs/ECL.md`, `backend/tests` (`139 passed`) |
| `server-blank-recognition` | Added C++20 bridge, transient in-RAM recognition service, `/submissions/recognize*` endpoints | `backend/tests/test_blank_recognition.py` (`8 passed`), full suite `147 passed` |
| `local-backend-deployment` | Fixed missing `logger` in `backend/app/main.py`; slimmed `backend/Dockerfile` via `backend/requirements.txt` & `.dockerignore` with non-root `appuser` | `backend/tests` (`139 passed`), `docker build` + container smoke (`CV2_NUMPY_OK 4.10.0 2.5.3`, `/health`, PDF `73614` B) |
| `russian-primary-tasks-and-ocr-mask` | Added 14 Russian primary-school (1–4) generators, all 4 addition modes with `tt`/`ru` isolation, and 33-letter Russian OCR mask | `169 passed`; C++20 `ON`/`OFF` builds pass; Russian mask `97.59%` (`0` Tatar leaks); `reviews/review.md` (`accept`) |
| `backend-critical-correctness` | Fixed Stage 1 runtime logger, DB FKs/constraints, `checker.py`, `submissions.py` recalculation/upsert, `analytics.py`, `reports.py`, `classes.py` *(archived)* | `backend/tests` (`148 passed`), `harness/changes/archive/backend-critical-correctness/reviews/review.md` (`accept-with-notes`) |
| `backend-ocr-pdf-generators-contract` | Aligned `pipeline_spec.json` & C++20 `native/` (`short_h`, 1–12 cells), PDF SSOT, 39-letter OCR alphabet, phonetics/lexicon/24 generators, `constructor.py`, `assignments.py`, and `task_extractor` *(archived)* | `backend/tests` (`157 passed`), `evaluate_blanks.py` (`Acc_char=96.88%`, `R_Tatar=97.33%`), `harness/changes/archive/backend-ocr-pdf-generators-contract/reviews/review.md` (`accept-with-notes`) |
| `backend-auth-manifest-openapi-sync` | Enforced `X-Teacher-UUID` multi-teacher isolation & `/auth/device-handshake` validation, aligned `/model/manifest` to `pipeline_spec.json` SSOT, fixed Stage 2 minor tails & `HTTP_422_UNPROCESSABLE_CONTENT`, synced `openapi_specification.yaml` & `api_specification.md` *(archived)* | `backend/tests` (`162 passed`, 0 warnings), `harness/changes/archive/backend-auth-manifest-openapi-sync/reviews/review.md` (`accept-with-notes`) |
