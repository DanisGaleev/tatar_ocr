# Project Status & Session Handoff

## State Snapshot (updated 2026-11-18)

- **Repository baseline:** commit `f27a4b1` (`README.md` showcase), native C++20 OCR core (`d27477b`), ML model v4 INT8 (`models/tatar_ocr_v4_handwritten39_int8.onnx`, ~1.0 MB), FastAPI backend (`backend/app/`).
- **Verified Backend Suite:** `.\.venv\Scripts\python.exe -m pytest backend\tests -q` → **162 passed** (0 warnings, Stage 3 verified 2026-11-28).
- **Production OCR Model (`v4 INT8`):** `models/tatar_ocr_v4_handwritten39_int8.onnx` (`1,024,619` bytes, SHA-256 `656096ed97d25cfdd07390f292a7b9a9f838481adf126587e40dccae25c9f191`). Stage 2 C++20 `evaluate_blanks.py` run (`449` chars, `57` fields): `Acc_char = 96.88% (435/449)`, `R_Tatar = 97.33% (73/75)`, `FEM = 78.95% (45/57)`, `FFM (Lev<=1) = 96.49% (55/57)` (0-diff parity vs `HEAD`-equiv spec).
- **Agent Harness:** Initialized unified ECL + OpenSpec + Anthropic Harness + ADR setup (`AGENTS.md`, `docs/ECL.md`, `harness/templates/change/`, `.agents/skills/tatar-ocr-delivery/`).

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
| `backend-critical-correctness` | Fixed Stage 1 runtime logger, DB FKs/constraints, `checker.py`, `submissions.py` recalculation/upsert, `analytics.py`, `reports.py`, `classes.py` *(archived)* | `backend/tests` (`148 passed`), `harness/changes/archive/backend-critical-correctness/reviews/review.md` (`accept-with-notes`) |
| `backend-ocr-pdf-generators-contract` | Aligned `pipeline_spec.json` & C++20 `native/` (`short_h`, 1–12 cells), PDF SSOT, 39-letter OCR alphabet, phonetics/lexicon/24 generators, `constructor.py`, `assignments.py`, and `task_extractor` *(archived)* | `backend/tests` (`157 passed`), `evaluate_blanks.py` (`Acc_char=96.88%`, `R_Tatar=97.33%`), `harness/changes/archive/backend-ocr-pdf-generators-contract/reviews/review.md` (`accept-with-notes`) |
| `backend-auth-manifest-openapi-sync` | Enforced `X-Teacher-UUID` multi-teacher isolation & `/auth/device-handshake` validation, aligned `/model/manifest` to `pipeline_spec.json` SSOT, fixed Stage 2 minor tails & `HTTP_422_UNPROCESSABLE_CONTENT`, synced `openapi_specification.yaml` (30 endpoints) & `api_specification.md` *(archived)* | `backend/tests` (`162 passed`, 0 warnings), `harness/changes/archive/backend-auth-manifest-openapi-sync/reviews/review.md` (`accept-with-notes`) |
