# Tasks — Server-Side Blank Photo Recognition & Verification

| # | Task | Files owned | Check IDs | Status | Verification evidence |
|---|---|---|---|---|---|
| 1 | Add OCR/native config settings and recognition Pydantic schemas | `backend/app/core/config.py`, `backend/app/schemas/submission.py` | `AC-1`, `AC-3` | done | Verified via `pytest backend\tests\test_blank_recognition.py -v` (8 passed) |
| 2 | Implement C++20 `tatar_ocr_native` bridge (`blank_scanner_bridge.py`) | `backend/app/services/blank_scanner_bridge.py` | `AC-1` | done | `TestBlankScannerBridgeUnit` (3/3) + `TestRealBlankPhotoRecognitionE2E` (2/2) passed |
| 3 | Implement recognition & grading service (`blank_recognition_service.py`) | `backend/app/services/blank_recognition_service.py` | `AC-2` | done | `TestBlankRecognitionServiceAndAPI` (3/3) + E2E passed |
| 4 | Add `/recognize` & `/recognize-base64` endpoints in `submissions.py` | `backend/app/routers/submissions.py` | `AC-3` | done | Multipart & base64 endpoints + status codes 200/400/404/413/422/503 passed |
| 5 | Write unit, API, and real-photo tests (`test_blank_recognition.py`) and run full suite | `backend/tests/test_blank_recognition.py` | `AC-4`, `AC-5`, `AC-6` | done | `147 passed, 3 warnings in 11.96s` |
| 6 | Sync living specs, ADR-0004, ARCHITECTURE.md, STATUS.md and run Fresh-Context Evaluator | `api_specification.md`, `openapi_specification.yaml`, `docs/adr/0004-zero-photo-backend-contract.md`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`, `reviews/review.md` | `AC-7` | done | `reviews/review.md` verdict `accept`; OpenAPI valid; `147 passed in 9.75s` |

## Session Handoff

- **Last completed slice:** Task 6 complete (`AC-7` PASS; archived to `harness/changes/archive/server-blank-recognition/`).
- **Next immediate step:** Ready for next feature/bugfix via `docs/ECL.md`.
- **Uncommitted / tricky state:** None.

## Completion Definition

- [x] `AC-1`..`AC-6` in `checks.json` have `"status": "PASS"` with recorded `command` + `evidence`.
- [x] `AC-7` in `checks.json` has `"status": "PASS"` after Fresh-Context Evaluator verdict `accept`.
- [x] No open question in `spec.md` was silently guessed.
- [x] Spec Deltas from `spec.md` synced into living docs (`docs/specs/README.md`).
- [x] Independent Fresh-Context Review in `reviews/review.md` recorded with verdict `accept`.
