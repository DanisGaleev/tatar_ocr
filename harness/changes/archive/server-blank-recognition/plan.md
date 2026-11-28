# Plan (ExecPlan) — Server-Side Blank Photo Recognition & Verification

## Approach

Build a clean 3-layer integration inside `backend/app/`:
1. **C++ Bridge Layer (`backend/app/services/blank_scanner_bridge.py`):** `BlankScannerBridgeProtocol` + `NativeBlankScannerBridge` wrapping `tatar_ocr_native.NativeBlankScanner` (lazy thread-safe singleton, path/DLL discovery matching `blank_pipeline.py` + `NATIVE_BUILD_DIR`/`TATAR_OCR_NATIVE_DIR`), decoding image bytes in RAM (`cv2.imdecode`), decoding header QR payloads (`cv2.QRCodeDetector` on raw + rectified sheet header), and returning typed `RawBlankScanResult`. Rejected alternative: shelling out to a Python CLI subprocess per request (too slow due to ONNX session reload and harder error mapping).
2. **Domain Service (`backend/app/services/blank_recognition_service.py`):** `BlankRecognitionService` validating image payloads in RAM, running the C++ bridge in Starlette `run_in_threadpool` (C++ releases GIL), resolving assignment/variant/student/teacher preferences, running `verify_test_submission`, computing HITL `review_required` & `zero_touch`, returning all local-parity OCR + verification fields plus `submission_draft` (and optional `auto_save`).
3. **API Layer (`backend/app/schemas/submission.py`, `backend/app/routers/submissions.py`):** `POST /api/v1/submissions/recognize` (multipart) and `POST /api/v1/submissions/recognize-base64` (JSON).

## Milestones

1. **M1 — Config, Schemas & C++ Bridge:** Update `config.py`, add recognition schemas to `schemas/submission.py`, implement `services/blank_scanner_bridge.py`.
2. **M2 — Recognition Service & Endpoints:** Implement `services/blank_recognition_service.py` and mount `/recognize` and `/recognize-base64` on `routers/submissions.py`.
3. **M3 — Tests & Real Blank Photo Verification:** Add `backend/tests/test_blank_recognition.py` (unit + API + real blank photo tests) and run full `pytest backend\tests`.
4. **M4 — Living Specs Sync & Fresh-Context Review:** Sync `api_specification.md`, `openapi_specification.yaml`, `docs/adr/0004-zero-photo-backend-contract.md`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`; run Fresh-Context Evaluator subagent; commit and push branch.

## File Ownership

| File / Area | Writer | Notes |
|---|---|---|
| `backend/app/core/config.py` | Main Agent | OCR model/spec/native dir & size limit settings |
| `backend/app/schemas/submission.py` | Main Agent | Recognition request/response schemas |
| `backend/app/services/blank_scanner_bridge.py` | Main Agent | C++20 `tatar_ocr_native` bridge + QR decode |
| `backend/app/services/blank_recognition_service.py` | Main Agent | Validation, grading, HITL flags, submission draft |
| `backend/app/routers/submissions.py` | Main Agent | `/recognize` and `/recognize-base64` endpoints |
| `backend/tests/test_blank_recognition.py` | Main Agent | Unit, API, and real blank photo tests |
| Living specs & ADR & STATUS | Main Agent | `api_specification.md`, `openapi_specification.yaml`, `docs/adr/0004*`, `docs/ARCHITECTURE.md`, `docs/STATUS.md` |
| `harness/changes/active/server-blank-recognition/reviews/review.md` | Evaluator Subagent | Fresh-Context 4-lens review |

## Risks & Mitigations

- **Worktree missing `.venv` / `native/build`:** Support `TATAR_OCR_NATIVE_DIR` / `NATIVE_BUILD_DIR` in the bridge and verify with main repo `.venv` + `native/build` (without hardcoding machine paths in source code).
- **152-ФЗ / ADR 0004 compliance:** Never write uploaded bytes to temp files, DB, or logs; update ADR-0004 explicitly.
- **1-based C++ `cell_idx` vs 0-based `checker.py` `cell_index`:** Map `cell_idx - 1` explicitly when passing cells to `verify_test_submission`, and keep native 1-based `cell_idx` in `RecognizedCellItem` for local-scanner parity.

## Surprises & Discoveries

- `2026-11-27`: Worktree lacks gitignored `.venv` and `native/build`, whereas main repo at `C:\Users\galee\PycharmProjects\tatar_ocr` has Python 3.12.11 `.venv` and compiled `tatar_ocr_native.cp312-win_amd64.pyd`. Verified `BlankOCRScanner` runs cleanly on `benchmark/images/blank_01.jpg` and `samples/test_blank_phone_photo.jpg` (QR decoded: `{"tid":"TAT-2026-Q1","var":1,"page":1,"tot":1,"n_q":8}`).

## Decision Log

- `2026-11-27`: Provide both `POST /api/v1/submissions/recognize` (multipart) and `POST /api/v1/submissions/recognize-base64` (JSON) under `/api/v1/submissions`, matching `/constructor/scan-task` and `/scan-task-base64` conventions.
- `2026-11-27`: Keep `confidence` on `0..100` in native-parity `RecognizedCellItem` and normalized `0..1` in `SubmissionCellResult` / `submission_draft` so both local scanner consumers and `POST /api/v1/submissions` / `analytics.py` work seamlessly.
