# Independent Review (Fresh-Context Evaluator) — Server-Side Blank Photo Recognition & Verification

Evaluated in a fresh context / independent review pass against `spec.md`, `checks.json`, and actual workspace state.

## Verdict

`accept` <!-- accept | accept-with-notes | changes-required -->

## Evaluator Verification

- [x] Read `spec.md` deltas and `checks.json` before inspecting code.
- [x] Re-ran key test/build/eval commands independently (never trust narrated results).
- [x] Verified every `"PASS"` item in `checks.json` reflects the **final** edited state.
- [x] Verified living specs (`docs/specs/README.md`) match `spec.md` Spec Deltas.

## Subsystem Lenses Applied

- **API Contract & Zero-Photo (152-ФЗ):** `PASS` — `POST /api/v1/submissions/recognize` and `POST /api/v1/submissions/recognize-base64` in `backend/app/routers/submissions.py` and schemas in `backend/app/schemas/submission.py` match `openapi_specification.yaml` (validated via `yaml.safe_load`: 20 paths, 49 schemas) and `api_specification.md`. `docs/adr/0004-zero-photo-backend-contract.md` explicitly documents the transient in-RAM opt-in fallback. Code inspection confirms uploaded bytes are decoded via `np.frombuffer` + `cv2.imdecode` in RAM only, with zero disk, DB, or log persistence of student photos.
- **C++20 / Python / Android NDK Parity:** `PASS` — `git status --short` confirms zero modifications to `native/`, `pipeline_spec.json`, or `models/`. `blank_scanner_bridge.py` wraps `tatar_ocr_native.NativeBlankScanner` cleanly without altering the C++20 core or `TATAR_OCR_BUILD_PYTHON=OFF` NDK compatibility, and explicitly maps 1-based native `cell_idx` to 0-based `cell_index` for `checker.py`.
- **ML Benchmark & Metrics (`ocr_metrics_specification.md`):** `N/A` — No changes to model weights (`models/tatar_ocr_v4_handwritten39_int8.onnx`), preprocessing, or `pipeline_spec.json`. Real blank photo E2E tests (`samples/test_blank_phone_photo.jpg` and `benchmark/images/blank_01.jpg`) pass through FastAPI + `tatar_ocr_native`.
- **Backend QA, Migrations & Hygiene:** `PASS` — Independently executed `$env:TATAR_OCR_NATIVE_DIR="C:\Users\galee\PycharmProjects\tatar_ocr\native\build"; C:\Users\galee\PycharmProjects\tatar_ocr\.venv\Scripts\python.exe -m pytest backend\tests -q` → `147 passed, 3 warnings in 9.75s`. No temporary `tmp_*`, `debug_*`, or `_peek_*` files exist in the repository.

## Deviations from Plan

| Deviation | Justified? | Note |
|---|---|---|
| `docs/STATUS.md` already lists `server-blank-recognition` in `Completed Structured Changes` before archival | Yes (harmless) | Main agent pre-populated `STATUS.md` ahead of `AC-7` closure; no functional impact. |

## Issues Found

- none

## Residual Risks

- Runtime availability of `POST /api/v1/submissions/recognize*` in deployment requires `tatar_ocr_native` compiled for the host Python/OS (or `NATIVE_BUILD_DIR` / `TATAR_OCR_NATIVE_DIR` configured) alongside `models/tatar_ocr_v4_handwritten39_int8.onnx` and `pipeline_spec.json`; otherwise endpoints fail cleanly with HTTP `503`.
- QR metadata decoding uses `cv2.QRCodeDetector` on the rectified header crop, full rectified sheet, and raw image; severely degraded QR codes fall back gracefully to `qr_metadata=None` (or explicit `assignment_id`/`variant` query/body params).
