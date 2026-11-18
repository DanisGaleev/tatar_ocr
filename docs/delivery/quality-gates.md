# Quality Gates

Every gate follows the **Default-FAIL** rule: a gate passes only with an observed command + output recorded after the final edit.

| Contour | Gate | Command / Check | Current Baseline |
|---|---|---|---|
| **Backend API & Domain** | Pytest suite | `.\.venv\Scripts\python.exe -m pytest backend\tests -q` | ✅ `139 passed` (2026-11-18) |
| **API Contract** | OpenAPI & prose sync | Diff `backend/app/routers/` + `schemas/` against `openapi_specification.yaml` & `api_specification.md` | ⚠️ Manual diff required on API edits |
| **Native C++20 Core** | CMake Release build | `cmake -S native -B native\build -DCMAKE_BUILD_TYPE=Release` then `cmake --build native\build --config Release` | ⏳ Verify whenever `native/` changes |
| **Android NDK Compatibility** | No Python-only leak in `tatar_ocr_core` | Keep `tatar_ocr.hpp`/`tatar_ocr.cpp` free of `<pybind11/...>` (`TATAR_OCR_BUILD_PYTHON=OFF` safe) | ✅ Enforced by `native/CMakeLists.txt` |
| **ML / OCR Pipeline** | 12-blank benchmark | `.\.venv\Scripts\python.exe v3\evaluate_blanks.py --model models\tatar_ocr_v4_handwritten39_int8.onnx` | ✅ Recorded in `docs/PIPELINE.md` (`96.88%` Acc_char) |
| **Dataset / Augmentation** | Pipeline test script | `.\.venv\Scripts\python.exe test_pipeline.py` | ⏳ Verify when `tatar_ocr_augmentation.py` / `tatar_ocr_dataset.py` change |
| **Privacy (Zero-Photo / 152-ФЗ)** | No student blank persistence | Audit `backend/app/routers/` & `services/` for disk/DB image writes | ✅ Enforced by ADR-0004 |
