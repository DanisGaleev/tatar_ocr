# ADR 0001: C++20 Native OCR Core as Single Source of Truth

- **Status:** Accepted
- **Date:** 2026-11-18 (documenting decision established in commit `d27477b`)

## Context

`tatar_ocr` runs both in Python (research, benchmarking, `visualize_pipeline.py`) and on Android devices inside `kara_kuyan_frontend`. Maintaining separate OpenCV/ONNX preprocessing implementations in Python and Kotlin/C++ caused parity drift.

## Decision

All critical CV preprocessing and batched ONNX Runtime inference live in `native/` (`tatar_ocr_core` static library, `tatar_ocr::NativeBlankScanner` in `native/include/tatar_ocr.hpp` and `native/src/tatar_ocr.cpp`).
- Python consumes it via `pybind11` (`native/src/py_module.cpp`, `blank_pipeline.py`) when `TATAR_OCR_BUILD_PYTHON=ON`.
- Android NDK consumes `tatar_ocr_core` directly as a Git submodule with `TATAR_OCR_BUILD_PYTHON=OFF`.

## Consequences

- `tatar_ocr.hpp` and `tatar_ocr.cpp` must never depend on `pybind11` or Python headers.
- Changes to `NativeBlankScanner` public types or `native/CMakeLists.txt` require checking downstream compatibility with `kara_kuyan_frontend`.
