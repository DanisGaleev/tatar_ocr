# ADR 0003: Offline INT8 ONNX Edge Model (`<= 1 MB`) with HITL Verification

- **Status:** Accepted
- **Date:** 2026-11-18 (documenting decision recorded in `docs/PIPELINE.md`)

## Context

School classrooms lack guaranteed connectivity, and student handwriting must run fast on-device without exceeding APK asset budgets or sending raw images to servers.

## Decision

- Use quantized INT8 ONNX (`models/tatar_ocr_v4_handwritten39_int8.onnx`, ~1.0 MB, `TatarOCRNetV3_MultiScale`, 39 classes) for 100% offline batched inference `(N, 1, 64, 64)`.
- Pair model predictions with Human-in-the-Loop (HITL) Top-3 candidates and confidence flags (`docs/ocr_metrics_specification.md`).

## Consequences

- Model replacements or preprocessing tweaks must not regress key metrics on the 12-blank benchmark (`v3/evaluate_blanks.py`) or exceed the ~1 MB edge budget without an explicit ADR update.
