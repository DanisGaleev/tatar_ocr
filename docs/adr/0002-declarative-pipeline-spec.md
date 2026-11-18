# ADR 0002: Declarative `pipeline_spec.json` as SSOT

- **Status:** Accepted
- **Date:** 2026-11-18 (documenting decision established in commit `d27477b`)

## Context

Blank dimensions (`2100×2970`), ArUco dictionary (`DICT_4X4_50`), corner/question marker IDs, canonical anchor coordinates, thresholding parameters, 39-character Tatar uppercase alphabet, and default ONNX model paths are shared across C++, Python, PDF blank generation, and the mobile app.

## Decision

`pipeline_spec.json` is the single declarative source of truth for pipeline geometry, alphabet ordering, and inference defaults.

## Consequences

- Never hardcode differing coordinates, thresholds, or class-index orderings in Python or C++ without updating `pipeline_spec.json` and verifying class-index alignment (`А, Ә, Б, ...`).
- Any schema change to `pipeline_spec.json` is a Structured Change and must be checked against `native/src/tatar_ocr.cpp` parsing and `kara_kuyan_frontend` asset loading.
