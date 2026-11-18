# Living Specifications Registry (OpenSpec)

These files are the canonical **living specifications** of `tatar_ocr`. Every Structured Change must declare `ADDED / MODIFIED / REMOVED` deltas against them in its `spec.md` and sync them before closure.

| Domain | Living Spec File | Scope |
|---|---|---|
| **HTTP API (prose)** | [`api_specification.md`](../../api_specification.md) | Zero-Photo flow, routers, payloads, grading/HITL semantics |
| **HTTP API (schema)** | [`openapi_specification.yaml`](../../openapi_specification.yaml) | Machine-readable OpenAPI contract for `kara_kuyan_frontend` |
| **Pipeline & Geometry (SSOT)** | [`pipeline_spec.json`](../../pipeline_spec.json) | Canvas `2100×2970`, ArUco IDs/anchors, thresholds, 39-char alphabet, ONNX paths |
| **OCR Pipeline Algorithms** | [`docs/PIPELINE.md`](../PIPELINE.md) | Model v4 choice, 7-stage rectification/cell-extraction/inference pipeline |
| **OCR Metrics & Calibration** | [`docs/ocr_metrics_specification.md`](../ocr_metrics_specification.md) | `Acc_char`, `CER`, `R_Tatar`, `FEM`, `FFM`, `ZTSR`, `FAR`, `HCER` |
| **Backend Config & Env** | [`backend/app/core/config.py`](../../backend/app/core/config.py) | Runtime settings and physical A4 template constants |
