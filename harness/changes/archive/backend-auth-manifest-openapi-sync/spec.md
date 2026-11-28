# Spec — Stage 3: Teacher UUID Isolation, Model Manifest SSOT & OpenAPI Sync

## Acceptance Criteria

- **AC-1 (`auth.py` & `schemas/auth.py`):**
  - `POST /api/v1/auth/device-handshake` validates non-empty `X-Teacher-UUID` (`<= 64` chars, `422` when blank), `GradingScale` ordering (`0 <= grade_3 < grade_4 < grade_5 <= 100`), non-empty stripped `teacher_name` / `school_name`, and optional `confidence_flag_threshold` (`0.0..1.0`), preserving existing threshold when omitted.
- **AC-2 (`X-Teacher-UUID` multi-teacher isolation):**
  - Shared `backend/app/core/teacher_scope.py` (`normalize_teacher_uuid`, `assert_class_access`) enforced across `classes.py`, `assignments.py`, `submissions.py`, `analytics.py`, and `reports.py`.
  - Requests carrying a different `X-Teacher-UUID` than a class owner receive `403 Forbidden`; `GET /submissions` and `GET /analytics/assignments/{id}` filter out other teachers' submissions; headerless calls remain backward-compatible; whitespace-only `X-Teacher-UUID` returns `422`.
- **AC-3 (`/api/v1/model/manifest` SSOT):**
  - `GET /api/v1/model/manifest` loads `model_name`, `model_version`, `num_classes`, `alphabet_classes`, real ONNX file `sha256`, and `preprocessing` parameters (`margin_trim_pct=4.0`, `target_occupancy_px=48.0`, `input_resolution=[64, 64]`, etc.) directly from `pipeline_spec.json` and `models/tatar_ocr_v4_handwritten39_int8.onnx`.
- **AC-4 (Stage 2 review minor tails & Starlette deprecation cleanup):**
  - `get_task_extractor_service()` `ValueError` in `/constructor/scan-task` and `/scan-task-base64` returns structured `503 Service Unavailable`.
  - `ge=1` enforced on `assignments.py` `offline-bundle?variant=` and `constructor.py` `tasks?max_cells=`.
  - `HTTP_422_UNPROCESSABLE_ENTITY` replaced with `HTTP_422_UNPROCESSABLE_CONTENT` across `backend/app/routers/`.
- **AC-5 (Living Spec Sync):**
  - `openapi_specification.yaml` matches live `app.openapi()` paths/methods 100% (no phantom `/tests/{test_id}/fork`).
  - `api_specification.md` documents all live endpoints, ONNX Runtime (`tatar_ocr_core`), and Stage 1–3 contract deltas cleanly.
- **AC-6 (Verification Suite):**
  - `backend/tests/test_stage3_auth_manifest_openapi.py` covers `AC-1`..`AC-5` (including automated OpenAPI YAML parity test) and full `pytest backend\tests -q` passes with `0` failures and `0` Starlette deprecation warnings.
