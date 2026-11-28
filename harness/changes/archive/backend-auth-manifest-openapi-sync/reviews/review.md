# Independent Review — Stage 3 (`backend-auth-manifest-openapi-sync`)

**Verdict:** `accept-with-notes` *(0 Critical, 0 Major blockers; 3 Minor non-blocking observations)*

## Scope & Archive Verification

- **Stage 3 active artifacts:** Reviewed `harness/changes/active/backend-auth-manifest-openapi-sync/{summary.md,spec.md,plan.md,tasks.md,checks.json}` and `docs/STATUS.md`.
- **Previous stages archived:** Confirmed `harness/changes/archive/backend-critical-correctness/` (`summary.md` status `done`, `tasks.md` 8/8 `done`, `checks.json` 9/9 `PASS`) and `harness/changes/archive/backend-ocr-pdf-generators-contract/` (`summary.md` status `done`, `tasks.md` 6/6 `done`, `checks.json` 11/11 `PASS`) exist and are marked complete.

---

## Per-Lens Assessment

### 1. API & Zero-Photo / 152-ФЗ — PASS
- **Shared teacher isolation (`backend/app/core/teacher_scope.py`):**
  - `normalize_teacher_uuid(raw, required=False)` trims input, enforces `422` (`HTTP_422_UNPROCESSABLE_CONTENT`) when missing with `required=True` or when blank / `> 64` chars, and returns `None` when omitted on optional endpoints.
  - `assert_class_access(cls_obj, teacher_uuid)` raises `403 Forbidden` whenever both `teacher_uuid` and `cls_obj.teacher_uuid` are set and differ.
- **Auth handshake (`backend/app/schemas/auth.py`, `backend/app/routers/auth.py`):**
  - `POST /api/v1/auth/device-handshake` requires valid `X-Teacher-UUID` (`required=True`), validates `0 <= grade_3_min_pct < grade_4_min_pct < grade_5_min_pct <= 100`, strips and rejects whitespace-only `teacher_name` / `school_name`, enforces `0.0 <= confidence_flag_threshold <= 1.0`, defaults to `0.65` on insert, and preserves existing `teacher.confidence_flag_threshold` when omitted on update.
- **Router enforcement (`classes.py`, `assignments.py`, `submissions.py`, `analytics.py`, `reports.py`):**
  - `classes.py`: `GET /classes` filters to `teacher_uuid` or shared legacy (`None`); `GET /{class_id}/students`, `POST /{class_id}/students/bulk-import`, `POST /{class_id}/assignments`, `GET /{class_id}/assignments`, and `GET /{class_id}/assignments/{assignment_id}/batch-blanks.pdf` enforce `normalize_teacher_uuid` + `assert_class_access`.
  - `assignments.py`: `GET /assignments` scopes `assigned_classes` to `teacher_uuid` or `None`; `GET /{assignment_id}/batch-blanks.pdf` delegates `x_teacher_uuid` to `get_class_batch_blanks_pdf`; `GET /{assignment_id}/blank.pdf` enforces `assert_class_access` whenever `student_id` is supplied.
  - `submissions.py`: `_load_submission_context` enforces `assert_class_access` for `POST /submissions` and `POST /submissions/batch-sync`; `GET /submissions` enforces `assert_class_access` on explicit `class_id` and joins `ClassModel` to exclude other teachers' submissions.
  - `analytics.py`: `GET /analytics/students/{student_id}` and `GET /analytics/classes/{class_id}` enforce `assert_class_access` and resolve `confidence_flag_threshold` via `_get_teacher_flag_threshold`; `GET /analytics/assignments/{assignment_id}` joins `ClassModel` to scope submissions to the caller's classes (plus legacy `None`).
  - `reports.py`: `GET /reports/assignments/{assignment_id}/gradebook.xlsx` enforces `assert_class_access`.
- **Stage 2 minor tails (`constructor.py`, `assignments.py`):**
  - `/constructor/scan-task` and `/constructor/scan-task-base64` catch `ValueError` from `get_task_extractor_service()` and return `503 Service Unavailable`.
  - `ge=1` enforced on `GET /api/v1/assignments/{assignment_id}/offline-bundle?variant=` and `GET /api/v1/constructor/tasks?max_cells=`.
- **Zero-Photo / 152-ФЗ alignment:** Student worksheet photos remain exclusively on-device; `submissions.py` ingests structured JSON only, `/constructor/scan-task*` processes textbook images strictly in memory, and `X-Teacher-UUID` prevents cross-teacher access to student rosters, submissions, analytics, and Excel/CSV gradebooks.

### 2. SSOT & Spec Parity — PASS
- **`GET /api/v1/model/manifest` (`backend/app/routers/model.py`):**
  - Dynamically reads `pipeline_spec.json` (`REPO_ROOT / "pipeline_spec.json"`): `model_name="tatar_ocr_v4_handwritten39_int8"`, `model_version="1.0"`, `num_classes=39`, `alphabet_classes` matching `spec["model"]["alphabet"]`, `input_resolution=[64, 64]`, `margin_trim_pct=4.0`, `ink_threshold_offset=22.0`, `ink_min_area_pixels=35`, `bounding_box_padding_px=2`, `target_occupancy_px=48.0`, `canvas_background_val=250`.
  - Live file digest of `models/tatar_ocr_v4_handwritten39_int8.onnx` independently verified as `656096ed97d25cfdd07390f292a7b9a9f838481adf126587e40dccae25c9f191`, matching `api_specification.md` and `/api/v1/model/manifest`.
- **`openapi_specification.yaml` vs live `app.openapi()`:**
  - Exact parity verified across all **30** `(path, method)` operations (`live["paths"] == yaml["paths"]` is `True`, `live["components"]["schemas"] == yaml["components"]["schemas"]` is `True`).
  - Phantom `/api/v1/constructor/tests/{test_id}/fork` is removed from both `openapi_specification.yaml` and `api_specification.md`.
  - Top-level diffs between `app.openapi()` and `openapi_specification.yaml` are intentional contract additions (`servers`, `info.version: 1.1.0`, and `components.securitySchemes.TeacherUUID`).
- **`api_specification.md`:** Covers all active routers, `X-Teacher-UUID` isolation, `503` on missing `YANDEX_API_KEY`, `ge=1` parameters, ONNX Runtime (`tatar_ocr_core`), and the SSOT `/api/v1/model/manifest` contract.

### 3. Backward Compatibility & Frontend Contract (`kara_kuyan_frontend`) — PASS
- Headerless requests (omitting `X-Teacher-UUID`) continue to work cleanly across `/classes`, `/assignments`, `/submissions`, `/analytics`, and `/reports`, and legacy shared classes (`teacher_uuid=None`) remain accessible when `X-Teacher-UUID` is provided.
- `DeviceHandshakeResponse` and `ModelManifestResponse` shapes are unchanged; omitting `confidence_flag_threshold` on repeat handshakes preserves the stored value.
- `HTTP_422_UNPROCESSABLE_ENTITY` has `0` occurrences across `backend/` (replaced with `HTTP_422_UNPROCESSABLE_CONTENT`).

### 4. Backend QA & Workspace Hygiene — PASS
- **Independent pytest execution:**
  - Command: `C:\Users\galee\PycharmProjects\tatar_ocr\.venv\Scripts\python.exe -m pytest -p no:cacheprovider backend\tests -q`
  - Result: **`162 passed in 18.57s`** (exit code `0`, `0` warnings).
- **Workspace hygiene:** Recursive scan confirmed zero stray `tmp_*`, `debug_*`, or `build_core` artifacts in the repository.

---

## Findings

1. **[Minor] `bulk_import_students` empty-input early return before ownership check (`backend/app/routers/classes.py:118-137`):**
   - If `payload.raw_text` contains only whitespace, `bulk_import_students` returns `BulkImportStudentsResponse(added_count=0, students=[])` before checking `assert_class_access(cls_obj, teacher_uuid)`. No data is leaked or mutated, so this is non-blocking, though moving the class lookup + `assert_class_access` ahead of the empty-lines return would make cross-teacher `403` behavior completely uniform.
2. **[Minor] Minor documentation nits in `api_specification.md` (`api_specification.md:67,228,461-563`):**
   - Line 67 reads `blank headers > 64 chars` (intended: `blank headers or > 64 chars`).
   - Line 228 lists `X-Teacher-UUID` on `GET /api/v1/assignments/{assignment_id}/offline-bundle`, whereas `get_offline_bundle` (`backend/app/routers/assignments.py:106-110`) does not accept or enforce `X-Teacher-UUID` because `AssembledTestModel` has no `teacher_uuid` owner.
   - Sample JSON blocks for `/analytics/students/{student_id}` and `/analytics/classes/{class_id}` do not show the Stage 1 `handwriting_*` response fields, though `openapi_specification.yaml` schemas match `app.openapi()` 100%.

---

## Recommendation

- **Approve Stage 3 (`backend-auth-manifest-openapi-sync`) for closure and archival** (`accept-with-notes`).
