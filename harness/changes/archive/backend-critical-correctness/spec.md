# Spec — Stage 1: Backend Critical Correctness

## Goal

Eliminate critical correctness, data-integrity, and runtime-crash defects across core runtime/DB wiring, answer checking (`checker.py`), submission ingestion (`submissions.py`), analytics (`analytics.py`), gradebook export (`reports.py`), and class management (`classes.py`).

## Non-goals

- Stage 2 scope: `pipeline_spec.json` / `native/src/tatar_ocr.cpp` 1-cell blank support, PDF geometry alignment, task generators (`possessive_case`, `registry.py`, `phonetics.py`, `lexicon.py`), and `task_extractor`.
- Stage 3 scope: `X-Teacher-UUID` multi-tenant isolation & DB `teacher_uuid` columns, `/api/v1/model/manifest` update, `offline-bundle?variant=N` `total_variants` contract change, and full `openapi_specification.yaml` / `api_specification.md` sync for undocumented endpoints.
- Stage 4 scope: N+1 query optimization in `assignments.py`/`classes.py` and cross-router helper deduplication.

## Assumptions

- `SubmissionModel` remains Zero-Photo compliant (no student worksheet photos persisted or logged).
- Existing SQLite dev databases on disk (`tatar_ocr.db`) are not automatically migrated by `Base.metadata.create_all`; new FK and uniqueness constraints apply to freshly created tables and in-memory test DBs unless recreated or migrated separately.

## Spec Deltas (OpenSpec — against `docs/specs/README.md`)

### ADDED
- `api_specification.md` / `openapi_specification.yaml`: `QuestionAnalyticsItem.variant_id` (optional integer, default `1`) to distinguish multi-variant questions in `GET /api/v1/analytics/assignments/{assignment_id}`.
- `api_specification.md` / `openapi_specification.yaml`: `NO_DATA` handwriting status and nullable `handwriting_quality_pct` / `average_handwriting_quality_pct` (`null` when 0 non-empty cells exist).

### MODIFIED
- `api_specification.md` / `openapi_specification.yaml`: `POST /api/v1/submissions/batch-sync` and `POST /api/v1/submissions` validate referential integrity (`assignment_id`, `class_id`, `student_id`, `variant`), bounds/enums (`SubmissionCellResult`, `SubmissionItem`), recompute scores/grades server-side via `checker.py` using `TeacherModel.grading_scale` (preserving `teacher_override`), enforce unique `(assignment_id, class_id, student_id)` upsert, and return `409` on cross-student `client_submission_uuid` collision.
- `api_specification.md` / `openapi_specification.yaml`: `GET /api/v1/analytics/{students,classes,assignments}/*` and `GET /api/v1/reports/assignments/{id}/gradebook.xlsx` return `404` when target entity does not exist; `teacher_override="CORRECT"` is not counted as an error; `StudentHistoryItem.assignment_title` returns the real test title.
- `backend/app/core/config.py`: Default `DATABASE_URL` anchors `tatar_ocr.db` to `backend/` regardless of process CWD.

### REMOVED
- None.

## Acceptance Criteria

- `AC-1`: Global exception handler logs via defined `logger` without `NameError` and returns JSON 500; SQLite `PRAGMA foreign_keys=ON` enabled; `get_db()` rolls back on exception; startup seed does not overwrite existing `TAT-2026-Q1.bundle_json`; `CORSMiddleware` disables `allow_credentials` with wildcard origin; `ClassAssignmentModel` and `SubmissionModel` enforce uniqueness and FKs.
- `AC-2`: `verify_question_cells` and `/constructor/verify-answer` reject `cell_count < 1` and `cell_count < len(expected_answer)` (HTTP 422), reject extra trailing student chars, normalize Latin homoglyphs on `expected_answer` and NFC-decomposed characters, handle `confidence=None`, reject empty `questions` in `verify_test_submission`, and align `calculate_final_grade` with rounded `score_percentage`.
- `AC-3`: `submissions.py` enforces strict bounds/enums/FK existence, recomputes tampered client `overall_score`/`final_grade` via `checker.py` respecting `teacher_override` and `TeacherModel.grading_scale`, deduplicates `(assignment_id, class_id, student_id)` and in-batch `client_submission_uuid`, and returns `409` on cross-record UUID reuse.
- `AC-4`: `analytics.py` counts `teacher_override="CORRECT"` as non-error, separates multi-variant item analysis by `(variant_id, question_number)`, excludes `EMPTY_MATCH` from handwriting metrics, returns `NO_DATA`/`None` for 0-cell/0-submission records, returns `404` for missing entities, populates real `assignment_title`, and excludes 0-mistake topics from weak-topic lists.
- `AC-5`: `reports.py` returns `404` for missing class/test, sanitizes Excel sheet titles (`/` etc. and >31 chars), escapes spreadsheet formula injection (`=`, `+`, `-`, `@`, `\t`, `\r`) in CSV/XLSX, sanitizes `Content-Disposition` filenames, and runs `openpyxl` workbook generation in threadpool.
- `AC-6`: `classes.py` preserves 4-token Tatar patronymics (`Нияз кызы`), avoids creating ghost classes on blank `bulk-import`, deduplicates imported students atomically, validates `due_date` and `status` (`422`) in `assign_test_to_class`, reports real submission progress on re-assign, and fixes `average_score_pct` denominator + `Content-Disposition` sanitization.
- `AC-7`: Full `backend/tests` suite (including updated `test_zero_photo_batch_sync_and_analytics` and new Stage 1 regression tests) passes cleanly after final edit.

## Open Questions

- None (Stage 1 decisions confirmed by user on 2026-11-28).
