# Independent Review (Fresh-Context Evaluator) — Stage 1: Backend Critical Correctness

Evaluated in a fresh context / independent review pass against `spec.md`, `plan.md`, `tasks.md`, `checks.json`, and actual workspace state.

## Verdict

`accept-with-notes`

All code acceptance criteria (`AC-1`–`AC-7`) and `GATE-BACKEND` pass independent verification (`148 passed, 6 warnings in 14.96s`) with clean workspace hygiene and Zero-Photo (152-ФЗ) compliance. Accepted with notes because `GATE-SPECS` remains `FAIL` until living specs (`openapi_specification.yaml`, `api_specification.md`) are synced at closure (or explicitly deferred to Stage 3), alongside documented residual notes below.

## Evaluator Verification Checklist

- [x] `AC-1` Core runtime & DB integrity:
  - `backend/app/main.py`: `logger = logging.getLogger("tatar_ocr.api")` defined; `unhandled_exception_handler` logs via `logger.exception(...)` and returns JSON `500`; `allow_credentials = "*" not in settings.CORS_ORIGINS`; `seed_initial_data()` skips `TAT-2026-Q1` insert when `existing_asm` exists.
  - `backend/app/core/database.py`: `_set_sqlite_pragma` sets `PRAGMA foreign_keys=ON` for SQLite; `get_db()` rolls back on exception.
  - `backend/app/core/config.py`: `DATABASE_URL` anchored via `BACKEND_DIR = Path(__file__).resolve().parents[2]`.
  - `backend/app/models/school.py` & `backend/app/models/submission.py`: `ClassAssignmentModel` (`uq_class_assignment`, FKs with `ondelete="CASCADE"`) and `SubmissionModel` (`uq_submission_assignment_class_student`, unique `client_submission_uuid`, FKs with `ondelete="CASCADE"`).
- [x] `AC-2` Answer checking & constructor validation:
  - `backend/app/generators/checker.py`: `verify_question_cells` rejects `cell_count < 1`, empty `expected_answer`, and `len(clean_expected) > cell_count`; normalizes NFC and Latin homoglyphs on both `expected_answer` and `predicted_char`; defaults `confidence=None` to `1.0`; rejects out-of-range non-empty student cells (`has_extra_student_chars`); `verify_test_submission` rejects empty `questions` and computes `calculate_final_grade(pct, scale)` using rounded `score_percentage`.
  - `backend/app/schemas/task.py` & `backend/app/routers/constructor.py`: `VerifyAnswerRequest.cell_count` bounded `1..12`; `verify_student_answer` NFC-normalizes `student_answer`, passes `max(cell_count, len(st_clean))` slots so extra chars fail check, and maps `ValueError` to HTTP `422`.
- [x] `AC-3` Submission validation, recalculation & upsert:
  - `backend/app/schemas/submission.py`: Enforces cell bounds (`0..11`), finite confidence (`0.0..1.0`), `CellStatus` and `TeacherOverride` literals, `variant >= 1`, `max_score > 0`, `0 <= overall_score <= max_score`, `final_grade` in `2..5`.
  - `backend/app/routers/submissions.py`: Validates `AssembledTestModel` (`404`), `ClassModel` (`404`), `StudentModel` membership (`422`), and `variant` existence (`422`); recalculates scores/grades via `verify_test_submission` using `TeacherModel.grading_scale` and preserving `teacher_override`; enforces `409 Conflict` on cross-submission `client_submission_uuid` collision; deduplicates by `(assignment_id, class_id, student_id)` and in-batch maps (`batch_by_uuid`, `batch_by_student`).
- [x] `AC-4` Analytics correctness & 404 handling:
  - `backend/app/schemas/analytics.py` & `backend/app/routers/analytics.py`: `is_cell_error` treats `teacher_override="CORRECT"` as non-error; `get_assignment_analytics` isolates items by `(variant_id, question_number)` with `QuestionAnalyticsItem.variant_id`; `compute_handwriting_metrics` excludes `EMPTY_MATCH` and returns `quality_pct=None` / `status="NO_DATA"` when zero active cells or submissions exist; returns `404` on missing student/class/assignment; `StudentHistoryItem.assignment_title` reads `AssembledTestModel.title`; 0-mistake topics excluded from `frequent_weak_topics` and `top_class_mistakes`.
- [x] `AC-5` Reports safety & threadpool export:
  - `backend/app/routers/reports.py`: Returns `404` on missing `AssembledTestModel` or `ClassModel`; `_safe_sheet_title` strips invalid Excel sheet characters and truncates to `31` chars; `_sanitize_spreadsheet_value` prefixes `'` for values (or `lstrip()`) starting with `=`, `+`, `-`, `@`, `\t`, `\r` in both CSV and XLSX; `_safe_filename_part` sanitizes `Content-Disposition` filenames; `run_in_threadpool(_build_gradebook_xlsx, ...)` offloads `openpyxl`.
- [x] `AC-6` Classes & assignments:
  - `backend/app/schemas/classes.py` & `backend/app/routers/classes.py`: `bulk_import_students` exits before creating `ClassModel` on blank `raw_text`, preserves multi-token patronymics (`middle_name = " ".join(parts[2:])`), and deduplicates case-insensitively across existing and incoming rows in a single commit; `assign_test_to_class` validates `due_date` (ISO, `422` on error) and `AssignTestToClassRequest.status` (`Literal["active", "completed", "archived"]`), and computes real submission progress + `average_score_pct` (using `len(valid_pcts)` denominator where `max_score > 0`) on re-assign and in `list_class_assignments`; `get_class_batch_blanks_pdf` sanitizes `Content-Disposition` filename.
- [x] `AC-7` & `GATE-BACKEND` Independent test run & workspace hygiene:
  - Command: `C:\Users\galee\PycharmProjects\tatar_ocr\.venv\Scripts\python.exe -m pytest backend\tests -q`
  - Observed result: `148 passed, 6 warnings in 14.96s` (exit code `0`).
  - Workspace hygiene check (`tmp_*`, `debug_*`, `_peek_*`, `check_*`, stray `*.db` in `backend`): zero stray temporary scripts or test DB artifacts found.

## Subsystem Lenses

1. **API Contract & Zero-Photo (152-ФЗ):**
   - **Zero-Photo:** Verified `SubmissionModel`, `schemas/submission.py`, `routers/submissions.py`, `analytics.py`, `reports.py`, and `classes.py` — no student worksheet photos, raw image bytes, or base64 image payloads are accepted, persisted, or logged.
   - **OpenAPI / API Spec Alignment:** `spec.md` declares the Stage 1 deltas (`QuestionAnalyticsItem.variant_id`, `NO_DATA` / nullable `handwriting_quality_pct` & `average_handwriting_quality_pct`, `409` conflict / server recalculation on submissions, `404` responses on analytics & reports). Living specs (`openapi_specification.yaml` — e.g. `QuestionAnalyticsItem` still lacks `variant_id` — and `api_specification.md`) have **not** been synced yet; `GATE-SPECS` in `checks.json` is accurately marked `FAIL` pending closure confirmation.
2. **C++20 / Python / Android NDK Parity:**
   - **N/A for Stage 1 scope.** No edits to `native/` or `pipeline_spec.json` (deferred to Stage 2).
3. **ML Benchmark & Metrics:**
   - **N/A for Stage 1 scope.** No edits to `v3/`, preprocessing, `pipeline_spec.json`, or `.onnx` weights.
4. **Backend QA, Migrations & Hygiene:**
   - Full backend suite independently verified after final edits: `148 passed, 6 warnings in 14.96s`.
   - `checks.json` statuses and evidences match observed behavior (`AC-1`..`AC-7`, `GATE-BACKEND` = `PASS`; `GATE-SPECS` = `FAIL`).
   - Hygiene clean: zero `tmp_*`/`debug_*`/`_peek_*` scripts or stray `.db` files in `backend/`.

## Deviations from Plan

- None. Implementation followed Milestones `M1`–`M4` and stayed strictly within Stage 1 boundaries (deferring Stage 2 OCR/generator/geometry items and Stage 3 `X-Teacher-UUID` isolation / manifest / full OpenAPI bulk sync).

## Issues Found

1. **Non-blocking / Open Gate (`GATE-SPECS`):** Living specifications (`openapi_specification.yaml` and `api_specification.md`) do not yet reflect the Stage 1 `spec.md` deltas (`QuestionAnalyticsItem.variant_id`, `NO_DATA`/nullable handwriting fields, `409` collision response, `404` responses). Must either be synced before archiving this change or explicitly deferred to Stage 3 by user decision.
2. **Minor Deprecation Warnings:** `pytest` emitted 6 `StarletteDeprecationWarning` messages for `HTTP_422_UNPROCESSABLE_ENTITY` (pre-existing convention across FastAPI routers, including `submissions.py:357`).

## Residual Risks

- **Pre-existing on-disk `tatar_ocr.db`:** `Base.metadata.create_all` does not alter existing SQLite tables; any local DB file created before Stage 1 will not enforce the new `SubmissionModel` / `ClassAssignmentModel` FKs or unique constraints unless recreated or manually migrated (documented in `spec.md`, `plan.md`, and `docs/STATUS.md`).
- **Empty `questions_results: []` submission fallback:** In `submissions.py:_recalculate_submission`, payloads with `questions_results: []` retain client-supplied `overall_score` and `max_score` (validated only for `0 <= overall_score <= max_score`) and recompute `final_grade` from `TeacherModel.grading_scale`. This preserves compatibility with summary-only saves (`test_save_single_submission_idempotent_upsert`), though cell-level verification is skipped when `questions_results` is empty.
- **Topic-level override assumption on legacy rows:** `analytics.py` (`get_student_analytics` and `get_class_analytics`) checks `q.get("is_correct", False)` for topic-level mistake counts; newly ingested submissions have `is_correct` recomputed by `checker.py` (respecting `teacher_override`), whereas any un-recalculated pre-Stage-1 rows in an existing DB would reflect their stored `is_correct` flag.
