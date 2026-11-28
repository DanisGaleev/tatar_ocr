# Tasks — Stage 1: Backend Critical Correctness

| # | Task | Files owned | Check IDs | Status | Verification evidence |
|---|---|---|---|---|---|
| 1 | Fix core runtime, SQLite FKs, rollback, seeding, and DB constraints | `backend/app/core/{config.py,database.py}`, `backend/app/main.py`, `backend/app/models/{school.py,submission.py}` | `AC-1` | done | `pytest backend\tests -q` → `148 passed, 6 warnings in 14.24s` |
| 2 | Harden `checker.py`, `VerifyAnswerRequest`, and `/constructor/verify-answer` | `backend/app/generators/checker.py`, `backend/app/schemas/task.py`, `backend/app/routers/constructor.py` | `AC-2` | done | `pytest backend\tests -q` → `148 passed, 6 warnings in 14.24s` |
| 3 | Add strict submission validation, `checker.py` recalculation, and safe upsert | `backend/app/schemas/submission.py`, `backend/app/routers/submissions.py` | `AC-3` | done | `pytest backend\tests -q` → `148 passed, 6 warnings in 14.24s` |
| 4 | Fix analytics overrides, multi-variant isolation, `NO_DATA`, 404s, and real titles | `backend/app/schemas/analytics.py`, `backend/app/routers/analytics.py` | `AC-4` | done | `pytest backend\tests -q` → `148 passed, 6 warnings in 14.24s` |
| 5 | Fix Excel sheet-title crash, formula injection, threadpool export, and 404s | `backend/app/routers/reports.py` | `AC-5` | done | `pytest backend\tests -q` → `148 passed, 6 warnings in 14.24s` |
| 6 | Fix `bulk_import_students` and `assign_test_to_class` / `list_class_assignments` | `backend/app/schemas/classes.py`, `backend/app/routers/classes.py` | `AC-6` | done | `pytest backend\tests -q` → `148 passed, 6 warnings in 14.24s` |
| 7 | Update `test_zero_photo_batch_sync_and_analytics` & add Stage 1 regression tests | `backend/tests/{test_school_and_analytics_api.py,test_checker.py,test_stage1_critical_fixes.py}` | `AC-7`, `GATE-BACKEND` | done | `pytest backend\tests -q` → `148 passed, 6 warnings in 14.24s` |
| 8 | Run Fresh-Context Evaluator & update `checks.json`, `review.md`, `STATUS.md` | `harness/changes/active/backend-critical-correctness/*`, `docs/STATUS.md` | `GATE-SPECS` | done | `reviews/review.md` verdict `accept-with-notes`; evaluator verified `148 passed, 6 warnings in 14.96s` |

## Session Handoff

- **Last completed slice:** Task 8 (Fresh-Context Evaluator complete with verdict `accept-with-notes`).
- **Next immediate step:** Confirm with user whether to sync Stage 1 deltas into `openapi_specification.yaml` / `api_specification.md` and archive `backend-critical-correctness`, then start Stage 2.
- **Uncommitted / tricky state:** `GATE-SPECS` remains `FAIL` until living specs (`openapi_specification.yaml`, `api_specification.md`) are synced on closure (or user confirms deferring OpenAPI bulk sync to Stage 3).
