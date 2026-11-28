# Plan (ExecPlan) — Stage 1: Backend Critical Correctness

## Approach

Fix critical bugs across core runtime, `checker.py`, `submissions.py`, `analytics.py`, `reports.py`, and `classes.py` in bounded slices while preserving Stage 2 (`pipeline_spec.json`/OCR/generators) and Stage 3 (`X-Teacher-UUID`/manifest/OpenAPI bulk sync) boundaries. Rejected: trusting client scores without `checker.py` recalculation (leaves tampered/stale scores in analytics and gradebooks).

## Milestones

1. **M1 — Core runtime, DB integrity & answer checking (`AC-1`, `AC-2`):** Fix `main.py`, `database.py`, `config.py`, `models/school.py`, `models/submission.py`, `checker.py`, `schemas/task.py`, `constructor.py`.
2. **M2 — Submission validation, server-side recalculation & upsert (`AC-3`):** Harden `schemas/submission.py` and `routers/submissions.py`.
3. **M3 — Analytics, reports & classes correctness (`AC-4`, `AC-5`, `AC-6`):** Fix `schemas/analytics.py`, `routers/analytics.py`, `routers/reports.py`, `schemas/classes.py`, `routers/classes.py`.
4. **M4 — Regression tests, verification & independent review (`AC-7`):** Update `test_school_and_analytics_api.py`, expand `test_checker.py`, add `test_stage1_critical_fixes.py`, run full `pytest backend\tests -q`, run Fresh-Context Evaluator, update `checks.json` & `docs/STATUS.md`.

## File Ownership

| File / Area | Writer | Notes |
|---|---|---|
| `backend/app/core/{config.py,database.py}`, `backend/app/main.py` | coordinator | Core runtime & seeding |
| `backend/app/models/{school.py,submission.py}` | coordinator | FKs & unique constraints |
| `backend/app/generators/checker.py`, `backend/app/schemas/task.py`, `backend/app/routers/constructor.py` | coordinator | Answer verification hardening |
| `backend/app/schemas/submission.py`, `backend/app/routers/submissions.py` | coordinator | Validation & server recalculation |
| `backend/app/schemas/analytics.py`, `backend/app/routers/analytics.py`, `backend/app/routers/reports.py` | coordinator | Analytics & gradebook export |
| `backend/app/schemas/classes.py`, `backend/app/routers/classes.py` | coordinator | Bulk import & class assignments |
| `backend/tests/{test_school_and_analytics_api.py,test_checker.py,test_stage1_critical_fixes.py}` | coordinator | Verification suite |
| `harness/changes/active/backend-critical-correctness/reviews/review.md` | evaluator | Independent review pass |

## Risks & Mitigations

- Existing on-disk `tatar_ocr.db` lacks new FK/unique constraints via `create_all` → document schema migration note in `Surprises & Discoveries` and `STATUS.md`.
- Existing test `test_zero_photo_batch_sync_and_analytics` sent partial 2-cell payloads with mismatched `overall_score=8.0` → update fixture payload to realistic 4-question Variant 1 responses matching `TAT-2026-Q1`.

## Surprises & Discoveries

*(Update during execution whenever codebase/tests reveal unexpected facts)*
- `2026-11-28`: Worktree uses the main repository's virtualenv at `C:\Users\galee\PycharmProjects\tatar_ocr\.venv\Scripts\python.exe` rather than a local `.venv` inside the worktree.
- `2026-11-28`: Existing on-disk SQLite DB (`tatar_ocr.db`) if previously created without `ForeignKey`/`UniqueConstraint` on `submissions` and `class_assignments` will not be altered by `Base.metadata.create_all`; fresh DB creation or manual migration is required to enforce new table-level constraints on pre-existing `.db` files.

## Decision Log

- `2026-11-28`: Enforce strict Pydantic + referential validation and server-side score/grade recalculation via `checker.py` in `submissions.py`, respecting `teacher_override` and `TeacherModel.grading_scale` (user-confirmed).
- `2026-11-28`: Defer `pipeline_spec.json` + `tatar_ocr.cpp` 1-cell updates to Stage 2 and `X-Teacher-UUID` + `/model/manifest` sync to Stage 3.
