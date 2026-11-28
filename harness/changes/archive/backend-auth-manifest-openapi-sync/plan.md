# Plan — Stage 3 (`backend-auth-manifest-openapi-sync`)

1. **ECL Init:** Archive Stage 2 (`backend-ocr-pdf-generators-contract`) and bootstrap Stage 3 artifacts (`summary.md`, `spec.md`, `plan.md`, `tasks.md`, `checks.json`, `reviews/review.md`) + `docs/STATUS.md`.
2. **Teacher Scope & Auth:** Create `backend/app/core/teacher_scope.py`, harden `schemas/auth.py` and `routers/auth.py`, wire `X-Teacher-UUID` ownership checks and scoping across `classes.py`, `assignments.py`, `submissions.py`, `analytics.py`, and `reports.py`.
3. **Model Manifest & Minor Tails:** Update `routers/model.py` + `test_school_and_analytics_api.py:494`, fix `constructor.py` (`503` on missing API key, `ge=1` on `max_cells`), `assignments.py` (`ge=1` on `offline-bundle?variant=`), and replace `HTTP_422_UNPROCESSABLE_ENTITY` with `HTTP_422_UNPROCESSABLE_CONTENT`.
4. **Living Specs Sync:** Regenerate `openapi_specification.yaml` from `app.openapi()` with `TeacherUUID` security scheme and update `api_specification.md`.
5. **Tests & Verification:** Add `backend/tests/test_stage3_auth_manifest_openapi.py`, run `pytest backend\tests -q`, verify `git status --short`.
6. **Evaluation:** Update `checks.json`, `tasks.md`, and `docs/STATUS.md`; run read-only Fresh-Context Evaluator to produce `reviews/review.md`.
