# Tasks — Stage 3: Teacher UUID Isolation, Model Manifest SSOT & OpenAPI Sync

| # | Task | Files owned | Check IDs | Status | Verification evidence |
|---|---|---|---|---|---|
| 1 | Archive Stage 2 & initialize Stage 3 ECL + `docs/STATUS.md` | `harness/changes/*`, `docs/STATUS.md` | — | done | Stage 1 & Stage 2 archived; Stage 3 active |
| 2 | Add `teacher_scope.py`, harden `auth.py`/`schemas/auth.py`, enforce `X-Teacher-UUID` across routers | `backend/app/core/teacher_scope.py`, `backend/app/schemas/auth.py`, `backend/app/routers/{auth,classes,assignments,submissions,analytics,reports}.py` | `AC-1`, `AC-2` | done | `test_ac1_*` & `test_ac2_*` pass inside `162 passed` suite |
| 3 | Align `/model/manifest` to `pipeline_spec.json` & fix Stage 2 minor tails + 422 constant | `backend/app/routers/{model,constructor,assignments,classes,submissions}.py`, `backend/tests/test_school_and_analytics_api.py` | `AC-3`, `AC-4` | done | `test_ac3_*`, `test_ac4_*`, and `test_school_and_analytics_api.py` pass with 0 warnings |
| 4 | Sync `openapi_specification.yaml` and `api_specification.md` with Stage 1–3 contracts | `openapi_specification.yaml`, `api_specification.md` | `AC-5` | done | `Synced endpoints: 30`; `test_ac5_*` passes |
| 5 | Add `test_stage3_auth_manifest_openapi.py` and run full `pytest backend\tests -q` | `backend/tests/test_stage3_auth_manifest_openapi.py` | `AC-6`, `GATE-BACKEND` | done | `162 passed in 18.26s` (0 warnings) |
| 6 | Update `checks.json`, `tasks.md`, `STATUS.md`, and run Fresh-Context Evaluator (`reviews/review.md`) | `harness/changes/active/backend-auth-manifest-openapi-sync/*`, `docs/STATUS.md` | `GATE-SPECS` | done | `reviews/review.md` (`accept-with-notes`, 0 Critical/Major blockers) |
