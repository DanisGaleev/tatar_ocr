# Tasks — Minimal Backend Deployment & Logger Fix

Atomic, ordered slices. Each task names its writer and maps to `checks.json`.

| # | Task | Files owned | Check IDs | Status | Verification evidence |
|---|---|---|---|---|---|
| 1 | Define `logger` in `backend/app/main.py` | `backend/app/main.py` | `AC-1` | done | `500_HANDLER_OK {'detail': 'Internal server error', 'status_code': 500}` |
| 2 | Add `.dockerignore`, `backend/requirements.txt`, and update `backend/Dockerfile` | `.dockerignore`, `backend/requirements.txt`, `backend/Dockerfile` | `AC-2` | done | Import AST audit `MISSING: []`; `.dockerignore` tightened for `**/.env*` |
| 3 | Run backend `pytest` suite & import audit | — | `AC-1`, `AC-2`, `AC-3`, `GATE-SPECS` | done | `139 passed, 3 warnings in 14.77s` |
| 4 | Fresh-Context Evaluator review | `reviews/review.md` | All | done | Verdict: `accept-with-notes` |

## Session Handoff

- **Last completed slice:** Independent Fresh-Context Evaluator review recorded (`accept-with-notes`), `.dockerignore` tightened.
- **Next immediate step:** Ready for user confirmation to close/archive `harness/changes/active/local-backend-deployment/` into `harness/changes/archive/local-backend-deployment/` and run `docker compose build` on host.
- **Uncommitted / tricky state:** Docker image build was not executed inside the sandbox.

## Completion Definition

- [x] Every check in `checks.json` has `"status": "PASS"` with recorded `command` + `evidence`.
- [x] No open question in `spec.md` was silently guessed.
- [x] Spec Deltas from `spec.md` synced into living docs (`docs/specs/README.md` — N/A, no living spec deltas).
- [x] Independent Fresh-Context Review in `reviews/review.md` recorded with verdict `accept` / `accept-with-notes`.
