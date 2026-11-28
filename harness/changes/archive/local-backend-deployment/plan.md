# Plan (ExecPlan) — Minimal Backend Deployment & Logger Fix

HOW the spec is delivered. Living plan updated across sessions.

## Approach

Minimal surgical fix agreed with the user: keep all API contracts, CORS, seeding, and `docker-compose.yml` ports unchanged for hackathon demo compatibility, while fixing the `logger` `NameError` in `backend/app/main.py` and removing unused PyTorch/ML weight bloat from `backend/Dockerfile` via `backend/requirements.txt` and `.dockerignore`.

## Milestones

1. **M1 — Fix & Slim Packaging:** Patch `backend/app/main.py`, add `.dockerignore` and `backend/requirements.txt`, update `backend/Dockerfile`.
2. **M2 — Verification & Fresh-Context Review:** Run backend `pytest`, validate all runtime imports in `backend/requirements.txt` against `backend/app/`, run Fresh-Context Evaluator review, and sync/close.

## File Ownership

| File / Area | Writer | Notes |
|---|---|---|
| `backend/app/main.py` | Main agent | Define `logger` |
| `.dockerignore`, `backend/requirements.txt`, `backend/Dockerfile` | Main agent | Slim non-root backend image |
| `harness/changes/active/local-backend-deployment/reviews/review.md` | Fresh-Context Evaluator subagent | Independent post-implementation review |

## Risks & Mitigations

- Missing runtime package in `backend/requirements.txt` → Verify every third-party module imported under `backend/app/` is present in `backend/requirements.txt`.

## Surprises & Discoveries

- `2026-11-28`: User scoped down full local hardening to a minimal hackathon-friendly fix (no `BACKEND_API_KEY`, no Caddy/CORS/seeding changes).

## Decision Log

- `2026-11-28`: Scoped change to `backend/app/main.py` (`logger` fix), `.dockerignore`, `backend/requirements.txt`, and `backend/Dockerfile` per user confirmation.
