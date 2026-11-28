# Change: Minimal Backend Deployment & Logger Fix

- **ID:** local-backend-deployment
- **Type:** bugfix
- **Status:** done
- **Created:** 2026-11-28
- **Owner (lead):** Antigravity
- **One-line goal:** Slim down the backend Docker image (remove unused PyTorch/ML weights, add `.dockerignore`, run as non-root) and fix missing `logger` in `backend/app/main.py` without changing API contracts or adding `BACKEND_API_KEY`.
- **Blocker (if parked):** none
