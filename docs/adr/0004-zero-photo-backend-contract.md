# ADR 0004: Zero-Photo Backend Contract Boundary

- **Status:** Accepted
- **Date:** 2026-11-18 (documenting architectural contract in `api_specification.md`)

## Context

Under 152-ФЗ privacy constraints and classroom bandwidth limits, student worksheet photos must never leave the teacher's device.

## Decision

- `backend/app/routers/submissions.py` ingests only lightweight structured JSON results (`~1.5 KB` per sheet: recognized letters, confidences, HITL overrides, scores).
- Textbook-photo exercise extraction (`backend/app/routers/constructor.py` / `task_extractor`) is an isolated exception for printed textbook snippets and must not persist uploaded images.

## Consequences

- Never add student worksheet photo upload, storage, or logging in `backend/app/`.
- Any change to `/api/v1/*` routes or schemas requires updating `openapi_specification.yaml` and `api_specification.md` (and noting any breaking changes for `kara_kuyan_frontend`).
