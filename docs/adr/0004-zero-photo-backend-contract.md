# ADR 0004: Zero-Photo Backend Contract Boundary

- **Status:** Accepted (Amended 2026-11-27)
- **Date:** 2026-11-18 (updated 2026-11-27 for transient server-side blank recognition fallback)

## Context

Under 152-ФЗ privacy constraints and classroom bandwidth limits, student worksheet photos normally stay on the teacher's device. For devices or web flows that require server-side C++20 OCR fallback, an opt-in recognition endpoint must process the photo strictly in memory without storing or logging it.

## Decision

- `backend/app/routers/submissions.py` primarily ingests lightweight structured JSON results (`~1.5 KB` per sheet: recognized letters, confidences, HITL overrides, scores) via `/submissions` and `/submissions/batch-sync`.
- Opt-in server-side blank recognition (`POST /api/v1/submissions/recognize` and `/recognize-base64`) and textbook-photo exercise extraction (`backend/app/routers/constructor.py` / `task_extractor`) are explicit transient exceptions: uploaded images are decoded and processed **strictly in RAM** via `tatar_ocr_native` and discarded immediately when the request completes.

## Consequences

- Never persist student worksheet photos on disk, object storage, database tables, or application logs in `backend/app/`.
- Any change to `/api/v1/*` routes or schemas requires updating `openapi_specification.yaml` and `api_specification.md` (and noting any breaking changes for `kara_kuyan_frontend`).
