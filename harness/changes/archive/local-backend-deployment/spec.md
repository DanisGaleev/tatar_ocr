# Spec — Minimal Backend Deployment & Logger Fix

WHAT and WHY only. No low-level implementation code.

## Goal

Make the existing backend Docker build fast and clean for hackathon/local server deployment (exclude datasets/weights/PyTorch/tests from Docker context & image, run container as non-root `appuser` with writable `/data` and `/tmp/mplconfig`) and fix the undefined `logger` bug in `backend/app/main.py`'s unhandled exception handler.

## Non-goals

- Adding `BACKEND_API_KEY`, changing `X-Teacher-UUID` auth, restricting `CORS_ORIGINS`, toggling `/docs`, or disabling `seed_initial_data()` (explicitly excluded by user for hackathon demo readiness).
- Adding Caddy TLS, `deploy.ps1`/`deploy.sh`, or changing `api_specification.md` / `openapi_specification.yaml`.

## Assumptions

- `backend/app/` continues to require no PyTorch, torchvision, ONNX Runtime, or root ML scripts (`blank_pipeline.py`).

## Spec Deltas (OpenSpec — against `docs/specs/README.md`)

### ADDED
- `.dockerignore`: Exclude non-backend assets (`dataset/`, `realdataset/`, `models/`, `v3/`, `native/`, `.venv/`, `backend/tests/`, `.env`, `*.db`) from Docker build context.
- `backend/requirements.txt`: Slim runtime dependency list for `backend/app/` without `torch`, `torchvision`, `onnxruntime`, `scipy`, or `pytest`.

### MODIFIED
- `backend/app/main.py`: Import `logging` and define `logger = logging.getLogger(__name__)` so `unhandled_exception_handler` does not raise `NameError`.
- `backend/Dockerfile`: Install `backend/requirements.txt`, copy only `fonts/` and `backend/app/`, set `MPLCONFIGDIR=/tmp/mplconfig`, and run as non-root `appuser`.

### REMOVED
- PyTorch/CPU extra-index install and copying of `models/` and root ML scripts (`blank_pipeline.py`, `tatar_ocr_dataset.py`, `tatar_ocr_augmentation.py`) from `backend/Dockerfile`.

## Acceptance Criteria

- `AC-1`: `backend/app/main.py` defines `logger` and `unhandled_exception_handler` returns HTTP `500` JSON instead of `NameError`.
- `AC-2`: `.dockerignore`, `backend/requirements.txt`, and `backend/Dockerfile` exclude PyTorch/ML weights/datasets, include all packages imported by `backend/app/`, and configure non-root `appuser` with `MPLCONFIGDIR=/tmp/mplconfig`.
- `AC-3`: Full backend pytest suite (`backend/tests`) passes after the final edit.

## Open Questions

- None.
