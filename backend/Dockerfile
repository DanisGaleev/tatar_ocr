FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MPLCONFIGDIR=/tmp/mplconfig

WORKDIR /app

# Install system dependencies for headless OpenCV and fonts for PDF generation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reproducible dependency installation
RUN pip install --no-cache-dir uv

# Install only slim backend runtime dependencies (no PyTorch / ML training packages)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN uv pip install --system -r /app/backend/requirements.txt

# Copy fonts and backend source code
COPY fonts /app/fonts
COPY backend/app /app/backend/app

# Run as non-root user with writable /data and /tmp/mplconfig
RUN useradd --system --uid 10001 --create-home appuser \
    && mkdir -p /data /tmp/mplconfig \
    && chown -R appuser:appuser /app /data /tmp/mplconfig

USER appuser

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
