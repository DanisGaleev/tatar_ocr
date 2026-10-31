FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies for OpenCV and GUI/image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable package management
RUN pip install --no-cache-dir uv

# Copy project dependencies configuration
COPY pyproject.toml requirements.txt /app/

# Install dependencies into system environment using uv
RUN uv pip install --system -r /app/requirements.txt

# Copy root OCR modules, models, fonts, and backend source code
COPY blank_pipeline.py tatar_ocr_dataset.py tatar_ocr_augmentation.py /app/
COPY models /app/models
COPY fonts /app/fonts
COPY backend/app /app/backend/app

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
