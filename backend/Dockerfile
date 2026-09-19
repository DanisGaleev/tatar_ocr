FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install uv for fast, reliable package management
RUN pip install --no-cache-dir uv

# Copy project dependencies configuration
COPY pyproject.toml /app/

# Install dependencies into system environment using uv
RUN uv pip install --system fastapi "uvicorn[standard]" pydantic-settings "sqlalchemy>=2.0" aiosqlite pytest pytest-asyncio httpx

# Copy source code
COPY backend/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
