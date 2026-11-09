from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Кара куян (Tatar OCR) Task Generator"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite+aiosqlite:///./tatar_ocr.db"
    
    # OCR and A4 Physical Template Specs
    MAX_QUESTIONS_PER_PAGE: int = 8
    CELL_WIDTH_MM: float = 10.0
    CELL_HEIGHT_MM: float = 10.0
    ARUCO_DICT: str = "DICT_4X4_50"
    CORNER_ARUCO_IDS: list[int] = [0, 1, 2, 3]

    # Yandex Cloud (OCR & YandexGPT)
    YANDEX_API_KEY: str | None = None
    YANDEX_FOLDER_ID: str | None = None
    YANDEX_GPT_MODEL_URI: str | None = None
    YANDEX_OCR_URL: str = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
    YANDEX_GPT_URL: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    TASK_EXTRACTOR_MOCK_MODE: bool = False
    
    # Security & JWT Auth
    JWT_SECRET_KEY: str = "test-secret-key-tatar-ocr-2026-very-secure"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    RATE_LIMIT_ENABLED: bool = True
    ALLOW_LEGACY_TEST_AUTH: bool = False
    AUTH_ENFORCED_STRICT: bool = True
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "https://karakuyan.tatar",
    ]

    # Email / SMTP
    EMAIL_MOCK_MODE: bool = True
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_TLS: bool = True
    EMAILS_FROM_EMAIL: str = "noreply@tatar-ocr.ru"

    # OAuth Credentials
    YANDEX_CLIENT_ID: str = "test_yandex_client_id"
    YANDEX_CLIENT_SECRET: str = "test_yandex_secret"
    VK_CLIENT_ID: str = "test_vk_client_id"
    VK_CLIENT_SECRET: str = "test_vk_secret"

    # Default Seed Admin
    ADMIN_EMAIL: str = "admin@example.com"
    ADMIN_PASSWORD: str = "AdminPassword123!"
    
    model_config = {
        "case_sensitive": True,
        "env_file": (".env", "../.env")
    }

settings = Settings()

