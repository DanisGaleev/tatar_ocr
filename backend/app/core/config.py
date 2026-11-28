from pathlib import Path
from pydantic_settings import BaseSettings

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "Кара куян (Tatar OCR) Task Generator"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = f"sqlite+aiosqlite:///{(BACKEND_DIR / 'tatar_ocr.db').as_posix()}"
    CORS_ORIGINS: list[str] = ["*"]
    
    # OCR and A4 Physical Template Specs
    MAX_QUESTIONS_PER_PAGE: int = 8
    CELL_WIDTH_MM: float = 10.0
    CELL_HEIGHT_MM: float = 10.0
    ARUCO_DICT: str = "DICT_4X4_50"
    CORNER_ARUCO_IDS: list[int] = [0, 1, 2, 3]

    # Server-side Blank Recognition (C++20 native OCR core)
    BLANK_MAX_IMAGE_SIZE_BYTES: int = 15 * 1024 * 1024
    OCR_MODEL_PATH: str = "models/tatar_ocr_v4_handwritten39_int8.onnx"
    OCR_SPEC_PATH: str = "pipeline_spec.json"
    NATIVE_BUILD_DIR: str | None = None
    OCR_DEFAULT_CONFIDENCE_THRESHOLD_PCT: float = 85.0

    # Yandex Cloud (OCR & YandexGPT)
    YANDEX_API_KEY: str | None = None
    YANDEX_FOLDER_ID: str | None = None
    YANDEX_GPT_MODEL_URI: str | None = None
    YANDEX_OCR_URL: str = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"
    YANDEX_GPT_URL: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    TASK_EXTRACTOR_MOCK_MODE: bool = False
    
    model_config = {
        "case_sensitive": True,
        "env_file": (".env", "../.env")
    }

settings = Settings()

