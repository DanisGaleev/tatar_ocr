from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Дәресханә (Tatar OCR) Task Generator"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite+aiosqlite:///./tatar_ocr.db"
    
    # OCR and A4 Physical Template Specs
    MAX_QUESTIONS_PER_PAGE: int = 8
    CELL_WIDTH_MM: float = 10.0
    CELL_HEIGHT_MM: float = 10.0
    ARUCO_DICT: str = "DICT_4X4_50"
    CORNER_ARUCO_IDS: list[int] = [0, 1, 2, 3]
    
    model_config = {
        "case_sensitive": True,
        "env_file": ".env"
    }

settings = Settings()

