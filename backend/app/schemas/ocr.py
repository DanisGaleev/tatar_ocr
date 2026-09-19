from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field


class ScanBlankRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded image string (with or without data URI prefix)")
    annotate: bool = Field(True, description="Whether to return an annotated visualization image")


class CellResult(BaseModel):
    cell_idx: int
    char: str
    confidence: float
    is_empty: bool
    bbox: List[int]


class QuestionResult(BaseModel):
    q_num: int
    text: str
    cells: List[CellResult]


class ScanBlankStats(BaseModel):
    total_letters: int
    avg_confidence: float
    questions_count: int


class ScanBlankResponse(BaseModel):
    status: str = "success"
    rectification_method: str
    student_name: str
    questions: List[QuestionResult]
    stats: ScanBlankStats
    annotated_b64: Optional[str] = None


class PredictBoxRequest(BaseModel):
    image_b64: str = Field(..., description="Base64-encoded cropped cell image")
    contrast_mode: str = Field("percentile", description="Contrast normalization mode: 'percentile' or 'clahe'")


class PredictBoxCharResult(BaseModel):
    char: str
    unicode: str
    confidence: float


class PredictBoxResponse(BaseModel):
    stages: dict
    predictions_39: List[PredictBoxCharResult]
    predictions_102: Optional[List[PredictBoxCharResult]] = None
