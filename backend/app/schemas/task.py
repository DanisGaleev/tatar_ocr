from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict

class TaskBankItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_id: str
    prompt_tt: str
    expected_answer: str
    cell_count: int = Field(..., ge=1, le=12)
    grade_level: int = Field(..., ge=5, le=9)
    topic_tag: str
    topic_name_tt: str
    is_public_in_bank: bool = False


class CreateTaskRequest(BaseModel):
    prompt_tt: str = Field(..., min_length=3, max_length=512)
    expected_answer: str = Field(..., min_length=1, max_length=12)
    cell_count: int = Field(..., ge=1, le=12)
    grade_level: int = Field(..., ge=5, le=9)
    topic_tag: str = Field(..., min_length=2, max_length=64)
    topic_name_tt: str = Field(..., min_length=2, max_length=128)
    is_public_in_bank: bool = False
    task_type: Optional[str] = "custom"

class GenerateTasksRequest(BaseModel):
    task_type: str = Field(..., description="One of: case_inflection, plural_affixes, antonyms, translation")
    count: int = Field(default=1, ge=1, le=10)
    seed: Optional[int] = Field(default=None, description="Optional seed for deterministic reproducibility")
    grade_level: int = Field(default=7, ge=5, le=9)
    custom_stems: Optional[List[str]] = Field(default=None, description="Optional list of custom words/stems for the generator")
    case_code: Optional[str] = Field(default=None, description="Specific case code for case_inflection (e.g. case_dative)")
    save_to_bank: bool = Field(default=False, description="Whether to persist generated tasks into task bank")

class GenerateTasksResponse(BaseModel):
    task_type: str
    seed: Optional[int]
    count: int
    tasks: List[TaskBankItem]

class VerifyAnswerRequest(BaseModel):
    expected_answer: str
    student_answer: str
    cell_count: int = 8
    topic_tag: str = "general"

class CellStatusItem(BaseModel):
    cell_index: int
    expected_char: str
    predicted_char: str
    status: str

class VerifyAnswerResponse(BaseModel):
    is_correct: bool
    points_earned: float
    written_word: str
    cells: List[CellStatusItem]


class ScanTaskRequestBase64(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string (JPEG/PNG/WEBP)")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image")
    grade_level: Optional[int] = Field(default=None, ge=5, le=9, description="Optional grade level hint")
    save_to_bank: bool = Field(default=False, description="Persist directly into task bank if valid")


class ScanTaskResponse(BaseModel):
    is_supported: bool = Field(..., description="True if task matches a supported type and fits 12-cell physical blanks")
    unsupported_reason: Optional[str] = Field(default=None, description="Explanation why the task is not supported")
    task_type: Optional[str] = Field(default=None, description="Identified task generator type or 'custom'")
    task: Optional[TaskBankItem] = Field(default=None, description="Primary formatted task ready for task bank and test blanks")
    tasks: List[TaskBankItem] = Field(default_factory=list, description="All extracted sub-tasks if exercise contains multiple items")
    raw_ocr_text: str = Field(default="", description="Recognized text from image OCR")
    confidence: float = Field(default=1.0, description="Overall confidence score of OCR and classification")
    saved_to_bank: bool = Field(default=False, description="Whether the task was persisted in DB")

