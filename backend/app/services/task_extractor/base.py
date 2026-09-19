from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ExtractedTaskSubItem(BaseModel):
    task_type: Optional[str] = Field(default=None)
    prompt_tt: str = Field(...)
    expected_answer: str = Field(...)
    cell_count: Optional[int] = Field(default=None)
    grade_level: Optional[int] = Field(default=None)
    topic_tag: Optional[str] = Field(default=None)
    topic_name_tt: Optional[str] = Field(default=None)

class ExtractedTaskResult(BaseModel):
    is_supported: bool = Field(..., description="Whether the task type is supported and fits 1-12 cell physical blanks")
    unsupported_reason: Optional[str] = Field(default=None, description="Detailed explanation if task is unsupported")
    task_type: Optional[str] = Field(default=None, description="Identified generator task_type or 'custom'")
    prompt_tt: Optional[str] = Field(default=None, description="Cleaned prompt instruction in Tatar/Russian")
    expected_answer: Optional[str] = Field(default=None, description="Upper-case normalized expected answer word")
    cell_count: Optional[int] = Field(default=None, description="Required number of handwriting cells (1..12)")
    grade_level: Optional[int] = Field(default=None, description="Target grade level (5..9)")
    topic_tag: Optional[str] = Field(default=None, description="Internal topic tag")
    topic_name_tt: Optional[str] = Field(default=None, description="Human-readable topic title in Tatar")
    items: List[ExtractedTaskSubItem] = Field(default_factory=list, description="List of sub-tasks if exercise is split into multiple items")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    raw_ocr_text: str = Field(default="", description="Original raw recognized OCR text")

class BaseOCRClient(ABC):
    @abstractmethod
    async def recognize_text(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """Extract raw text from image bytes."""
        pass

class BaseLLMTaskClassifier(ABC):
    @abstractmethod
    async def classify_and_extract(
        self,
        ocr_text: str,
        supported_types: List[Dict[str, str]],
        grade_hint: Optional[int] = None,
    ) -> ExtractedTaskResult:
        """Classify task type and extract structured fields using LLM."""
        pass
