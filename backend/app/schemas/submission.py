from typing import List, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator

CellStatus = Literal[
    "MATCH",
    "MISMATCH",
    "EMPTY_MATCH",
    "EMPTY_MISMATCH",
    "FLAGGED",
    "FLAG_OVERRIDDEN_BY_TEACHER",
]
TeacherOverride = Literal["CORRECT", "WRONG"]


class SubmissionCellResult(BaseModel):
    cell_index: int = Field(..., ge=0, le=11)
    expected_char: str
    predicted_char: str
    confidence: float = Field(..., ge=0.0, le=1.0, allow_inf_nan=False)
    status: CellStatus
    teacher_override: Optional[TeacherOverride] = None

    @field_validator("teacher_override", mode="before")
    @classmethod
    def _normalize_override(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip().upper()
            return cleaned or None
        return v


class SubmissionQuestionResult(BaseModel):
    question_number: int = Field(..., ge=1)
    marker_id: int
    topic_tag: str
    is_correct: bool
    points_earned: float = Field(..., ge=0.0, allow_inf_nan=False)
    cells: List[SubmissionCellResult]


class SubmissionItem(BaseModel):
    client_submission_uuid: Optional[str] = None
    student_id: str
    student_name: str
    variant: int = Field(default=1, ge=1)
    checked_at: Optional[datetime] = None
    overall_score: float = Field(..., ge=0.0, allow_inf_nan=False)
    max_score: float = Field(default=8.0, gt=0.0, allow_inf_nan=False)
    final_grade: int = Field(default=2, ge=2, le=5)
    teacher_reviewed_flags: int = Field(default=0, ge=0)
    questions_results: List[SubmissionQuestionResult]

    @model_validator(mode="after")
    def _check_score_bounds(self):
        if self.overall_score > self.max_score:
            raise ValueError("overall_score cannot exceed max_score")
        return self

class BatchSyncSubmissionsRequest(BaseModel):
    assignment_id: str
    class_id: str
    synced_at: Optional[datetime] = None
    submissions: List[SubmissionItem]

class BatchSyncSubmissionsResponse(BaseModel):
    status: str = "synced"
    received_count: int
    inserted_count: int
    updated_count: int
    analytics_recalculated: bool = True

class SingleSubmissionSaveRequest(BaseModel):
    assignment_id: str
    class_id: str
    submission: SubmissionItem

class SingleSubmissionSaveResponse(BaseModel):
    status: str = "saved"
    submission_id: str
