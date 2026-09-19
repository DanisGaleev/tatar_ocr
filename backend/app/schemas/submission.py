from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class SubmissionCellResult(BaseModel):
    cell_index: int
    expected_char: str
    predicted_char: str
    confidence: float
    status: str  # MATCH, MISMATCH, FLAG_OVERRIDDEN_BY_TEACHER
    teacher_override: Optional[str] = None  # e.g. WRONG, CORRECT

class SubmissionQuestionResult(BaseModel):
    question_number: int
    marker_id: int
    topic_tag: str
    is_correct: bool
    points_earned: float
    cells: List[SubmissionCellResult]

class SubmissionItem(BaseModel):
    client_submission_uuid: Optional[str] = None
    student_id: str
    student_name: str
    variant: int = 1
    checked_at: Optional[datetime] = None
    overall_score: float
    max_score: float = 8.0
    final_grade: int = 2
    teacher_reviewed_flags: int = 0
    questions_results: List[SubmissionQuestionResult]

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
