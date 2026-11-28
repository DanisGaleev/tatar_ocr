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


class TopPredictionItem(BaseModel):
    char: str
    confidence: float


class RecognizedCellItem(BaseModel):
    cell_idx: int
    char: str
    confidence: float
    is_empty: bool
    review_required: bool = False
    top3: List[TopPredictionItem] = Field(default_factory=list)
    bbox: List[int] = Field(default_factory=list)


class RecognizedQuestionItem(BaseModel):
    q_num: int
    marker_id: int
    text: str
    prompt: Optional[str] = None
    topic_tag: Optional[str] = None
    expected_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    points_earned: Optional[float] = None
    cells: List[RecognizedCellItem] = Field(default_factory=list)


class RecognizedBlankStats(BaseModel):
    total_letters: int
    avg_confidence: float
    questions_count: int
    uncertain_cells_count: int = 0
    zero_touch: bool = False


class QRMetadataItem(BaseModel):
    raw: str
    assignment_id: Optional[str] = None
    variant: Optional[int] = None
    page: Optional[int] = None
    total_pages: Optional[int] = None
    questions_count: Optional[int] = None
    student_id: Optional[str] = None


class RecognizeBlankRequestBase64(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded blank photo (JPEG/PNG/WEBP)")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image")
    assignment_id: Optional[str] = Field(default=None, description="Optional assignment ID overriding QR code")
    variant: Optional[int] = Field(default=None, ge=1, le=10, description="Optional variant number overriding QR code")
    student_id: Optional[str] = Field(default=None, description="Optional student ID")
    class_id: Optional[str] = Field(default=None, description="Optional class ID (required when auto_save=True)")
    confidence_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Optional cell review threshold in percent (0-100); defaults to teacher setting or 85.0",
    )
    auto_save: bool = Field(default=False, description="Persist verified submission immediately into DB")


class RecognizeBlankResponse(BaseModel):
    status: str = "success"
    rectification_method: str
    qr_metadata: Optional[QRMetadataItem] = None
    assignment_id: Optional[str] = None
    variant: int = 1
    student_id: Optional[str] = None
    class_id: Optional[str] = None
    student_name: str = ""
    student_name_cells: List[RecognizedCellItem] = Field(default_factory=list)
    questions: List[RecognizedQuestionItem] = Field(default_factory=list)
    stats: RecognizedBlankStats
    confidence_threshold_pct: float = 85.0
    is_verified: bool = False
    verification_warning: Optional[str] = None
    overall_score: Optional[float] = None
    max_score: Optional[float] = None
    score_percentage: Optional[float] = None
    final_grade: Optional[int] = None
    teacher_reviewed_flags: int = 0
    questions_results: List[SubmissionQuestionResult] = Field(default_factory=list)
    submission_draft: Optional[SubmissionItem] = None
    saved_submission_id: Optional[str] = None
