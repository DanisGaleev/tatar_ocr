from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, Text
from app.core.database import Base

class SubmissionModel(Base):
    __tablename__ = "submissions"

    submission_id = Column(String(64), primary_key=True, index=True, default=lambda: f"sub_{uuid.uuid4().hex[:8]}")
    client_submission_uuid = Column(String(64), unique=True, index=True, nullable=True)
    assignment_id = Column(String(64), index=True, nullable=False)
    class_id = Column(String(64), index=True, nullable=False)
    student_id = Column(String(64), index=True, nullable=False)
    student_name = Column(String(192), nullable=False)
    variant = Column(Integer, default=1)
    checked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    overall_score = Column(Float, default=0.0)
    max_score = Column(Float, default=8.0)
    final_grade = Column(Integer, default=2)
    teacher_reviewed_flags = Column(Integer, default=0)
    questions_results_json = Column(Text, nullable=False)  # Serialized questions_results array
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
