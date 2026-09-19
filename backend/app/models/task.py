from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Boolean, DateTime
from app.core.database import Base

class TaskBankModel(Base):
    __tablename__ = "task_bank"

    task_id = Column(String(64), primary_key=True, index=True, default=lambda: f"tsk_{uuid.uuid4().hex[:8]}")
    prompt_tt = Column(String(512), nullable=False)
    expected_answer = Column(String(32), nullable=False)
    cell_count = Column(Integer, nullable=False, default=8)
    grade_level = Column(Integer, nullable=False, default=7)
    topic_tag = Column(String(64), nullable=False, index=True)
    topic_name_tt = Column(String(128), nullable=False)
    task_type = Column(String(64), nullable=False, default="custom")
    is_public_in_bank = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

