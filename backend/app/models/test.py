from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text
from app.core.database import Base

class AssembledTestModel(Base):
    __tablename__ = "assembled_tests"

    test_id = Column(String(64), primary_key=True, index=True, default=lambda: f"TAT-{uuid.uuid4().hex[:6].upper()}")
    title = Column(String(256), nullable=False)
    grade_level = Column(Integer, nullable=False, default=7)
    total_variants = Column(Integer, nullable=False, default=1)
    bundle_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

