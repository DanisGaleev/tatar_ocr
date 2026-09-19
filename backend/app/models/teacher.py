from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime
from app.core.database import Base

class TeacherModel(Base):
    __tablename__ = "teachers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    teacher_uuid = Column(String(64), unique=True, index=True, nullable=False)
    device_os = Column(String(32), default="android")
    app_version = Column(String(32), default="1.0.4")
    teacher_name = Column(String(256), nullable=False)
    school_name = Column(String(256), nullable=False)
    grade_5_min_pct = Column(Integer, default=85)
    grade_4_min_pct = Column(Integer, default=70)
    grade_3_min_pct = Column(Integer, default=50)
    confidence_flag_threshold = Column(Float, default=0.65)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
