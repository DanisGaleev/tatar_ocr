from datetime import datetime, timezone
from sqlalchemy import Column, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.core.database import Base

class ClassModel(Base):
    __tablename__ = "classes"

    class_id = Column(String(64), primary_key=True, index=True)
    teacher_uuid = Column(String(64), index=True, nullable=True)
    name = Column(String(64), nullable=False)
    subject = Column(String(64), default="tatar_language")
    academic_year = Column(String(32), default="2026-2027")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    students = relationship("StudentModel", back_populates="school_class", cascade="all, delete-orphan")
    assignments = relationship("ClassAssignmentModel", backref="school_class", cascade="all, delete-orphan")


class StudentModel(Base):
    __tablename__ = "students"

    student_id = Column(String(64), primary_key=True, index=True)
    class_id = Column(String(64), ForeignKey("classes.class_id"), index=True, nullable=False)
    last_name = Column(String(64), nullable=False)
    first_name = Column(String(64), nullable=False)
    middle_name = Column(String(64), nullable=True, default="")
    full_name = Column(String(192), nullable=False)
    short_name = Column(String(96), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    school_class = relationship("ClassModel", back_populates="students")


class ClassAssignmentModel(Base):
    __tablename__ = "class_assignments"

    id = Column(String(64), primary_key=True, index=True)
    class_id = Column(String(64), ForeignKey("classes.class_id"), index=True, nullable=False)
    assignment_id = Column(String(64), ForeignKey("assembled_tests.test_id"), index=True, nullable=False)
    assigned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    due_date = Column(DateTime, nullable=True)
    status = Column(String(32), default="active")  # active, completed, archived
