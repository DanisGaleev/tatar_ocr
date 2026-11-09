from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Text, ForeignKey, DateTime
from app.core.database import Base


def generate_audit_id() -> str:
    return f"aud_{uuid.uuid4().hex}"


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True, default=generate_audit_id)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    user_email = Column(String(255), nullable=True)
    role = Column(String(32), nullable=True)
    action = Column(String(64), nullable=False, index=True)  # LOGIN, FREEZE_USER, etc.
    target_entity = Column(String(64), nullable=True)  # "user", "student", "class", "submission"
    target_id = Column(String(64), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(255), nullable=True)
    details_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
