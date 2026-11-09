from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from app.core.database import Base


def generate_session_id() -> str:
    return f"ses_{uuid.uuid4().hex}"


class UserSessionModel(Base):
    __tablename__ = "user_sessions"

    id = Column(String(64), primary_key=True, index=True, default=generate_session_id)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    refresh_token_hash = Column(String(128), unique=True, index=True, nullable=False)
    device_info = Column(String(255), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("UserModel", back_populates="sessions")
