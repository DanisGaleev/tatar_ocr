from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Integer, Boolean, DateTime
from app.core.database import Base


def generate_verification_id() -> str:
    return f"vcd_{uuid.uuid4().hex}"


class VerificationCodeModel(Base):
    __tablename__ = "verification_codes"

    id = Column(String(64), primary_key=True, default=generate_verification_id)
    email = Column(String(255), index=True, nullable=False)
    code = Column(String(16), nullable=False)  # 6-digit numeric string
    code_type = Column(String(32), nullable=False)  # "register_verify", "login_2fa", "password_reset"
    attempts_left = Column(Integer, default=5, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
