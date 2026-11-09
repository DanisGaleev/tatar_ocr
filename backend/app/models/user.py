from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, Boolean, Integer, DateTime
from sqlalchemy.orm import relationship
from app.core.database import Base


def generate_user_id() -> str:
    return f"usr_{uuid.uuid4().hex}"


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True, default=generate_user_id)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)  # Argon2id / bcrypt; null for pure OAuth
    full_name = Column(String(255), nullable=False)
    school_name = Column(String(255), nullable=True)
    role = Column(String(32), default="teacher", nullable=False)  # "teacher", "admin", "superadmin"
    status = Column(String(32), default="unverified", nullable=False)  # "unverified", "active", "frozen", "deleted"

    # Verification and 2FA flags
    email_verified = Column(Boolean, default=False, nullable=False)
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    two_factor_method = Column(String(16), default="none", nullable=False)  # "none", "email", "totp"
    totp_secret = Column(String(64), nullable=True)

    # Brute force protection
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)

    # Timestamps & auditing
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_active_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    sessions = relationship("UserSessionModel", back_populates="user", cascade="all, delete-orphan")
    oauth_accounts = relationship("OAuthAccountModel", back_populates="user", cascade="all, delete-orphan")
    classes = relationship("ClassModel", back_populates="teacher_user")
