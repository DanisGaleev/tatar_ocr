from datetime import datetime, timezone
import uuid
from sqlalchemy import Column, String, ForeignKey, DateTime, UniqueConstraint
from sqlalchemy.orm import relationship
from app.core.database import Base


def generate_oauth_id() -> str:
    return f"oa_{uuid.uuid4().hex}"


class OAuthAccountModel(Base):
    __tablename__ = "oauth_accounts"

    id = Column(String(64), primary_key=True, default=generate_oauth_id)
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    provider = Column(String(32), nullable=False)  # "yandex", "vk"
    provider_user_id = Column(String(128), nullable=False)
    provider_email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (UniqueConstraint("provider", "provider_user_id", name="uq_provider_user"),)

    user = relationship("UserModel", back_populates="oauth_accounts")
