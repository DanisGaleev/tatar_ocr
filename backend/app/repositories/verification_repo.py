from datetime import datetime, timedelta, timezone
from typing import Optional, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.models.verification import VerificationCodeModel, generate_verification_id
from app.repositories.base import BaseRepository


class VerificationCodeRepository(BaseRepository[VerificationCodeModel]):
    """Data access repository for VerificationCodeModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, VerificationCodeModel)

    async def create_code(
        self,
        *,
        email: str,
        code: str,
        code_type: str,
        ttl_minutes: int = 10,
        attempts_left: int = 5,
        auto_commit: bool = True,
    ) -> VerificationCodeModel:
        """Create a new verification OTP code record."""
        now = self._now_utc()
        record = VerificationCodeModel(
            id=generate_verification_id(),
            email=email.strip().lower(),
            code=code.strip(),
            code_type=code_type,
            attempts_left=attempts_left,
            expires_at=now + timedelta(minutes=ttl_minutes),
            is_used=False,
            created_at=now,
        )
        self._session.add(record)
        await self._commit_or_flush(record, auto_commit=auto_commit)
        return record

    async def get_valid_code(
        self,
        email: str,
        code_type: str,
    ) -> Optional[VerificationCodeModel]:
        """Fetch latest active code that is not used, not expired, and has attempts remaining."""
        return await self.get_latest_valid_code(email=email, code_type=code_type)

    async def get_latest_valid_code(
        self,
        email: str,
        code_type: str,
    ) -> Optional[VerificationCodeModel]:
        """Fetch latest active code that is not used, not expired, and has attempts remaining."""
        now = self._now_utc()
        stmt = (
            select(VerificationCodeModel)
            .where(
                VerificationCodeModel.email == email.strip().lower(),
                VerificationCodeModel.code_type == code_type,
                VerificationCodeModel.is_used == False,
                VerificationCodeModel.expires_at > now,
                VerificationCodeModel.attempts_left > 0,
            )
            .order_by(VerificationCodeModel.created_at.desc())
        )
        res = await self._session.execute(stmt)
        return res.scalars().first()

    async def decrement_attempts(
        self,
        code_obj: VerificationCodeModel,
        *,
        auto_commit: bool = True,
    ) -> int:
        """Decrement attempts_left by 1."""
        if code_obj.attempts_left > 0:
            code_obj.attempts_left -= 1
        await self._commit_or_flush(code_obj, auto_commit=auto_commit)
        return code_obj.attempts_left

    async def consume_code(
        self,
        code_obj: Union[VerificationCodeModel, str],
        *,
        auto_commit: bool = True,
    ) -> Optional[VerificationCodeModel]:
        """Mark code as consumed/used."""
        if isinstance(code_obj, str):
            code_entity = await self.get_by_id(code_obj)
        else:
            code_entity = code_obj
        if code_entity:
            code_entity.is_used = True
            await self._commit_or_flush(code_entity, auto_commit=auto_commit)
        return code_entity

    async def invalidate_all(
        self,
        email: str,
        code_type: str,
        *,
        auto_commit: bool = True,
    ) -> int:
        """Invalidate all pending codes for email and code_type."""
        stmt = (
            update(VerificationCodeModel)
            .where(
                VerificationCodeModel.email == email.strip().lower(),
                VerificationCodeModel.code_type == code_type,
                VerificationCodeModel.is_used == False,
            )
            .values(is_used=True)
        )
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0

    async def get_last_created_at(
        self,
        email: str,
        code_type: str,
    ) -> Optional[datetime]:
        """Get timestamp of most recently issued code for cooldown verification."""
        stmt = (
            select(VerificationCodeModel.created_at)
            .where(
                VerificationCodeModel.email == email.strip().lower(),
                VerificationCodeModel.code_type == code_type,
            )
            .order_by(VerificationCodeModel.created_at.desc())
        )
        res = await self._session.execute(stmt)
        return res.scalars().first()

    def _now_utc(self) -> datetime:
        """Internal helper returning current UTC time."""
        return datetime.now(timezone.utc)
