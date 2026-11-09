from datetime import datetime, timezone
from typing import Optional, List, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from sqlalchemy.orm import selectinload

from app.models.session import UserSessionModel, generate_session_id
from app.repositories.base import BaseRepository


class SessionRepository(BaseRepository[UserSessionModel]):
    """Data access repository for UserSessionModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, UserSessionModel)

    async def create_session(
        self,
        *,
        user_id: str,
        refresh_token_hash: str,
        expires_at: datetime,
        device_info: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        auto_commit: bool = True,
    ) -> UserSessionModel:
        """Create new active user session."""
        session_obj = UserSessionModel(
            id=generate_session_id(),
            user_id=user_id,
            refresh_token_hash=refresh_token_hash,
            device_info=device_info,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=expires_at,
            created_at=self._now_utc(),
        )
        self._session.add(session_obj)
        await self._commit_or_flush(session_obj, auto_commit=auto_commit)
        return session_obj

    async def get_by_token_hash(self, token_hash: str) -> Optional[UserSessionModel]:
        """Fetch session by SHA-256 hash of refresh token."""
        stmt = select(UserSessionModel).where(UserSessionModel.refresh_token_hash == token_hash)
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def get_active_by_token_hash(self, token_hash: str) -> Optional[UserSessionModel]:
        """Fetch session by hash only if unrevoked and not expired."""
        now = self._now_utc()
        stmt = select(UserSessionModel).where(
            UserSessionModel.refresh_token_hash == token_hash,
            UserSessionModel.revoked_at.is_(None),
            UserSessionModel.expires_at > now,
        )
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: str,
        *,
        active_only: bool = True,
        skip: int = 0,
        limit: int = 50,
    ) -> List[UserSessionModel]:
        """List sessions for a specific user."""
        stmt = select(UserSessionModel).where(UserSessionModel.user_id == user_id)
        if active_only:
            now = self._now_utc()
            stmt = stmt.where(UserSessionModel.revoked_at.is_(None), UserSessionModel.expires_at > now)
        stmt = stmt.order_by(UserSessionModel.created_at.desc()).offset(skip).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def list_all_active(
        self,
        *,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[UserSessionModel], int]:
        """List all active sessions platform-wide for admin overview."""
        now = self._now_utc()
        where_clause = (UserSessionModel.revoked_at.is_(None), UserSessionModel.expires_at > now)
        count_stmt = select(func.count()).select_from(UserSessionModel).where(*where_clause)
        count_res = await self._session.execute(count_stmt)
        total = count_res.scalar() or 0

        stmt = (
            select(UserSessionModel)
            .options(selectinload(UserSessionModel.user))
            .where(*where_clause)
            .order_by(UserSessionModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        res = await self._session.execute(stmt)
        sessions = list(res.scalars().all())
        return sessions, total

    async def revoke_session(self, session_id: str, *, auto_commit: bool = True) -> bool:
        """Revoke a single session by ID (admin kill or user logout)."""
        session = await self.get_by_id(session_id)
        if session and session.revoked_at is None:
            session.revoked_at = self._now_utc()
            await self._commit_or_flush(session, auto_commit=auto_commit)
            return True
        return False

    async def revoke_by_token_hash(self, token_hash: str, *, auto_commit: bool = True) -> bool:
        """Revoke a session by token hash."""
        session = await self.get_by_token_hash(token_hash)
        if session and session.revoked_at is None:
            session.revoked_at = self._now_utc()
            await self._commit_or_flush(session, auto_commit=auto_commit)
            return True
        return False

    async def revoke_all_for_user(self, user_id: str, *, auto_commit: bool = True) -> int:
        """Revoke all active sessions for a user (used during freeze or password reset)."""
        now = self._now_utc()
        stmt = (
            update(UserSessionModel)
            .where(
                UserSessionModel.user_id == user_id,
                UserSessionModel.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0

    def _now_utc(self) -> datetime:
        """Internal helper returning current UTC time."""
        return datetime.now(timezone.utc)
