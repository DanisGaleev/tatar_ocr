from datetime import datetime, timedelta, timezone
from typing import Optional, List, Tuple, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_

from app.models.user import UserModel, generate_user_id
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[UserModel]):
    """Data access repository for UserModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, UserModel)

    async def get_by_id(self, user_id: str) -> Optional[UserModel]:
        """Fetch user by primary key usr_..."""
        return await self._session.get(UserModel, user_id)

    async def get_by_email(self, email: str) -> Optional[UserModel]:
        """Fetch user by case-insensitive normalized email."""
        norm_email = email.strip().lower()
        stmt = select(UserModel).where(func.lower(UserModel.email) == norm_email)
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def list_users(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        role: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[UserModel], int]:
        """Query users with pagination, filters, and full text search."""
        skip = max(0, (page - 1) * page_size)
        base_stmt = select(UserModel)
        count_stmt = select(func.count()).select_from(UserModel)

        filters = self._build_user_filters(status=status, role=role, search=search)
        if filters:
            base_stmt = base_stmt.where(*filters)
            count_stmt = count_stmt.where(*filters)

        total_res = await self._session.execute(count_stmt)
        total = total_res.scalar() or 0

        stmt = base_stmt.order_by(UserModel.created_at.desc()).offset(skip).limit(page_size)
        users_res = await self._session.execute(stmt)
        users = list(users_res.scalars().all())
        return users, total

    async def create_user(
        self,
        *,
        email: str,
        hashed_password: Optional[str] = None,
        full_name: str,
        school_name: Optional[str] = None,
        role: str = "teacher",
        status: str = "unverified",
        email_verified: bool = False,
        user_id: Optional[str] = None,
        auto_commit: bool = True,
    ) -> UserModel:
        """Create and persist a new user entity."""
        now = self._now_utc()
        user = UserModel(
            id=user_id or generate_user_id(),
            email=email.strip().lower(),
            hashed_password=hashed_password,
            full_name=full_name.strip(),
            school_name=school_name.strip() if school_name else None,
            role=role,
            status=status,
            email_verified=email_verified,
            created_at=now,
            updated_at=now,
        )
        self._session.add(user)
        await self._commit_or_flush(user, auto_commit=auto_commit)
        return user

    async def update_user(
        self,
        user: UserModel,
        update_data: Dict[str, Any],
        *,
        auto_commit: bool = True,
    ) -> UserModel:
        """Update user attributes."""
        for key, value in update_data.items():
            if hasattr(user, key):
                setattr(user, key, value)
        user.updated_at = self._now_utc()
        await self._commit_or_flush(user, auto_commit=auto_commit)
        return user

    async def set_status(
        self,
        user: Union[UserModel, str],
        status: str,
        *,
        auto_commit: bool = True,
    ) -> Optional[UserModel]:
        """Update account status (e.g. active, frozen, deleted)."""
        target_user = await self._resolve_user(user)
        if target_user is None:
            return None
        target_user.status = status
        target_user.updated_at = self._now_utc()
        await self._commit_or_flush(target_user, auto_commit=auto_commit)
        return target_user

    async def soft_delete(
        self,
        user: Union[UserModel, str],
        *,
        auto_commit: bool = True,
    ) -> Optional[UserModel]:
        """Soft delete user by setting status to deleted."""
        return await self.set_status(user, "deleted", auto_commit=auto_commit)

    async def update_password(
        self,
        user: UserModel,
        hashed_password: str,
        *,
        auto_commit: bool = True,
    ) -> UserModel:
        """Update user hashed password and reset failed attempts."""
        user.hashed_password = hashed_password
        user.failed_login_attempts = 0
        user.locked_until = None
        user.updated_at = self._now_utc()
        await self._commit_or_flush(user, auto_commit=auto_commit)
        return user

    async def increment_failed_attempts(
        self,
        user: Union[UserModel, str],
        *,
        max_attempts: int = 5,
        lockout_minutes: int = 15,
        auto_commit: bool = True,
    ) -> Optional[UserModel]:
        """Increment failed attempts and lock account if max attempts exceeded."""
        target_user = await self._resolve_user(user)
        if target_user is None:
            return None
        target_user.failed_login_attempts += 1
        if target_user.failed_login_attempts >= max_attempts:
            target_user.locked_until = self._now_utc() + timedelta(minutes=lockout_minutes)
        await self._commit_or_flush(target_user, auto_commit=auto_commit)
        return target_user

    async def reset_failed_attempts(
        self,
        user: Union[UserModel, str],
        *,
        auto_commit: bool = True,
    ) -> Optional[UserModel]:
        """Reset failed login counter and unlock account."""
        target_user = await self._resolve_user(user)
        if target_user is None:
            return None
        target_user.failed_login_attempts = 0
        target_user.locked_until = None
        await self._commit_or_flush(target_user, auto_commit=auto_commit)
        return target_user

    async def update_last_login(
        self,
        user: UserModel,
        *,
        auto_commit: bool = True,
    ) -> UserModel:
        """Record successful login and activity timestamps."""
        now = self._now_utc()
        user.last_login_at = now
        user.last_active_at = now
        await self._commit_or_flush(user, auto_commit=auto_commit)
        return user

    async def set_2fa_status(
        self,
        user: UserModel,
        *,
        enabled: bool,
        method: str = "none",
        totp_secret: Optional[str] = None,
        auto_commit: bool = True,
    ) -> UserModel:
        """Update two-factor authentication configuration."""
        user.two_factor_enabled = enabled
        user.two_factor_method = method
        user.totp_secret = totp_secret
        user.updated_at = self._now_utc()
        await self._commit_or_flush(user, auto_commit=auto_commit)
        return user

    async def mark_email_verified(
        self,
        user: UserModel,
        *,
        auto_commit: bool = True,
    ) -> UserModel:
        """Mark email verified and transition unverified account to active."""
        user.email_verified = True
        if user.status == "unverified":
            user.status = "active"
        user.updated_at = self._now_utc()
        await self._commit_or_flush(user, auto_commit=auto_commit)
        return user

    async def _resolve_user(self, user: Union[UserModel, str]) -> Optional[UserModel]:
        """Internal helper to resolve user instance if ID was passed."""
        if isinstance(user, UserModel):
            return user
        return await self.get_by_id(user)

    def _build_user_filters(
        self,
        *,
        status: Optional[str],
        role: Optional[str],
        search: Optional[str],
    ) -> List[Any]:
        """Internal helper to construct filter criteria."""
        clauses = []
        if status:
            clauses.append(UserModel.status == status)
        if role:
            clauses.append(UserModel.role == role)
        if search:
            raw = search.strip()
            pattern = f"%{raw}%"
            lower_pattern = f"%{raw.lower()}%"
            title_pattern = f"%{raw.title()}%"
            upper_pattern = f"%{raw.upper()}%"
            clauses.append(
                or_(
                    UserModel.full_name.ilike(pattern),
                    UserModel.email.ilike(pattern),
                    UserModel.school_name.ilike(pattern),
                    UserModel.full_name.like(pattern),
                    UserModel.full_name.like(lower_pattern),
                    UserModel.full_name.like(title_pattern),
                    UserModel.full_name.like(upper_pattern),
                    UserModel.school_name.like(pattern),
                    UserModel.school_name.like(lower_pattern),
                    UserModel.school_name.like(title_pattern),
                    UserModel.school_name.like(upper_pattern),
                )
            )
        return clauses

    def _now_utc(self) -> datetime:
        """Internal helper returning current UTC time."""
        return datetime.now(timezone.utc)
