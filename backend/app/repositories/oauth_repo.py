from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.models.oauth import OAuthAccountModel, generate_oauth_id
from app.repositories.base import BaseRepository


class OAuthRepository(BaseRepository[OAuthAccountModel]):
    """Data access repository for OAuthAccountModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, OAuthAccountModel)

    async def get_by_provider_and_id(
        self,
        provider: str,
        provider_user_id: str,
    ) -> Optional[OAuthAccountModel]:
        """Fetch linked account by provider ('yandex', 'vk') and external user ID."""
        stmt = select(OAuthAccountModel).where(
            OAuthAccountModel.provider == provider.strip().lower(),
            OAuthAccountModel.provider_user_id == str(provider_user_id).strip(),
        )
        res = await self._session.execute(stmt)
        return res.scalar_one_or_none()

    async def get_by_user(self, user_id: str) -> List[OAuthAccountModel]:
        """Fetch all linked OAuth accounts for a user."""
        stmt = select(OAuthAccountModel).where(OAuthAccountModel.user_id == user_id)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def get_by_user_id(self, user_id: str) -> List[OAuthAccountModel]:
        """Alias for get_by_user."""
        return await self.get_by_user(user_id)

    async def link_oauth_account(
        self,
        *,
        user_id: str,
        provider: str,
        provider_user_id: str,
        provider_email: Optional[str] = None,
        auto_commit: bool = True,
    ) -> OAuthAccountModel:
        """Link external provider to user."""
        account = OAuthAccountModel(
            id=generate_oauth_id(),
            user_id=user_id,
            provider=provider.strip().lower(),
            provider_user_id=str(provider_user_id).strip(),
            provider_email=provider_email.strip().lower() if provider_email else None,
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(account)
        await self._commit_or_flush(account, auto_commit=auto_commit)
        return account

    async def link_account(
        self,
        *,
        user_id: str,
        provider: str,
        provider_user_id: str,
        provider_email: Optional[str] = None,
        auto_commit: bool = True,
    ) -> OAuthAccountModel:
        """Alias for link_oauth_account."""
        return await self.link_oauth_account(
            user_id=user_id,
            provider=provider,
            provider_user_id=provider_user_id,
            provider_email=provider_email,
            auto_commit=auto_commit,
        )

    async def unlink_account(
        self,
        user_id: str,
        provider: str,
        *,
        auto_commit: bool = True,
    ) -> bool:
        """Unlink provider from user."""
        stmt = select(OAuthAccountModel).where(
            OAuthAccountModel.user_id == user_id,
            OAuthAccountModel.provider == provider.strip().lower(),
        )
        res = await self._session.execute(stmt)
        acc = res.scalar_one_or_none()
        if acc:
            await self._session.delete(acc)
            if auto_commit:
                await self._session.commit()
            return True
        return False

    async def delete_by_user_id(
        self,
        user_id: str,
        *,
        auto_commit: bool = True,
    ) -> int:
        """Delete all OAuth links for user."""
        stmt = delete(OAuthAccountModel).where(OAuthAccountModel.user_id == user_id)
        res = await self._session.execute(stmt)
        if auto_commit:
            await self._session.commit()
        return res.rowcount or 0
