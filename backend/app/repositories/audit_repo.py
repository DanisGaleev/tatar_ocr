import json
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.models.audit import AuditLogModel, generate_audit_id
from app.repositories.base import BaseRepository


class AuditRepository(BaseRepository[AuditLogModel]):
    """Data access repository for AuditLogModel."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, AuditLogModel)

    async def log_event(
        self,
        *,
        action: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        role: Optional[str] = None,
        target_entity: Optional[str] = None,
        target_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        auto_commit: bool = True,
    ) -> AuditLogModel:
        """Append an immutable audit log entry."""
        log_entry = AuditLogModel(
            id=generate_audit_id(),
            user_id=user_id,
            user_email=user_email,
            role=role,
            action=action.strip().upper(),
            target_entity=target_entity,
            target_id=target_id,
            ip_address=ip_address,
            user_agent=user_agent[:255] if user_agent else None,
            details_json=self._serialize_details(details),
            created_at=datetime.now(timezone.utc),
        )
        self._session.add(log_entry)
        await self._commit_or_flush(log_entry, auto_commit=auto_commit)
        return log_entry

    async def query_logs(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        action: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[List[AuditLogModel], int]:
        """Query audit trail with pagination and filters."""
        skip = max(0, (page - 1) * page_size)
        base_stmt = select(AuditLogModel)
        count_stmt = select(func.count()).select_from(AuditLogModel)

        filters = []
        if action:
            filters.append(AuditLogModel.action == action.strip().upper())
        if user_id:
            filters.append(AuditLogModel.user_id == user_id)
        if start_date:
            filters.append(AuditLogModel.created_at >= start_date)
        if end_date:
            filters.append(AuditLogModel.created_at <= end_date)

        if filters:
            base_stmt = base_stmt.where(*filters)
            count_stmt = count_stmt.where(*filters)

        total_res = await self._session.execute(count_stmt)
        total = total_res.scalar() or 0

        stmt = base_stmt.order_by(AuditLogModel.created_at.desc()).offset(skip).limit(page_size)
        res = await self._session.execute(stmt)
        logs = list(res.scalars().all())
        return logs, total

    def _serialize_details(self, details: Optional[Dict[str, Any]]) -> Optional[str]:
        """Internal helper to JSON-serialize audit details."""
        if not details:
            return None
        try:
            return json.dumps(details, ensure_ascii=False)
        except Exception:
            return str(details)
