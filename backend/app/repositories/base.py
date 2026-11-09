from typing import Generic, TypeVar, Type, Optional, List, Any, Dict, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, inspect
from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Generic asynchronous repository base class."""

    def __init__(self, session: AsyncSession, model: Type[ModelType]):
        self._session: AsyncSession = session
        self._model: Type[ModelType] = model

    @property
    def session(self) -> AsyncSession:
        return self._session

    @property
    def db(self) -> AsyncSession:
        return self._session

    @property
    def model(self) -> Type[ModelType]:
        return self._model

    def _pk_column(self) -> Any:
        """Internal helper returning primary key column attribute."""
        return inspect(self._model).primary_key[0]

    async def get(self, id: Any) -> Optional[ModelType]:
        """Fetch a single record by primary key."""
        return await self._session.get(self._model, id)

    async def get_by_id(self, id: Any) -> Optional[ModelType]:
        """Fetch a single record by primary key (alias for get)."""
        return await self.get(id)

    async def get_all(self, *, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Fetch multiple records with offset and limit."""
        stmt = select(self._model).offset(skip).limit(limit)
        res = await self._session.execute(stmt)
        return list(res.scalars().all())

    async def get_multi(self, *, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Fetch multiple records with offset and limit (alias for get_all)."""
        return await self.get_all(skip=skip, limit=limit)

    async def count(self) -> int:
        """Count total records for the model."""
        stmt = select(func.count()).select_from(self._model)
        res = await self._session.execute(stmt)
        return res.scalar() or 0

    async def create(
        self,
        obj_in: Union[Dict[str, Any], ModelType],
        *,
        auto_commit: bool = True,
    ) -> ModelType:
        """Create and persist a new record."""
        if isinstance(obj_in, dict):
            db_obj = self._model(**obj_in)
        else:
            db_obj = obj_in
        self._session.add(db_obj)
        await self._commit_or_flush(db_obj, auto_commit=auto_commit)
        return db_obj

    async def update(
        self,
        db_obj: ModelType,
        obj_in: Union[Dict[str, Any], ModelType],
        *,
        auto_commit: bool = True,
    ) -> ModelType:
        """Update existing record attributes."""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = {
                c.name: getattr(obj_in, c.name)
                for c in self._model.__table__.columns
                if getattr(obj_in, c.name, None) is not None
            }
        for field, value in update_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        await self._commit_or_flush(db_obj, auto_commit=auto_commit)
        return db_obj

    async def delete(
        self,
        db_obj: Union[ModelType, Any],
        *,
        auto_commit: bool = True,
    ) -> bool:
        """Delete record from database. Accepts model instance or primary key value."""
        if not isinstance(db_obj, self._model):
            db_obj = await self.get(db_obj)
            if db_obj is None:
                return False
        await self._session.delete(db_obj)
        await self._commit_or_flush(auto_commit=auto_commit)
        return True

    async def flush(self) -> None:
        """Flush pending operations to database within transaction."""
        await self._session.flush()

    async def commit(self) -> None:
        """Commit current transaction."""
        await self._session.commit()

    async def rollback(self) -> None:
        """Rollback current transaction."""
        await self._session.rollback()

    async def _commit_or_flush(
        self,
        db_obj: Optional[ModelType] = None,
        *,
        auto_commit: bool = True,
    ) -> None:
        """Internal helper: commits or flushes according to transaction strategy."""
        if auto_commit:
            await self._session.commit()
            if db_obj is not None:
                await self._session.refresh(db_obj)
        else:
            await self._session.flush()
