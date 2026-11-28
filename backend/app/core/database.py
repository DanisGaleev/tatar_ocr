from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db() -> None:
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        for table_name in ("task_bank", "assembled_tests"):
            res = await conn.execute(text(f"PRAGMA table_info({table_name})"))
            cols = {row[1] for row in res.fetchall()}
            if "language" not in cols:
                await conn.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN language VARCHAR(8) NOT NULL DEFAULT 'tt'")
                )
