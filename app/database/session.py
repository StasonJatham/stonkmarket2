"""Database session dependency injection for FastAPI routes.

This module provides request-scoped database sessions via FastAPI's
dependency injection system. Each request gets its own session that
is automatically committed on success or rolled back on error.

Usage in routes:
    from app.database.session import DbSession
    
    @router.get("/symbols/{symbol}")
    async def get_symbol(symbol: str, db: DbSession) -> SymbolResponse:
        result = await db.execute(select(Symbol).where(Symbol.symbol == symbol))
        symbol = result.scalar_one_or_none()
        ...

Usage in services (when not in request context):
    from app.database.connection import get_session
    
    async def some_background_task():
        async with get_session() as session:
            ...
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Request-scoped database session with auto-commit/rollback.
    
    This dependency creates a new session for each request and ensures:
    - Automatic commit on successful completion
    - Automatic rollback on any exception
    - Proper resource cleanup
    
    The session uses expire_on_commit=False to allow accessing
    ORM objects after the session is closed.
    """
    session_factory = await get_session_factory()
    
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# Type alias for dependency injection - use this in route signatures
DbSession = Annotated[AsyncSession, Depends(get_db_session)]
