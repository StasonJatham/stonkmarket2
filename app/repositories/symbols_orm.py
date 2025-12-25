"""Symbol repository using SQLAlchemy ORM.

This is the modern ORM-based implementation. The legacy raw SQL version
is in symbols.py and will be deprecated.

Usage:
    from app.repositories import symbols_orm as symbols
    
    # List all active symbols
    async with get_session() as session:
        symbols = await symbols.list_symbols(session)
    
    # Or use the convenience functions that manage their own sessions
    symbols = await symbols.list_symbols_auto()
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional, Sequence

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.connection import get_session
from app.database.orm import Symbol
from app.core.logging import get_logger

logger = get_logger("repositories.symbols_orm")


# =============================================================================
# CACHE INVALIDATION (shared with legacy implementation)
# =============================================================================

async def invalidate_symbol_caches(symbol: str | None = None) -> None:
    """Invalidate caches affected by symbol changes.
    
    Args:
        symbol: If provided, invalidate caches for specific symbol.
                If None, invalidate all ranking/chart caches.
    """
    from app.cache.cache import Cache
    
    try:
        ranking_cache = Cache(prefix="ranking")
        await ranking_cache.invalidate_pattern("*")
        
        if symbol:
            chart_cache = Cache(prefix="chart")
            await chart_cache.invalidate_pattern(f"{symbol}:*")
            logger.debug(f"Invalidated caches for symbol {symbol}")
        else:
            chart_cache = Cache(prefix="chart")
            await chart_cache.invalidate_pattern("*")
            logger.debug("Invalidated all symbol caches")
    except Exception as e:
        logger.warning(f"Failed to invalidate symbol caches: {e}")


# =============================================================================
# ORM REPOSITORY FUNCTIONS (session-managed)
# =============================================================================

async def list_symbols(session: AsyncSession) -> Sequence[Symbol]:
    """List all active symbols."""
    result = await session.execute(
        select(Symbol)
        .where(Symbol.is_active == True)
        .order_by(Symbol.symbol)
    )
    return result.scalars().all()


async def get_symbol(session: AsyncSession, symbol: str) -> Optional[Symbol]:
    """Get a symbol by ticker."""
    result = await session.execute(
        select(Symbol).where(Symbol.symbol == symbol.upper())
    )
    return result.scalar_one_or_none()


async def upsert_symbol(
    session: AsyncSession,
    symbol: str,
    min_dip_pct: float = 0.15,
    min_days: int = 5,
) -> Symbol:
    """Create or update a symbol."""
    symbol_upper = symbol.upper()
    existing = await get_symbol(session, symbol_upper)
    is_new = existing is None
    
    if existing:
        existing.min_dip_pct = Decimal(str(min_dip_pct))
        existing.min_days = min_days
        existing.is_active = True
        existing.updated_at = datetime.utcnow()
        symbol_obj = existing
    else:
        symbol_obj = Symbol(
            symbol=symbol_upper,
            min_dip_pct=Decimal(str(min_dip_pct)),
            min_days=min_days,
            is_active=True,
        )
        session.add(symbol_obj)
    
    await session.flush()
    
    # Queue new symbols for initial data ingest
    if is_new:
        from app.jobs.definitions import add_to_ingest_queue
        await add_to_ingest_queue(symbol_upper, priority=0)
    
    # Invalidate caches
    await invalidate_symbol_caches(symbol_upper)
    
    return symbol_obj


async def update_symbol(
    session: AsyncSession,
    symbol: str,
    min_dip_pct: float | None = None,
    min_days: int | None = None,
) -> Optional[Symbol]:
    """Update a symbol's configuration."""
    existing = await get_symbol(session, symbol)
    if not existing:
        return None

    if min_dip_pct is not None:
        existing.min_dip_pct = Decimal(str(min_dip_pct))
    if min_days is not None:
        existing.min_days = min_days
    existing.updated_at = datetime.utcnow()
    
    await session.flush()
    await invalidate_symbol_caches(symbol.upper())
    
    return existing


async def delete_symbol(session: AsyncSession, symbol: str) -> bool:
    """Delete a symbol."""
    result = await session.execute(
        delete(Symbol).where(Symbol.symbol == symbol.upper())
    )
    
    if result.rowcount > 0:
        await invalidate_symbol_caches(symbol.upper())
        return True
    return False


async def symbol_exists(session: AsyncSession, symbol: str) -> bool:
    """Check if a symbol exists."""
    result = await session.execute(
        select(func.count()).select_from(Symbol).where(Symbol.symbol == symbol.upper())
    )
    return (result.scalar() or 0) > 0


async def count_symbols(session: AsyncSession) -> int:
    """Count total symbols."""
    result = await session.execute(
        select(func.count()).select_from(Symbol)
    )
    return result.scalar() or 0


# =============================================================================
# CONVENIENCE FUNCTIONS (auto-manage sessions)
# These create their own sessions for simple one-off operations.
# For multiple operations, use the session-managed versions above.
# =============================================================================

async def list_symbols_auto() -> Sequence[Symbol]:
    """List all active symbols (auto-managed session)."""
    async with get_session() as session:
        return await list_symbols(session)


async def get_symbol_auto(symbol: str) -> Optional[Symbol]:
    """Get a symbol by ticker (auto-managed session)."""
    async with get_session() as session:
        return await get_symbol(session, symbol)


async def upsert_symbol_auto(
    symbol: str,
    min_dip_pct: float = 0.15,
    min_days: int = 5,
) -> Symbol:
    """Create or update a symbol (auto-managed session)."""
    async with get_session() as session:
        result = await upsert_symbol(session, symbol, min_dip_pct, min_days)
        await session.commit()
        return result


async def delete_symbol_auto(symbol: str) -> bool:
    """Delete a symbol (auto-managed session)."""
    async with get_session() as session:
        result = await delete_symbol(session, symbol)
        await session.commit()
        return result


async def symbol_exists_auto(symbol: str) -> bool:
    """Check if a symbol exists (auto-managed session)."""
    async with get_session() as session:
        return await symbol_exists(session, symbol)


async def count_symbols_auto() -> int:
    """Count total symbols (auto-managed session)."""
    async with get_session() as session:
        return await count_symbols(session)
