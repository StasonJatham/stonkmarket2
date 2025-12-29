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

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import Symbol


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
# INTERNAL ORM REPOSITORY FUNCTIONS (session-managed)
# =============================================================================

async def _list_symbols(session: AsyncSession) -> Sequence[Symbol]:
    """List all active symbols."""
    result = await session.execute(
        select(Symbol)
        .where(Symbol.is_active == True)
        .order_by(Symbol.symbol)
    )
    return result.scalars().all()


async def _get_symbol(session: AsyncSession, symbol: str) -> Symbol | None:
    """Get a symbol by ticker."""
    result = await session.execute(
        select(Symbol).where(Symbol.symbol == symbol.upper())
    )
    return result.scalar_one_or_none()


async def _upsert_symbol(
    session: AsyncSession,
    symbol: str,
    min_dip_pct: float = 0.15,
    min_days: int = 5,
) -> Symbol:
    """Create or update a symbol."""
    symbol_upper = symbol.upper()
    existing = await _get_symbol(session, symbol_upper)
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


async def _update_symbol(
    session: AsyncSession,
    symbol: str,
    min_dip_pct: float | None = None,
    min_days: int | None = None,
) -> Symbol | None:
    """Update a symbol's configuration."""
    existing = await _get_symbol(session, symbol)
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


async def _delete_symbol(session: AsyncSession, symbol: str) -> bool:
    """Delete a symbol."""
    result = await session.execute(
        delete(Symbol).where(Symbol.symbol == symbol.upper())
    )

    if result.rowcount > 0:
        await invalidate_symbol_caches(symbol.upper())
        return True
    return False


async def _symbol_exists(session: AsyncSession, symbol: str) -> bool:
    """Check if a symbol exists."""
    result = await session.execute(
        select(func.count()).select_from(Symbol).where(Symbol.symbol == symbol.upper())
    )
    return (result.scalar() or 0) > 0


async def _count_symbols(session: AsyncSession) -> int:
    """Count total symbols."""
    result = await session.execute(
        select(func.count()).select_from(Symbol)
    )
    return result.scalar() or 0


# =============================================================================
# PUBLIC API FUNCTIONS (auto-manage sessions)
# These match the original symbols.py API for drop-in compatibility.
# =============================================================================

async def list_symbols() -> Sequence[Symbol]:
    """List all active symbols."""
    async with get_session() as session:
        return await _list_symbols(session)


async def get_symbol(symbol: str) -> Symbol | None:
    """Get a symbol by ticker."""
    async with get_session() as session:
        return await _get_symbol(session, symbol)


async def upsert_symbol(
    symbol: str,
    min_dip_pct: float = 0.15,
    min_days: int = 5,
) -> Symbol:
    """Create or update a symbol."""
    async with get_session() as session:
        result = await _upsert_symbol(session, symbol, min_dip_pct, min_days)
        await session.commit()
        return result


async def update_symbol(
    symbol: str,
    min_dip_pct: float | None = None,
    min_days: int | None = None,
) -> Symbol | None:
    """Update a symbol's configuration."""
    async with get_session() as session:
        result = await _update_symbol(session, symbol, min_dip_pct, min_days)
        if result:
            await session.commit()
        return result


async def delete_symbol(symbol: str) -> bool:
    """Delete a symbol."""
    async with get_session() as session:
        result = await _delete_symbol(session, symbol)
        await session.commit()
        return result


async def symbol_exists(symbol: str) -> bool:
    """Check if a symbol exists."""
    async with get_session() as session:
        return await _symbol_exists(session, symbol)


async def count_symbols() -> int:
    """Count total symbols."""
    async with get_session() as session:
        return await _count_symbols(session)


async def get_symbols_by_status(fetch_status: str) -> list[Symbol]:
    """Get all symbols with a specific fetch status.
    
    Args:
        fetch_status: Status to filter by ('pending', 'fetching', 'fetched', 'error')
    
    Returns:
        List of Symbol objects matching the status
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol).where(Symbol.fetch_status == fetch_status)
        )
        return list(result.scalars().all())


async def update_fetch_status(
    symbol: str,
    fetch_status: str,
    fetch_error: str | None = None,
    fetched_at: datetime | None = None,
) -> bool:
    """Update a symbol's fetch status.
    
    Args:
        symbol: Symbol ticker
        fetch_status: Status string ('pending', 'fetching', 'fetched', 'error')
        fetch_error: Error message if status is 'error'
        fetched_at: Timestamp when fetch completed
    
    Returns:
        True if symbol was updated, False if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol).where(Symbol.symbol == symbol.upper())
        )
        sym = result.scalar_one_or_none()

        if not sym:
            return False

        sym.fetch_status = fetch_status
        sym.fetch_error = fetch_error
        if fetched_at:
            sym.fetched_at = fetched_at
        sym.updated_at = datetime.utcnow()

        await session.commit()
        return True


async def update_symbol_info(
    symbol: str,
    *,
    name: str | None = None,
    sector: str | None = None,
    summary_ai: str | None = None,
) -> bool:
    """Update a symbol's name, sector, and/or AI summary.
    
    Only updates fields that are explicitly provided (not None).
    
    Returns:
        True if symbol was updated, False if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol).where(Symbol.symbol == symbol.upper())
        )
        sym = result.scalar_one_or_none()

        if not sym:
            return False

        if name is not None:
            sym.name = name
        if sector is not None:
            sym.sector = sector
        if summary_ai is not None:
            sym.summary_ai = summary_ai
        sym.updated_at = datetime.utcnow()

        await session.commit()
        await invalidate_symbol_caches(symbol.upper())
        return True


async def get_symbol_summary_ai(symbol: str) -> str | None:
    """Get a symbol's AI summary.
    
    Returns:
        AI summary string or None if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.summary_ai).where(Symbol.symbol == symbol.upper())
        )
        row = result.one_or_none()
        return row[0] if row else None


async def update_stock_info_cache(
    symbol: str,
    *,
    fifty_two_week_low: float | None = None,
    fifty_two_week_high: float | None = None,
    previous_close: float | None = None,
    avg_volume: int | None = None,
    pe_ratio: float | None = None,
    market_cap: int | None = None,
) -> bool:
    """Update a symbol's cached stock info fields.
    
    Used by scheduled jobs to cache live market data.
    
    Args:
        symbol: Stock symbol
        fifty_two_week_low: 52-week low price
        fifty_two_week_high: 52-week high price
        previous_close: Previous day's close price
        avg_volume: Average trading volume
        pe_ratio: Price-to-earnings ratio
        market_cap: Market capitalization
        
    Returns:
        True if updated, False if symbol not found
    """
    from decimal import Decimal
    
    async with get_session() as session:
        result = await session.execute(
            select(Symbol).where(Symbol.symbol == symbol.upper())
        )
        sym = result.scalar_one_or_none()

        if not sym:
            return False

        if fifty_two_week_low is not None:
            sym.fifty_two_week_low = Decimal(str(fifty_two_week_low))
        if fifty_two_week_high is not None:
            sym.fifty_two_week_high = Decimal(str(fifty_two_week_high))
        if previous_close is not None:
            sym.previous_close = Decimal(str(previous_close))
        if avg_volume is not None:
            sym.avg_volume = avg_volume
        if pe_ratio is not None:
            sym.pe_ratio = Decimal(str(pe_ratio))
        if market_cap is not None:
            sym.market_cap = market_cap
        
        sym.stock_info_updated_at = datetime.utcnow()
        await session.commit()
        return True


async def batch_update_stock_info_cache(updates: list[dict]) -> int:
    """Batch update stock info cache for multiple symbols.
    
    Args:
        updates: List of dicts with 'symbol' and optional fields:
                 fifty_two_week_low, fifty_two_week_high, previous_close,
                 avg_volume, pe_ratio, market_cap
                 
    Returns:
        Number of symbols updated
    """
    from decimal import Decimal
    
    updated = 0
    async with get_session() as session:
        for upd in updates:
            symbol = upd.get("symbol")
            if not symbol:
                continue
                
            result = await session.execute(
                select(Symbol).where(Symbol.symbol == symbol.upper())
            )
            sym = result.scalar_one_or_none()
            
            if not sym:
                continue
            
            if "fifty_two_week_low" in upd and upd["fifty_two_week_low"] is not None:
                sym.fifty_two_week_low = Decimal(str(upd["fifty_two_week_low"]))
            if "fifty_two_week_high" in upd and upd["fifty_two_week_high"] is not None:
                sym.fifty_two_week_high = Decimal(str(upd["fifty_two_week_high"]))
            if "previous_close" in upd and upd["previous_close"] is not None:
                sym.previous_close = Decimal(str(upd["previous_close"]))
            if "avg_volume" in upd and upd["avg_volume"] is not None:
                sym.avg_volume = upd["avg_volume"]
            if "pe_ratio" in upd and upd["pe_ratio"] is not None:
                sym.pe_ratio = Decimal(str(upd["pe_ratio"]))
            if "market_cap" in upd and upd["market_cap"] is not None:
                sym.market_cap = upd["market_cap"]
            
            sym.stock_info_updated_at = datetime.utcnow()
            updated += 1
        
        await session.commit()
    
    return updated
