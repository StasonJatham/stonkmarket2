"""Cleanup repository using SQLAlchemy ORM.

Repository for cleaning up expired data from various tables.

Usage:
    from app.repositories import cleanup_orm as cleanup_repo
    
    await cleanup_repo.delete_expired_signals()
    await cleanup_repo.delete_expired_yfinance_cache()
    await cleanup_repo.delete_old_price_history(days=730)
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import delete, func

from app.database.connection import get_session
from app.database.orm import DipfinderSignal, YfinanceInfoCache, PriceHistory
from app.core.logging import get_logger

logger = get_logger("repositories.cleanup_orm")


async def delete_expired_signals() -> int:
    """Delete expired dipfinder signals.
    
    Returns:
        Number of deleted rows
    """
    async with get_session() as session:
        result = await session.execute(
            delete(DipfinderSignal).where(DipfinderSignal.expires_at < func.now())
        )
        await session.commit()
        return result.rowcount


async def delete_expired_yfinance_cache() -> int:
    """Delete expired yfinance cache entries.
    
    Returns:
        Number of deleted rows
    """
    async with get_session() as session:
        result = await session.execute(
            delete(YfinanceInfoCache).where(YfinanceInfoCache.expires_at < func.now())
        )
        await session.commit()
        return result.rowcount


async def delete_old_price_history(days: int = 730) -> int:
    """Delete price history older than specified days.
    
    Args:
        days: Number of days to keep (default 730 = 2 years)
    
    Returns:
        Number of deleted rows
    """
    cutoff_date = date.today() - timedelta(days=days)
    
    async with get_session() as session:
        result = await session.execute(
            delete(PriceHistory).where(PriceHistory.date < cutoff_date)
        )
        await session.commit()
        return result.rowcount
