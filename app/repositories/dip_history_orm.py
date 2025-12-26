"""Dip history repository using SQLAlchemy ORM.

This is the modern ORM-based implementation replacing raw SQL in dip_history.py.

Usage:
    from app.repositories.dip_history_orm import get_dip_changes, get_dip_changes_summary
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import select, delete, func, distinct, case, and_

from app.database.connection import get_session
from app.database.orm import DipHistory
from app.core.logging import get_logger

logger = get_logger("repositories.dip_history_orm")


async def get_dip_changes(
    hours: int = 24,
    action: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    """
    Get dip changes within the last X hours.

    Args:
        hours: Number of hours to look back (default 24)
        action: Filter by action type ('added', 'removed', 'updated')
        limit: Maximum number of results

    Returns:
        List of change records with symbol, action, price info, and timestamp
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with get_session() as session:
        stmt = (
            select(DipHistory)
            .where(DipHistory.recorded_at >= since)
        )
        
        if action:
            stmt = stmt.where(DipHistory.action == action)
        
        stmt = stmt.order_by(DipHistory.recorded_at.desc()).limit(limit)
        
        result = await session.execute(stmt)
        rows = result.scalars().all()

    return [
        {
            "symbol": r.symbol,
            "action": r.action,
            "current_price": float(r.current_price) if r.current_price else None,
            "ath_price": float(r.ath_price) if r.ath_price else None,
            "dip_percentage": float(r.dip_percentage) if r.dip_percentage else None,
            "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
        }
        for r in rows
    ]


async def get_dip_changes_summary(hours: int = 24) -> dict:
    """
    Get a summary of dip changes in the last X hours.

    Returns:
        Summary with counts of added, removed, updated dips
    """
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with get_session() as session:
        result = await session.execute(
            select(
                func.count().filter(DipHistory.action == "added").label("added_count"),
                func.count().filter(DipHistory.action == "removed").label("removed_count"),
                func.count().filter(DipHistory.action == "updated").label("updated_count"),
                func.count(distinct(DipHistory.symbol)).label("unique_symbols"),
                func.min(DipHistory.recorded_at).label("earliest_change"),
                func.max(DipHistory.recorded_at).label("latest_change"),
            ).where(DipHistory.recorded_at >= since)
        )
        row = result.one()

    return {
        "hours": hours,
        "since": since.isoformat(),
        "added": row.added_count or 0,
        "removed": row.removed_count or 0,
        "updated": row.updated_count or 0,
        "unique_symbols": row.unique_symbols or 0,
        "earliest_change": row.earliest_change.isoformat() if row.earliest_change else None,
        "latest_change": row.latest_change.isoformat() if row.latest_change else None,
    }


async def get_symbols_added_since(hours: int = 24) -> list[str]:
    """Get list of symbols added in the last X hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with get_session() as session:
        result = await session.execute(
            select(distinct(DipHistory.symbol))
            .where(
                and_(
                    DipHistory.recorded_at >= since,
                    DipHistory.action == "added"
                )
            )
        )
        return [r[0] for r in result.all()]


async def get_symbols_removed_since(hours: int = 24) -> list[str]:
    """Get list of symbols removed in the last X hours."""
    since = datetime.now(timezone.utc) - timedelta(hours=hours)

    async with get_session() as session:
        result = await session.execute(
            select(distinct(DipHistory.symbol))
            .where(
                and_(
                    DipHistory.recorded_at >= since,
                    DipHistory.action == "removed"
                )
            )
        )
        return [r[0] for r in result.all()]


async def get_symbol_history(symbol: str, days: int = 30) -> list[dict]:
    """Get the history of changes for a specific symbol."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    async with get_session() as session:
        result = await session.execute(
            select(DipHistory)
            .where(
                and_(
                    DipHistory.symbol == symbol.upper(),
                    DipHistory.recorded_at >= since
                )
            )
            .order_by(DipHistory.recorded_at.desc())
        )
        rows = result.scalars().all()

    return [
        {
            "action": r.action,
            "current_price": float(r.current_price) if r.current_price else None,
            "ath_price": float(r.ath_price) if r.ath_price else None,
            "dip_percentage": float(r.dip_percentage) if r.dip_percentage else None,
            "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
        }
        for r in rows
    ]


async def manually_record_change(
    symbol: str,
    action: str,
    current_price: Optional[float] = None,
    ath_price: Optional[float] = None,
    dip_percentage: Optional[float] = None,
) -> int:
    """
    Manually record a dip change (used for migrations or manual adjustments).
    The trigger on dip_state handles this automatically for normal operations.
    """
    async with get_session() as session:
        record = DipHistory(
            symbol=symbol.upper(),
            action=action,
            current_price=Decimal(str(current_price)) if current_price else None,
            ath_price=Decimal(str(ath_price)) if ath_price else None,
            dip_percentage=Decimal(str(dip_percentage)) if dip_percentage else None,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        return record.id


async def cleanup_old_history(days: int = 90) -> int:
    """
    Remove history entries older than X days.

    Returns:
        Number of records deleted
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with get_session() as session:
        result = await session.execute(
            delete(DipHistory).where(DipHistory.recorded_at < cutoff)
        )
        await session.commit()
        count = result.rowcount
        
    logger.info(f"Cleaned up {count} old dip history records")
    return count
