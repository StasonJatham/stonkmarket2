"""Dip history repository for tracking changes."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
from decimal import Decimal

from app.database.connection import get_db, fetch_all, fetch_one, fetch_val
from app.core.logging import get_logger

logger = get_logger("dip_history")


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
    since = datetime.utcnow() - timedelta(hours=hours)
    
    if action:
        rows = await fetch_all(
            """
            SELECT symbol, action, current_price, ath_price, dip_percentage, recorded_at
            FROM dip_history
            WHERE recorded_at >= $1 AND action = $2
            ORDER BY recorded_at DESC
            LIMIT $3
            """,
            since, action, limit
        )
    else:
        rows = await fetch_all(
            """
            SELECT symbol, action, current_price, ath_price, dip_percentage, recorded_at
            FROM dip_history
            WHERE recorded_at >= $1
            ORDER BY recorded_at DESC
            LIMIT $2
            """,
            since, limit
        )
    
    return [
        {
            "symbol": r["symbol"],
            "action": r["action"],
            "current_price": float(r["current_price"]) if r["current_price"] else None,
            "ath_price": float(r["ath_price"]) if r["ath_price"] else None,
            "dip_percentage": float(r["dip_percentage"]) if r["dip_percentage"] else None,
            "recorded_at": r["recorded_at"].isoformat() if r["recorded_at"] else None,
        }
        for r in rows
    ]


async def get_dip_changes_summary(hours: int = 24) -> dict:
    """
    Get a summary of dip changes in the last X hours.
    
    Returns:
        Summary with counts of added, removed, updated dips
    """
    since = datetime.utcnow() - timedelta(hours=hours)
    
    async with get_db() as conn:
        row = await conn.fetchrow(
            """
            SELECT 
                COUNT(*) FILTER (WHERE action = 'added') as added_count,
                COUNT(*) FILTER (WHERE action = 'removed') as removed_count,
                COUNT(*) FILTER (WHERE action = 'updated') as updated_count,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(recorded_at) as earliest_change,
                MAX(recorded_at) as latest_change
            FROM dip_history
            WHERE recorded_at >= $1
            """,
            since
        )
    
    return {
        "hours": hours,
        "since": since.isoformat(),
        "added": row["added_count"] or 0,
        "removed": row["removed_count"] or 0,
        "updated": row["updated_count"] or 0,
        "unique_symbols": row["unique_symbols"] or 0,
        "earliest_change": row["earliest_change"].isoformat() if row["earliest_change"] else None,
        "latest_change": row["latest_change"].isoformat() if row["latest_change"] else None,
    }


async def get_symbols_added_since(hours: int = 24) -> list[str]:
    """Get list of symbols added in the last X hours."""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    rows = await fetch_all(
        """
        SELECT DISTINCT symbol
        FROM dip_history
        WHERE recorded_at >= $1 AND action = 'added'
        """,
        since
    )
    
    return [r["symbol"] for r in rows]


async def get_symbols_removed_since(hours: int = 24) -> list[str]:
    """Get list of symbols removed in the last X hours."""
    since = datetime.utcnow() - timedelta(hours=hours)
    
    rows = await fetch_all(
        """
        SELECT DISTINCT symbol
        FROM dip_history
        WHERE recorded_at >= $1 AND action = 'removed'
        """,
        since
    )
    
    return [r["symbol"] for r in rows]


async def get_symbol_history(symbol: str, days: int = 30) -> list[dict]:
    """Get the history of changes for a specific symbol."""
    since = datetime.utcnow() - timedelta(days=days)
    
    rows = await fetch_all(
        """
        SELECT action, current_price, ath_price, dip_percentage, recorded_at
        FROM dip_history
        WHERE symbol = $1 AND recorded_at >= $2
        ORDER BY recorded_at DESC
        """,
        symbol.upper(), since
    )
    
    return [
        {
            "action": r["action"],
            "current_price": float(r["current_price"]) if r["current_price"] else None,
            "ath_price": float(r["ath_price"]) if r["ath_price"] else None,
            "dip_percentage": float(r["dip_percentage"]) if r["dip_percentage"] else None,
            "recorded_at": r["recorded_at"].isoformat() if r["recorded_at"] else None,
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
    async with get_db() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO dip_history (symbol, action, current_price, ath_price, dip_percentage)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            symbol.upper(), action, current_price, ath_price, dip_percentage
        )
    
    return row["id"]


async def cleanup_old_history(days: int = 90) -> int:
    """
    Remove history entries older than X days.
    
    Returns:
        Number of records deleted
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    async with get_db() as conn:
        result = await conn.execute(
            "DELETE FROM dip_history WHERE recorded_at < $1",
            cutoff
        )
    
    # Parse "DELETE X" from result
    try:
        count = int(result.split()[-1])
        logger.info(f"Cleaned up {count} old dip history records")
        return count
    except (ValueError, IndexError):
        return 0
