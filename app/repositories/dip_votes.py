"""Dip voting repository - buy/sell votes on current dips (PostgreSQL)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from app.core.config import settings
from app.database.connection import get_pg_connection, fetch_one, fetch_all, execute, fetch_val
from app.core.logging import get_logger

logger = get_logger("dip_votes")


async def get_vote_cooldown_remaining(
    symbol: str,
    fingerprint: str,
) -> Optional[int]:
    """
    Check if user is on cooldown for voting on this symbol.
    
    Returns:
        Seconds remaining on cooldown, or None if can vote
    """
    cooldown_days = settings.vote_cooldown_days
    cutoff = datetime.utcnow() - timedelta(days=cooldown_days)
    
    row = await fetch_one(
        """
        SELECT created_at FROM dip_votes
        WHERE symbol = $1 AND fingerprint = $2 AND created_at > $3
        ORDER BY created_at DESC
        LIMIT 1
        """,
        symbol.upper(), fingerprint, cutoff,
    )
    
    if not row:
        return None
    
    last_vote = row["created_at"]
    cooldown_end = last_vote + timedelta(days=cooldown_days)
    remaining = (cooldown_end - datetime.utcnow()).total_seconds()
    
    return int(remaining) if remaining > 0 else None


async def add_vote(
    symbol: str,
    fingerprint: str,
    vote_type: str,
    vote_weight: int = 1,
    api_key_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """
    Add a vote for a dip.
    
    Args:
        symbol: Stock symbol
        fingerprint: Hashed voter identifier
        vote_type: 'buy' or 'sell'
        vote_weight: Vote weight multiplier (default 1, API key users get 10)
        api_key_id: Optional API key ID if using authenticated voting
        
    Returns:
        Tuple of (success, error_message)
    """
    if vote_type not in ("buy", "sell"):
        return False, "Invalid vote type. Must be 'buy' or 'sell'"
    
    # Check cooldown
    cooldown = await get_vote_cooldown_remaining(symbol, fingerprint)
    if cooldown:
        hours = cooldown // 3600
        minutes = (cooldown % 3600) // 60
        return False, f"Vote cooldown active. Try again in {hours}h {minutes}m"
    
    # Check symbol exists in dip_state
    exists = await fetch_val(
        "SELECT EXISTS(SELECT 1 FROM dip_state WHERE symbol = $1)",
        symbol.upper(),
    )
    if not exists:
        return False, f"Symbol {symbol.upper()} is not currently in a dip"
    
    # Insert vote with weight
    async with get_pg_connection() as conn:
        await conn.execute(
            """
            INSERT INTO dip_votes (symbol, fingerprint, vote_type, vote_weight, api_key_id, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (symbol, fingerprint) DO UPDATE SET
                vote_type = EXCLUDED.vote_type,
                vote_weight = EXCLUDED.vote_weight,
                api_key_id = EXCLUDED.api_key_id,
                created_at = NOW()
            """,
            symbol.upper(), fingerprint, vote_type, vote_weight, api_key_id,
        )
    
    logger.info(f"Vote recorded: {symbol.upper()} {vote_type} (weight: {vote_weight})")
    return True, None


async def get_vote_counts(symbol: str) -> dict:
    """Get vote counts for a symbol (with weighted totals)."""
    row = await fetch_one(
        """
        SELECT 
            COUNT(*) FILTER (WHERE vote_type = 'buy') as buy_count,
            COUNT(*) FILTER (WHERE vote_type = 'sell') as sell_count,
            COALESCE(SUM(vote_weight) FILTER (WHERE vote_type = 'buy'), 0) as buy_weighted,
            COALESCE(SUM(vote_weight) FILTER (WHERE vote_type = 'sell'), 0) as sell_weighted
        FROM dip_votes
        WHERE symbol = $1
        """,
        symbol.upper(),
    )
    
    if not row:
        return {"buy": 0, "sell": 0, "buy_weighted": 0, "sell_weighted": 0, "net_score": 0}
    
    return {
        "buy": row["buy_count"] or 0,
        "sell": row["sell_count"] or 0,
        "buy_weighted": row["buy_weighted"] or 0,
        "sell_weighted": row["sell_weighted"] or 0,
        "net_score": (row["buy_weighted"] or 0) - (row["sell_weighted"] or 0),
    }


async def get_all_vote_counts() -> dict[str, dict]:
    """Get vote counts for all symbols with votes."""
    rows = await fetch_all(
        """
        SELECT 
            symbol,
            COUNT(*) FILTER (WHERE vote_type = 'buy') as buy_count,
            COUNT(*) FILTER (WHERE vote_type = 'sell') as sell_count,
            COALESCE(SUM(vote_weight) FILTER (WHERE vote_type = 'buy'), 0) as buy_weighted,
            COALESCE(SUM(vote_weight) FILTER (WHERE vote_type = 'sell'), 0) as sell_weighted
        FROM dip_votes
        GROUP BY symbol
        """
    )
    
    result = {}
    for row in rows:
        result[row["symbol"]] = {
            "buy": row["buy_count"] or 0,
            "sell": row["sell_count"] or 0,
            "buy_weighted": row["buy_weighted"] or 0,
            "sell_weighted": row["sell_weighted"] or 0,
            "net_score": (row["buy_weighted"] or 0) - (row["sell_weighted"] or 0),
        }
    
    return result


async def get_user_votes(
    fingerprint: str,
    limit: int = 50,
) -> list[dict]:
    """Get recent votes by a user."""
    rows = await fetch_all(
        """
        SELECT id, symbol, fingerprint, vote_type, vote_weight, created_at
        FROM dip_votes
        WHERE fingerprint = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        fingerprint, limit,
    )
    return [dict(r) for r in rows]


async def get_user_vote_for_symbol(
    symbol: str,
    fingerprint: str,
) -> Optional[dict]:
    """Get user's existing vote for a symbol."""
    row = await fetch_one(
        """
        SELECT id, symbol, fingerprint, vote_type, vote_weight, created_at
        FROM dip_votes
        WHERE symbol = $1 AND fingerprint = $2
        """,
        symbol.upper(), fingerprint,
    )
    return dict(row) if row else None


# ============================================================================
# AI Analysis Functions (PostgreSQL)
# ============================================================================

async def get_ai_analysis(symbol: str) -> Optional[dict]:
    """Get cached AI analysis for a symbol."""
    row = await fetch_one(
        """
        SELECT symbol, tinder_bio, ai_rating, rating_reasoning as ai_reasoning,
               model_used, is_batch_generated, generated_at, expires_at
        FROM dip_ai_analysis
        WHERE symbol = $1 AND (expires_at IS NULL OR expires_at > NOW())
        """,
        symbol.upper(),
    )
    return dict(row) if row else None


async def upsert_ai_analysis(
    symbol: str,
    tinder_bio: Optional[str] = None,
    ai_rating: Optional[float] = None,
    ai_reasoning: Optional[str] = None,
    model_used: str = "gpt-4o-mini",
    is_batch: bool = False,
    expires_hours: int = 168,  # 7 days
) -> dict:
    """Create or update AI analysis for a symbol."""
    expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
    
    async with get_pg_connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO dip_ai_analysis (
                symbol, tinder_bio, ai_rating, rating_reasoning,
                model_used, is_batch_generated, generated_at, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7)
            ON CONFLICT (symbol) DO UPDATE SET
                tinder_bio = EXCLUDED.tinder_bio,
                ai_rating = EXCLUDED.ai_rating,
                rating_reasoning = EXCLUDED.rating_reasoning,
                model_used = EXCLUDED.model_used,
                is_batch_generated = EXCLUDED.is_batch_generated,
                generated_at = NOW(),
                expires_at = EXCLUDED.expires_at
            RETURNING symbol, tinder_bio, ai_rating, rating_reasoning as ai_reasoning,
                      model_used, is_batch_generated, generated_at, expires_at
            """,
            symbol.upper(), tinder_bio, ai_rating, ai_reasoning,
            model_used, is_batch, expires_at,
        )
    
    return dict(row) if row else {"symbol": symbol.upper()}


async def delete_expired_analyses() -> int:
    """Delete expired AI analyses."""
    result = await execute(
        "DELETE FROM dip_ai_analysis WHERE expires_at < NOW()"
    )
    
    try:
        count = int(result.split()[-1])
        return count
    except (ValueError, IndexError):
        return 0
