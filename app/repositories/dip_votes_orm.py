"""Dip voting repository using SQLAlchemy ORM.

Usage (recommended - auto session management):
    from app.repositories import dip_votes_orm as dip_votes
    
    # Simple API - session managed automatically
    success, error = await dip_votes.add_vote("AAPL", fingerprint, "buy")
    counts = await dip_votes.get_vote_counts("AAPL")

Usage (advanced - manual session control):
    from app.repositories import dip_votes_orm as dip_votes
    from app.database.connection import get_session
    
    async with get_session() as session:
        success, error = await dip_votes.add_vote_with_session(session, "AAPL", fingerprint, "buy")
        await session.commit()
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import DipAIAnalysis, DipVote, Symbol


logger = get_logger("repositories.dip_votes_orm")


# =============================================================================
# SESSION-BASED FUNCTIONS (for advanced use with manual session control)
# =============================================================================

async def get_vote_cooldown_remaining_with_session(
    session: AsyncSession,
    symbol: str,
    fingerprint: str,
) -> int | None:
    """
    Check if user is on cooldown for voting on this symbol.

    Returns:
        Seconds remaining on cooldown, or None if can vote
    """
    cooldown_days = settings.vote_cooldown_days
    cutoff = datetime.now(UTC) - timedelta(days=cooldown_days)

    result = await session.execute(
        select(DipVote.created_at)
        .where(
            and_(
                DipVote.symbol == symbol.upper(),
                DipVote.fingerprint == fingerprint,
                DipVote.created_at > cutoff,
            )
        )
        .order_by(DipVote.created_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()

    if not row:
        return None

    cooldown_end = row + timedelta(days=cooldown_days)
    remaining = (cooldown_end - datetime.now(UTC)).total_seconds()

    return int(remaining) if remaining > 0 else None


async def add_vote_with_session(
    session: AsyncSession,
    symbol: str,
    fingerprint: str,
    vote_type: str,
    vote_weight: int = 1,
    api_key_id: int | None = None,
    skip_cooldown: bool = False,
) -> tuple[bool, str | None]:
    """
    Add a vote for a dip.

    Returns:
        Tuple of (success, error_message)
    """
    if vote_type not in ("buy", "sell"):
        return False, "Invalid vote type. Must be 'buy' or 'sell'"

    # Check cooldown (unless skipped for admin)
    if not skip_cooldown:
        cooldown = await get_vote_cooldown_remaining_with_session(session, symbol, fingerprint)
        if cooldown:
            hours = cooldown // 3600
            minutes = (cooldown % 3600) // 60
            return False, f"Vote cooldown active. Try again in {hours}h {minutes}m"

    # Check symbol is tracked
    result = await session.execute(
        select(Symbol.symbol)
        .where(and_(Symbol.symbol == symbol.upper(), Symbol.is_active == True))
    )
    if not result.scalar_one_or_none():
        return False, f"Symbol {symbol.upper()} is not being tracked"

    # Upsert vote using PostgreSQL INSERT ... ON CONFLICT
    stmt = insert(DipVote).values(
        symbol=symbol.upper(),
        fingerprint=fingerprint,
        vote_type=vote_type,
        vote_weight=vote_weight,
        api_key_id=api_key_id,
        created_at=datetime.now(UTC),
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_dip_vote",
        set_={
            "vote_type": stmt.excluded.vote_type,
            "vote_weight": stmt.excluded.vote_weight,
            "api_key_id": stmt.excluded.api_key_id,
            "created_at": stmt.excluded.created_at,
        },
    )
    await session.execute(stmt)

    logger.info(f"Vote recorded: {symbol.upper()} {vote_type} (weight: {vote_weight})")
    return True, None


async def get_vote_counts_with_session(session: AsyncSession, symbol: str) -> dict:
    """Get vote counts for a symbol (with weighted totals)."""
    result = await session.execute(
        select(
            func.count().filter(DipVote.vote_type == "buy").label("buy_count"),
            func.count().filter(DipVote.vote_type == "sell").label("sell_count"),
            func.coalesce(
                func.sum(DipVote.vote_weight).filter(DipVote.vote_type == "buy"), 0
            ).label("buy_weighted"),
            func.coalesce(
                func.sum(DipVote.vote_weight).filter(DipVote.vote_type == "sell"), 0
            ).label("sell_weighted"),
        ).where(DipVote.symbol == symbol.upper())
    )
    row = result.one_or_none()

    if not row:
        return {
            "buy": 0,
            "sell": 0,
            "buy_weighted": 0,
            "sell_weighted": 0,
            "net_score": 0,
        }

    return {
        "buy": row.buy_count or 0,
        "sell": row.sell_count or 0,
        "buy_weighted": row.buy_weighted or 0,
        "sell_weighted": row.sell_weighted or 0,
        "net_score": (row.buy_weighted or 0) - (row.sell_weighted or 0),
    }


async def get_all_vote_counts_with_session(session: AsyncSession) -> dict[str, dict]:
    """Get vote counts for all symbols with votes."""
    result = await session.execute(
        select(
            DipVote.symbol,
            func.count().filter(DipVote.vote_type == "buy").label("buy_count"),
            func.count().filter(DipVote.vote_type == "sell").label("sell_count"),
            func.coalesce(
                func.sum(DipVote.vote_weight).filter(DipVote.vote_type == "buy"), 0
            ).label("buy_weighted"),
            func.coalesce(
                func.sum(DipVote.vote_weight).filter(DipVote.vote_type == "sell"), 0
            ).label("sell_weighted"),
        ).group_by(DipVote.symbol)
    )

    counts = {}
    for row in result.all():
        counts[row.symbol] = {
            "buy": row.buy_count or 0,
            "sell": row.sell_count or 0,
            "buy_weighted": row.buy_weighted or 0,
            "sell_weighted": row.sell_weighted or 0,
            "net_score": (row.buy_weighted or 0) - (row.sell_weighted or 0),
        }

    return counts


async def get_user_votes_with_session(
    session: AsyncSession,
    fingerprint: str,
    limit: int = 50,
) -> list[dict]:
    """Get recent votes by a user."""
    result = await session.execute(
        select(DipVote)
        .where(DipVote.fingerprint == fingerprint)
        .order_by(DipVote.created_at.desc())
        .limit(limit)
    )
    votes = result.scalars().all()
    return [
        {
            "id": v.id,
            "symbol": v.symbol,
            "fingerprint": v.fingerprint,
            "vote_type": v.vote_type,
            "vote_weight": v.vote_weight,
            "created_at": v.created_at,
        }
        for v in votes
    ]


async def get_user_vote_for_symbol_with_session(
    session: AsyncSession,
    symbol: str,
    fingerprint: str,
) -> dict | None:
    """Get user's existing vote for a symbol within cooldown period."""
    cooldown_days = settings.vote_cooldown_days
    cutoff = datetime.now(UTC) - timedelta(days=cooldown_days)

    result = await session.execute(
        select(DipVote)
        .where(
            and_(
                DipVote.symbol == symbol.upper(),
                DipVote.fingerprint == fingerprint,
                DipVote.created_at > cutoff,
            )
        )
        .order_by(DipVote.created_at.desc())
        .limit(1)
    )
    vote = result.scalar_one_or_none()
    if vote:
        return {
            "id": vote.id,
            "symbol": vote.symbol,
            "fingerprint": vote.fingerprint,
            "vote_type": vote.vote_type,
            "vote_weight": vote.vote_weight,
            "created_at": vote.created_at,
        }
    return None


async def get_ai_analysis_with_session(
    session: AsyncSession,
    symbol: str
) -> dict | None:
    """Get cached AI analysis for a symbol."""
    result = await session.execute(
        select(DipAIAnalysis)
        .where(
            and_(
                DipAIAnalysis.symbol == symbol.upper(),
                or_(
                    DipAIAnalysis.expires_at.is_(None),
                    DipAIAnalysis.expires_at > datetime.now(UTC),
                ),
            )
        )
    )
    analysis = result.scalar_one_or_none()
    if analysis:
        return {
            "symbol": analysis.symbol,
            "swipe_bio": analysis.swipe_bio,
            "ai_rating": analysis.ai_rating,
            "ai_reasoning": analysis.rating_reasoning,
            "model_used": analysis.model_used,
            "is_batch_generated": analysis.is_batch_generated,
            "generated_at": analysis.generated_at,
            "expires_at": analysis.expires_at,
        }
    return None


async def upsert_ai_analysis_with_session(
    session: AsyncSession,
    symbol: str,
    swipe_bio: str | None = None,
    ai_rating: str | None = None,
    ai_reasoning: str | None = None,
    model_used: str = "gpt-5-mini",
    is_batch: bool = False,
    expires_hours: int = 168,  # 7 days
) -> dict:
    """Create or update AI analysis for a symbol."""
    expires_at = datetime.now(UTC) + timedelta(hours=expires_hours)
    now = datetime.now(UTC)

    stmt = insert(DipAIAnalysis).values(
        symbol=symbol.upper(),
        swipe_bio=swipe_bio,
        ai_rating=ai_rating,
        rating_reasoning=ai_reasoning,
        model_used=model_used,
        is_batch_generated=is_batch,
        generated_at=now,
        expires_at=expires_at,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol"],
        set_={
            "swipe_bio": stmt.excluded.swipe_bio,
            "ai_rating": stmt.excluded.ai_rating,
            "rating_reasoning": stmt.excluded.rating_reasoning,
            "model_used": stmt.excluded.model_used,
            "is_batch_generated": stmt.excluded.is_batch_generated,
            "generated_at": now,
            "expires_at": stmt.excluded.expires_at,
        },
    )
    await session.execute(stmt)

    return {
        "symbol": symbol.upper(),
        "swipe_bio": swipe_bio,
        "ai_rating": ai_rating,
        "ai_reasoning": ai_reasoning,
        "model_used": model_used,
        "is_batch_generated": is_batch,
        "generated_at": now,
        "expires_at": expires_at,
    }


# =============================================================================
# PUBLIC API - AUTO-MANAGED SESSIONS (drop-in replacement for legacy module)
# =============================================================================

async def get_vote_cooldown_remaining(symbol: str, fingerprint: str) -> int | None:
    """Check cooldown for voting on this symbol."""
    async with get_session() as session:
        return await get_vote_cooldown_remaining_with_session(session, symbol, fingerprint)


async def add_vote(
    symbol: str,
    fingerprint: str,
    vote_type: str,
    vote_weight: int = 1,
    api_key_id: int | None = None,
    skip_cooldown: bool = False,
) -> tuple[bool, str | None]:
    """Add a vote for a dip."""
    async with get_session() as session:
        result = await add_vote_with_session(
            session, symbol, fingerprint, vote_type, vote_weight, api_key_id, skip_cooldown
        )
        await session.commit()
        return result


async def get_vote_counts(symbol: str) -> dict:
    """Get vote counts for a symbol."""
    async with get_session() as session:
        return await get_vote_counts_with_session(session, symbol)


async def get_all_vote_counts() -> dict[str, dict]:
    """Get vote counts for all symbols with votes."""
    async with get_session() as session:
        return await get_all_vote_counts_with_session(session)


async def get_user_votes(fingerprint: str, limit: int = 50) -> list[dict]:
    """Get recent votes by a user."""
    async with get_session() as session:
        return await get_user_votes_with_session(session, fingerprint, limit)


async def get_user_vote_for_symbol(symbol: str, fingerprint: str) -> dict | None:
    """Get user's existing vote for a symbol within cooldown period."""
    async with get_session() as session:
        return await get_user_vote_for_symbol_with_session(session, symbol, fingerprint)


async def get_ai_analysis(symbol: str) -> dict | None:
    """Get cached AI analysis for a symbol."""
    async with get_session() as session:
        return await get_ai_analysis_with_session(session, symbol)


async def upsert_ai_analysis(
    symbol: str,
    swipe_bio: str | None = None,
    ai_rating: str | None = None,
    ai_reasoning: str | None = None,
    model_used: str = "gpt-5-mini",
    is_batch: bool = False,
    expires_hours: int = 168,
) -> dict:
    """Create or update AI analysis for a symbol."""
    async with get_session() as session:
        result = await upsert_ai_analysis_with_session(
            session, symbol, swipe_bio, ai_rating, ai_reasoning, model_used, is_batch, expires_hours
        )
        await session.commit()
        return result
