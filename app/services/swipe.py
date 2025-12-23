"""Swipe service - combines dips with AI analysis and voting (PostgreSQL)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.core.logging import get_logger
from app.database.connection import fetch_all, fetch_one
from app.repositories import dip_votes as dip_votes_repo
from app.schemas.swipe import DipCard, DipStats, VoteCounts
from app.services.openai_client import generate_bio, rate_dip
from app.services import stock_info

logger = get_logger("swipe")


async def get_dip_card(symbol: str) -> Optional[DipCard]:
    """
    Get a complete dip card with AI analysis for a symbol.

    Returns:
        DipCard with symbol, prices, AI analysis, and vote counts.
        None if symbol not found in dip state.
    """
    # Get dip state and symbol info from PostgreSQL
    dip_row = await fetch_one(
        """
        SELECT ds.symbol, ds.current_price, ds.ath_price, ds.dip_percentage,
               ds.first_seen, ds.last_updated,
               s.name, s.sector, s.summary_ai
        FROM dip_state ds
        LEFT JOIN symbols s ON s.symbol = ds.symbol
        WHERE ds.symbol = $1
        """,
        symbol.upper(),
    )

    if not dip_row:
        return None

    # Get cached AI analysis
    ai_analysis = await dip_votes_repo.get_ai_analysis(symbol)

    # Get vote counts
    vote_counts_dict = await dip_votes_repo.get_vote_counts(symbol)
    vote_counts = VoteCounts(**vote_counts_dict)

    # Calculate days in dip from first_seen
    days_below = 0
    if dip_row["first_seen"]:
        first_seen = dip_row["first_seen"]
        if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
            first_seen = first_seen.replace(tzinfo=timezone.utc)
        days_below = (datetime.now(timezone.utc) - first_seen).days

    # Build card with Pydantic model
    card = DipCard(
        symbol=symbol.upper(),
        name=dip_row.get("name"),
        sector=dip_row.get("sector"),
        current_price=float(dip_row["current_price"]) if dip_row["current_price"] else 0,
        ref_high=float(dip_row["ath_price"]) if dip_row["ath_price"] else 0,
        dip_pct=float(dip_row["dip_percentage"]) if dip_row["dip_percentage"] else 0,
        days_below=days_below,
        vote_counts=vote_counts,
        summary_ai=dip_row.get("summary_ai"),
        swipe_bio=ai_analysis.get("swipe_bio") if ai_analysis else None,
        ai_rating=ai_analysis.get("ai_rating") if ai_analysis else None,
        ai_reasoning=ai_analysis.get("ai_reasoning") if ai_analysis else None,
    )

    return card


async def get_dip_card_with_fresh_ai(symbol: str, force_refresh: bool = False) -> Optional[DipCard]:
    """
    Get dip card with fresh AI analysis (generates if needed or forced).

    This fetches stock info and generates AI content if not cached.
    
    Args:
        symbol: Stock symbol
        force_refresh: If True, regenerate AI even if cached
        
    Returns:
        DipCard with AI content populated.
        None if symbol not found in dip state.
    """
    card = await get_dip_card(symbol)
    if not card:
        return None

    # If AI analysis already cached and not forcing refresh, return as-is
    if card.swipe_bio and not force_refresh:
        return card

    # Fetch stock info for AI generation
    info = await stock_info.get_stock_info_async(symbol)

    # Generate AI content
    bio = await generate_bio(
        symbol=symbol,
        name=info.get("name") if info else None,
        sector=info.get("sector") if info else None,
        summary=info.get("summary") if info else None,
        dip_pct=card.dip_pct,
    )

    rating_result = await rate_dip(
        symbol=symbol,
        current_price=card.current_price,
        ref_high=card.ref_high,
        dip_pct=card.dip_pct,
        name=info.get("name") if info else None,
        sector=info.get("sector") if info else None,
        summary=info.get("summary") if info else None,
    )

    # Cache the results
    if bio or rating_result:
        await dip_votes_repo.upsert_ai_analysis(
            symbol=symbol,
            swipe_bio=bio,
            ai_rating=rating_result.get("rating") if rating_result else None,
            ai_reasoning=rating_result.get("reasoning") if rating_result else None,
            is_batch=False,
        )

    # Update card with fresh data
    if info:
        card = card.model_copy(update={
            "name": info.get("name"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "ipo_year": info.get("ipo_year"),
        })
    
    card = card.model_copy(update={
        "swipe_bio": bio,
        "ai_rating": rating_result.get("rating") if rating_result else None,
        "ai_reasoning": rating_result.get("reasoning") if rating_result else None,
        "ai_confidence": rating_result.get("confidence") if rating_result else None,
    })

    return card


async def regenerate_ai_field(
    symbol: str,
    field: str,
) -> Optional[DipCard]:
    """
    Regenerate a swipe-specific AI field for a dip card.

    Args:
        symbol: Stock symbol
        field: Field to regenerate: 'rating' or 'bio'
               Note: 'summary' should use /symbols/{symbol}/ai/summary endpoint

    Returns:
        Updated DipCard with the regenerated field.
        None if symbol not found in dip state.
    """
    card = await get_dip_card(symbol)
    if not card:
        return None

    # Fetch stock info for AI generation
    info = await stock_info.get_stock_info_async(symbol)

    if field == "bio":
        # Regenerate Swipe bio
        bio = await generate_bio(
            symbol=symbol,
            name=info.get("name") if info else None,
            sector=info.get("sector") if info else None,
            summary=info.get("summary") if info else None,
            dip_pct=card.dip_pct,
        )
        if bio:
            await dip_votes_repo.upsert_ai_analysis(
                symbol=symbol,
                swipe_bio=bio,
                ai_rating=None,  # Don't update rating
                ai_reasoning=None,
                is_batch=False,
            )
            card = card.model_copy(update={"swipe_bio": bio})

    elif field == "rating":
        # Regenerate AI rating
        rating_result = await rate_dip(
            symbol=symbol,
            current_price=card.current_price,
            ref_high=card.ref_high,
            dip_pct=card.dip_pct,
            name=info.get("name") if info else None,
            sector=info.get("sector") if info else None,
            summary=info.get("summary") if info else None,
        )
        if rating_result:
            await dip_votes_repo.upsert_ai_analysis(
                symbol=symbol,
                swipe_bio=None,  # Don't update bio
                ai_rating=rating_result.get("rating"),
                ai_reasoning=rating_result.get("reasoning"),
                is_batch=False,
            )
            card = card.model_copy(update={
                "ai_rating": rating_result.get("rating"),
                "ai_reasoning": rating_result.get("reasoning"),
                "ai_confidence": rating_result.get("confidence"),
            })

    # Update card with stock info
    if info:
        card = card.model_copy(update={
            "name": info.get("name"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "website": info.get("website"),
            "ipo_year": info.get("ipo_year"),
        })

    return card


async def get_all_dip_cards(include_ai: bool = False) -> list[DipCard]:
    """
    Get all current dips as cards.

    Args:
        include_ai: If True, fetch fresh AI analysis for cards without it (slower)
        
    Returns:
        List of DipCard objects sorted by dip percentage.
    """
    # Get all dip states with symbol info
    dip_rows = await fetch_all(
        """
        SELECT ds.symbol, ds.current_price, ds.ath_price, ds.dip_percentage, 
               ds.first_seen, ds.last_updated,
               s.name, s.sector, s.summary_ai
        FROM dip_state ds
        JOIN symbols s ON s.symbol = ds.symbol
        ORDER BY ds.dip_percentage DESC
        """
    )

    # Get all vote counts at once
    all_vote_counts = await dip_votes_repo.get_all_vote_counts()

    # Get all AI analyses
    ai_rows = await fetch_all(
        """
        SELECT symbol, swipe_bio, ai_rating, rating_reasoning
        FROM dip_ai_analysis
        WHERE expires_at IS NULL OR expires_at > NOW()
        """
    )
    ai_by_symbol = {r["symbol"]: dict(r) for r in ai_rows}

    cards: list[DipCard] = []
    now = datetime.now(timezone.utc)
    for dip in dip_rows:
        symbol = dip["symbol"]

        # Calculate days in dip from first_seen
        days_below = 0
        if dip["first_seen"]:
            first_seen = dip["first_seen"]
            if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=timezone.utc)
            days_below = (now - first_seen).days

        # Get vote counts for this symbol
        vote_counts_dict = all_vote_counts.get(
            symbol,
            {"buy": 0, "sell": 0, "buy_weighted": 0, "sell_weighted": 0, "net_score": 0},
        )
        vote_counts = VoteCounts(**vote_counts_dict)

        # Get AI analysis if available
        ai = ai_by_symbol.get(symbol, {})

        card = DipCard(
            symbol=symbol,
            name=dip["name"],
            sector=dip["sector"],
            current_price=float(dip["current_price"]) if dip["current_price"] else 0,
            ref_high=float(dip["ath_price"]) if dip["ath_price"] else 0,
            dip_pct=float(dip["dip_percentage"]) if dip["dip_percentage"] else 0,
            days_below=days_below,
            vote_counts=vote_counts,
            summary_ai=dip.get("summary_ai"),
            swipe_bio=ai.get("swipe_bio"),
            ai_rating=ai.get("ai_rating"),
            ai_reasoning=ai.get("rating_reasoning"),
        )

        cards.append(card)

    return cards


async def vote_on_dip(
    symbol: str,
    voter_identifier: str,
    vote_type: str,
    vote_weight: int = 1,
    api_key_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """
    Record a vote on a dip.

    Args:
        symbol: Stock symbol
        voter_identifier: Hashed voter ID (fingerprint)
        vote_type: 'buy' or 'sell'
        vote_weight: Vote weight multiplier (default 1, API key users get 10)
        api_key_id: Optional API key ID if using authenticated voting

    Returns:
        Tuple of (success, error_message)
    """
    return await dip_votes_repo.add_vote(
        symbol=symbol,
        fingerprint=voter_identifier,
        vote_type=vote_type,
        vote_weight=vote_weight,
        api_key_id=api_key_id,
    )


async def get_vote_stats(symbol: str) -> DipStats:
    """Get voting statistics for a symbol."""
    counts_dict = await dip_votes_repo.get_vote_counts(symbol)
    vote_counts = VoteCounts(**counts_dict)

    total = counts_dict["buy"] + counts_dict["sell"]
    weighted_total = counts_dict["buy_weighted"] + counts_dict["sell_weighted"]

    return DipStats(
        symbol=symbol,
        vote_counts=vote_counts,
        total_votes=total,
        weighted_total=weighted_total,
        buy_pct=round(counts_dict["buy"] / total * 100, 1) if total > 0 else 0,
        sell_pct=round(counts_dict["sell"] / total * 100, 1) if total > 0 else 0,
        sentiment=_calculate_sentiment(counts_dict),
    )


def _calculate_sentiment(counts: dict) -> str:
    """Calculate overall sentiment from weighted vote counts."""
    buy = counts.get("buy_weighted", counts.get("buy", 0))
    sell = counts.get("sell_weighted", counts.get("sell", 0))

    if buy == 0 and sell == 0:
        return "neutral"

    ratio = buy / (buy + sell) if (buy + sell) > 0 else 0.5

    if ratio >= 0.7:
        return "very_bullish"
    elif ratio >= 0.55:
        return "bullish"
    elif ratio >= 0.45:
        return "neutral"
    elif ratio >= 0.3:
        return "bearish"
    else:
        return "very_bearish"
