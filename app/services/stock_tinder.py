"""Stock Tinder service - combines dips with AI analysis and voting (PostgreSQL)."""

from __future__ import annotations

import json
from typing import Optional, Any

from app.core.logging import get_logger
from app.database.connection import get_db, fetch_all, fetch_one
from app.repositories import dip_votes as dip_votes_repo
from app.services.openai_batch import generate_dip_bio_realtime, rate_dip_realtime
from app.services import stock_info

logger = get_logger("stock_tinder")


async def get_dip_card(symbol: str) -> Optional[dict[str, Any]]:
    """
    Get a complete dip card with AI analysis for a symbol.
    
    Returns dict with:
        - symbol, current_price, ath_price, dip_pct
        - tinder_bio (AI generated)
        - ai_rating, ai_reasoning (if available)
        - vote_counts (buy/sell with weighted totals)
    """
    # Get dip state from PostgreSQL
    dip_row = await fetch_one(
        """
        SELECT ds.symbol, ds.current_price, ds.ath_price, ds.dip_percentage,
               ds.first_seen, ds.last_updated
        FROM dip_state ds
        WHERE ds.symbol = $1
        """,
        symbol.upper(),
    )
    
    if not dip_row:
        return None
    
    # Get cached AI analysis
    ai_analysis = await dip_votes_repo.get_ai_analysis(symbol)
    
    # Get vote counts
    vote_counts = await dip_votes_repo.get_vote_counts(symbol)
    
    # Build base card
    card = {
        "symbol": symbol.upper(),
        "current_price": float(dip_row["current_price"]) if dip_row["current_price"] else 0,
        "ref_high": float(dip_row["ath_price"]) if dip_row["ath_price"] else 0,
        "dip_pct": float(dip_row["dip_percentage"]) if dip_row["dip_percentage"] else 0,
        "days_below": 0,  # Not tracked in new schema, could add if needed
        "vote_counts": vote_counts,
    }
    
    # Add AI analysis if cached
    if ai_analysis:
        card["tinder_bio"] = ai_analysis.get("tinder_bio")
        card["ai_rating"] = float(ai_analysis["ai_rating"]) if ai_analysis.get("ai_rating") else None
        card["ai_reasoning"] = ai_analysis.get("ai_reasoning")
    
    return card


async def get_dip_card_with_fresh_ai(symbol: str) -> Optional[dict[str, Any]]:
    """
    Get dip card with fresh AI analysis (generates if needed).
    
    This fetches stock info and generates AI content if not cached.
    """
    card = await get_dip_card(symbol)
    if not card:
        return None
    
    # If AI analysis already cached, return as-is
    if card.get("tinder_bio"):
        return card
    
    # Fetch stock info for AI generation
    info = await stock_info.get_stock_info_async(symbol)
    
    # Generate AI content
    bio = await generate_dip_bio_realtime(
        symbol=symbol,
        current_price=card["current_price"],
        ath_price=card["ref_high"],
        dip_percentage=card["dip_pct"],
    )
    
    rating_result = await rate_dip_realtime(
        symbol=symbol,
        current_price=card["current_price"],
        ath_price=card["ref_high"],
        dip_percentage=card["dip_pct"],
    )
    
    # Cache the results
    if bio or rating_result:
        await dip_votes_repo.upsert_ai_analysis(
            symbol=symbol,
            tinder_bio=bio,
            ai_rating=rating_result.get("rating") if rating_result else None,
            ai_reasoning=rating_result.get("reasoning") if rating_result else None,
            is_batch=False,
        )
    
    # Update card with fresh AI data
    if info:
        card["name"] = info.get("name")
        card["sector"] = info.get("sector")
        card["industry"] = info.get("industry")
    
    card["tinder_bio"] = bio
    if rating_result:
        card["ai_rating"] = rating_result.get("rating")
        card["ai_reasoning"] = rating_result.get("reasoning")
        card["ai_confidence"] = rating_result.get("confidence")
    
    return card


async def get_all_dip_cards(include_ai: bool = False) -> list[dict[str, Any]]:
    """
    Get all current dips as cards.
    
    Args:
        include_ai: If True, fetch fresh AI analysis for cards without it (slower)
    """
    # Get all dip states
    dip_rows = await fetch_all(
        """
        SELECT symbol, current_price, ath_price, dip_percentage, first_seen, last_updated
        FROM dip_state
        ORDER BY dip_percentage DESC
        """
    )
    
    # Get all vote counts at once
    all_vote_counts = await dip_votes_repo.get_all_vote_counts()
    
    # Get all AI analyses
    ai_rows = await fetch_all(
        """
        SELECT symbol, tinder_bio, ai_rating, rating_reasoning
        FROM dip_ai_analysis
        WHERE expires_at IS NULL OR expires_at > NOW()
        """
    )
    ai_by_symbol = {r["symbol"]: dict(r) for r in ai_rows}
    
    cards = []
    for dip in dip_rows:
        symbol = dip["symbol"]
        
        card = {
            "symbol": symbol,
            "current_price": float(dip["current_price"]) if dip["current_price"] else 0,
            "ref_high": float(dip["ath_price"]) if dip["ath_price"] else 0,
            "dip_pct": float(dip["dip_percentage"]) if dip["dip_percentage"] else 0,
            "days_below": 0,
            "vote_counts": all_vote_counts.get(symbol, {
                "buy": 0, "sell": 0, "buy_weighted": 0, "sell_weighted": 0, "net_score": 0
            }),
        }
        
        # Add cached AI analysis
        if symbol in ai_by_symbol:
            ai = ai_by_symbol[symbol]
            card["tinder_bio"] = ai.get("tinder_bio")
            card["ai_rating"] = float(ai["ai_rating"]) if ai.get("ai_rating") else None
            card["ai_reasoning"] = ai.get("rating_reasoning")
        
        if include_ai and not card.get("tinder_bio"):
            # Get cached AI or generate
            full_card = await get_dip_card_with_fresh_ai(symbol)
            if full_card:
                card = full_card
        
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


async def get_vote_stats(symbol: str) -> dict[str, Any]:
    """Get voting statistics for a symbol."""
    counts = await dip_votes_repo.get_vote_counts(symbol)
    
    total = counts["buy"] + counts["sell"]
    weighted_total = counts["buy_weighted"] + counts["sell_weighted"]
    
    return {
        "symbol": symbol,
        "vote_counts": counts,
        "total_votes": total,
        "weighted_total": weighted_total,
        "buy_pct": round(counts["buy"] / total * 100, 1) if total > 0 else 0,
        "sell_pct": round(counts["sell"] / total * 100, 1) if total > 0 else 0,
        "sentiment": _calculate_sentiment(counts),
    }


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
