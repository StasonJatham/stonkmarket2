"""Stock Tinder service - combines dips with AI analysis and voting."""

from __future__ import annotations

import json
from typing import Optional, Any

from app.core.fingerprint import get_vote_identifier
from app.core.logging import get_logger
from app.database.connection import get_db
from app.repositories import dips as dips_repo
from app.repositories import dip_votes as dip_votes_repo
from app.repositories import symbols as symbols_repo
from app.services import openai_service
from app.services import stock_info

logger = get_logger("stock_tinder")


async def get_dip_card(symbol: str) -> Optional[dict[str, Any]]:
    """
    Get a complete dip card with AI analysis for a symbol.
    
    Returns dict with:
        - symbol, name, sector, industry, summary
        - current_price, ref_high, dip_pct, days_below
        - tinder_bio (AI generated)
        - ai_rating, ai_reasoning (if available)
        - vote_counts (buy/sell/skip)
    """
    with get_db() as conn:
        # Get dip state
        dip_state = dips_repo.get_dip_state(conn, symbol)
        if not dip_state:
            return None
        
        # Get symbol config
        symbol_config = symbols_repo.get_symbol(conn, symbol)
        
        # Get cached AI analysis
        ai_analysis = dip_votes_repo.get_ai_analysis(conn, symbol)
        
        # Get vote counts
        vote_counts = dip_votes_repo.get_vote_counts(conn, symbol)
    
    # Calculate dip percentage
    dip_pct = ((dip_state.ref_high - dip_state.last_price) / dip_state.ref_high) * 100 if dip_state.ref_high > 0 else 0
    
    # Build base card
    card = {
        "symbol": symbol,
        "current_price": dip_state.last_price,
        "ref_high": dip_state.ref_high,
        "dip_pct": round(dip_pct, 2),
        "days_below": dip_state.days_below,
        "min_dip_pct": symbol_config.min_dip_pct if symbol_config else 10,
        "vote_counts": vote_counts,
    }
    
    # Add AI analysis if cached
    if ai_analysis:
        card["tinder_bio"] = ai_analysis.tinder_bio
        card["ai_rating"] = ai_analysis.ai_rating
        card["ai_reasoning"] = ai_analysis.ai_reasoning
    
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
    
    dip_data = {
        "name": info.get("name") if info else None,
        "sector": info.get("sector") if info else None,
        "industry": info.get("industry") if info else None,
        "summary": info.get("summary") if info else None,
        "current_price": card["current_price"],
        "ref_high": card["ref_high"],
        "dip_pct": card["dip_pct"],
        "days_below": card["days_below"],
    }
    
    # Add fundamentals if available
    if info:
        dip_data["fundamentals"] = {
            "pe_ratio": info.get("pe_ratio"),
            "market_cap": info.get("market_cap"),
            "dividend_yield": info.get("dividend_yield"),
            "52_week_change": info.get("52_week_change"),
        }
    
    # Generate AI content
    bio = await openai_service.generate_tinder_bio_for_dip(symbol, dip_data)
    rating_result = await openai_service.rate_dip(symbol, dip_data)
    
    # Cache the results
    if bio or rating_result:
        with get_db() as conn:
            dip_votes_repo.upsert_ai_analysis(
                conn,
                symbol,
                tinder_bio=bio,
                ai_rating=rating_result.get("rating") if rating_result else None,
                ai_reasoning=rating_result.get("reasoning") if rating_result else None,
                analysis_data=json.dumps(dip_data),
                expires_hours=24,
            )
    
    # Update card with fresh AI data
    card["name"] = dip_data.get("name")
    card["sector"] = dip_data.get("sector")
    card["industry"] = dip_data.get("industry")
    card["summary"] = dip_data.get("summary")
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
    with get_db() as conn:
        dip_states = dips_repo.get_all_dip_states(conn)
        all_vote_counts = dip_votes_repo.get_all_vote_counts(conn)
    
    cards = []
    for dip in dip_states:
        dip_pct = ((dip.ref_high - dip.last_price) / dip.ref_high) * 100 if dip.ref_high > 0 else 0
        
        card = {
            "symbol": dip.symbol,
            "current_price": dip.last_price,
            "ref_high": dip.ref_high,
            "dip_pct": round(dip_pct, 2),
            "days_below": dip.days_below,
            "vote_counts": all_vote_counts.get(dip.symbol, {"buy": 0, "sell": 0, "skip": 0}),
        }
        
        if include_ai:
            # Get cached AI or generate
            full_card = await get_dip_card_with_fresh_ai(dip.symbol)
            if full_card:
                card = full_card
        
        cards.append(card)
    
    # Sort by dip percentage (biggest dips first)
    cards.sort(key=lambda x: x["dip_pct"], reverse=True)
    
    return cards


def vote_on_dip(
    symbol: str,
    voter_identifier: str,
    vote_type: str,
) -> tuple[bool, Optional[str]]:
    """
    Record a vote on a dip.
    
    Args:
        symbol: Stock symbol
        voter_identifier: Hashed voter ID
        vote_type: 'buy', 'sell', or 'skip'
        
    Returns:
        Tuple of (success, error_message)
    """
    with get_db() as conn:
        return dip_votes_repo.add_vote(conn, symbol, voter_identifier, vote_type)


def get_vote_stats(symbol: str) -> dict[str, Any]:
    """Get voting statistics for a symbol."""
    with get_db() as conn:
        counts = dip_votes_repo.get_vote_counts(conn, symbol)
    
    total = counts["buy"] + counts["sell"] + counts["skip"]
    
    return {
        "symbol": symbol,
        "vote_counts": counts,
        "total_votes": total,
        "buy_pct": round(counts["buy"] / total * 100, 1) if total > 0 else 0,
        "sell_pct": round(counts["sell"] / total * 100, 1) if total > 0 else 0,
        "skip_pct": round(counts["skip"] / total * 100, 1) if total > 0 else 0,
        "sentiment": _calculate_sentiment(counts),
    }


def _calculate_sentiment(counts: dict[str, int]) -> str:
    """Calculate overall sentiment from vote counts."""
    buy = counts.get("buy", 0)
    sell = counts.get("sell", 0)
    
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
