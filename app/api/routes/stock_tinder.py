"""Stock Tinder routes - dip voting and AI analysis."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request, Header

from app.core.exceptions import NotFoundError, ValidationError
from app.core.fingerprint import get_vote_identifier
from app.repositories import user_api_keys
from app.schemas.stock_tinder import (
    DipCard,
    DipCardList,
    DipVoteRequest,
    DipVoteResponse,
    DipStats,
    VoteCounts,
)
from app.services import stock_tinder

router = APIRouter()


@router.get(
    "/cards",
    response_model=DipCardList,
    summary="Get all dip cards",
    description="Get all current dips as tinder-style cards.",
)
async def get_dip_cards(
    include_ai: bool = Query(
        False, description="Fetch fresh AI analysis for cards without it (slower)"
    ),
) -> DipCardList:
    """Get all current dips as swipeable cards."""
    cards = await stock_tinder.get_all_dip_cards(include_ai=include_ai)

    return DipCardList(
        cards=[
            DipCard(
                symbol=c["symbol"],
                name=c.get("name"),
                sector=c.get("sector"),
                industry=c.get("industry"),
                current_price=c["current_price"],
                ref_high=c["ref_high"],
                dip_pct=c["dip_pct"],
                days_below=c["days_below"],
                min_dip_pct=c.get("min_dip_pct"),
                tinder_bio=c.get("tinder_bio"),
                ai_rating=c.get("ai_rating"),
                ai_reasoning=c.get("ai_reasoning"),
                ai_confidence=c.get("ai_confidence"),
                vote_counts=VoteCounts(**c["vote_counts"]),
            )
            for c in cards
        ],
        total=len(cards),
    )


@router.get(
    "/cards/{symbol}",
    response_model=DipCard,
    summary="Get single dip card",
    description="Get a specific dip card with AI analysis.",
)
async def get_dip_card(
    symbol: str,
    refresh_ai: bool = Query(False, description="Force refresh AI analysis"),
) -> DipCard:
    """Get a single dip card with full details."""
    if refresh_ai:
        card = await stock_tinder.get_dip_card_with_fresh_ai(symbol.upper())
    else:
        card = await stock_tinder.get_dip_card(symbol.upper())

    if not card:
        raise NotFoundError(f"Symbol {symbol.upper()} is not currently in a dip")

    return DipCard(
        symbol=card["symbol"],
        name=card.get("name"),
        sector=card.get("sector"),
        industry=card.get("industry"),
        current_price=card["current_price"],
        ref_high=card["ref_high"],
        dip_pct=card["dip_pct"],
        days_below=card["days_below"],
        min_dip_pct=card.get("min_dip_pct"),
        tinder_bio=card.get("tinder_bio"),
        ai_rating=card.get("ai_rating"),
        ai_reasoning=card.get("ai_reasoning"),
        ai_confidence=card.get("ai_confidence"),
        vote_counts=VoteCounts(**card["vote_counts"]),
    )


@router.put(
    "/cards/{symbol}/vote",
    response_model=DipVoteResponse,
    summary="Vote on a dip",
    description="Submit a buy/sell vote for a stock dip. API key holders get 10x vote weight.",
)
async def vote_on_dip(
    request: Request,
    symbol: str,
    payload: DipVoteRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> DipVoteResponse:
    """
    Vote on a stock dip.

    Uses PUT since voting is idempotent within the cooldown period.

    Vote cooldown: Each user can only vote once per stock every 7 days.
    Users are identified by a fingerprint combining IP + browser headers.

    API key holders get 10x vote weight. Use X-API-Key header for authenticated voting.
    """
    # Get voter identifier
    vote_id = get_vote_identifier(request, symbol.upper())

    # Check for API key and get vote weight
    vote_weight = 1
    api_key_id = None

    if x_api_key:
        key_data = await user_api_keys.validate_api_key(x_api_key)
        if key_data:
            vote_weight = key_data.get("vote_weight", 10)
            api_key_id = key_data.get("id")

    success, error = await stock_tinder.vote_on_dip(
        symbol=symbol.upper(),
        voter_identifier=vote_id,
        vote_type=payload.vote_type,
        vote_weight=vote_weight,
        api_key_id=api_key_id,
    )

    if not success:
        raise ValidationError(error or "Failed to record vote")

    weight_msg = f" (weight: {vote_weight}x)" if vote_weight > 1 else ""

    return DipVoteResponse(
        symbol=symbol.upper(),
        vote_type=payload.vote_type,
        message=f"Your '{payload.vote_type}' vote has been recorded{weight_msg}",
    )


@router.get(
    "/cards/{symbol}/stats",
    response_model=DipStats,
    summary="Get voting stats",
    description="Get detailed voting statistics for a dip.",
)
async def get_dip_stats(symbol: str) -> DipStats:
    """Get voting statistics for a dip."""
    stats = await stock_tinder.get_vote_stats(symbol.upper())

    return DipStats(
        symbol=stats["symbol"],
        vote_counts=VoteCounts(**stats["vote_counts"]),
        total_votes=stats["total_votes"],
        weighted_total=stats.get("weighted_total", 0),
        buy_pct=stats["buy_pct"],
        sell_pct=stats["sell_pct"],
        sentiment=stats["sentiment"],
    )


@router.post(
    "/cards/{symbol}/refresh-ai",
    response_model=DipCard,
    summary="Refresh AI analysis",
    description="Force refresh AI-generated content for a dip.",
)
async def refresh_ai_analysis(symbol: str) -> DipCard:
    """Force refresh AI analysis for a dip."""
    card = await stock_tinder.get_dip_card_with_fresh_ai(symbol.upper())

    if not card:
        raise NotFoundError(f"Symbol {symbol.upper()} is not currently in a dip")

    return DipCard(
        symbol=card["symbol"],
        name=card.get("name"),
        sector=card.get("sector"),
        industry=card.get("industry"),
        current_price=card["current_price"],
        ref_high=card["ref_high"],
        dip_pct=card["dip_pct"],
        days_below=card["days_below"],
        min_dip_pct=card.get("min_dip_pct"),
        tinder_bio=card.get("tinder_bio"),
        ai_rating=card.get("ai_rating"),
        ai_reasoning=card.get("ai_reasoning"),
        ai_confidence=card.get("ai_confidence"),
        vote_counts=VoteCounts(**card["vote_counts"]),
    )
