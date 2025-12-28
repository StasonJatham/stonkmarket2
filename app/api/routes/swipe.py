"""Swipe routes - dip voting and AI analysis."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Header, Query, Request

from app.cache.cache import Cache
from app.celery_app import celery_app
from app.core.client_identity import (
    check_vote_allowed,
    get_vote_identifier,
    record_vote,
)
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import decode_access_token
from app.repositories import dip_votes_orm as dip_votes_repo
from app.repositories import user_api_keys_orm as user_api_keys
from app.schemas.swipe import (
    DipCard,
    DipCardList,
    DipStats,
    DipVoteRequest,
    DipVoteResponse,
)
from app.services import swipe


router = APIRouter()

_ai_refresh_cache = Cache(prefix="swipe_ai_refresh", default_ttl=900)


async def _enqueue_swipe_ai_refresh(symbol: str, field: str | None = None) -> str:
    """Queue swipe AI refresh with lightweight dedupe."""
    key = f"{symbol}:{field or 'all'}"
    cached = await _ai_refresh_cache.get(key)
    if cached and isinstance(cached, dict) and cached.get("task_id"):
        return cached["task_id"]

    task = celery_app.send_task(
        "jobs.refresh_dip_ai",
        args=[symbol.upper()],
        kwargs={"field": field},
    )
    await _ai_refresh_cache.set(key, {"task_id": task.id})
    return task.id


@router.get(
    "/cards",
    response_model=DipCardList,
    summary="Get all dip cards",
    description="Get all current dips as swipe-style cards.",
)
async def get_dip_cards(
    request: Request,
    include_ai: bool = Query(
        False, description="Queue AI refresh for cards missing analysis (async)"
    ),
    exclude_voted: bool = Query(
        False, description="Exclude cards the user has already voted on"
    ),
) -> DipCardList:
    """Get all current dips as swipeable cards."""
    cards = await swipe.get_all_dip_cards(include_ai=False)

    if include_ai:
        refreshed_cards = []
        for card in cards:
            if card.swipe_bio is None or card.ai_rating is None:
                task_id = await _enqueue_swipe_ai_refresh(card.symbol)
                card = card.model_copy(
                    update={"ai_pending": True, "ai_task_id": task_id}
                )
            refreshed_cards.append(card)
        cards = refreshed_cards

    # If exclude_voted, filter out cards the user has already voted on
    if exclude_voted:
        # Compute vote_id for each symbol and check which have votes
        voted_symbols = set()
        for c in cards:
            vote_id = get_vote_identifier(request, c.symbol)
            vote = await dip_votes_repo.get_user_vote_for_symbol(c.symbol, vote_id)
            if vote:
                voted_symbols.add(c.symbol)
        cards = [c for c in cards if c.symbol not in voted_symbols]

    return DipCardList(
        cards=cards,
        total=len(cards),
    )


@router.get(
    "/cards/{symbol}",
    response_model=DipCard,
    summary="Get single dip card",
    description="Get a specific dip card with cached AI analysis.",
)
async def get_dip_card(
    symbol: str,
    refresh_ai: bool = Query(False, description="Force refresh AI analysis"),
) -> DipCard:
    """Get a single dip card with full details."""
    card = await swipe.get_dip_card(symbol.upper())

    if not card:
        raise NotFoundError(f"Symbol {symbol.upper()} is not currently in a dip")

    if refresh_ai:
        task_id = await _enqueue_swipe_ai_refresh(symbol.upper())
        card = card.model_copy(update={"ai_pending": True, "ai_task_id": task_id})

    return card


@router.put(
    "/cards/{symbol}/vote",
    response_model=DipVoteResponse,
    summary="Vote on a dip",
    description="Submit a buy/sell vote for a stock dip. API key holders get 10x vote weight. Admins bypass cooldown.",
)
async def vote_on_dip(
    request: Request,
    symbol: str,
    payload: DipVoteRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> DipVoteResponse:
    """
    Vote on a stock dip.

    Uses PUT since voting is idempotent within the cooldown period.

    Vote cooldown: Each user can only vote once per stock every 7 days.
    Users are identified by a fingerprint combining IP + browser headers.
    
    Admins bypass the cooldown entirely.
    API key holders get 10x vote weight. Use X-API-Key header for authenticated voting.
    """
    symbol = symbol.upper()

    # Check if admin - bypass cooldown for admins
    is_admin = False
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            token_data = decode_access_token(token)
            if token_data and token_data.is_admin:
                is_admin = True
        except Exception:
            pass

    # === Check if vote is allowed (cooldown + risk assessment) ===
    check = None
    if not is_admin:
        check = await check_vote_allowed(request, symbol)

        if not check.allowed:
            raise ValidationError(check.reason, details={"retry_after": 3600})

    # Get voter identifier
    vote_id = get_vote_identifier(request, symbol)

    # Check for API key and get vote weight
    vote_weight = 1
    api_key_id = None

    if x_api_key:
        key_data = await user_api_keys.validate_api_key(x_api_key)
        if key_data:
            vote_weight = key_data.get("vote_weight", 10)
            api_key_id = key_data.get("id")

    # Reduce vote weight for high-risk votes (soft penalty) - skip for admin
    if check and check.reduce_weight:
        vote_weight = max(1, vote_weight // 2)

    success, error = await swipe.vote_on_dip(
        symbol=symbol,
        voter_identifier=vote_id,
        vote_type=payload.vote_type,
        vote_weight=vote_weight,
        api_key_id=api_key_id,
        skip_cooldown=is_admin,
    )

    if not success:
        raise ValidationError(error or "Failed to record vote")

    # Record successful vote for tracking
    await record_vote(request, symbol)

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
    return await swipe.get_vote_stats(symbol.upper())


@router.post(
    "/cards/{symbol}/refresh-ai",
    response_model=DipCard,
    summary="Refresh AI analysis",
    description="Queue refresh of AI-generated content for a dip.",
)
async def refresh_ai_analysis(symbol: str) -> DipCard:
    """Force refresh AI analysis for a dip."""
    card = await swipe.get_dip_card(symbol.upper())

    if not card:
        raise NotFoundError(f"Symbol {symbol.upper()} is not currently in a dip")

    task_id = await _enqueue_swipe_ai_refresh(symbol.upper())
    return card.model_copy(update={"ai_pending": True, "ai_task_id": task_id})


@router.post(
    "/cards/{symbol}/refresh-ai/{field}",
    response_model=DipCard,
    summary="Refresh specific AI field",
    description="Queue regeneration for a swipe AI field: bio only. For summary, use /symbols/{symbol}/ai/summary.",
)
async def refresh_ai_field(
    symbol: str,
    field: Literal["bio"],
) -> DipCard:
    """Regenerate a swipe-specific AI field for a dip (bio only)."""
    card = await swipe.get_dip_card(symbol.upper())

    if not card:
        raise NotFoundError(f"Symbol {symbol.upper()} is not currently in a dip")

    task_id = await _enqueue_swipe_ai_refresh(symbol.upper(), field)
    return card.model_copy(update={"ai_pending": True, "ai_task_id": task_id})
