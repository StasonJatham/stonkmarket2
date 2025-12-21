"""Stock suggestion API routes with REST conventions and fingerprint-based deduplication."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from app.api.dependencies import (
    get_db,
    require_admin,
    get_request_fingerprint,
    get_vote_identifier,
)
from app.core.exceptions import ValidationError, NotFoundError
from app.core.logging import get_logger
from app.repositories import suggestions as suggestions_repo
from app.schemas.suggestions import (
    SuggestionCreate,
    SuggestionResponse,
    SuggestionListResponse,
    SuggestionApprove,
    SuggestionReject,
    TopSuggestion,
    SuggestionStatus,
)
from app.services import suggestion_service

logger = get_logger("api.routes.suggestions")

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


# =============================================================================
# Public endpoints (fingerprint-based vote deduplication with 7-day cooldown)
# =============================================================================

@router.post("", response_model=dict, status_code=201)
async def suggest_stock(
    request: Request,
    data: SuggestionCreate,
):
    """
    Suggest a new stock to be tracked.
    
    This is a public endpoint. Users can suggest stocks using Yahoo Finance
    symbol format:
    - US stocks: AAPL, MSFT, GOOGL
    - International: 7203.T (Japan), BHP.AX (Australia)
    - Crypto: BTC-USD, ETH-USD
    - Indices: ^GSPC (S&P 500)
    
    If the stock was already suggested, this adds a vote instead.
    Vote cooldown: 7 days per stock per user (based on fingerprint).
    """
    # Use fingerprint for vote deduplication
    voter_fingerprint = get_request_fingerprint(request)
    
    success, error, result = suggestion_service.suggest_stock(
        symbol=data.symbol,
        voter_identifier=voter_fingerprint,
    )
    
    if not success:
        raise ValidationError(error)
    
    return {
        "message": "Stock suggested successfully" if result["is_new"] else "Vote added successfully",
        "symbol": result["symbol"],
        "vote_count": result["vote_count"],
        "status": result["status"],
    }


@router.put("/{symbol}/vote", response_model=dict)
async def vote_for_suggestion(
    request: Request,
    symbol: str,
):
    """
    Vote for an existing stock suggestion.
    
    Uses PUT since voting is idempotent - voting twice has no additional effect.
    
    Vote cooldown: Each user can only vote once per stock every 7 days.
    Users are identified by a fingerprint combining IP + browser headers.
    
    If auto-approve is enabled and the suggestion meets all criteria
    (vote count, unique voters, age), it will be automatically approved.
    """
    # Create a vote-specific identifier that includes the symbol
    vote_id = get_vote_identifier(request, symbol)
    
    success, error, was_auto_approved = await suggestion_service.vote_for_suggestion(
        symbol=symbol.upper(),
        voter_identifier=vote_id,
    )
    
    if not success:
        raise ValidationError(error)
    
    if was_auto_approved:
        return {
            "message": "Vote recorded - suggestion auto-approved!",
            "symbol": symbol.upper(),
            "auto_approved": True,
        }
    
    return {"message": "Vote recorded", "symbol": symbol.upper()}


@router.get("/top", response_model=list[TopSuggestion])
async def get_top_suggestions(
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get top voted pending suggestions.
    
    This shows what stocks the community wants tracked next.
    No authentication or rate limiting required.
    """
    return suggestion_service.get_top_suggestions(limit=limit)


@router.get("/pending", response_model=SuggestionListResponse)
async def list_pending_suggestions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    conn=Depends(get_db),
):
    """List all pending suggestions (for public viewing)."""
    items, total = suggestions_repo.list_suggestions(
        conn,
        status="pending",
        page=page,
        page_size=page_size,
    )
    
    return SuggestionListResponse(
        items=[SuggestionResponse(
            id=s.id,
            symbol=s.symbol,
            status=SuggestionStatus(s.status),
            vote_count=s.vote_count,
            name=s.name,
            sector=s.sector,
            summary=s.summary[:300] + "..." if s.summary and len(s.summary) > 300 else s.summary,
            last_price=s.last_price,
            price_change_90d=s.price_change_90d,
            created_at=s.created_at,
            updated_at=s.updated_at,
            fetched_at=s.fetched_at,
        ) for s in items],
        total=total,
        page=page,
        page_size=page_size,
    )


# =============================================================================
# Admin endpoints (authenticated)
# =============================================================================

@router.get("", response_model=SuggestionListResponse)
async def list_all_suggestions(
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    order_by: str = Query("vote_count"),
    order_dir: str = Query("DESC"),
    conn=Depends(get_db),
    _admin=Depends(require_admin),
):
    """List all suggestions (admin only)."""
    items, total = suggestions_repo.list_suggestions(
        conn,
        status=status,
        page=page,
        page_size=page_size,
        order_by=order_by,
        order_dir=order_dir,
    )
    
    return SuggestionListResponse(
        items=[SuggestionResponse(
            id=s.id,
            symbol=s.symbol,
            status=SuggestionStatus(s.status),
            vote_count=s.vote_count,
            name=s.name,
            sector=s.sector,
            summary=s.summary,
            last_price=s.last_price,
            price_change_90d=s.price_change_90d,
            created_at=s.created_at,
            updated_at=s.updated_at,
            fetched_at=s.fetched_at,
            rejection_reason=s.rejection_reason,
        ) for s in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{suggestion_id}", response_model=SuggestionResponse)
async def get_suggestion(
    suggestion_id: int,
    conn=Depends(get_db),
    _admin=Depends(require_admin),
):
    """Get a single suggestion by ID (admin only)."""
    suggestion = suggestions_repo.get_by_id(conn, suggestion_id)
    
    if not suggestion:
        raise NotFoundError("Suggestion", str(suggestion_id))
    
    return SuggestionResponse(
        id=suggestion.id,
        symbol=suggestion.symbol,
        status=SuggestionStatus(suggestion.status),
        vote_count=suggestion.vote_count,
        name=suggestion.name,
        sector=suggestion.sector,
        summary=suggestion.summary,
        last_price=suggestion.last_price,
        price_change_90d=suggestion.price_change_90d,
        created_at=suggestion.created_at,
        updated_at=suggestion.updated_at,
        fetched_at=suggestion.fetched_at,
        rejection_reason=suggestion.rejection_reason,
    )


@router.put("/{suggestion_id}/approve", response_model=dict)
async def approve_suggestion(
    suggestion_id: int,
    data: SuggestionApprove = None,
    _admin=Depends(require_admin),
):
    """
    Approve a suggestion and add it to tracked symbols.
    
    Uses PUT since approving is idempotent.
    """
    success, error, result = await suggestion_service.admin_action(
        suggestion_id=suggestion_id,
        action="approve",
        reason=None,
    )
    
    if not success:
        raise ValidationError(error)
    
    return {
        "message": "Suggestion approved successfully",
        **result,
    }


@router.put("/{suggestion_id}/reject", response_model=dict)
async def reject_suggestion(
    suggestion_id: int,
    data: SuggestionReject,
    _admin=Depends(require_admin),
):
    """
    Reject a suggestion with a reason.
    
    Uses PUT since rejecting is idempotent.
    """
    success, error, result = await suggestion_service.admin_action(
        suggestion_id=suggestion_id,
        action="reject",
        reason=data.reason,
    )
    
    if not success:
        raise ValidationError(error)
    
    return {
        "message": "Suggestion rejected",
        **result,
    }


@router.delete("/{suggestion_id}", response_model=dict)
async def remove_suggestion(
    suggestion_id: int,
    _admin=Depends(require_admin),
):
    """
    Remove a previously approved stock from tracking.
    
    The stock enters a 90-day cooldown before it can be suggested again.
    Uses DELETE as this is a removal operation.
    """
    success, error, result = await suggestion_service.admin_action(
        suggestion_id=suggestion_id,
        action="remove",
        reason=None,
    )
    
    if not success:
        raise ValidationError(error)
    
    return {
        "message": "Stock removed from tracking",
        **result,
    }


@router.post("/fetch", response_model=dict)
async def trigger_fetch(
    _admin=Depends(require_admin),
):
    """
    Manually trigger fetching data for pending suggestions.
    
    This is normally done by the scheduled job, but can be triggered manually.
    Progress is broadcast via WebSocket.
    """
    result = await suggestion_service.process_pending_suggestions()
    return {
        "message": "Fetch job completed",
        **result,
    }
