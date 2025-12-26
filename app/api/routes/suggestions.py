"""Stock suggestion API routes - PostgreSQL async version."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query, Request, Header

from app.api.dependencies import require_admin
from app.core.exceptions import ValidationError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData, decode_access_token
from app.celery_app import celery_app
from app.core.client_identity import (
    get_client_fingerprint,
    get_server_fingerprint,
    check_vote_allowed,
    record_vote,
    check_can_suggest,
    record_suggestion,
    get_voted_symbols,
    RiskLevel,
)
from app.database.connection import get_session
from sqlalchemy import text
from app.repositories import suggestions_orm as suggestions_repo
from app.repositories import symbols_orm as symbols_repo
from app.repositories import dip_state_orm as dip_state_repo
from app.repositories import auth_user_orm as auth_repo
from app.services.runtime_settings import get_runtime_setting
from app.services.suggestion_stock_info import (
    validate_symbol_format as _validate_symbol_format,
    get_stock_info_full as _get_stock_info_full,
    get_stock_info_basic as _get_stock_info,
    get_ipo_year as _get_ipo_year,
    get_stock_info_full_async,
)

logger = get_logger("api.routes.suggestions")

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


def _enqueue_symbol_processing(symbol: str) -> str:
    """Queue background processing for an approved symbol."""
    task = celery_app.send_task("jobs.process_approved_symbol", args=[symbol.upper()])
    return task.id


# =============================================================================
# Auto-approval helper
# =============================================================================


async def _check_and_apply_auto_approval(
    suggestion_id: int,
    symbol: str,
    new_score: int,
) -> bool:
    """Check all auto-approval conditions and approve if met.
    
    Returns True if auto-approved, False otherwise.
    
    Conditions (all must be met):
    1. auto_approve_enabled must be True
    2. Vote count >= auto_approve_votes threshold
    3. Unique voters (by fingerprint) >= auto_approve_unique_voters
    4. Suggestion age >= auto_approve_min_age_hours
    """
    from app.core.config import settings
    
    # 1. Check if auto-approval is enabled
    if not settings.auto_approve_enabled:
        logger.debug(f"Auto-approval disabled for {symbol}")
        return False
    
    # 2. Check vote count threshold (runtime-configurable)
    auto_approve_threshold = get_runtime_setting("auto_approve_votes", settings.auto_approve_votes)
    if new_score < auto_approve_threshold:
        logger.debug(f"Auto-approval: {symbol} has {new_score} votes, need {auto_approve_threshold}")
        return False
    
    # 3. Check unique voter count
    unique_voters = await suggestions_repo.get_unique_voter_count(suggestion_id)
    
    if unique_voters < settings.auto_approve_unique_voters:
        logger.debug(
            f"Auto-approval: {symbol} has {unique_voters} unique voters, "
            f"need {settings.auto_approve_unique_voters}"
        )
        return False
    
    # 4. Check suggestion age
    age_hours = await suggestions_repo.get_suggestion_age_hours(suggestion_id)
    
    if age_hours < settings.auto_approve_min_age_hours:
        logger.debug(
            f"Auto-approval: {symbol} is {age_hours:.1f}h old, "
            f"need {settings.auto_approve_min_age_hours}h"
        )
        return False
    
    # All conditions met - auto-approve
    await suggestions_repo.auto_approve_suggestion(suggestion_id)
    
    logger.info(
        f"Auto-approved {symbol}: {new_score} votes (>={auto_approve_threshold}), "
        f"{unique_voters} unique voters (>={settings.auto_approve_unique_voters}), "
        f"{age_hours:.1f}h old (>={settings.auto_approve_min_age_hours}h)"
    )
    
    return True


# =============================================================================
# Public endpoints
# =============================================================================


@router.get("/settings", response_model=dict)
async def get_suggestion_settings():
    """
    Get public suggestion settings.
    
    Returns auto-approval threshold and other public settings.
    """
    from app.core.config import settings
    return {
        "auto_approve_votes": get_runtime_setting("auto_approve_votes", settings.auto_approve_votes),
    }


@router.post("", response_model=dict, status_code=201)
async def suggest_stock(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=10, description="Stock symbol (1-10 chars, letters/numbers/dots only)"),
    authorization: Optional[str] = Header(default=None),
):
    """
    Suggest a new stock to be tracked.

    Public endpoint - no auth required.
    If stock already suggested, adds a vote instead.
    Uses multi-layer fingerprinting for abuse prevention.
    Admins bypass cooldown restrictions.
    """
    # Check if admin (bypass cooldown)
    is_admin = False
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.split(" ", 1)[1]
            token_data = decode_access_token(token)
            if token_data and token_data.is_admin:
                is_admin = True
        except Exception:
            pass  # Not a valid token, continue as anonymous
    
    # Validate symbol format
    symbol = _validate_symbol_format(symbol)
    
    # Get server-generated fingerprint (not client-provided)
    server_fp = get_server_fingerprint(request)
    fingerprint_hash = hashlib.sha256(server_fp.encode()).hexdigest()[:32]

    # Check if suggestion already exists
    existing = await suggestions_repo.get_suggestion_by_symbol(symbol)

    if existing:
        task_id = None
        # Already exists - try to add a vote (admins bypass cooldown)
        if not is_admin:
            check = await check_vote_allowed(request, symbol)
            if not check.allowed:
                raise ValidationError(message=check.reason, details={"symbol": symbol})

        suggestion_id = existing.id

        # Record vote for tracking (skip for admins to avoid polluting rate limit data)
        if not is_admin:
            await record_vote(request, symbol)

        # Add vote to database (upsert)
        new_score, _ = await suggestions_repo.upsert_vote(
            suggestion_id=suggestion_id,
            fingerprint=fingerprint_hash,
            vote_type="up",
        )

        current_status = existing.status
        auto_approved = False
        
        # Check for auto-approval if still pending
        if current_status == "pending":
            auto_approved = await _check_and_apply_auto_approval(
                suggestion_id=suggestion_id,
                symbol=symbol,
                new_score=new_score,
            )
            if auto_approved:
                task_id = _enqueue_symbol_processing(symbol)
                current_status = "approved"

        response = {
            "message": "Vote added successfully",
            "symbol": symbol,
            "vote_count": new_score,
            "status": current_status,
            "auto_approved": auto_approved,
        }
        if task_id:
            response["task_id"] = task_id

        return response

    # NEW SUGGESTION - check rate limit first
    # Check suggestion limits (admins bypass)
    if not is_admin:
        can_suggest, reason = await check_can_suggest(request)
        if not can_suggest:
            raise ValidationError(message=reason, details={"symbol": symbol})

    # Fetch stock info from yfinance
    stock_info = await get_stock_info_full_async(symbol)
    
    # If symbol is completely invalid (not rate limited), reject the suggestion
    if stock_info["fetch_status"] == "invalid":
        raise ValidationError(
            message=stock_info["fetch_error"] or "Symbol not found",
            details={"symbol": symbol}
        )
    
    # Create suggestion with fetched data (or with pending fetch status if rate limited)
    new_suggestion = await suggestions_repo.create_suggestion(
        symbol=symbol,
        fingerprint=fingerprint_hash,
        company_name=stock_info["name"],
        sector=stock_info["sector"],
        summary=stock_info["summary"],
        website=stock_info["website"],
        ipo_year=stock_info["ipo_year"],
        current_price=stock_info["current_price"],
        ath_price=stock_info["ath_price"],
    )

    # Record suggestion for rate limiting (skip for admins)
    if not is_admin:
        await record_suggestion(request)
    
    # Record vote (suggesting also counts as voting - skip for admins)
    if not is_admin:
        await record_vote(request, symbol)

    # Add initial vote to database
    await suggestions_repo.upsert_vote(
        suggestion_id=new_suggestion.id,
        fingerprint=fingerprint_hash,
        vote_type="up",
    )
    
    response = {
        "message": "Stock suggested successfully",
        "symbol": symbol,
        "vote_count": 1,
        "status": "pending",
    }
    
    # Include fetch status info if not fully fetched
    if stock_info["fetch_status"] == "rate_limited":
        response["fetch_status"] = "rate_limited"
        response["fetch_message"] = "Data will be fetched once rate limit resets"
    
    return response


@router.put("/{symbol}/vote", response_model=dict)
async def vote_for_suggestion(
    request: Request,
    symbol: str,
    authorization: Optional[str] = Header(default=None),
):
    """
    Vote for an existing stock suggestion.

    Public endpoint - no auth required.
    Uses server-side fingerprinting for abuse prevention.
    Vote cooldown: 7 days per stock per user/device/IP.
    Admins bypass the cooldown entirely.
    """
    symbol = symbol.strip().upper()
    
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
    
    # === Risk Assessment + Cooldown Check (skip for admin) ===
    if not is_admin:
        check = await check_vote_allowed(request, symbol)
        
        if not check.allowed:
            raise ValidationError(
                check.reason,
                details={"retry_after": 3600}
            )

    # Get suggestion
    suggestion = await suggestions_repo.get_suggestion_by_symbol(symbol)

    if not suggestion:
        raise NotFoundError(
            message=f"No suggestion found for '{symbol}'", details={"symbol": symbol}
        )

    suggestion_id = suggestion.id

    # Record vote for tracking
    await record_vote(request, symbol)

    # Store vote in database
    server_fp = get_server_fingerprint(request)
    fingerprint_hash = hashlib.sha256(server_fp.encode()).hexdigest()[:32]
    new_score, _ = await suggestions_repo.upsert_vote(
        suggestion_id=suggestion_id,
        fingerprint=fingerprint_hash,
        vote_type="up",
    )

    current_status = suggestion.status
    auto_approved = False
    task_id = None
    
    # Check for auto-approval if still pending
    if current_status == "pending":
        auto_approve_threshold = get_runtime_setting("auto_approve_votes", 10)
        if new_score >= auto_approve_threshold:
            # Auto-approve the suggestion
            await suggestions_repo.auto_approve_suggestion(suggestion_id)
            current_status = "approved"
            auto_approved = True
            logger.info(f"Auto-approved suggestion {symbol} with {new_score} votes (threshold: {auto_approve_threshold})")
            # Trigger background processing for data + AI
            task_id = _enqueue_symbol_processing(symbol)

    response = {
        "message": "Vote recorded", 
        "symbol": symbol,
        "vote_count": new_score,
        "status": current_status,
        "auto_approved": auto_approved,
    }
    if task_id:
        response["task_id"] = task_id

    return response


@router.get("/top", response_model=List[dict])
async def get_top_suggestions(
    request: Request,
    limit: int = Query(10, ge=1, le=50),
    exclude_voted: bool = Query(False, description="Exclude suggestions the user has already voted on"),
):
    """
    Get top voted pending suggestions.

    Public endpoint - shows what stocks the community wants tracked.
    Uses session-based tracking for exclude_voted filter.
    """
    # Get voted symbols from Valkey
    voted_symbols = set()
    if exclude_voted:
        voted_symbols = await get_voted_symbols(request)
    
    # Use ORM to get pending suggestions
    suggestions, _ = await suggestions_repo.list_all_suggestions(
        status="pending",
        limit=limit + len(voted_symbols),
        offset=0,
    )
    
    # Filter out voted symbols and convert to response format
    results = []
    for suggestion in suggestions:
        if suggestion.symbol not in voted_symbols:
            results.append({
                "symbol": suggestion.symbol,
                "name": suggestion.company_name,
                "vote_count": suggestion.vote_score,
                "sector": suggestion.sector,
                "summary": suggestion.summary,
                "ipo_year": suggestion.ipo_year,
                "website": suggestion.website,
                "fetch_status": suggestion.fetch_status,
            })
            if len(results) >= limit:
                break
    
    return results


@router.get("/search", response_model=List[dict])
async def search_stored_suggestions(
    q: str = Query(..., min_length=1, max_length=50, description="Search query"),
    limit: int = Query(10, ge=1, le=20),
):
    """
    Search stored suggestions and tracked symbols for quick autocomplete.
    
    This searches only cached/stored data (no yfinance API calls).
    Returns results from:
    1. Pending suggestions (community suggested)
    2. Already tracked symbols
    
    Fast endpoint suitable for real-time debounced search.
    Results are ordered by relevance: exact match > prefix match > contains match.
    """
    # Search in suggestions (pending ones) using ORM
    suggestion_rows = await suggestions_repo.search_pending_suggestions(q, limit)
    
    # Search in tracked symbols using ORM
    symbol_rows = await suggestions_repo.search_tracked_symbols(q, limit)
    
    # Combine and dedupe (prefer tracked over suggestions)
    seen_symbols = set()
    results = []
    
    # Add tracked symbols first (they're already in our system)
    for row in symbol_rows:
        if row["symbol"] not in seen_symbols:
            seen_symbols.add(row["symbol"])
            results.append({
                "symbol": row["symbol"],
                "name": row["name"],
                "sector": row["sector"],
                "source": "tracked",
                "vote_count": None,
            })
    
    # Add pending suggestions
    for row in suggestion_rows:
        if row["symbol"] not in seen_symbols:
            seen_symbols.add(row["symbol"])
            results.append({
                "symbol": row["symbol"],
                "name": row["name"],
                "sector": row["sector"],
                "source": "suggestion",
                "vote_count": row["vote_score"],
            })
    
    return results[:limit]


@router.get("/pending", response_model=dict)
async def list_pending_suggestions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List all pending suggestions (public)."""
    offset = (page - 1) * page_size

    # Get pending suggestions using ORM
    suggestions, total = await suggestions_repo.list_all_suggestions(
        status="pending",
        limit=page_size,
        offset=offset,
    )

    return {
        "items": [
            {
                "id": suggestion.id,
                "symbol": suggestion.symbol,
                "name": suggestion.company_name,
                "sector": suggestion.sector,
                "summary": suggestion.summary,
                "website": suggestion.website,
                "ipo_year": suggestion.ipo_year,
                "fetch_status": suggestion.fetch_status,
                "status": suggestion.status,
                "vote_count": suggestion.vote_score,
                "created_at": suggestion.created_at.isoformat()
                if suggestion.created_at
                else None,
            }
            for suggestion in suggestions
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# =============================================================================
# Admin endpoints
# =============================================================================


@router.get("", response_model=dict)
async def list_all_suggestions(
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    admin: TokenData = Depends(require_admin),
):
    """List all suggestions (admin only)."""
    offset = (page - 1) * page_size

    # Get suggestions using ORM
    suggestions, total = await suggestions_repo.list_all_suggestions(
        status=status,
        limit=page_size,
        offset=offset,
    )

    return {
        "items": [
            {
                "id": suggestion.id,
                "symbol": suggestion.symbol,
                "name": suggestion.company_name,
                "sector": suggestion.sector,
                "summary": suggestion.summary,
                "website": suggestion.website,
                "ipo_year": suggestion.ipo_year,
                "current_price": float(suggestion.current_price) if suggestion.current_price else None,
                "ath_price": float(suggestion.ath_price) if suggestion.ath_price else None,
                "fetch_status": suggestion.fetch_status,
                "fetch_error": suggestion.fetch_error,
                "fetched_at": suggestion.fetched_at.isoformat() if suggestion.fetched_at else None,
                "status": suggestion.status,
                "vote_count": suggestion.vote_score,
                "approved_by": suggestion.approved_by_id,
                "created_at": suggestion.created_at.isoformat()
                if suggestion.created_at
                else None,
                "reviewed_at": suggestion.reviewed_at.isoformat()
                if suggestion.reviewed_at
                else None,
            }
            for suggestion in suggestions
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("/{suggestion_id}/approve", response_model=dict)
async def approve_suggestion(
    suggestion_id: int,
    admin: TokenData = Depends(require_admin),
):
    """Approve a suggestion and add the stock to tracking (admin only).
    
    This will:
    1. Update the suggestion status to 'approved'
    2. Add the symbol to tracked symbols
    3. Fetch Yahoo Finance data in background
    4. Generate AI bio in background
    """
    # Get suggestion
    suggestion = await suggestions_repo.get_suggestion_by_id(suggestion_id)

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion {suggestion_id} not found",
            details={"id": suggestion_id},
        )

    if suggestion.status != "pending":
        raise ValidationError(
            message=f"Suggestion is already {suggestion.status}",
            details={"status": suggestion.status},
        )

    symbol = suggestion.symbol

    # Approve suggestion using ORM
    from app.repositories import auth_user_orm as auth_repo
    admin_user = await auth_repo.get_user(admin.sub)
    admin_id = admin_user.id if admin_user else None

    await suggestions_repo.approve_suggestion(suggestion_id, admin_id)

    # Add symbol to tracked symbols using ORM
    await suggestions_repo.add_symbol_from_suggestion(
        symbol=symbol,
        name=suggestion.company_name,
        sector=suggestion.sector,
    )
    
    # Queue for initial data ingest (high priority since user is waiting)
    from app.jobs.definitions import add_to_ingest_queue
    await add_to_ingest_queue(symbol, priority=5)

    task_id = _enqueue_symbol_processing(symbol)

    return {
        "message": f"Approved and added {symbol} to tracking. AI content generating in background.",
        "symbol": symbol,
        "task_id": task_id,
    }


@router.post("/{suggestion_id}/refresh", response_model=dict)
async def refresh_suggestion_data(
    suggestion_id: int,
    admin: TokenData = Depends(require_admin),
):
    """Refresh data and AI content for an approved suggestion (admin only).
    
    Fetches latest Yahoo Finance data and regenerates AI bio/rating in real-time.
    """
    # Get suggestion
    suggestion = await suggestions_repo.get_suggestion_by_id(suggestion_id)

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion #{suggestion_id} not found",
            details={"id": suggestion_id},
        )

    if suggestion.status != "approved":
        raise ValidationError(
            message="Only approved suggestions can be refreshed",
            details={"status": suggestion.status},
        )

    symbol = suggestion.symbol
    
    task_id = _enqueue_symbol_processing(symbol)

    return {
        "message": f"Refresh queued for {symbol}",
        "symbol": symbol,
        "task_id": task_id,
    }


@router.patch("/{suggestion_id}", response_model=dict)
async def update_suggestion(
    suggestion_id: int,
    new_symbol: str = Query(None, description="New symbol to use (e.g., change .F to .DE)"),
    admin: TokenData = Depends(require_admin),
):
    """Update a suggestion's symbol (admin only).
    
    Useful for correcting symbols (e.g., German stocks listed as .F vs .DE).
    """
    # Get suggestion
    suggestion = await suggestions_repo.get_suggestion_by_id(suggestion_id)

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion {suggestion_id} not found",
            details={"id": suggestion_id},
        )

    updates = {}
    
    if new_symbol:
        # Validate the new symbol
        new_symbol = _validate_symbol_format(new_symbol)
        
        # Check if the new symbol already exists
        existing = await suggestions_repo.check_symbol_exists_in_other_suggestion(
            new_symbol, suggestion_id
        )
        
        if existing:
            raise ValidationError(
                message=f"Symbol {new_symbol} already exists as a suggestion",
                details={"new_symbol": new_symbol},
            )
        
        updates["symbol"] = new_symbol

    if not updates:
        raise ValidationError(
            message="No updates provided",
            details={"suggestion_id": suggestion_id},
        )

    # Update symbol
    if "symbol" in updates:
        await suggestions_repo.update_suggestion_symbol(suggestion_id, updates["symbol"])

    return {
        "message": f"Updated suggestion #{suggestion_id}",
        "old_symbol": suggestion.symbol,
        "new_symbol": updates.get("symbol", suggestion.symbol),
    }


@router.post("/{suggestion_id}/retry", response_model=dict)
async def retry_suggestion_fetch(
    suggestion_id: int,
    admin: TokenData = Depends(require_admin),
):
    """Retry fetching yfinance data for a suggestion with rate_limited or error status (admin only).
    
    This is useful for retrying individual suggestions that failed due to rate limiting.
    """
    # Get suggestion
    suggestion = await suggestions_repo.get_suggestion_by_id(suggestion_id)

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion {suggestion_id} not found",
            details={"id": suggestion_id},
        )

    # Only allow retry for rate_limited, error, or pending status
    if suggestion.fetch_status not in ("rate_limited", "error", "pending", None):
        raise ValidationError(
            message=f"Suggestion already has status '{suggestion.fetch_status}'",
            details={"fetch_status": suggestion.fetch_status},
        )

    symbol = suggestion.symbol
    
    # Fetch stock info
    stock_info = await get_stock_info_full_async(symbol)
    
    # Update suggestion with fetched data
    await suggestions_repo.update_suggestion_stock_info(
        symbol=symbol,
        company_name=stock_info["name"],
        sector=stock_info["sector"],
        summary=stock_info["summary"],
        website=stock_info["website"],
        ipo_year=stock_info["ipo_year"],
        current_price=stock_info["current_price"],
        ath_price=stock_info["ath_price"],
        fetch_status=stock_info["fetch_status"],
        fetch_error=stock_info["fetch_error"],
    )
    
    logger.info(f"Retried fetch for {symbol}: {stock_info['fetch_status']}")
    
    return {
        "message": f"Retried fetch for {symbol}",
        "symbol": symbol,
        "fetch_status": stock_info["fetch_status"],
        "fetch_error": stock_info["fetch_error"],
    }


@router.post("/backfill", response_model=dict)
async def backfill_suggestion_data(
    limit: int = Query(10, ge=1, le=100, description="Max suggestions to process"),
    admin: TokenData = Depends(require_admin),
):
    """Backfill yfinance data for suggestions with pending fetch status (admin only).
    
    This is useful for populating data for suggestions created before the
    data fetch was implemented.
    """
    # Get suggestions with pending fetch status
    suggestions = await suggestions_repo.list_suggestions_needing_backfill(limit)
    
    if not suggestions:
        return {"message": "No suggestions need backfilling", "processed": 0}
    
    processed = 0
    errors = []
    
    for suggestion in suggestions:
        symbol = suggestion.symbol
        suggestion_id = suggestion.id
        
        try:
            # Fetch stock info
            stock_info = await get_stock_info_full_async(symbol)
            
            # Update suggestion with fetched data
            await suggestions_repo.update_suggestion_stock_info(
                symbol=symbol,
                company_name=stock_info["name"],
                sector=stock_info["sector"],
                summary=stock_info["summary"],
                website=stock_info["website"],
                ipo_year=stock_info["ipo_year"],
                current_price=stock_info["current_price"],
                ath_price=stock_info["ath_price"],
                fetch_status=stock_info["fetch_status"],
                fetch_error=stock_info["fetch_error"],
            )
            processed += 1
            logger.info(f"Backfilled data for {symbol}: {stock_info['fetch_status']}")
            
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
            logger.error(f"Error backfilling {symbol}: {e}")
    
    return {
        "message": f"Processed {processed} suggestions",
        "processed": processed,
        "total": len(suggestions),
        "errors": errors if errors else None,
    }


async def _process_approved_symbol(symbol: str) -> None:
    """Background task to process newly approved symbol.
    
    Fetches Yahoo Finance data, 365 days of price history, and generates AI content.
    Also generates AI summary of company description on first import.
    Updates fetch_status throughout for real-time admin feedback.
    """
    from datetime import date, timedelta
    from app.services.stock_info import get_stock_info_async
    from app.services.openai_client import generate_bio, rate_dip, summarize_company
    from app.repositories import dip_votes_orm as dip_votes_repo
    from app.dipfinder.service import get_dipfinder_service
    from app.cache.cache import Cache
    
    logger.info(f"Processing newly approved symbol: {symbol}")
    
    # Set fetch_status to 'fetching' so admin UI shows loading state
    await suggestions_repo.set_suggestion_fetching(symbol.upper())
    
    try:
        # Step 1: Fetch Yahoo Finance info
        info = await get_stock_info_async(symbol)
        if not info:
            logger.warning(f"Could not fetch Yahoo data for {symbol}")
            return
            
        current_price = info.get("current_price", 0)
        ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
        dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0
        
        logger.info(f"Fetched data for {symbol}: price=${current_price}, ATH=${ath_price}, dip={dip_pct:.1f}%")
        
        # Step 1.5: Update symbols table with name, sector and generate AI summary
        name = info.get("name") or info.get("short_name")
        sector = info.get("sector")
        full_summary = info.get("summary")  # longBusinessSummary from yfinance
        
        # Generate AI summary from the long description (only on first import)
        # Respects ai_enrichment_enabled setting
        ai_summary = None
        ai_enabled = get_runtime_setting("ai_enrichment_enabled", True)
        if ai_enabled and full_summary and len(full_summary) > 100:
            # Check if we already have an AI summary
            existing_summary = await symbols_repo.get_symbol_summary_ai(symbol.upper())
            if not existing_summary:
                ai_summary = await summarize_company(
                    symbol=symbol,
                    name=name,
                    description=full_summary,
                )
                if ai_summary:
                    logger.info(f"Generated AI summary for {symbol}: {len(ai_summary)} chars")
        
        # Update symbols with name, sector, and AI summary
        if name or sector or ai_summary:
            await symbols_repo.update_symbol_info(
                symbol.upper(),
                name=name,
                sector=sector,
                summary_ai=ai_summary,
            )
            logger.info(f"Updated symbol info for {symbol}: name='{name}', sector='{sector}', summary_ai={'yes' if ai_summary else 'no'}")
        
        # Also update stock_suggestions table with name/sector for admin UI display
        await suggestions_repo.update_suggestion_stock_info(
            symbol=symbol.upper(),
            company_name=name,
            sector=sector,
            summary=full_summary[:1000] if full_summary else None,  # Truncate for storage
            current_price=current_price,
            ath_price=ath_price,
        )
        
        # Step 1.6: Fetch 365 days of price history
        try:
            service = get_dipfinder_service()
            prices = await service.price_provider.get_prices(
                symbol.upper(),
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
            )
            if prices is not None and not prices.empty:
                logger.info(f"Fetched {len(prices)} days of price history for {symbol}")
            else:
                logger.warning(f"No price history returned for {symbol}")
        except Exception as price_err:
            logger.warning(f"Failed to fetch price history for {symbol}: {price_err}")
        
        # Step 2: Create or update dip_state entry
        await dip_state_repo.upsert_dip_state(
            symbol=symbol.upper(),
            current_price=current_price,
            ath_price=ath_price,
            dip_percentage=dip_pct,
        )
        
        # Check if AI enrichment is enabled before running AI work
        ai_enabled = get_runtime_setting("ai_enrichment_enabled", True)
        
        if ai_enabled:
            # Step 3: Generate AI bio
            bio = await generate_bio(
                symbol=symbol,
                dip_pct=dip_pct,
            )
            
            # Step 4: Generate AI rating
            rating_data = await rate_dip(
                symbol=symbol,
                current_price=current_price,
                ref_high=ath_price,
                dip_pct=dip_pct,
            )
            
            # Step 5: Store AI analysis
            if bio or rating_data:
                await dip_votes_repo.upsert_ai_analysis(
                    symbol=symbol,
                    swipe_bio=bio,
                    ai_rating=rating_data.get("rating") if rating_data else None,
                    ai_reasoning=rating_data.get("reasoning") if rating_data else None,
                    is_batch=False,
                )
                logger.info(f"Generated AI content for {symbol}: bio={'yes' if bio else 'no'}, rating={rating_data.get('rating') if rating_data else 'none'}")
            else:
                logger.warning(f"No AI content generated for {symbol}")
            
            # Step 5.5: Run AI agent analysis (Warren Buffett, Peter Lynch, etc.)
            try:
                from app.services.ai_agents import run_agent_analysis
                agent_result = await run_agent_analysis(symbol)
                if agent_result:
                    logger.info(f"AI agents for {symbol}: {agent_result.overall_signal} ({agent_result.overall_confidence}%)")
                else:
                    logger.warning(f"No agent analysis generated for {symbol}")
            except Exception as agent_err:
                logger.warning(f"AI agents error for {symbol}: {agent_err}")
        else:
            logger.info(f"AI enrichment disabled, skipping AI content generation for {symbol}")
        
        # Step 6: Mark as fetched and invalidate ranking cache
        # Update both stock_suggestions AND symbols tables
        await suggestions_repo.set_suggestion_fetched(symbol.upper())
        await symbols_repo.update_fetch_status(
            symbol.upper(),
            fetch_status="fetched",
            fetch_error=None,
        )
        
        # Invalidate ranking cache so the new stock appears immediately
        ranking_cache = Cache(prefix="ranking", default_ttl=3600)
        deleted = await ranking_cache.invalidate_pattern("*")
        logger.info(f"Completed processing {symbol}, invalidated {deleted} ranking cache keys")
            
    except Exception as e:
        logger.error(f"Error processing approved symbol {symbol}: {e}")
        # Set error status so admin can retry
        await suggestions_repo.set_suggestion_error(symbol.upper(), str(e))


@router.post("/{suggestion_id}/reject", response_model=dict)
async def reject_suggestion(
    suggestion_id: int,
    reason: str = Query(None, description="Rejection reason"),
    admin: TokenData = Depends(require_admin),
):
    """Reject a suggestion (admin only)."""
    # Get suggestion
    suggestion = await suggestions_repo.get_suggestion_by_id(suggestion_id)

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion {suggestion_id} not found",
            details={"id": suggestion_id},
        )

    if suggestion.status != "pending":
        raise ValidationError(
            message=f"Suggestion is already {suggestion.status}",
            details={"status": suggestion.status},
        )

    # Look up admin user ID from username
    admin_user = await auth_repo.get_user(admin.sub)
    admin_id = admin_user.id if admin_user else None

    # Update suggestion status
    await suggestions_repo.reject_suggestion_with_reason(
        suggestion_id=suggestion_id,
        admin_id=admin_id,
        reason=reason,
    )

    return {
        "message": f"Rejected suggestion for {suggestion.symbol}",
        "symbol": suggestion.symbol,
    }
