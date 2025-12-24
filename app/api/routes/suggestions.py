"""Stock suggestion API routes - PostgreSQL async version."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Header

from app.api.dependencies import require_admin
from app.core.exceptions import ValidationError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData, decode_access_token
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
from app.database.connection import fetch_all, fetch_one, execute
from app.services.runtime_settings import get_runtime_setting
from app.services.suggestion_stock_info import (
    validate_symbol_format as _validate_symbol_format,
    get_stock_info_full as _get_stock_info_full,
    get_stock_info_basic as _get_stock_info,
    get_ipo_year as _get_ipo_year,
    get_stock_info_full_async,
    RATE_LIMIT_INDICATORS,
    _executor,
)

logger = get_logger("api.routes.suggestions")

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

# Vote cooldown in days (now managed in anon_session.py)
VOTE_COOLDOWN_DAYS = 7


# =============================================================================
# Public endpoints
# =============================================================================


@router.get("/settings", response_model=dict)
async def get_suggestion_settings():
    """
    Get public suggestion settings.
    
    Returns auto-approval threshold and other public settings.
    """
    return {
        "auto_approve_votes": get_runtime_setting("auto_approve_votes", 10),
    }


@router.post("", response_model=dict, status_code=201)
async def suggest_stock(
    request: Request,
    background_tasks: BackgroundTasks,
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
    existing = await fetch_one(
        "SELECT id, symbol, status, vote_score FROM stock_suggestions WHERE symbol = $1",
        symbol,
    )

    if existing:
        # Already exists - try to add a vote (admins bypass cooldown)
        if not is_admin:
            check = await check_vote_allowed(request, symbol)
            if not check.allowed:
                raise ValidationError(message=check.reason, details={"symbol": symbol})

        suggestion_id = existing["id"]

        # Record vote for tracking (skip for admins to avoid polluting rate limit data)
        if not is_admin:
            await record_vote(request, symbol)

        # Also store in database for historical record
        await execute(
            """INSERT INTO suggestion_votes (suggestion_id, fingerprint, vote_type, created_at)
               VALUES ($1, $2, 'up', NOW())
               ON CONFLICT (suggestion_id, fingerprint) DO NOTHING""",
            suggestion_id,
            fingerprint_hash,
        )

        # Update vote score
        await execute(
            "UPDATE stock_suggestions SET vote_score = vote_score + 1 WHERE id = $1",
            suggestion_id,
        )

        new_score = existing["vote_score"] + 1
        current_status = existing["status"]
        auto_approved = False
        
        # Check for auto-approval if still pending
        if current_status == "pending":
            auto_approve_threshold = get_runtime_setting("auto_approve_votes", 10)
            if new_score >= auto_approve_threshold:
                # Auto-approve the suggestion
                await execute(
                    """UPDATE stock_suggestions 
                       SET status = 'approved', approved_by = NULL, reviewed_at = NOW()
                       WHERE id = $1 AND status = 'pending'""",
                    suggestion_id,
                )
                current_status = "approved"
                auto_approved = True
                logger.info(f"Auto-approved suggestion {symbol} with {new_score} votes (threshold: {auto_approve_threshold})")
                # Trigger background processing for data + AI
                background_tasks.add_task(_process_approved_symbol, symbol)

        return {
            "message": "Vote added successfully",
            "symbol": symbol,
            "vote_count": new_score,
            "status": current_status,
            "auto_approved": auto_approved,
        }

    # NEW SUGGESTION - check rate limit first
    # Check suggestion limits (admins bypass)
    if not is_admin:
        can_suggest, reason = await check_can_suggest(request)
        if not can_suggest:
            raise ValidationError(message=reason, details={"symbol": symbol})

    # Fetch stock info from yfinance
    loop = asyncio.get_event_loop()
    stock_info = await loop.run_in_executor(_executor, _get_stock_info_full, symbol)
    
    # If symbol is completely invalid (not rate limited), reject the suggestion
    if stock_info["fetch_status"] == "invalid":
        raise ValidationError(
            message=stock_info["fetch_error"] or "Symbol not found",
            details={"symbol": symbol}
        )
    
    # Create suggestion with fetched data (or with pending fetch status if rate limited)
    result = await fetch_one(
        """INSERT INTO stock_suggestions (
               symbol, fingerprint, status, vote_score, created_at,
               company_name, sector, summary, website, ipo_year, 
               current_price, ath_price, fetch_status, fetch_error, fetched_at
           ) VALUES ($1, $2, 'pending', 1, NOW(), $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
           RETURNING id, vote_score""",
        symbol,
        fingerprint_hash,
        stock_info["name"],
        stock_info["sector"],
        stock_info["summary"],
        stock_info["website"],
        stock_info["ipo_year"],
        stock_info["current_price"],
        stock_info["ath_price"],
        stock_info["fetch_status"],
        stock_info["fetch_error"],
    )

    # Record suggestion for rate limiting (skip for admins)
    if not is_admin:
        await record_suggestion(request)
    
    # Record vote (suggesting also counts as voting - skip for admins)
    if not is_admin:
        await record_vote(request, symbol)

    # Add initial vote to database
    await execute(
        """INSERT INTO suggestion_votes (suggestion_id, fingerprint, vote_type, created_at)
           VALUES ($1, $2, 'up', NOW())
           ON CONFLICT (suggestion_id, fingerprint) DO NOTHING""",
        result["id"],
        fingerprint_hash,
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
    background_tasks: BackgroundTasks,
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
    suggestion = await fetch_one(
        "SELECT id, symbol, status, vote_score FROM stock_suggestions WHERE symbol = $1",
        symbol,
    )

    if not suggestion:
        raise NotFoundError(
            message=f"No suggestion found for '{symbol}'", details={"symbol": symbol}
        )

    suggestion_id = suggestion["id"]

    # Record vote for tracking
    await record_vote(request, symbol)

    # Also store in database for historical record
    server_fp = get_server_fingerprint(request)
    fingerprint_hash = hashlib.sha256(server_fp.encode()).hexdigest()[:32]
    await execute(
        """INSERT INTO suggestion_votes (suggestion_id, fingerprint, vote_type, created_at)
           VALUES ($1, $2, 'up', NOW())
           ON CONFLICT (suggestion_id, fingerprint) DO NOTHING""",
        suggestion_id,
        fingerprint_hash,
    )

    # Update vote score
    await execute(
        "UPDATE stock_suggestions SET vote_score = vote_score + 1 WHERE id = $1",
        suggestion_id,
    )

    new_score = suggestion["vote_score"] + 1
    current_status = suggestion["status"]
    auto_approved = False
    
    # Check for auto-approval if still pending
    if current_status == "pending":
        auto_approve_threshold = get_runtime_setting("auto_approve_votes", 10)
        if new_score >= auto_approve_threshold:
            # Auto-approve the suggestion
            await execute(
                """UPDATE stock_suggestions 
                   SET status = 'approved', approved_by = NULL, reviewed_at = NOW()
                   WHERE id = $1 AND status = 'pending'""",
                suggestion_id,
            )
            current_status = "approved"
            auto_approved = True
            logger.info(f"Auto-approved suggestion {symbol} with {new_score} votes (threshold: {auto_approve_threshold})")
            # Trigger background processing for data + AI
            background_tasks.add_task(_process_approved_symbol, symbol)

    return {
        "message": "Vote recorded", 
        "symbol": symbol,
        "vote_count": new_score,
        "status": current_status,
        "auto_approved": auto_approved,
    }


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
    
    rows = await fetch_all(
        """SELECT symbol, company_name as name, vote_score as vote_count, 
                  sector, summary, website, ipo_year, fetch_status
           FROM stock_suggestions 
           WHERE status = 'pending'
           ORDER BY vote_score DESC
           LIMIT $1""",
        limit + len(voted_symbols),  # Fetch extra to account for filtering
    )
    
    # Filter out voted symbols
    if voted_symbols:
        rows = [row for row in rows if row["symbol"] not in voted_symbols][:limit]

    # Use stored data - no live yfinance calls needed!
    return [
        {
            "symbol": row["symbol"],
            "name": row["name"],
            "vote_count": row["vote_count"],
            "sector": row["sector"],
            "summary": row["summary"],
            "ipo_year": row["ipo_year"],
            "website": row["website"],
            "fetch_status": row["fetch_status"],
        }
        for row in rows
    ]


@router.get("/pending", response_model=dict)
async def list_pending_suggestions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """List all pending suggestions (public)."""
    offset = (page - 1) * page_size

    # Get count
    count_row = await fetch_one(
        "SELECT COUNT(*) as cnt FROM stock_suggestions WHERE status = 'pending'"
    )
    total = count_row["cnt"] if count_row else 0

    # Get items
    rows = await fetch_all(
        """SELECT id, symbol, company_name, sector, summary, website, ipo_year,
                  fetch_status, status, vote_score, created_at
           FROM stock_suggestions 
           WHERE status = 'pending'
           ORDER BY vote_score DESC
           LIMIT $1 OFFSET $2""",
        page_size,
        offset,
    )

    return {
        "items": [
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "name": row["company_name"],
                "sector": row["sector"],
                "summary": row["summary"],
                "website": row["website"],
                "ipo_year": row["ipo_year"],
                "fetch_status": row["fetch_status"],
                "status": row["status"],
                "vote_count": row["vote_score"],
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
            }
            for row in rows
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

    # Build query
    where = ""
    params = []
    param_idx = 1

    if status:
        where = f"WHERE status = ${param_idx}"
        params.append(status)
        param_idx += 1

    # Get count
    count_row = await fetch_one(
        f"SELECT COUNT(*) as cnt FROM stock_suggestions {where}", *params
    )
    total = count_row["cnt"] if count_row else 0

    # Get items
    params.extend([page_size, offset])
    rows = await fetch_all(
        f"""SELECT id, symbol, company_name, sector, summary, website, ipo_year,
                   current_price, ath_price, fetch_status, fetch_error, fetched_at,
                   status, vote_score, approved_by, created_at, reviewed_at
            FROM stock_suggestions {where}
            ORDER BY vote_score DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}""",
        *params,
    )

    return {
        "items": [
            {
                "id": row["id"],
                "symbol": row["symbol"],
                "name": row["company_name"],
                "sector": row["sector"],
                "summary": row["summary"],
                "website": row["website"],
                "ipo_year": row["ipo_year"],
                "current_price": float(row["current_price"]) if row["current_price"] else None,
                "ath_price": float(row["ath_price"]) if row["ath_price"] else None,
                "fetch_status": row["fetch_status"],
                "fetch_error": row["fetch_error"],
                "fetched_at": row["fetched_at"].isoformat() if row["fetched_at"] else None,
                "status": row["status"],
                "vote_count": row["vote_score"],
                "approved_by": row["approved_by"],
                "created_at": row["created_at"].isoformat()
                if row["created_at"]
                else None,
                "reviewed_at": row["reviewed_at"].isoformat()
                if row["reviewed_at"]
                else None,
            }
            for row in rows
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.post("/{suggestion_id}/approve", response_model=dict)
async def approve_suggestion(
    suggestion_id: int,
    background_tasks: BackgroundTasks,
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
    suggestion = await fetch_one(
        "SELECT id, symbol, status FROM stock_suggestions WHERE id = $1", suggestion_id
    )

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion {suggestion_id} not found",
            details={"id": suggestion_id},
        )

    if suggestion["status"] != "pending":
        raise ValidationError(
            message=f"Suggestion is already {suggestion['status']}",
            details={"status": suggestion["status"]},
        )

    symbol = suggestion["symbol"]

    # Look up admin user ID from username
    admin_user = await fetch_one(
        "SELECT id FROM auth_user WHERE username = $1",
        admin.sub
    )
    admin_id = admin_user["id"] if admin_user else None

    # Update suggestion status
    await execute(
        """UPDATE stock_suggestions 
           SET status = 'approved', approved_by = $1, reviewed_at = NOW()
           WHERE id = $2""",
        admin_id,
        suggestion_id,
    )

    # Add symbol to tracked symbols
    await execute(
        """INSERT INTO symbols (symbol, is_active, added_at)
           VALUES ($1, TRUE, NOW())
           ON CONFLICT (symbol) DO NOTHING""",
        symbol,
    )

    # Schedule background task to fetch data and generate AI content
    background_tasks.add_task(_process_approved_symbol, symbol)

    return {
        "message": f"Approved and added {symbol} to tracking. AI content generating in background.",
        "symbol": symbol,
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
    suggestion = await fetch_one(
        "SELECT id, symbol, status FROM stock_suggestions WHERE id = $1",
        suggestion_id,
    )

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion #{suggestion_id} not found",
            details={"id": suggestion_id},
        )

    if suggestion["status"] != "approved":
        raise ValidationError(
            message="Only approved suggestions can be refreshed",
            details={"status": suggestion["status"]},
        )

    symbol = suggestion["symbol"]
    
    # Process the symbol synchronously (not in background) for immediate feedback
    await _process_approved_symbol(symbol)

    return {
        "message": f"Refreshed data and AI content for {symbol}",
        "symbol": symbol,
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
    suggestion = await fetch_one(
        "SELECT id, symbol, status FROM stock_suggestions WHERE id = $1", suggestion_id
    )

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
        existing = await fetch_one(
            "SELECT id FROM stock_suggestions WHERE symbol = $1 AND id != $2",
            new_symbol,
            suggestion_id,
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

    # Build update query
    set_parts = []
    params = []
    param_idx = 1
    
    for key, value in updates.items():
        set_parts.append(f"{key} = ${param_idx}")
        params.append(value)
        param_idx += 1
    
    params.append(suggestion_id)
    
    await execute(
        f"UPDATE stock_suggestions SET {', '.join(set_parts)} WHERE id = ${param_idx}",
        *params,
    )

    return {
        "message": f"Updated suggestion #{suggestion_id}",
        "old_symbol": suggestion["symbol"],
        "new_symbol": updates.get("symbol", suggestion["symbol"]),
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
    suggestion = await fetch_one(
        "SELECT id, symbol, fetch_status FROM stock_suggestions WHERE id = $1", suggestion_id
    )

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion {suggestion_id} not found",
            details={"id": suggestion_id},
        )

    # Only allow retry for rate_limited, error, or pending status
    if suggestion["fetch_status"] not in ("rate_limited", "error", "pending", None):
        raise ValidationError(
            message=f"Suggestion already has status '{suggestion['fetch_status']}'",
            details={"fetch_status": suggestion["fetch_status"]},
        )

    symbol = suggestion["symbol"]
    
    # Fetch stock info
    loop = asyncio.get_event_loop()
    stock_info = await loop.run_in_executor(_executor, _get_stock_info_full, symbol)
    
    # Update suggestion with fetched data
    await execute(
        """UPDATE stock_suggestions SET
               company_name = COALESCE($2, company_name),
               sector = $3,
               summary = $4,
               website = $5,
               ipo_year = $6,
               current_price = $7,
               ath_price = $8,
               fetch_status = $9,
               fetch_error = $10,
               fetched_at = NOW()
           WHERE id = $1""",
        suggestion_id,
        stock_info["name"],
        stock_info["sector"],
        stock_info["summary"],
        stock_info["website"],
        stock_info["ipo_year"],
        stock_info["current_price"],
        stock_info["ath_price"],
        stock_info["fetch_status"],
        stock_info["fetch_error"],
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
    rows = await fetch_all(
        """SELECT id, symbol FROM stock_suggestions 
           WHERE fetch_status = 'pending' OR fetch_status IS NULL
           ORDER BY vote_score DESC
           LIMIT $1""",
        limit,
    )
    
    if not rows:
        return {"message": "No suggestions need backfilling", "processed": 0}
    
    processed = 0
    errors = []
    
    for row in rows:
        symbol = row["symbol"]
        suggestion_id = row["id"]
        
        try:
            # Fetch stock info
            loop = asyncio.get_event_loop()
            stock_info = await loop.run_in_executor(_executor, _get_stock_info_full, symbol)
            
            # Update suggestion with fetched data
            await execute(
                """UPDATE stock_suggestions SET
                       company_name = COALESCE($2, company_name),
                       sector = $3,
                       summary = $4,
                       website = $5,
                       ipo_year = $6,
                       current_price = $7,
                       ath_price = $8,
                       fetch_status = $9,
                       fetch_error = $10,
                       fetched_at = NOW()
                   WHERE id = $1""",
                suggestion_id,
                stock_info["name"],
                stock_info["sector"],
                stock_info["summary"],
                stock_info["website"],
                stock_info["ipo_year"],
                stock_info["current_price"],
                stock_info["ath_price"],
                stock_info["fetch_status"],
                stock_info["fetch_error"],
            )
            processed += 1
            logger.info(f"Backfilled data for {symbol}: {stock_info['fetch_status']}")
            
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
            logger.error(f"Error backfilling {symbol}: {e}")
    
    return {
        "message": f"Processed {processed} suggestions",
        "processed": processed,
        "total": len(rows),
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
    from app.repositories import dip_votes as dip_votes_repo
    from app.dipfinder.service import get_dipfinder_service
    from app.cache.cache import Cache
    
    logger.info(f"Processing newly approved symbol: {symbol}")
    
    # Set fetch_status to 'fetching' so admin UI shows loading state
    await execute(
        """UPDATE stock_suggestions 
           SET fetch_status = 'fetching', fetch_error = NULL
           WHERE symbol = $1""",
        symbol.upper(),
    )
    
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
        ai_summary = None
        if full_summary and len(full_summary) > 100:
            # Check if we already have an AI summary
            existing = await fetch_one(
                "SELECT summary_ai FROM symbols WHERE symbol = $1",
                symbol.upper(),
            )
            if not existing or not existing.get("summary_ai"):
                ai_summary = await summarize_company(
                    symbol=symbol,
                    name=name,
                    description=full_summary,
                )
                if ai_summary:
                    logger.info(f"Generated AI summary for {symbol}: {len(ai_summary)} chars")
        
        # Update symbols with name, sector, and AI summary
        if name or sector or ai_summary:
            await execute(
                """UPDATE symbols SET 
                       name = COALESCE($2, name),
                       sector = COALESCE($3, sector),
                       summary_ai = COALESCE($4, summary_ai),
                       updated_at = NOW()
                   WHERE symbol = $1""",
                symbol.upper(),
                name,
                sector,
                ai_summary,
            )
            logger.info(f"Updated symbol info for {symbol}: name='{name}', sector='{sector}', summary_ai={'yes' if ai_summary else 'no'}")
        
        # Also update stock_suggestions table with name/sector for admin UI display
        await execute(
            """UPDATE stock_suggestions SET
                   company_name = COALESCE($2, company_name),
                   sector = COALESCE($3, sector),
                   summary = COALESCE($4, summary),
                   current_price = $5,
                   ath_price = $6
               WHERE symbol = $1""",
            symbol.upper(),
            name,
            sector,
            full_summary[:1000] if full_summary else None,  # Truncate for storage
            current_price,
            ath_price,
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
        await execute(
            """INSERT INTO dip_state (symbol, current_price, ath_price, dip_percentage, first_seen, last_updated)
               VALUES ($1, $2, $3, $4, NOW(), NOW())
               ON CONFLICT (symbol) DO UPDATE SET
                   current_price = EXCLUDED.current_price,
                   ath_price = EXCLUDED.ath_price,
                   dip_percentage = EXCLUDED.dip_percentage,
                   last_updated = NOW()""",
            symbol.upper(),
            current_price,
            ath_price,
            dip_pct,
        )
        
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
        
        # Step 6: Mark as fetched and invalidate ranking cache
        await execute(
            """UPDATE stock_suggestions 
               SET fetch_status = 'fetched', fetched_at = NOW()
               WHERE symbol = $1""",
            symbol.upper(),
        )
        
        # Invalidate ranking cache so the new stock appears immediately
        ranking_cache = Cache(prefix="ranking", default_ttl=3600)
        deleted = await ranking_cache.invalidate_pattern("*")
        logger.info(f"Completed processing {symbol}, invalidated {deleted} ranking cache keys")
            
    except Exception as e:
        logger.error(f"Error processing approved symbol {symbol}: {e}")
        # Set error status so admin can retry
        await execute(
            """UPDATE stock_suggestions 
               SET fetch_status = 'error', fetch_error = $2
               WHERE symbol = $1""",
            symbol.upper(),
            str(e)[:500],  # Truncate error message
        )


@router.post("/{suggestion_id}/reject", response_model=dict)
async def reject_suggestion(
    suggestion_id: int,
    reason: str = Query(None, description="Rejection reason"),
    admin: TokenData = Depends(require_admin),
):
    """Reject a suggestion (admin only)."""
    # Get suggestion
    suggestion = await fetch_one(
        "SELECT id, symbol, status FROM stock_suggestions WHERE id = $1", suggestion_id
    )

    if not suggestion:
        raise NotFoundError(
            message=f"Suggestion {suggestion_id} not found",
            details={"id": suggestion_id},
        )

    if suggestion["status"] != "pending":
        raise ValidationError(
            message=f"Suggestion is already {suggestion['status']}",
            details={"status": suggestion["status"]},
        )

    # Look up admin user ID from username
    admin_user = await fetch_one(
        "SELECT id FROM auth_user WHERE username = $1",
        admin.sub
    )
    admin_id = admin_user["id"] if admin_user else None

    # Update suggestion status
    await execute(
        """UPDATE stock_suggestions 
           SET status = 'rejected', approved_by = $1, reviewed_at = NOW(), reason = $2
           WHERE id = $3""",
        admin_id,
        reason,
        suggestion_id,
    )

    return {
        "message": f"Rejected suggestion for {suggestion['symbol']}",
        "symbol": suggestion["symbol"],
    }
