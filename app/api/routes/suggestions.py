"""Stock suggestion API routes - PostgreSQL async version."""

from __future__ import annotations

import asyncio
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Optional

import yfinance as yf
from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request

from app.api.dependencies import require_admin
from app.core.exceptions import ValidationError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.core.anon_session import (
    get_device_fingerprint,
    check_can_vote,
    record_vote,
    check_can_suggest,
    record_suggestion,
    get_voted_symbols,
)
from app.database.connection import fetch_all, fetch_one, execute
from app.services.runtime_settings import get_runtime_setting

logger = get_logger("api.routes.suggestions")

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

# Vote cooldown in days (now managed in anon_session.py)
VOTE_COOLDOWN_DAYS = 7

# Symbol validation pattern: 1-10 chars, alphanumeric + dot only
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.]{1,10}$')

# Thread pool for blocking yfinance calls
_executor = ThreadPoolExecutor(max_workers=2)

# Rate limit error messages to detect
RATE_LIMIT_INDICATORS = ["rate limit", "too many requests", "429"]


def _get_ipo_year(symbol: str) -> Optional[int]:
    """Get IPO/first trade year for a symbol from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            first_trade_date = datetime.utcfromtimestamp(first_trade_ms / 1000)
            return first_trade_date.year
    except Exception as e:
        logger.debug(f"Failed to get IPO year for {symbol}: {e}")
    return None


def _get_stock_info_full(symbol: str) -> dict:
    """
    Get comprehensive stock info from Yahoo Finance for suggestions.
    
    Returns:
        dict with keys:
        - valid: bool - whether symbol is valid
        - name: str | None
        - sector: str | None
        - summary: str | None
        - website: str | None
        - ipo_year: int | None
        - current_price: float | None
        - ath_price: float | None (52-week high as proxy)
        - fetch_status: 'fetched' | 'rate_limited' | 'error' | 'invalid'
        - fetch_error: str | None
    """
    result = {
        "valid": False,
        "name": None,
        "sector": None,
        "summary": None,
        "website": None,
        "ipo_year": None,
        "current_price": None,
        "ath_price": None,
        "fetch_status": "error",
        "fetch_error": None,
    }
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Check if valid symbol (must have at least a name or price)
        name = info.get("shortName") or info.get("longName")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        
        if not name and not current_price:
            result["fetch_status"] = "invalid"
            result["fetch_error"] = "Symbol not found on Yahoo Finance"
            return result
        
        result["valid"] = True
        result["fetch_status"] = "fetched"
        result["name"] = name
        result["sector"] = info.get("sector")
        result["summary"] = info.get("longBusinessSummary")
        result["website"] = info.get("website")
        result["current_price"] = current_price
        result["ath_price"] = info.get("fiftyTwoWeekHigh")
        
        # Get IPO year
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            first_trade_date = datetime.utcfromtimestamp(first_trade_ms / 1000)
            result["ipo_year"] = first_trade_date.year
            
    except Exception as e:
        error_str = str(e).lower()
        # Check if rate limited
        if any(indicator in error_str for indicator in RATE_LIMIT_INDICATORS):
            result["fetch_status"] = "rate_limited"
            result["fetch_error"] = "Yahoo Finance rate limit reached. Will retry automatically."
            logger.warning(f"Rate limited fetching {symbol}: {e}")
        else:
            result["fetch_status"] = "error"
            result["fetch_error"] = str(e) if str(e) else "Failed to fetch stock info"
            logger.debug(f"Failed to get stock info for {symbol}: {e}")
    
    return result


def _get_stock_info(symbol: str) -> dict:
    """Get IPO year and website for a symbol from Yahoo Finance."""
    result = {"ipo_year": None, "website": None}
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        # Get IPO year
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            first_trade_date = datetime.utcfromtimestamp(first_trade_ms / 1000)
            result["ipo_year"] = first_trade_date.year
        
        # Get website
        result["website"] = info.get("website")
    except Exception as e:
        logger.debug(f"Failed to get stock info for {symbol}: {e}")
    return result


def _validate_symbol_format(symbol: str) -> str:
    """Validate and normalize symbol format."""
    normalized = symbol.strip().upper()
    if not normalized:
        raise ValidationError(
            message="Symbol cannot be empty",
            details={"symbol": symbol}
        )
    if len(normalized) > 10:
        raise ValidationError(
            message="Symbol must be 10 characters or less",
            details={"symbol": symbol, "max_length": 10}
        )
    if not SYMBOL_PATTERN.match(normalized):
        raise ValidationError(
            message="Symbol can only contain letters, numbers, and dots",
            details={"symbol": symbol}
        )
    return normalized


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
):
    """
    Suggest a new stock to be tracked.

    Public endpoint - no auth required.
    If stock already suggested, adds a vote instead.
    Uses multi-layer fingerprinting for abuse prevention.
    """
    # Validate symbol format
    symbol = _validate_symbol_format(symbol)
    
    # Get device fingerprint from header (sent by frontend)
    device_id = get_device_fingerprint(request)
    if not device_id:
        raise ValidationError(
            message="Device fingerprint required",
            details={"hint": "Please enable cookies and try again"}
        )
    
    # Hash for database storage
    fingerprint_hash = hashlib.sha256(device_id.encode()).hexdigest()[:32]

    # Check if suggestion already exists
    existing = await fetch_one(
        "SELECT id, symbol, status, vote_score FROM stock_suggestions WHERE symbol = $1",
        symbol,
    )

    if existing:
        # Already exists - try to add a vote
        # Check if this device/IP can vote (Valkey-based session)
        can_vote, reason = await check_can_vote(device_id, symbol, request)
        if not can_vote:
            raise ValidationError(message=reason, details={"symbol": symbol})

        suggestion_id = existing["id"]

        # Record vote in Valkey (for device/IP tracking)
        await record_vote(device_id, symbol, request)

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
    can_suggest, reason = await check_can_suggest(device_id, request)
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

    # Record suggestion count in session (for rate limiting)
    await record_suggestion(device_id, request)
    
    # Record vote in Valkey (suggesting also counts as voting)
    await record_vote(device_id, symbol, request)

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
):
    """
    Vote for an existing stock suggestion.

    Public endpoint - no auth required.
    Uses multi-layer fingerprinting: client device ID + IP address.
    Vote cooldown: 7 days per stock per user/device/IP.
    """
    symbol = symbol.strip().upper()
    
    # Get device fingerprint from header (sent by frontend)
    device_id = get_device_fingerprint(request)
    if not device_id:
        raise ValidationError(
            message="Device fingerprint required",
            details={"hint": "Please enable cookies and try again"}
        )

    # Check if this device/IP can vote (Valkey-based session)
    can_vote, reason = await check_can_vote(device_id, symbol, request)
    if not can_vote:
        raise ValidationError(message=reason, details={"symbol": symbol})

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

    # Record vote in Valkey (for device/IP tracking)
    await record_vote(device_id, symbol, request)

    # Also store in database for historical record (using hashed device_id)
    fingerprint_hash = hashlib.sha256(device_id.encode()).hexdigest()[:32]
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
    # Get voted symbols from Valkey session
    voted_symbols = set()
    if exclude_voted:
        device_id = get_device_fingerprint(request)
        if device_id:
            voted_symbols = await get_voted_symbols(device_id, request)
    
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
    """
    from datetime import date, timedelta
    from app.services.stock_info import get_stock_info_async
    from app.services.openai_batch import generate_dip_bio_realtime, rate_dip_realtime
    from app.repositories import dip_votes as dip_votes_repo
    from app.dipfinder.service import get_dipfinder_service
    
    logger.info(f"Processing newly approved symbol: {symbol}")
    
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
        
        # Step 1.5: Update symbols table with name and sector
        name = info.get("name") or info.get("short_name")
        sector = info.get("sector")
        if name or sector:
            await execute(
                """UPDATE symbols SET 
                       name = COALESCE($2, name),
                       sector = COALESCE($3, sector),
                       updated_at = NOW()
                   WHERE symbol = $1""",
                symbol.upper(),
                name,
                sector,
            )
            logger.info(f"Updated symbol info for {symbol}: name='{name}', sector='{sector}'")
        
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
        bio = await generate_dip_bio_realtime(
            symbol=symbol,
            current_price=current_price,
            ath_price=ath_price,
            dip_percentage=dip_pct,
        )
        
        # Step 4: Generate AI rating
        rating_data = await rate_dip_realtime(
            symbol=symbol,
            current_price=current_price,
            ath_price=ath_price,
            dip_percentage=dip_pct,
        )
        
        # Step 5: Store AI analysis
        if bio or rating_data:
            await dip_votes_repo.upsert_ai_analysis(
                symbol=symbol,
                tinder_bio=bio,
                ai_rating=rating_data.get("rating") if rating_data else None,
                ai_reasoning=rating_data.get("reasoning") if rating_data else None,
                is_batch=False,
            )
            logger.info(f"Generated AI content for {symbol}: bio={'yes' if bio else 'no'}, rating={rating_data.get('rating') if rating_data else 'none'}")
        else:
            logger.warning(f"No AI content generated for {symbol}")
            
    except Exception as e:
        logger.error(f"Error processing approved symbol {symbol}: {e}")


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
