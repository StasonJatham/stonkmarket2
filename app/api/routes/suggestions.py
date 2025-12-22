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

from app.api.dependencies import require_admin, get_request_fingerprint
from app.core.exceptions import ValidationError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.database.connection import fetch_all, fetch_one, execute
from app.services.runtime_settings import get_runtime_setting

logger = get_logger("api.routes.suggestions")

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

# Vote cooldown in days
VOTE_COOLDOWN_DAYS = 7

# Symbol validation pattern: 1-10 chars, alphanumeric + dot only
SYMBOL_PATTERN = re.compile(r'^[A-Z0-9.]{1,10}$')

# Thread pool for blocking yfinance calls
_executor = ThreadPoolExecutor(max_workers=2)


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


def _hash_fingerprint(fingerprint: str) -> str:
    """Hash fingerprint for storage."""
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:32]


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
    """
    # Validate symbol format
    symbol = _validate_symbol_format(symbol)
    fingerprint = get_request_fingerprint(request)
    fingerprint_hash = _hash_fingerprint(fingerprint)

    # Check if suggestion already exists
    existing = await fetch_one(
        "SELECT id, symbol, status, vote_score FROM stock_suggestions WHERE symbol = $1",
        symbol,
    )

    if existing:
        # Already exists - try to add a vote
        suggestion_id = existing["id"]

        # Check cooldown
        cooldown_start = datetime.utcnow() - timedelta(days=VOTE_COOLDOWN_DAYS)
        recent_vote = await fetch_one(
            """SELECT id FROM suggestion_votes 
               WHERE suggestion_id = $1 AND fingerprint = $2 AND created_at > $3""",
            suggestion_id,
            fingerprint_hash,
            cooldown_start,
        )

        if recent_vote:
            raise ValidationError(
                message="You already voted for this stock recently",
                details={"symbol": symbol, "cooldown_days": VOTE_COOLDOWN_DAYS},
            )

        # Add vote
        await execute(
            """INSERT INTO suggestion_votes (suggestion_id, fingerprint, vote_type, created_at)
               VALUES ($1, $2, 'up', NOW())""",
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

    # Create new suggestion
    result = await fetch_one(
        """INSERT INTO stock_suggestions (symbol, fingerprint, status, vote_score, created_at)
           VALUES ($1, $2, 'pending', 1, NOW())
           RETURNING id, vote_score""",
        symbol,
        fingerprint_hash,
    )

    # Add initial vote
    await execute(
        """INSERT INTO suggestion_votes (suggestion_id, fingerprint, vote_type, created_at)
           VALUES ($1, $2, 'up', NOW())""",
        result["id"],
        fingerprint_hash,
    )

    return {
        "message": "Stock suggested successfully",
        "symbol": symbol,
        "vote_count": 1,
        "status": "pending",
    }


@router.put("/{symbol}/vote", response_model=dict)
async def vote_for_suggestion(
    request: Request,
    symbol: str,
    background_tasks: BackgroundTasks,
):
    """
    Vote for an existing stock suggestion.

    Public endpoint - no auth required.
    Vote cooldown: 7 days per stock per user.
    """
    symbol = symbol.strip().upper()
    fingerprint = get_request_fingerprint(request)
    fingerprint_hash = _hash_fingerprint(fingerprint)

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

    # Check cooldown
    cooldown_start = datetime.utcnow() - timedelta(days=VOTE_COOLDOWN_DAYS)
    recent_vote = await fetch_one(
        """SELECT id FROM suggestion_votes 
           WHERE suggestion_id = $1 AND fingerprint = $2 AND created_at > $3""",
        suggestion_id,
        fingerprint_hash,
        cooldown_start,
    )

    if recent_vote:
        raise ValidationError(
            message="You already voted for this stock recently",
            details={"symbol": symbol, "cooldown_days": VOTE_COOLDOWN_DAYS},
        )

    # Add vote
    await execute(
        """INSERT INTO suggestion_votes (suggestion_id, fingerprint, vote_type, created_at)
           VALUES ($1, $2, 'up', NOW())""",
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
    """
    # Get user fingerprint hash
    fingerprint = get_request_fingerprint(request)
    fingerprint_hash = _hash_fingerprint(fingerprint)
    
    # If exclude_voted, get suggestions the user has already voted on
    voted_symbols = set()
    if exclude_voted:
        cooldown_start = datetime.utcnow() - timedelta(days=VOTE_COOLDOWN_DAYS)
        voted_rows = await fetch_all(
            """SELECT DISTINCT ss.symbol 
               FROM suggestion_votes sv
               JOIN stock_suggestions ss ON sv.suggestion_id = ss.id
               WHERE sv.fingerprint = $1 AND sv.created_at > $2""",
            fingerprint_hash,
            cooldown_start,
        )
        voted_symbols = {row["symbol"] for row in voted_rows}
    
    rows = await fetch_all(
        """SELECT symbol, company_name as name, vote_score as vote_count, 
                  NULL as sector, NULL as summary
           FROM stock_suggestions 
           WHERE status = 'pending'
           ORDER BY vote_score DESC
           LIMIT $1""",
        limit + len(voted_symbols),  # Fetch extra to account for filtering
    )
    
    # Filter out voted symbols
    if voted_symbols:
        rows = [row for row in rows if row["symbol"] not in voted_symbols][:limit]

    # Fetch IPO years and websites in parallel using thread pool
    loop = asyncio.get_event_loop()
    symbols = [row["symbol"] for row in rows]
    stock_infos = await asyncio.gather(*[
        loop.run_in_executor(_executor, _get_stock_info, symbol)
        for symbol in symbols
    ])
    stock_info_map = dict(zip(symbols, stock_infos))

    return [
        {
            "symbol": row["symbol"],
            "name": row["name"],
            "vote_count": row["vote_count"],
            "sector": row["sector"],
            "summary": row["summary"],
            "ipo_year": stock_info_map.get(row["symbol"], {}).get("ipo_year"),
            "website": stock_info_map.get(row["symbol"], {}).get("website"),
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
        """SELECT id, symbol, company_name, status, vote_score, created_at
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
        f"""SELECT id, symbol, company_name, status, vote_score, 
                   approved_by, created_at, reviewed_at
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


async def _process_approved_symbol(symbol: str) -> None:
    """Background task to process newly approved symbol.
    
    Fetches Yahoo Finance data and generates AI content.
    """
    from app.services.stock_info import get_stock_info_async
    from app.services.openai_batch import generate_dip_bio_realtime, rate_dip_realtime
    from app.repositories import dip_votes as dip_votes_repo
    
    logger.info(f"Processing newly approved symbol: {symbol}")
    
    try:
        # Step 1: Fetch Yahoo Finance data
        info = await get_stock_info_async(symbol)
        if not info:
            logger.warning(f"Could not fetch Yahoo data for {symbol}")
            return
            
        current_price = info.get("current_price", 0)
        ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
        dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0
        
        logger.info(f"Fetched data for {symbol}: price=${current_price}, ATH=${ath_price}, dip={dip_pct:.1f}%")
        
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
