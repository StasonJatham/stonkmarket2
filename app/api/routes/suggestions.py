"""Stock suggestion API routes - PostgreSQL async version."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request

from app.api.dependencies import require_admin, get_request_fingerprint
from app.core.exceptions import ValidationError, NotFoundError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.database.connection import fetch_all, fetch_one, execute

logger = get_logger("api.routes.suggestions")

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

# Vote cooldown in days
VOTE_COOLDOWN_DAYS = 7


def _hash_fingerprint(fingerprint: str) -> str:
    """Hash fingerprint for storage."""
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:32]


# =============================================================================
# Public endpoints
# =============================================================================


@router.post("", response_model=dict, status_code=201)
async def suggest_stock(
    request: Request,
    symbol: str = Query(..., min_length=1, max_length=20, description="Stock symbol"),
):
    """
    Suggest a new stock to be tracked.

    Public endpoint - no auth required.
    If stock already suggested, adds a vote instead.
    """
    symbol = symbol.strip().upper()
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

        return {
            "message": "Vote added successfully",
            "symbol": symbol,
            "vote_count": new_score,
            "status": existing["status"],
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

    return {"message": "Vote recorded", "symbol": symbol}


@router.get("/top", response_model=List[dict])
async def get_top_suggestions(
    limit: int = Query(10, ge=1, le=50),
):
    """
    Get top voted pending suggestions.

    Public endpoint - shows what stocks the community wants tracked.
    """
    rows = await fetch_all(
        """SELECT symbol, company_name as name, vote_score as vote_count, 
                  NULL as sector, NULL as summary
           FROM stock_suggestions 
           WHERE status = 'pending'
           ORDER BY vote_score DESC
           LIMIT $1""",
        limit,
    )

    return [
        {
            "symbol": row["symbol"],
            "name": row["name"],
            "vote_count": row["vote_count"],
            "sector": row["sector"],
            "summary": row["summary"],
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
