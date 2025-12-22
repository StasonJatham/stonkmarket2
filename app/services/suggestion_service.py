"""Stock suggestion service with batch fetching."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import yfinance as yf

from app.core.logging import get_logger
from app.core.rate_limiter import get_yfinance_limiter
from app.database import get_db
from app.repositories import suggestions as suggestions_repo
from app.websocket import get_connection_manager, WSEvent, WSEventType
from app.services.runtime_settings import get_runtime_setting

logger = get_logger("services.suggestions")

# Rate limiting for Yahoo Finance - now using centralized limiter
DEFAULT_BATCH_SIZE = 5  # Default number of suggestions to fetch per batch run


def get_batch_size() -> Optional[int]:
    """Get batch size from settings. Returns None for 'all' mode (0 setting)."""
    batch_size = get_runtime_setting("ai_batch_size", DEFAULT_BATCH_SIZE)
    if batch_size == 0:
        return None  # No limit - process all
    return batch_size


async def fetch_suggestion_data(symbol: str) -> dict:
    """Fetch stock data from Yahoo Finance with rate limiting.

    Returns dict with: name, sector, industry, summary, last_price, price_90d_ago
    """
    result = {
        "name": None,
        "sector": None,
        "industry": None,
        "summary": None,
        "last_price": None,
        "price_90d_ago": None,
        "success": False,
    }

    # Acquire rate limit token
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync(timeout=60.0):
        logger.warning(f"Rate limit timeout for suggestion {symbol}")
        return result

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # Get basic info
        result["name"] = info.get("shortName") or info.get("longName")
        result["sector"] = info.get("sector")
        result["industry"] = info.get("industry")
        result["summary"] = info.get("longBusinessSummary")

        # Get current price
        result["last_price"] = info.get("regularMarketPrice") or info.get(
            "previousClose"
        )

        # Get price from 90 days ago
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=95)  # Extra buffer
            history = ticker.history(start=start_date, end=end_date)

            if not history.empty and len(history) > 0:
                # Get the oldest price in range (approximately 90 days ago)
                result["price_90d_ago"] = float(history["Close"].iloc[0])
                # Update last_price with most recent if available
                if len(history) > 1:
                    result["last_price"] = float(history["Close"].iloc[-1])
        except Exception as e:
            logger.warning(f"Failed to get historical prices for {symbol}: {e}")

        # Consider success if we got at least the name
        result["success"] = result["name"] is not None

    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")

    return result


async def process_pending_suggestions() -> dict:
    """Process pending suggestions that need data fetching.

    This runs as a slow batch job to avoid hitting rate limits.
    """
    manager = get_connection_manager()

    # Broadcast start
    await manager.broadcast_to_admins(
        WSEvent(
            type=WSEventType.FETCH_STARTED,
            message="Processing pending stock suggestions",
            data={"job": "suggestion_fetch"},
        )
    )

    batch_size = get_batch_size()
    with get_db() as conn:
        pending = suggestions_repo.list_pending_for_fetch(conn, limit=batch_size)

    if not pending:
        logger.info("No pending suggestions to fetch")
        return {"processed": 0, "success": 0, "failed": 0}

    results = {"processed": 0, "success": 0, "failed": 0, "symbols": []}

    for i, suggestion in enumerate(pending):
        # Broadcast progress
        await manager.broadcast_to_admins(
            WSEvent(
                type=WSEventType.FETCH_PROGRESS,
                message=f"Fetching {suggestion.symbol}",
                data={
                    "job": "suggestion_fetch",
                    "current": i + 1,
                    "total": len(pending),
                    "symbol": suggestion.symbol,
                },
            )
        )

        logger.info(f"Fetching data for suggestion: {suggestion.symbol}")

        try:
            data = await fetch_suggestion_data(suggestion.symbol)

            with get_db() as conn:
                suggestions_repo.update_fetch_data(
                    conn,
                    suggestion.id,
                    name=data["name"],
                    sector=data["sector"],
                    industry=data["industry"],
                    summary=data["summary"],
                    last_price=data["last_price"],
                    price_90d_ago=data["price_90d_ago"],
                    success=data["success"],
                )

            results["processed"] += 1
            if data["success"]:
                results["success"] += 1
                results["symbols"].append(
                    {"symbol": suggestion.symbol, "status": "success"}
                )
            else:
                results["failed"] += 1
                results["symbols"].append(
                    {"symbol": suggestion.symbol, "status": "failed"}
                )

        except Exception as e:
            logger.error(f"Error processing suggestion {suggestion.symbol}: {e}")
            results["failed"] += 1
            results["symbols"].append(
                {"symbol": suggestion.symbol, "status": "error", "error": str(e)}
            )

        # Rate limit delay between fetches
        if i < len(pending) - 1:
            await asyncio.sleep(FETCH_DELAY_SECONDS)

    # Broadcast completion
    await manager.broadcast_to_admins(
        WSEvent(
            type=WSEventType.FETCH_COMPLETE,
            message=f"Processed {results['processed']} suggestions",
            data={
                "job": "suggestion_fetch",
                **results,
            },
        )
    )

    logger.info(f"Suggestion fetch complete: {results}")
    return results


def suggest_stock(
    symbol: str, voter_identifier: str
) -> tuple[bool, Optional[str], Optional[dict]]:
    """Submit a stock suggestion.

    Args:
        symbol: Yahoo Finance symbol
        voter_identifier: IP or session ID for vote deduplication

    Returns:
        Tuple of (success, error_message, suggestion_data)
    """
    symbol = symbol.upper().strip()

    with get_db() as conn:
        # Check if can suggest
        can_suggest, error = suggestions_repo.can_suggest(conn, symbol)

        if not can_suggest and error:
            return False, error, None

        # Check if already exists (for voting)
        existing = suggestions_repo.get_by_symbol(conn, symbol)

        if existing:
            # Add vote to existing suggestion
            success, vote_error = suggestions_repo.add_vote(
                conn, symbol, voter_identifier
            )
            if not success:
                return False, vote_error, None

            # Refresh and return
            updated = suggestions_repo.get_by_symbol(conn, symbol)
            return (
                True,
                None,
                {
                    "symbol": updated.symbol,
                    "vote_count": updated.vote_count,
                    "status": updated.status,
                    "is_new": False,
                },
            )

        # Create new suggestion
        suggestion = suggestions_repo.create(conn, symbol)

        return (
            True,
            None,
            {
                "symbol": suggestion.symbol,
                "vote_count": suggestion.vote_count,
                "status": suggestion.status,
                "is_new": True,
            },
        )


async def vote_for_suggestion(
    symbol: str, voter_identifier: str
) -> tuple[bool, Optional[str], bool]:
    """Vote for an existing suggestion.

    Args:
        symbol: Stock symbol
        voter_identifier: Fingerprint for deduplication

    Returns:
        Tuple of (success, error_message, was_auto_approved)
    """
    with get_db() as conn:
        success, error, should_auto_approve = suggestions_repo.add_vote(
            conn, symbol, voter_identifier
        )

        if success and should_auto_approve:
            # Trigger auto-approval
            suggestion = suggestions_repo.get_by_symbol(conn, symbol)
            if suggestion:
                approve_success, approve_error, _ = await admin_action(
                    suggestion_id=suggestion.id,
                    action="approve",
                    reason=None,
                    is_auto=True,
                )
                if approve_success:
                    logger.info(
                        f"Auto-approved suggestion {symbol} after meeting criteria"
                    )
                    return True, None, True

        return success, error, False


def get_top_suggestions(limit: int = 10) -> list[dict]:
    """Get top voted suggestions for admin review."""
    with get_db() as conn:
        suggestions = suggestions_repo.get_top_voted(conn, limit=limit)
        return [
            {
                "id": s.id,
                "symbol": s.symbol,
                "name": s.name,
                "sector": s.sector,
                "industry": s.industry,
                "summary": s.summary[:200] + "..."
                if s.summary and len(s.summary) > 200
                else s.summary,
                "vote_count": s.vote_count,
                "last_price": s.last_price,
                "price_change_90d": s.price_change_90d,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in suggestions
        ]


async def admin_action(
    suggestion_id: int,
    action: str,
    reason: Optional[str] = None,
    is_auto: bool = False,
) -> tuple[bool, Optional[str], Optional[dict]]:
    """Process admin action on a suggestion.

    Args:
        suggestion_id: Suggestion ID
        action: 'approve', 'reject', or 'remove'
        reason: Optional reason for rejection/removal
        is_auto: Whether this is an auto-approval (vs manual admin action)

    Returns:
        Tuple of (success, error_message, updated_suggestion)
    """
    manager = get_connection_manager()

    with get_db() as conn:
        suggestion = suggestions_repo.get_by_id(conn, suggestion_id)

        if not suggestion:
            return False, "Suggestion not found", None

        if action == "approve":
            updated = suggestions_repo.approve(conn, suggestion_id)
            event_type = WSEventType.SUGGESTION_APPROVED

        elif action == "reject":
            updated = suggestions_repo.reject(conn, suggestion_id, reason)
            event_type = WSEventType.SUGGESTION_REJECTED

        elif action == "remove":
            updated = suggestions_repo.remove(conn, suggestion_id, reason)
            event_type = WSEventType.SUGGESTION_REJECTED

        else:
            return False, f"Unknown action: {action}", None

        # Broadcast to admins
        await manager.broadcast_to_admins(
            WSEvent(
                type=event_type,
                message=f"Suggestion {updated.symbol} {action}d",
                data={
                    "symbol": updated.symbol,
                    "action": action,
                    "reason": reason,
                },
            )
        )

        return (
            True,
            None,
            {
                "id": updated.id,
                "symbol": updated.symbol,
                "status": updated.status,
            },
        )
