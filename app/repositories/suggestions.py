"""Repository for stock suggestions."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.database.models import StockSuggestion, SuggestionVote

logger = get_logger("repositories.suggestions")

# Removed stocks can be re-suggested after this many days
REMOVED_COOLDOWN_DAYS = 90


def get_by_symbol(conn: sqlite3.Connection, symbol: str) -> Optional[StockSuggestion]:
    """Get a suggestion by symbol."""
    row = conn.execute(
        "SELECT * FROM stock_suggestions WHERE symbol = ?",
        (symbol.upper(),)
    ).fetchone()
    return StockSuggestion.from_row(row) if row else None


def get_by_id(conn: sqlite3.Connection, suggestion_id: int) -> Optional[StockSuggestion]:
    """Get a suggestion by ID."""
    row = conn.execute(
        "SELECT * FROM stock_suggestions WHERE id = ?",
        (suggestion_id,)
    ).fetchone()
    return StockSuggestion.from_row(row) if row else None


def list_suggestions(
    conn: sqlite3.Connection,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    order_by: str = "vote_count",
    order_dir: str = "DESC",
) -> tuple[list[StockSuggestion], int]:
    """List suggestions with optional status filter."""
    # Build query
    where_clause = ""
    params: list = []
    
    if status:
        where_clause = "WHERE status = ?"
        params.append(status)
    
    # Get total count
    count_row = conn.execute(
        f"SELECT COUNT(*) as cnt FROM stock_suggestions {where_clause}",
        params
    ).fetchone()
    total = count_row["cnt"] if count_row else 0
    
    # Get paginated results
    allowed_order = {"vote_count", "created_at", "updated_at", "symbol"}
    if order_by not in allowed_order:
        order_by = "vote_count"
    order_dir = "DESC" if order_dir.upper() == "DESC" else "ASC"
    
    offset = (page - 1) * page_size
    rows = conn.execute(
        f"""SELECT * FROM stock_suggestions {where_clause}
            ORDER BY {order_by} {order_dir}
            LIMIT ? OFFSET ?""",
        params + [page_size, offset]
    ).fetchall()
    
    return [StockSuggestion.from_row(r) for r in rows], total


def list_pending_for_fetch(conn: sqlite3.Connection, limit: int = 10) -> list[StockSuggestion]:
    """Get pending suggestions that need data fetching."""
    rows = conn.execute(
        """SELECT * FROM stock_suggestions 
           WHERE status = 'pending' AND fetched_at IS NULL
           ORDER BY created_at ASC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    return [StockSuggestion.from_row(r) for r in rows]


def get_top_voted(conn: sqlite3.Connection, limit: int = 10) -> list[StockSuggestion]:
    """Get top voted pending suggestions for admin review."""
    rows = conn.execute(
        """SELECT * FROM stock_suggestions 
           WHERE status = 'pending'
           ORDER BY vote_count DESC, created_at ASC
           LIMIT ?""",
        (limit,)
    ).fetchall()
    return [StockSuggestion.from_row(r) for r in rows]


def create(conn: sqlite3.Connection, symbol: str) -> StockSuggestion:
    """Create a new suggestion."""
    now = datetime.utcnow().isoformat()
    cursor = conn.execute(
        """INSERT INTO stock_suggestions (symbol, status, vote_count, created_at)
           VALUES (?, 'pending', 1, ?)""",
        (symbol.upper(), now)
    )
    conn.commit()
    
    return get_by_id(conn, cursor.lastrowid)


def can_suggest(conn: sqlite3.Connection, symbol: str) -> tuple[bool, Optional[str]]:
    """Check if a symbol can be suggested.
    
    Returns:
        Tuple of (can_suggest, error_reason)
    """
    symbol = symbol.upper()
    
    existing = get_by_symbol(conn, symbol)
    if not existing:
        return True, None
    
    if existing.status == "approved":
        return False, "This stock is already being tracked"
    
    if existing.status == "pending":
        return False, "This stock has already been suggested and is pending review"
    
    if existing.status == "removed":
        # Check if cooldown has passed
        if existing.removed_at:
            cooldown_end = existing.removed_at + timedelta(days=REMOVED_COOLDOWN_DAYS)
            if datetime.utcnow() < cooldown_end:
                days_left = (cooldown_end - datetime.utcnow()).days
                return False, f"This stock was previously reviewed. It can be suggested again in {days_left} days"
            # Cooldown passed - reset the suggestion
            conn.execute(
                """UPDATE stock_suggestions 
                   SET status = 'pending', vote_count = 1, 
                       rejection_reason = NULL, removed_at = NULL,
                       updated_at = ?, fetched_at = NULL
                   WHERE id = ?""",
                (datetime.utcnow().isoformat(), existing.id)
            )
            conn.commit()
            return False, None  # Not an error, just already handled
    
    if existing.status == "rejected":
        return False, "This stock was reviewed and is not available for tracking"
    
    return True, None


def add_vote(
    conn: sqlite3.Connection,
    symbol: str,
    voter_identifier: str,
) -> tuple[bool, Optional[str]]:
    """Add a vote for a suggestion.
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        voter_identifier: Fingerprint hash for deduplication
        
    Returns:
        Tuple of (success, error_message, should_auto_approve)
    """
    symbol = symbol.upper()
    suggestion = get_by_symbol(conn, symbol)
    
    if not suggestion:
        return False, "Suggestion not found", False
    
    if suggestion.status != "pending":
        return False, "Can only vote on pending suggestions", False
    
    # Hash the voter identifier for privacy
    voter_hash = hashlib.sha256(voter_identifier.encode()).hexdigest()[:32]
    
    # Check if user has voted for this symbol within cooldown period
    cooldown_days = settings.vote_cooldown_days
    cooldown_start = (datetime.utcnow() - timedelta(days=cooldown_days)).isoformat()
    existing_vote = conn.execute(
        """SELECT id, created_at FROM suggestion_votes 
           WHERE suggestion_id = ? AND voter_hash = ? AND created_at > ?
           ORDER BY created_at DESC LIMIT 1""",
        (suggestion.id, voter_hash, cooldown_start)
    ).fetchone()
    
    if existing_vote:
        # Calculate days remaining in cooldown
        vote_date = datetime.fromisoformat(existing_vote["created_at"])
        cooldown_end = vote_date + timedelta(days=cooldown_days)
        days_left = (cooldown_end - datetime.utcnow()).days + 1
        return False, f"You can vote for this stock again in {days_left} day{'s' if days_left != 1 else ''}", False
    
    # Record the vote
    conn.execute(
        """INSERT INTO suggestion_votes (suggestion_id, voter_hash, created_at)
           VALUES (?, ?, ?)""",
        (suggestion.id, voter_hash, datetime.utcnow().isoformat())
    )
    # Increment vote count
    conn.execute(
        "UPDATE stock_suggestions SET vote_count = vote_count + 1 WHERE id = ?",
        (suggestion.id,)
    )
    conn.commit()
    
    # Check if should auto-approve
    should_auto_approve = check_auto_approve_eligibility(conn, suggestion.id)
    
    return True, None, should_auto_approve


def get_unique_voter_count(conn: sqlite3.Connection, suggestion_id: int) -> int:
    """Get count of unique voters for a suggestion."""
    result = conn.execute(
        "SELECT COUNT(DISTINCT voter_hash) FROM suggestion_votes WHERE suggestion_id = ?",
        (suggestion_id,)
    ).fetchone()
    return result[0] if result else 0


def check_auto_approve_eligibility(conn: sqlite3.Connection, suggestion_id: int) -> bool:
    """
    Check if a suggestion meets auto-approval criteria.
    
    All conditions must be met:
    1. Auto-approve is enabled in settings
    2. Vote count >= configured threshold
    3. Unique voters >= configured threshold
    4. Suggestion age >= configured minimum hours
    """
    if not settings.auto_approve_enabled:
        return False
    
    suggestion = get_by_id(conn, suggestion_id)
    if not suggestion or suggestion.status != "pending":
        return False
    
    # Check vote count
    if suggestion.vote_count < settings.auto_approve_votes:
        return False
    
    # Check unique voters
    unique_voters = get_unique_voter_count(conn, suggestion_id)
    if unique_voters < settings.auto_approve_unique_voters:
        return False
    
    # Check age
    if suggestion.created_at:
        age_hours = (datetime.utcnow() - suggestion.created_at).total_seconds() / 3600
        if age_hours < settings.auto_approve_min_age_hours:
            return False
    
    return True


def update_fetch_data(
    conn: sqlite3.Connection,
    suggestion_id: int,
    name: Optional[str],
    sector: Optional[str],
    industry: Optional[str],
    summary: Optional[str],
    last_price: Optional[float],
    price_90d_ago: Optional[float],
    success: bool,
) -> None:
    """Update suggestion with fetched data."""
    now = datetime.utcnow().isoformat()
    
    price_change = None
    if last_price and price_90d_ago and price_90d_ago > 0:
        price_change = ((last_price - price_90d_ago) / price_90d_ago) * 100
    
    status = "pending" if success else "fetch_failed"
    
    conn.execute(
        """UPDATE stock_suggestions 
           SET name = ?, sector = ?, industry = ?, summary = ?,
               last_price = ?, price_90d_ago = ?, price_change_90d = ?,
               status = ?, fetched_at = ?, updated_at = ?
           WHERE id = ?""",
        (name, sector, industry, summary, last_price, price_90d_ago,
         price_change, status, now, now, suggestion_id)
    )
    conn.commit()


def approve(conn: sqlite3.Connection, suggestion_id: int) -> Optional[StockSuggestion]:
    """Approve a suggestion."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        """UPDATE stock_suggestions 
           SET status = 'approved', updated_at = ?
           WHERE id = ?""",
        (now, suggestion_id)
    )
    conn.commit()
    return get_by_id(conn, suggestion_id)


def reject(
    conn: sqlite3.Connection,
    suggestion_id: int,
    reason: Optional[str] = None
) -> Optional[StockSuggestion]:
    """Reject a suggestion."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        """UPDATE stock_suggestions 
           SET status = 'rejected', rejection_reason = ?, updated_at = ?
           WHERE id = ?""",
        (reason or "Does not meet our criteria", now, suggestion_id)
    )
    conn.commit()
    return get_by_id(conn, suggestion_id)


def remove(
    conn: sqlite3.Connection,
    suggestion_id: int,
    reason: Optional[str] = None
) -> Optional[StockSuggestion]:
    """Remove/soft-delete a suggestion (was approved but now removed)."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        """UPDATE stock_suggestions 
           SET status = 'removed', rejection_reason = ?, 
               removed_at = ?, updated_at = ?
           WHERE id = ?""",
        (reason or "No longer tracked", now, now, suggestion_id)
    )
    conn.commit()
    return get_by_id(conn, suggestion_id)


def cleanup_old_removed(conn: sqlite3.Connection) -> int:
    """Clean up suggestions that have been removed for over 90 days.
    
    Resets them to allow re-suggestion.
    """
    cutoff = (datetime.utcnow() - timedelta(days=REMOVED_COOLDOWN_DAYS)).isoformat()
    cursor = conn.execute(
        """UPDATE stock_suggestions 
           SET status = 'pending', vote_count = 0, rejection_reason = NULL,
               removed_at = NULL, updated_at = ?
           WHERE status = 'removed' AND removed_at < ?""",
        (datetime.utcnow().isoformat(), cutoff)
    )
    conn.commit()
    return cursor.rowcount
