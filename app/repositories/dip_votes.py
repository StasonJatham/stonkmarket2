"""Dip voting repository - buy/sell/skip votes on current dips."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from app.core.config import settings
from app.database.models import DipVote, DipAIAnalysis


def get_vote_cooldown_remaining(
    conn: sqlite3.Connection,
    symbol: str,
    voter_hash: str,
) -> Optional[int]:
    """
    Check if user is on cooldown for voting on this symbol.
    
    Returns:
        Seconds remaining on cooldown, or None if can vote
    """
    cooldown_days = settings.vote_cooldown_days
    cutoff = (datetime.utcnow() - timedelta(days=cooldown_days)).isoformat()
    
    cur = conn.execute(
        """
        SELECT created_at FROM dip_votes
        WHERE symbol = ? AND voter_hash = ? AND created_at > ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (symbol.upper(), voter_hash, cutoff),
    )
    row = cur.fetchone()
    
    if not row:
        return None
    
    last_vote = datetime.fromisoformat(row["created_at"])
    cooldown_end = last_vote + timedelta(days=cooldown_days)
    remaining = (cooldown_end - datetime.utcnow()).total_seconds()
    
    return int(remaining) if remaining > 0 else None


def add_vote(
    conn: sqlite3.Connection,
    symbol: str,
    voter_hash: str,
    vote_type: str,
) -> tuple[bool, Optional[str]]:
    """
    Add a vote for a dip.
    
    Args:
        conn: Database connection
        symbol: Stock symbol
        voter_hash: Hashed voter identifier
        vote_type: 'buy', 'sell', or 'skip'
        
    Returns:
        Tuple of (success, error_message)
    """
    if vote_type not in ("buy", "sell", "skip"):
        return False, "Invalid vote type. Must be 'buy', 'sell', or 'skip'"
    
    # Check cooldown
    cooldown = get_vote_cooldown_remaining(conn, symbol, voter_hash)
    if cooldown:
        hours = cooldown // 3600
        minutes = (cooldown % 3600) // 60
        return False, f"Vote cooldown active. Try again in {hours}h {minutes}m"
    
    # Check symbol exists in dip_state
    cur = conn.execute(
        "SELECT symbol FROM dip_state WHERE symbol = ?",
        (symbol.upper(),),
    )
    if not cur.fetchone():
        return False, f"Symbol {symbol.upper()} is not currently in a dip"
    
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO dip_votes(symbol, voter_hash, vote_type, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (symbol.upper(), voter_hash, vote_type, now),
    )
    conn.commit()
    
    return True, None


def get_vote_counts(conn: sqlite3.Connection, symbol: str) -> dict[str, int]:
    """Get vote counts for a symbol."""
    cur = conn.execute(
        """
        SELECT vote_type, COUNT(*) as count
        FROM dip_votes
        WHERE symbol = ?
        GROUP BY vote_type
        """,
        (symbol.upper(),),
    )
    
    counts = {"buy": 0, "sell": 0, "skip": 0}
    for row in cur.fetchall():
        counts[row["vote_type"]] = row["count"]
    
    return counts


def get_all_vote_counts(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Get vote counts for all symbols with votes."""
    cur = conn.execute(
        """
        SELECT symbol, vote_type, COUNT(*) as count
        FROM dip_votes
        GROUP BY symbol, vote_type
        """
    )
    
    result: dict[str, dict[str, int]] = {}
    for row in cur.fetchall():
        symbol = row["symbol"]
        if symbol not in result:
            result[symbol] = {"buy": 0, "sell": 0, "skip": 0}
        result[symbol][row["vote_type"]] = row["count"]
    
    return result


def get_user_votes(
    conn: sqlite3.Connection,
    voter_hash: str,
    limit: int = 50,
) -> list[DipVote]:
    """Get recent votes by a user."""
    cur = conn.execute(
        """
        SELECT id, symbol, voter_hash, vote_type, created_at
        FROM dip_votes
        WHERE voter_hash = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (voter_hash, limit),
    )
    return [DipVote.from_row(row) for row in cur.fetchall()]


# --- AI Analysis cache ---

def get_ai_analysis(conn: sqlite3.Connection, symbol: str) -> Optional[DipAIAnalysis]:
    """Get cached AI analysis for a symbol."""
    cur = conn.execute(
        """
        SELECT symbol, tinder_bio, ai_rating, ai_reasoning, analysis_data, created_at, expires_at
        FROM dip_ai_analysis
        WHERE symbol = ? AND expires_at > ?
        """,
        (symbol.upper(), datetime.utcnow().isoformat()),
    )
    row = cur.fetchone()
    if row:
        return DipAIAnalysis.from_row(row)
    return None


def upsert_ai_analysis(
    conn: sqlite3.Connection,
    symbol: str,
    tinder_bio: Optional[str] = None,
    ai_rating: Optional[str] = None,
    ai_reasoning: Optional[str] = None,
    analysis_data: Optional[str] = None,
    expires_hours: int = 24,
) -> DipAIAnalysis:
    """Create or update AI analysis for a symbol."""
    now = datetime.utcnow()
    expires = now + timedelta(hours=expires_hours)
    
    conn.execute(
        """
        INSERT INTO dip_ai_analysis(symbol, tinder_bio, ai_rating, ai_reasoning, analysis_data, created_at, expires_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            tinder_bio = excluded.tinder_bio,
            ai_rating = excluded.ai_rating,
            ai_reasoning = excluded.ai_reasoning,
            analysis_data = excluded.analysis_data,
            created_at = excluded.created_at,
            expires_at = excluded.expires_at
        """,
        (symbol.upper(), tinder_bio, ai_rating, ai_reasoning, analysis_data, now.isoformat(), expires.isoformat()),
    )
    conn.commit()
    
    return get_ai_analysis(conn, symbol) or DipAIAnalysis(symbol=symbol.upper())


def delete_expired_analyses(conn: sqlite3.Connection) -> int:
    """Delete expired AI analyses."""
    cur = conn.execute(
        "DELETE FROM dip_ai_analysis WHERE expires_at < ?",
        (datetime.utcnow().isoformat(),),
    )
    conn.commit()
    return cur.rowcount
