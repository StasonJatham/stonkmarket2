"""Auth user repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from app.database.models import AuthUser


def get_user(conn: sqlite3.Connection, username: str) -> Optional[AuthUser]:
    """Get a user by username."""
    cur = conn.execute(
        "SELECT username, password_hash, updated_at FROM auth_user WHERE username = ?",
        (username.lower(),),
    )
    row = cur.fetchone()
    if row:
        return AuthUser.from_row(row)
    return None


def upsert_user(conn: sqlite3.Connection, username: str, password_hash: str) -> AuthUser:
    """Create or update a user."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO auth_user(username, password_hash, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(username) DO UPDATE SET 
            password_hash=excluded.password_hash, 
            updated_at=excluded.updated_at
        """,
        (username.lower(), password_hash, now),
    )
    conn.commit()
    return get_user(conn, username)  # type: ignore


def delete_user(conn: sqlite3.Connection, username: str) -> bool:
    """Delete a user."""
    cur = conn.execute("DELETE FROM auth_user WHERE username = ?", (username.lower(),))
    conn.commit()
    return cur.rowcount > 0


def any_user(conn: sqlite3.Connection) -> Optional[AuthUser]:
    """Get any user (for checking if users exist)."""
    cur = conn.execute("SELECT username, password_hash, updated_at FROM auth_user LIMIT 1")
    row = cur.fetchone()
    if row:
        return AuthUser.from_row(row)
    return None
