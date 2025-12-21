"""Auth user repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from app.database.models import AuthUser


def get_user(conn: sqlite3.Connection, username: str) -> Optional[AuthUser]:
    """Get a user by username."""
    cur = conn.execute(
        """
        SELECT username, password_hash, mfa_secret, mfa_enabled, mfa_backup_codes, updated_at 
        FROM auth_user WHERE username = ?
        """,
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


def set_mfa_secret(conn: sqlite3.Connection, username: str, mfa_secret: str) -> bool:
    """Set MFA secret for a user (not yet enabled)."""
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        UPDATE auth_user SET mfa_secret = ?, updated_at = ?
        WHERE username = ?
        """,
        (mfa_secret, now, username.lower()),
    )
    conn.commit()
    return cur.rowcount > 0


def enable_mfa(conn: sqlite3.Connection, username: str, backup_codes_json: str) -> bool:
    """Enable MFA for a user after verification."""
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        UPDATE auth_user 
        SET mfa_enabled = 1, mfa_backup_codes = ?, updated_at = ?
        WHERE username = ? AND mfa_secret IS NOT NULL
        """,
        (backup_codes_json, now, username.lower()),
    )
    conn.commit()
    return cur.rowcount > 0


def disable_mfa(conn: sqlite3.Connection, username: str) -> bool:
    """Disable MFA for a user."""
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        UPDATE auth_user 
        SET mfa_enabled = 0, mfa_secret = NULL, mfa_backup_codes = NULL, updated_at = ?
        WHERE username = ?
        """,
        (now, username.lower()),
    )
    conn.commit()
    return cur.rowcount > 0


def update_backup_codes(conn: sqlite3.Connection, username: str, backup_codes_json: str) -> bool:
    """Update backup codes (after one is used)."""
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        """
        UPDATE auth_user SET mfa_backup_codes = ?, updated_at = ?
        WHERE username = ?
        """,
        (backup_codes_json, now, username.lower()),
    )
    conn.commit()
    return cur.rowcount > 0


def delete_user(conn: sqlite3.Connection, username: str) -> bool:
    """Delete a user."""
    cur = conn.execute("DELETE FROM auth_user WHERE username = ?", (username.lower(),))
    conn.commit()
    return cur.rowcount > 0


def any_user(conn: sqlite3.Connection) -> Optional[AuthUser]:
    """Get any user (for checking if users exist)."""
    cur = conn.execute(
        "SELECT username, password_hash, mfa_secret, mfa_enabled, mfa_backup_codes, updated_at FROM auth_user LIMIT 1"
    )
    row = cur.fetchone()
    if row:
        return AuthUser.from_row(row)
    return None
