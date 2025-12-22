"""Secure API keys repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from app.core.encryption import encrypt_api_key, decrypt_api_key, get_key_hint
from app.database.models import SecureApiKey


def get_key(conn: sqlite3.Connection, key_name: str) -> Optional[SecureApiKey]:
    """Get an API key by name."""
    cur = conn.execute(
        """
        SELECT id, key_name, encrypted_key, key_hint, created_at, updated_at, created_by
        FROM secure_api_keys WHERE key_name = ?
        """,
        (key_name,),
    )
    row = cur.fetchone()
    if row:
        return SecureApiKey.from_row(row)
    return None


def get_decrypted_key(conn: sqlite3.Connection, key_name: str) -> Optional[str]:
    """Get and decrypt an API key by name."""
    key_record = get_key(conn, key_name)
    if key_record:
        return decrypt_api_key(key_record.encrypted_key)
    return None


def list_keys(conn: sqlite3.Connection) -> list[dict]:
    """
    List all API keys (without decrypted values).

    Returns list of dicts with key_name, key_hint, created_at, updated_at, created_by
    """
    cur = conn.execute(
        """
        SELECT id, key_name, key_hint, created_at, updated_at, created_by
        FROM secure_api_keys
        ORDER BY key_name
        """
    )
    return [
        {
            "id": row["id"],
            "key_name": row["key_name"],
            "key_hint": row["key_hint"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "created_by": row["created_by"],
        }
        for row in cur.fetchall()
    ]


def upsert_key(
    conn: sqlite3.Connection,
    key_name: str,
    api_key: str,
    created_by: str,
) -> SecureApiKey:
    """Create or update an API key."""
    now = datetime.utcnow().isoformat()
    encrypted = encrypt_api_key(api_key)
    hint = get_key_hint(api_key)

    conn.execute(
        """
        INSERT INTO secure_api_keys(key_name, encrypted_key, key_hint, created_at, updated_at, created_by)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(key_name) DO UPDATE SET
            encrypted_key = excluded.encrypted_key,
            key_hint = excluded.key_hint,
            updated_at = excluded.updated_at
        """,
        (key_name, encrypted, hint, now, now, created_by),
    )
    conn.commit()
    return get_key(conn, key_name)  # type: ignore


def delete_key(conn: sqlite3.Connection, key_name: str) -> bool:
    """Delete an API key."""
    cur = conn.execute("DELETE FROM secure_api_keys WHERE key_name = ?", (key_name,))
    conn.commit()
    return cur.rowcount > 0


# Convenience constants for well-known keys
OPENAI_API_KEY = "openai_api_key"
