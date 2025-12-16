from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional, Tuple


def get_user(conn: sqlite3.Connection, username: str) -> Optional[Tuple[str, str]]:
    cur = conn.execute(
        "SELECT username, password_hash FROM auth_user WHERE username = ?", (username,)
    )
    row = cur.fetchone()
    if row:
        return row[0], row[1]
    return None


def upsert_user(conn: sqlite3.Connection, username: str, password_hash: str) -> None:
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO auth_user(username, password_hash, updated_at) VALUES (?, ?, ?)\n        ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash, updated_at=excluded.updated_at",
        (username, password_hash, now),
    )
    conn.commit()


def any_user(conn: sqlite3.Connection) -> Optional[Tuple[str, str]]:
    cur = conn.execute("SELECT username, password_hash FROM auth_user LIMIT 1")
    row = cur.fetchone()
    if row:
        return row[0], row[1]
    return None
