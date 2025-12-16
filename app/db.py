from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .config import settings


DDL = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY,
    min_dip_pct REAL NOT NULL,
    min_days INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dip_state (
    symbol TEXT PRIMARY KEY,
    ref_high REAL NOT NULL,
    days_below INTEGER NOT NULL,
    last_price REAL NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cronjobs (
    name TEXT PRIMARY KEY,
    cron TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS cronjob_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(name) REFERENCES cronjobs(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS auth_user (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _configure_connection(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    resolved = Path(db_path or settings.db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(resolved.as_posix(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _configure_connection(conn)
    return conn


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    conn = get_connection(db_path)
    conn.executescript(DDL)
    _bootstrap_defaults(conn)
    conn.commit()
    return conn


def _bootstrap_defaults(conn: sqlite3.Connection) -> None:
    now = datetime.utcnow().isoformat()
    conn.executemany(
        """
        INSERT OR IGNORE INTO symbols(symbol, min_dip_pct, min_days, created_at)
        VALUES (?, ?, ?, ?)
        """,
        [
            (sym.upper(), settings.default_min_dip_pct, settings.default_min_days, now)
            for sym in settings.default_symbols
        ],
    )
    conn.executemany(
        """
        INSERT OR IGNORE INTO cronjobs(name, cron, description)
        VALUES (?, ?, ?)
        """,
        [
            ("data_grab", "0 6 * * 1-5", "Download fresh quotes"),
            ("analysis", "30 6 * * 1-5", "Run dip ranking"),
        ],
    )

    from .auth import hash_password  # local import to avoid cycle
    conn.execute(
        "INSERT OR IGNORE INTO auth_user(username, password_hash, updated_at) VALUES (?, ?, ?)",
        (settings.default_admin_user, hash_password(settings.default_admin_password), now),
    )


@contextmanager
def connection_scope(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = init_db(db_path)
    try:
        yield conn
    finally:
        conn.close()
