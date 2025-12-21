"""Symbol repository."""

from __future__ import annotations

import sqlite3
from typing import List, Optional

from app.database.models import SymbolConfig


def list_symbols(conn: sqlite3.Connection) -> List[SymbolConfig]:
    """List all symbols."""
    cur = conn.execute(
        "SELECT symbol, min_dip_pct, min_days, created_at FROM symbols ORDER BY symbol ASC"
    )
    return [SymbolConfig.from_row(row) for row in cur.fetchall()]


def get_symbol(conn: sqlite3.Connection, symbol: str) -> Optional[SymbolConfig]:
    """Get a symbol by ticker."""
    cur = conn.execute(
        "SELECT symbol, min_dip_pct, min_days, created_at FROM symbols WHERE symbol = ?",
        (symbol.upper(),),
    )
    row = cur.fetchone()
    return SymbolConfig.from_row(row) if row else None


def upsert_symbol(
    conn: sqlite3.Connection,
    symbol: str,
    min_dip_pct: float,
    min_days: int,
) -> SymbolConfig:
    """Create or update a symbol."""
    conn.execute(
        """
        INSERT INTO symbols(symbol, min_dip_pct, min_days, created_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(symbol) DO UPDATE SET
            min_dip_pct = excluded.min_dip_pct,
            min_days = excluded.min_days
        """,
        (symbol.upper(), float(min_dip_pct), int(min_days)),
    )
    conn.commit()
    return get_symbol(conn, symbol.upper())  # type: ignore


def delete_symbol(conn: sqlite3.Connection, symbol: str) -> bool:
    """Delete a symbol."""
    cur = conn.execute("DELETE FROM symbols WHERE symbol = ?", (symbol.upper(),))
    conn.commit()
    return cur.rowcount > 0
