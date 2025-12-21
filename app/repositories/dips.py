"""Dip state repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict

from app.database.models import DipState


def load_states(conn: sqlite3.Connection) -> Dict[str, DipState]:
    """Load all dip states."""
    cur = conn.execute(
        "SELECT symbol, ref_high, days_below, last_price, updated_at FROM dip_state"
    )
    states: Dict[str, DipState] = {}
    for row in cur.fetchall():
        states[row["symbol"]] = DipState.from_row(row)
    return states


def get_state(conn: sqlite3.Connection, symbol: str) -> DipState | None:
    """Get dip state for a specific symbol."""
    cur = conn.execute(
        "SELECT symbol, ref_high, days_below, last_price, updated_at FROM dip_state WHERE symbol = ?",
        (symbol.upper(),),
    )
    row = cur.fetchone()
    return DipState.from_row(row) if row else None


def save_states_batch(
    conn: sqlite3.Connection, states: Dict[str, DipState]
) -> None:
    """Save multiple dip states."""
    now = datetime.utcnow().isoformat()
    conn.executemany(
        """
        INSERT INTO dip_state (symbol, ref_high, days_below, last_price, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            ref_high=excluded.ref_high,
            days_below=excluded.days_below,
            last_price=excluded.last_price,
            updated_at=excluded.updated_at
        """,
        [
            (sym, s.ref_high, s.days_below, s.last_price, now)
            for sym, s in states.items()
        ],
    )
    conn.commit()


def delete_state(conn: sqlite3.Connection, symbol: str) -> bool:
    """Delete dip state for a symbol."""
    cur = conn.execute("DELETE FROM dip_state WHERE symbol = ?", (symbol.upper(),))
    conn.commit()
    return cur.rowcount > 0
