from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Dict, Iterable

from ..models import DipState


def load_states(conn: sqlite3.Connection) -> Dict[str, DipState]:
    cur = conn.execute(
        "SELECT symbol, ref_high, days_below, last_price, updated_at FROM dip_state"
    )
    states: Dict[str, DipState] = {}
    for row in cur.fetchall():
        updated_at = datetime.fromisoformat(row[4]) if row[4] else None
        states[row[0]] = DipState(row[1], row[2], row[3], updated_at)
    return states


def save_states_batch(
    conn: sqlite3.Connection, states: Dict[str, DipState]
) -> None:
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


def delete_state(conn: sqlite3.Connection, symbol: str) -> None:
    conn.execute("DELETE FROM dip_state WHERE symbol = ?", (symbol.upper(),))
    conn.commit()
