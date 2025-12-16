from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Optional

from ..models import CronJobConfig


def list_cronjobs(conn: sqlite3.Connection) -> List[CronJobConfig]:
    cur = conn.execute(
        "SELECT name, cron, description FROM cronjobs ORDER BY name ASC"
    )
    return [CronJobConfig(row[0], row[1], row[2]) for row in cur.fetchall()]


def get_cronjob(conn: sqlite3.Connection, name: str) -> Optional[CronJobConfig]:
    cur = conn.execute(
        "SELECT name, cron, description FROM cronjobs WHERE name = ?",
        (name,),
    )
    row = cur.fetchone()
    return CronJobConfig(row[0], row[1], row[2]) if row else None


def upsert_cronjob(conn: sqlite3.Connection, name: str, cron: str) -> CronJobConfig:
    conn.execute(
        """
        INSERT INTO cronjobs(name, cron, description)
        VALUES (?, ?, NULL)
        ON CONFLICT(name) DO UPDATE SET
            cron = excluded.cron
        """,
        (name, cron),
    )
    conn.commit()
    return get_cronjob(conn, name)  # type: ignore


def insert_log(conn: sqlite3.Connection, name: str, status: str, message: str | None) -> str:
    created_at = datetime.utcnow().isoformat()
    conn.execute(
        """
        INSERT INTO cronjob_logs(name, status, message, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (name, status, message, created_at),
    )
    conn.commit()
    return created_at


def list_logs(conn: sqlite3.Connection, name: str, limit: int = 50):
    cur = conn.execute(
        """
        SELECT name, status, message, created_at
        FROM cronjob_logs
        WHERE name = ?
        ORDER BY datetime(created_at) DESC
        LIMIT ?
        """,
        (name, limit),
    )
    return cur.fetchall()
