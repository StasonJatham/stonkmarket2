"""Cron job repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

from app.database.models import CronJobConfig, CronJobLog


def list_cronjobs(conn: sqlite3.Connection) -> List[CronJobConfig]:
    """List all cron jobs."""
    cur = conn.execute(
        "SELECT name, cron, description FROM cronjobs ORDER BY name ASC"
    )
    return [CronJobConfig.from_row(row) for row in cur.fetchall()]


def get_cronjob(conn: sqlite3.Connection, name: str) -> Optional[CronJobConfig]:
    """Get a cron job by name."""
    cur = conn.execute(
        "SELECT name, cron, description FROM cronjobs WHERE name = ?",
        (name,),
    )
    row = cur.fetchone()
    return CronJobConfig.from_row(row) if row else None


def upsert_cronjob(conn: sqlite3.Connection, name: str, cron: str) -> CronJobConfig:
    """Create or update a cron job."""
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
    """Insert a cron job log entry."""
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
    """List logs for a specific cron job."""
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


def list_all_logs(
    conn: sqlite3.Connection,
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
) -> Tuple[List, int]:
    """List all logs with pagination and optional filtering."""
    base_where = "1=1"
    params: List = []
    
    if search:
        # Sanitize search input
        search = search.replace("%", "\\%").replace("_", "\\_")
        base_where += " AND (name LIKE ? ESCAPE '\\' OR message LIKE ? ESCAPE '\\')"
        params.extend([f"%{search}%", f"%{search}%"])
    
    if status_filter:
        base_where += " AND status = ?"
        params.append(status_filter)
    
    # Get total count
    count_query = f"SELECT COUNT(*) FROM cronjob_logs WHERE {base_where}"
    total = conn.execute(count_query, params).fetchone()[0]
    
    # Get paginated results
    query = f"""
        SELECT name, status, message, created_at
        FROM cronjob_logs
        WHERE {base_where}
        ORDER BY datetime(created_at) DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    cur = conn.execute(query, params)
    
    return cur.fetchall(), total
