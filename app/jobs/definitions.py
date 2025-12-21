"""Built-in job definitions."""

from __future__ import annotations

import sqlite3

from app.core.logging import get_logger
from app.services.dip_service import compute_ranking_details, refresh_states

from .registry import register_job

logger = get_logger("jobs.definitions")


@register_job("data_grab")
def data_grab_job(conn: sqlite3.Connection) -> str:
    """Download fresh stock quotes and refresh dip states."""
    logger.info("Starting data_grab job")
    states = refresh_states(conn)
    count = len(states)
    logger.info(f"data_grab completed: refreshed {count} symbols")
    return f"Fetched quotes and refreshed {count} dip states"


@register_job("analysis")
def analysis_job(conn: sqlite3.Connection) -> str:
    """Recompute dip ranking."""
    logger.info("Starting analysis job")
    entries = compute_ranking_details(conn, force_refresh=True)
    count = len(entries)
    logger.info(f"analysis completed: {count} ranking entries")
    return f"Recomputed ranking with {count} entries"
