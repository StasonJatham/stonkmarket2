"""Cron job runner service."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.services import dip_service

if TYPE_CHECKING:
    import sqlite3

logger = get_logger("services.cron_runner")


def run_job(conn: "sqlite3.Connection", name: str) -> str:
    """Run a cron job by name.
    
    Args:
        conn: Database connection
        name: Job name ('data_grab' or 'analysis')
        
    Returns:
        Status message
        
    Raises:
        ValidationError: If job name is not recognized
    """
    logger.info(f"Running cron job: {name}")
    
    if name == "data_grab":
        dip_service.refresh_states(conn)
        logger.info("Data grab job completed")
        return "Fetched latest quotes and refreshed dip states"
    
    if name == "analysis":
        dip_service.compute_ranking_details(conn, force_refresh=True)
        logger.info("Analysis job completed")
        return "Recomputed dip ranking"
    
    logger.warning(f"Unknown cron job requested: {name}")
    raise ValidationError(f"Unknown cron job: {name}")

