"""Built-in job definitions."""

from __future__ import annotations

import asyncio
import sqlite3

from app.core.logging import get_logger
from app.services.dip_service import compute_ranking_details, refresh_states
from app.websocket import get_connection_manager, WSEvent, WSEventType

from .registry import register_job

logger = get_logger("jobs.definitions")


async def _broadcast_job_event(
    job_name: str,
    event_type: WSEventType,
    message: str,
    data: dict = None,
) -> None:
    """Broadcast job status via WebSocket."""
    manager = get_connection_manager()
    await manager.broadcast_to_admins(WSEvent(
        type=event_type,
        message=message,
        data={"job_name": job_name, **(data or {})},
    ))


def _run_async(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_running_loop()
        return asyncio.ensure_future(coro)
    except RuntimeError:
        return asyncio.run(coro)


@register_job("data_grab")
def data_grab_job(conn: sqlite3.Connection) -> str:
    """Download fresh stock quotes and refresh dip states."""
    logger.info("Starting data_grab job")
    
    _run_async(_broadcast_job_event(
        "data_grab",
        WSEventType.CRONJOB_STARTED,
        "Starting data grab job",
    ))
    
    try:
        states = refresh_states(conn)
        count = len(states)
        
        _run_async(_broadcast_job_event(
            "data_grab",
            WSEventType.CRONJOB_COMPLETE,
            f"Data grab completed: {count} symbols",
            {"symbols_refreshed": count},
        ))
        
        logger.info(f"data_grab completed: refreshed {count} symbols")
        return f"Fetched quotes and refreshed {count} dip states"
        
    except Exception as e:
        _run_async(_broadcast_job_event(
            "data_grab",
            WSEventType.CRONJOB_ERROR,
            f"Data grab failed: {e}",
            {"error": str(e)},
        ))
        raise


@register_job("analysis")
def analysis_job(conn: sqlite3.Connection) -> str:
    """Recompute dip ranking."""
    logger.info("Starting analysis job")
    
    _run_async(_broadcast_job_event(
        "analysis",
        WSEventType.CRONJOB_STARTED,
        "Starting analysis job",
    ))
    
    try:
        entries = compute_ranking_details(conn, force_refresh=True)
        count = len(entries)
        
        _run_async(_broadcast_job_event(
            "analysis",
            WSEventType.CRONJOB_COMPLETE,
            f"Analysis completed: {count} entries",
            {"ranking_entries": count},
        ))
        
        logger.info(f"analysis completed: {count} ranking entries")
        return f"Recomputed ranking with {count} entries"
        
    except Exception as e:
        _run_async(_broadcast_job_event(
            "analysis",
            WSEventType.CRONJOB_ERROR,
            f"Analysis failed: {e}",
            {"error": str(e)},
        ))
        raise


@register_job("suggestion_fetch")
def suggestion_fetch_job(conn: sqlite3.Connection) -> str:
    """Fetch data for pending stock suggestions (slow, rate-limited)."""
    from app.services.suggestion_service import process_pending_suggestions
    
    logger.info("Starting suggestion_fetch job")
    
    # This is async, run it properly
    result = asyncio.run(process_pending_suggestions())
    
    message = f"Processed {result['processed']} suggestions ({result['success']} success, {result['failed']} failed)"
    logger.info(f"suggestion_fetch completed: {message}")
    return message

