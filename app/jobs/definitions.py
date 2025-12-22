"""Built-in job definitions - exactly 4 jobs."""

from __future__ import annotations

from app.core.logging import get_logger
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
    await manager.broadcast_to_admins(
        WSEvent(
            type=event_type,
            message=message,
            data={"job_name": job_name, **(data or {})},
        )
    )


@register_job("data_grab")
async def data_grab_job() -> str:
    """
    Fetch stock data from yfinance and calculate signals/dips.

    Schedule: Mon-Fri at 11pm (after market close)
    """
    from app.dipfinder.service import get_dipfinder_service
    from app.database.connection import fetch_all
    from app.cache.cache import Cache

    logger.info("Starting data_grab job")

    await _broadcast_job_event(
        "data_grab", WSEventType.CRONJOB_STARTED, "Fetching stock data"
    )

    try:
        rows = await fetch_all("SELECT symbol FROM symbols WHERE is_active = TRUE")
        tickers = [row["symbol"] for row in rows]

        if not tickers:
            return "No active symbols"

        service = get_dipfinder_service()
        signals = await service.get_signals(tickers, force_refresh=True)
        dips = sum(1 for s in signals if s.dip_metrics and s.dip_metrics.in_dip)

        # Invalidate caches since new data is available
        ranking_cache = Cache(prefix="ranking", default_ttl=1800)
        chart_cache = Cache(prefix="chart", default_ttl=3600)
        await ranking_cache.invalidate_pattern("*")
        await chart_cache.invalidate_pattern("*")
        logger.info("Invalidated ranking and chart caches")

        message = f"Fetched {len(signals)} symbols, {dips} dips"
        await _broadcast_job_event("data_grab", WSEventType.CRONJOB_COMPLETE, message)
        logger.info(f"data_grab: {message}")
        return message

    except Exception as e:
        await _broadcast_job_event("data_grab", WSEventType.CRONJOB_ERROR, str(e))
        logger.error(f"data_grab failed: {e}")
        raise


@register_job("batch_ai_tinder")
async def batch_ai_tinder_job() -> str:
    """
    Generate tinder-style bios for dips using OpenAI Batch API.

    Schedule: Weekly Sunday 3am
    """
    from app.services.batch_scheduler import (
        schedule_batch_tinder_bios,
        process_completed_batch_jobs,
    )

    logger.info("Starting batch_ai_tinder job")

    try:
        # Process any completed batches first
        processed = await process_completed_batch_jobs()

        # Schedule new batch
        batch_id = await schedule_batch_tinder_bios()

        message = f"Batch: {batch_id or 'none needed'}, processed: {processed}"
        logger.info(f"batch_ai_tinder: {message}")
        return message

    except Exception as e:
        logger.error(f"batch_ai_tinder failed: {e}")
        raise


@register_job("batch_ai_analysis")
async def batch_ai_analysis_job() -> str:
    """
    Generate serious dip analysis using OpenAI Batch API.

    Schedule: Weekly Sunday 4am
    """
    from app.services.batch_scheduler import (
        schedule_batch_dip_analysis,
        process_completed_batch_jobs,
    )

    logger.info("Starting batch_ai_analysis job")

    try:
        # Process any completed batches first
        processed = await process_completed_batch_jobs()

        # Schedule new batch
        batch_id = await schedule_batch_dip_analysis()

        message = f"Batch: {batch_id or 'none needed'}, processed: {processed}"
        logger.info(f"batch_ai_analysis: {message}")
        return message

    except Exception as e:
        logger.error(f"batch_ai_analysis failed: {e}")
        raise


@register_job("batch_poll")
async def batch_poll_job() -> str:
    """
    Poll for completed OpenAI batch jobs.

    Schedule: Every 15 minutes

    OpenAI Batch API can take up to 24 hours to complete.
    This job polls for completed batches and processes their results.
    """
    from app.services.batch_scheduler import process_completed_batch_jobs

    logger.info("Starting batch_poll job")

    try:
        processed = await process_completed_batch_jobs()

        if processed > 0:
            await _broadcast_job_event(
                "batch_poll",
                WSEventType.CRONJOB_COMPLETE,
                f"Processed {processed} completed batch jobs",
            )

        message = f"Polled batches, processed: {processed}"
        logger.info(f"batch_poll: {message}")
        return message

    except Exception as e:
        logger.error(f"batch_poll failed: {e}")
        raise


@register_job("cleanup")
async def cleanup_job() -> str:
    """
    Clean up expired suggestions and old API keys.

    Schedule: Daily midnight
    """
    from app.database.connection import execute

    logger.info("Starting cleanup job")

    try:
        # Rejected suggestions > 7 days
        await execute(
            "DELETE FROM stock_suggestions WHERE status = 'rejected' AND updated_at < NOW() - INTERVAL '7 days'"
        )

        # Pending suggestions > 30 days
        await execute(
            "DELETE FROM stock_suggestions WHERE status = 'pending' AND created_at < NOW() - INTERVAL '30 days'"
        )

        # Expired AI analyses
        await execute(
            "DELETE FROM dip_ai_analysis WHERE expires_at IS NOT NULL AND expires_at < NOW()"
        )

        # Expired user API keys
        await execute(
            "DELETE FROM user_api_keys WHERE expires_at IS NOT NULL AND expires_at < NOW()"
        )

        message = "Cleanup completed"
        logger.info(f"cleanup: {message}")
        return message

    except Exception as e:
        logger.error(f"cleanup failed: {e}")
        raise
