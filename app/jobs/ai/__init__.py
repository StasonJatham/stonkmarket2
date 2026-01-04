"""AI-related job definitions.

This module contains jobs for:
- OpenAI Batch API processing (polls, watchdog)
- AI persona analysis (Warren Buffett, Peter Lynch, etc.)
- AI-generated bios for swipe interface
- Portfolio AI analysis

All jobs use the shared batch scheduler for cost-efficient processing.
"""

from __future__ import annotations

import time

from app.core.logging import get_logger

from ..registry import register_job
from ..utils import log_job_success


logger = get_logger("jobs.ai")


# =============================================================================
# AI BIOS - Weekly swipe-style bio generation
# =============================================================================


@register_job("ai_bios_weekly")
async def ai_bios_weekly_job() -> str:
    """
    Generate swipe-style bios for dips using OpenAI Batch API.

    Schedule: Weekly Sunday 4am
    """
    from app.services.batch_scheduler import (
        process_completed_batch_jobs,
        schedule_batch_swipe_bios,
    )

    logger.info("Starting ai_bios_weekly job")
    job_start = time.monotonic()

    try:
        # Process any completed batches first
        processed = await process_completed_batch_jobs()

        # Schedule new batch
        batch_id = await schedule_batch_swipe_bios()

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Batch: {batch_id or 'none needed'}, processed: {processed}"
        
        # Structured success log
        log_job_success(
            "ai_bios_weekly",
            message,
            batch_id=batch_id,
            items_processed=processed,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"ai_bios_weekly failed: {e}")
        raise


# =============================================================================
# AI BATCH POLL - Process completed OpenAI batches
# =============================================================================


@register_job("ai_batch_poll")
async def ai_batch_poll_job() -> str:
    """
    Poll for completed OpenAI batch jobs.

    Schedule: Every 5 minutes (idempotent - only calls API if pending jobs exist)

    OpenAI Batch API can take up to 24 hours to complete.
    This job polls for completed batches and processes their results.
    """
    from app.services.batch_scheduler import process_completed_batch_jobs

    logger.info("Starting ai_batch_poll job")
    job_start = time.monotonic()

    try:
        processed = await process_completed_batch_jobs()

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Polled batches, processed: {processed}"
        
        # Structured success log
        log_job_success(
            "ai_batch_poll",
            message,
            items_processed=processed,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"ai_batch_poll failed: {e}")
        raise


# =============================================================================
# BATCH WATCHDOG - Monitor and expire stale batch jobs
# =============================================================================


@register_job("batch_watchdog")
async def batch_watchdog_job() -> str:
    """
    Batch job watchdog - expires stale batch jobs and reports health.
    
    Actions:
    1. Mark jobs stuck in pending/in_progress for >24h as 'expired'
    2. Log warnings for stale jobs (useful for alerting)
    3. Return health summary
    
    Schedule: Every hour
    """
    from app.repositories import api_usage_orm
    
    logger.info("Starting batch_watchdog job")
    job_start = time.monotonic()
    
    try:
        # Expire stale jobs (>24h old)
        expired_jobs = await api_usage_orm.expire_stale_batch_jobs(max_age_hours=24)
        
        # Get current health metrics
        health = await api_usage_orm.get_batch_job_health()
        
        if expired_jobs:
            logger.warning(
                f"batch_watchdog: Expired {len(expired_jobs)} stale jobs: "
                f"{[j['batch_id'] for j in expired_jobs]}"
            )
        
        if health.get("has_stale_jobs"):
            logger.warning(
                f"batch_watchdog: Oldest pending job is {health['oldest_pending_hours']}h old"
            )
        
        pending = health.get("status_counts", {}).get("pending", 0)
        in_progress = health.get("status_counts", {}).get("in_progress", 0)
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Batch watchdog: expired={len(expired_jobs)}, pending={pending}, in_progress={in_progress}"
        
        # Structured success log
        log_job_success(
            "batch_watchdog",
            message,
            expired_count=len(expired_jobs),
            pending_count=pending,
            in_progress_count=in_progress,
            has_stale_jobs=health.get("has_stale_jobs", False),
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.error(f"batch_watchdog failed: {e}")
        raise


# =============================================================================
# AI PERSONAS - Weekly investor persona analysis
# =============================================================================


@register_job("ai_personas_weekly")
async def ai_personas_weekly_job() -> str:
    """
    Run AI persona analysis (Warren Buffett, Peter Lynch, etc.) on stocks.

    Schedule: Weekly Sunday 3am (first AI job of the day)

    Each persona analyzes stocks using their investment philosophy.
    Results are stored for frontend display.

    Uses Batch API for 50% cost savings. Results collected by ai_batch_poll job.
    Uses input version checking to skip symbols whose data hasn't changed.
    """
    from app.services.ai_agents import run_all_agent_analyses_batch
    from app.services.batch_scheduler import process_completed_batch_jobs

    logger.info("Starting ai_personas_weekly job")
    job_start = time.monotonic()

    try:
        # Process any completed agent batches first
        processed = await process_completed_batch_jobs()

        # Submit new batch
        result = await run_all_agent_analyses_batch()

        batch_id = result.get("batch_id")
        submitted = result.get("submitted", 0)
        skipped = result.get("skipped", 0)

        duration_ms = int((time.monotonic() - job_start) * 1000)
        message = f"Batch: {batch_id or 'none needed'}, submitted: {submitted}, skipped: {skipped}, processed: {processed}"
        
        # Structured success log
        log_job_success(
            "ai_personas_weekly",
            message,
            batch_id=batch_id,
            submitted_count=submitted,
            skipped_count=skipped,
            processed_count=processed,
            duration_ms=duration_ms,
        )
        return message

    except Exception as e:
        logger.error(f"ai_personas_weekly failed: {e}")
        raise


# =============================================================================
# PORTFOLIO AI ANALYSIS - Daily portfolio insights
# =============================================================================


@register_job("portfolio_ai_analysis")
async def portfolio_ai_analysis_job() -> str:
    """
    Run AI analysis on portfolios that have changed.
    
    Schedule: Daily at 6 AM UTC (after market data is updated)
    
    This job:
    1. Finds portfolios where holdings have changed since last analysis
    2. Schedules batch AI analysis using the central batch scheduler
    3. Results are collected later by cron_sync_batch_jobs
    
    Only runs for portfolios with at least one holding.
    Uses batch processing for cost efficiency.
    """
    from app.services.batch_scheduler import cron_portfolio_ai_analysis

    logger.info("Starting portfolio_ai_analysis job")
    job_start = time.monotonic()
    
    try:
        result = await cron_portfolio_ai_analysis()
        
        duration_ms = int((time.monotonic() - job_start) * 1000)
        batch_id = result.get("batch_id")
        
        if batch_id:
            message = f"Scheduled portfolio AI batch: {batch_id}"
        else:
            message = "No portfolios need AI analysis"
        
        log_job_success(
            "portfolio_ai_analysis",
            message,
            batch_id=batch_id,
            duration_ms=duration_ms,
        )
        return message
        
    except Exception as e:
        logger.error(f"portfolio_ai_analysis failed: {e}")
        raise


# Public exports
__all__ = [
    "ai_bios_weekly_job",
    "ai_batch_poll_job",
    "batch_watchdog_job",
    "ai_personas_weekly_job",
    "portfolio_ai_analysis_job",
]
