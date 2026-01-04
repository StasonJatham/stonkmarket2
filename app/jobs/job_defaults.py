"""Shared defaults for scheduled jobs.

Job Naming Convention:
    <domain>_<frequency> - Clear, descriptive names

Job Categories:
    1. REAL-TIME PROCESSING (every 5-15 min)
       - symbol_ingest: Process NEW symbols from queue
       - ai_batch_poll: Check OpenAI batch results
       - portfolio_worker: Process analytics queue
       - cache_warmup: Pre-cache chart data

    2. DAILY MARKET CLOSE PIPELINE (single orchestrator, Mon-Fri 10 PM UTC)
       - market_close_pipeline: Runs all steps sequentially, each waits for previous
         Steps: prices → fundamentals → signals → regime → strategy → quant_scoring → dipfinder → quant_analysis
       - Individual jobs exist for manual retries but are NOT scheduled

    3. WEEKLY AI PIPELINE (single orchestrator, Sunday 2 AM UTC)
       - weekly_ai_pipeline: Runs all AI jobs sequentially
         Steps: data_backfill → ai_personas_weekly → ai_bios_weekly
       - Individual jobs exist for manual retries but are NOT scheduled

    4. MONTHLY MAINTENANCE
       - quant_monthly: Portfolio optimization

    5. DAILY CLEANUP
       - cleanup_daily: Remove expired data
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy.dialects.postgresql import insert

from app.database.connection import get_session
from app.database.orm import CronJob


# =============================================================================
# SCHEDULE DEFINITIONS
# =============================================================================
# Format: job_name -> (cron_expression, human_description)

DEFAULT_SCHEDULES: dict[str, tuple[str, str]] = {
    # =========================================================================
    # 0. HEALTH CHECK - Every minute
    # =========================================================================
    "celery_health": (
        "* * * * *",
        "Celery health check - verifies Valkey connectivity and worker responsiveness. "
        "Every minute. If this fails, Celery/Valkey is unhealthy."
    ),

    # =========================================================================
    # 1. REAL-TIME PROCESSING - Continuous background tasks
    # =========================================================================
    "symbol_ingest": (
        "*/15 * * * *",
        "Process new symbols - fetches price history, fundamentals, and queues AI analysis. "
        "Batches symbols added in the last 15 minutes. Idempotent - skips already processed."
    ),
    "ai_batch_poll": (
        "*/5 * * * *",
        "OpenAI batch result collector - checks for completed AI jobs and stores results. "
        "Every 5 minutes."
    ),
    "batch_watchdog": (
        "0 * * * *",
        "Batch job watchdog - expires jobs stuck for >24h and logs health warnings. "
        "Every hour."
    ),
    "portfolio_worker": (
        "*/5 * * * *",
        "Portfolio analytics processor - handles queued risk analysis calculations. "
        "Every 5 minutes."
    ),
    "cache_warmup": (
        "*/30 * * * *",
        "Cache pre-warming - pre-generates chart data for top stocks to speed up page loads. "
        "Every 30 minutes."
    ),
    
    # =========================================================================
    # 2. DAILY MARKET DATA - After market close (Mon-Fri)
    # =========================================================================
    # SINGLE ORCHESTRATOR: market_close_pipeline runs all steps sequentially
    # Each step waits for previous to complete - no timing issues!
    # Individual jobs can still be triggered manually for debugging/retries.
    # =========================================================================
    "market_close_pipeline": (
        "0 22 * * 1-5",
        "Daily market close pipeline - runs all analysis jobs sequentially after market close. "
        "Steps: prices → signals → regime → strategy → quant_scoring → dipfinder → quant_analysis. "
        "Each step waits for previous to complete. Mon-Fri at 10 PM UTC."
    ),
    
    # Individual jobs (NOT scheduled - triggered by pipeline or manually)
    # These are kept for: manual retries, debugging, or partial re-runs
    "prices_daily": (
        "0 0 31 2 *",  # Never runs (Feb 31 doesn't exist) - triggered by pipeline
        "Daily price update - fetches closing prices and updates dip states. "
        "NOT SCHEDULED - runs as part of market_close_pipeline."
    ),
    "signals_daily": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "Technical signal scanner - RSI, MACD, Bollinger signals. "
        "NOT SCHEDULED - runs as part of market_close_pipeline."
    ),
    "regime_daily": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "Market regime detection - bull/bear, volatility. "
        "NOT SCHEDULED - runs as part of market_close_pipeline."
    ),
    "strategy_nightly": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "Strategy optimization - backtests and finds best strategies. "
        "NOT SCHEDULED - runs as part of market_close_pipeline."
    ),
    "quant_scoring_daily": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "Quant scoring - momentum, quality, value metrics. "
        "NOT SCHEDULED - runs as part of market_close_pipeline."
    ),
    "dipfinder_daily": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "DipFinder + Dip Entry Optimizer - dip metrics and optimal buy thresholds. "
        "NOT SCHEDULED - runs as part of market_close_pipeline."
    ),
    "quant_analysis_nightly": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "Quant analysis pre-compute - caches all quant results for API. "
        "NOT SCHEDULED - runs as part of market_close_pipeline."
    ),
    
    # =========================================================================
    # 3. WEEKLY AI PIPELINE - Sunday morning (orchestrated)
    # =========================================================================
    # SINGLE ORCHESTRATOR: weekly_ai_pipeline runs all AI jobs sequentially
    # Each step waits for previous to complete - proper data dependencies!
    # Individual jobs can still be triggered manually for debugging/retries.
    # =========================================================================
    "weekly_ai_pipeline": (
        "0 2 * * 0",
        "Weekly AI pipeline - runs data_backfill → ai_personas → ai_bios sequentially. "
        "Ensures data is complete before AI analysis. Sunday 2 AM UTC."
    ),
    
    # Individual jobs (NOT scheduled - triggered by pipeline or manually)
    "data_backfill": (
        "0 0 31 2 *",  # Never runs (Feb 31 doesn't exist) - triggered by pipeline
        "Comprehensive data backfill - fills ALL data gaps: missing sectors, summaries, "
        "price history, fundamentals, quant scores, dipfinder signals. "
        "NOT SCHEDULED - runs as part of weekly_ai_pipeline."
    ),
    "ai_personas_weekly": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "AI investor personas (Warren Buffett, Peter Lynch, Cathie Wood, Michael Burry) - "
        "each analyzes all stocks from their investment philosophy. "
        "NOT SCHEDULED - runs as part of weekly_ai_pipeline."
    ),
    "ai_bios_weekly": (
        "0 0 31 2 *",  # Never runs - triggered by pipeline
        "AI swipe bios - generates fun 'dating profile' style descriptions for stocks. "
        "NOT SCHEDULED - runs as part of weekly_ai_pipeline."
    ),
    
    # =========================================================================
    # 4. WEEKLY MARKET DATA
    # =========================================================================
    "market_data_sync": (
        "0 3 * * 0",
        "Market sector/industry sync - fetches all sectors, industries, top companies "
        "from yfinance. Used for competitor suggestions and similar stocks. "
        "Sunday 3 AM UTC."
    ),
    "calendar_sync": (
        "0 5 * * 6",
        "Calendar data sync - fetches earnings, IPOs, splits, economic events from yfinance "
        "for the next 5 weeks. Used for calendar widget and stock analysis. "
        "Saturday 5 AM UTC."
    ),
    
    # =========================================================================
    # 5. MONTHLY MAINTENANCE
    # =========================================================================
    "quant_monthly": (
        "0 3 1 * *",
        "Quant engine optimization - recalculates portfolio weights and risk models. "
        "1st of each month at 3 AM UTC."
    ),
    
    # =========================================================================
    # 6. DAILY CLEANUP
    # =========================================================================
    "cleanup_daily": (
        "0 0 * * *",
        "Data cleanup - removes expired cache entries, old analytics jobs, stale sessions. "
        "Daily at midnight UTC."
    ),
}


# =============================================================================
# JOB PRIORITIES
# =============================================================================
# Queue assignment and priority (higher = more important)

JOB_PRIORITIES: dict[str, dict[str, int | str]] = {
    # Health check - highest priority, fast queue
    "celery_health": {"queue": "high", "priority": 10},

    # Pipeline orchestrators - highest priority
    "market_close_pipeline": {"queue": "batch", "priority": 9},
    "weekly_ai_pipeline": {"queue": "batch", "priority": 9},
    
    # Pipeline component jobs (triggered by orchestrator, not scheduled)
    "prices_daily": {"queue": "high", "priority": 9},
    "fundamentals_daily": {"queue": "default", "priority": 8},
    "signals_daily": {"queue": "high", "priority": 8},
    "regime_daily": {"queue": "high", "priority": 8},
    "strategy_nightly": {"queue": "batch", "priority": 7},
    "quant_scoring_daily": {"queue": "default", "priority": 6},
    "dipfinder_daily": {"queue": "default", "priority": 6},
    "quant_analysis_nightly": {"queue": "batch", "priority": 6},
    
    # Real-time processing
    "ai_batch_poll": {"queue": "high", "priority": 8},
    "batch_watchdog": {"queue": "low", "priority": 3},
    "cache_warmup": {"queue": "high", "priority": 7},
    "symbol_ingest": {"queue": "default", "priority": 7},
    "portfolio_worker": {"queue": "default", "priority": 5},
    
    # Batch queue - heavy computation / AI jobs
    "ai_personas_weekly": {"queue": "batch", "priority": 6},
    "ai_bios_weekly": {"queue": "batch", "priority": 4},
    "data_backfill": {"queue": "batch", "priority": 3},
    
    # Weekly market data - medium priority
    "market_data_sync": {"queue": "default", "priority": 5},
    "calendar_sync": {"queue": "default", "priority": 5},
    
    # Monthly maintenance
    "quant_monthly": {"queue": "default", "priority": 4},
    
    # Low priority - maintenance
    "cleanup_daily": {"queue": "low", "priority": 2},
}


# =============================================================================
# JOB TIME LIMITS
# =============================================================================
# Per-task time limits to prevent runaway tasks and ensure visibility_timeout works.
#
# soft_limit: Task receives SIGTERM, can catch and cleanup gracefully.
# hard_limit: Task receives SIGKILL, forcefully terminated.
#
# Categories:
#   - Quick (60s/120s): Fast operations like polling, cleanup
#   - Standard (300s/600s): Normal jobs with API calls
#   - Heavy (1800s/2100s): Bulk data processing
#   - Pipeline (7200s/7500s): Full market close pipeline (~2.5 hours)
#   - AI (3600s/4200s): OpenAI batch operations
#
# IMPORTANT: visibility_timeout (3 hours) MUST exceed the longest hard_limit.

JOB_TIME_LIMITS: dict[str, dict[str, int]] = {
    # =========================================================================
    # QUICK JOBS (60s soft / 120s hard)
    # =========================================================================
    "batch_watchdog": {"soft_limit": 60, "hard_limit": 120},
    "cleanup_daily": {"soft_limit": 60, "hard_limit": 120},
    "cache_warmup": {"soft_limit": 60, "hard_limit": 120},
    "celery_health": {"soft_limit": 10, "hard_limit": 30},

    # =========================================================================
    # STANDARD JOBS (300s soft / 600s hard)
    # =========================================================================
    # ai_batch_poll collects results from potentially many completed batches
    # Each batch response can be large (AI analysis text). Give it 5 minutes.
    "ai_batch_poll": {"soft_limit": 300, "hard_limit": 600},
    "portfolio_worker": {"soft_limit": 300, "hard_limit": 600},
    "symbol_ingest": {"soft_limit": 300, "hard_limit": 600},
    "prices_daily": {"soft_limit": 300, "hard_limit": 600},
    "signals_daily": {"soft_limit": 300, "hard_limit": 600},
    "regime_daily": {"soft_limit": 300, "hard_limit": 600},
    "dipfinder_daily": {"soft_limit": 300, "hard_limit": 600},
    "quant_scoring_daily": {"soft_limit": 300, "hard_limit": 600},
    # fundamentals_daily fetches from yfinance for many symbols
    "fundamentals_daily": {"soft_limit": 600, "hard_limit": 900},

    # =========================================================================
    # HEAVY JOBS (1800s soft / 2100s hard) - 30-35 minutes
    # =========================================================================
    "strategy_nightly": {"soft_limit": 1800, "hard_limit": 2100},
    "quant_analysis_nightly": {"soft_limit": 1800, "hard_limit": 2100},
    "quant_monthly": {"soft_limit": 1800, "hard_limit": 2100},
    "market_data_sync": {"soft_limit": 1800, "hard_limit": 2100},
    "calendar_sync": {"soft_limit": 1800, "hard_limit": 2100},
    "data_backfill": {"soft_limit": 1800, "hard_limit": 2100},

    # =========================================================================
    # AI JOBS (3600s soft / 4200s hard) - 1 hour
    # =========================================================================
    "ai_personas_weekly": {"soft_limit": 3600, "hard_limit": 4200},
    "ai_bios_weekly": {"soft_limit": 3600, "hard_limit": 4200},

    # =========================================================================
    # PIPELINE ORCHESTRATORS (7200s soft / 7500s hard) - 2-2.5 hours
    # =========================================================================
    # These run multiple jobs sequentially and can take up to 2.5 hours.
    # visibility_timeout (3 hours) provides sufficient buffer.
    "market_close_pipeline": {"soft_limit": 7200, "hard_limit": 7500},
    "weekly_ai_pipeline": {"soft_limit": 7200, "hard_limit": 7500},
}

# Default time limits for unknown jobs (conservative)
DEFAULT_TIME_LIMITS: dict[str, int] = {"soft_limit": 300, "hard_limit": 600}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_time_limits(name: str) -> dict[str, int]:
    """Return soft_limit and hard_limit for a job.

    Args:
        name: Job name to look up.

    Returns:
        Dict with 'soft_limit' and 'hard_limit' in seconds.
    """
    return JOB_TIME_LIMITS.get(name, DEFAULT_TIME_LIMITS)


def get_job_schedule(name: str) -> tuple[str, str]:
    """Return (cron, description) for a job name."""
    return DEFAULT_SCHEDULES.get(name, ("0 * * * *", f"Job: {name}"))


def get_job_priority(name: str) -> dict[str, int | str]:
    """Return queue/priority for a job."""
    return JOB_PRIORITIES.get(name, {"queue": "default", "priority": 5})


async def seed_cronjobs(job_names: Iterable[str]) -> None:
    """Ensure cronjobs exist for all registered jobs."""
    async with get_session() as session:
        for job_name in job_names:
            cron_expr, description = get_job_schedule(job_name)
            stmt = insert(CronJob).values(
                name=job_name,
                cron=cron_expr,
                description=description,
                config={},
                is_active=True,
            ).on_conflict_do_nothing(index_elements=["name"])
            await session.execute(stmt)
        await session.commit()
