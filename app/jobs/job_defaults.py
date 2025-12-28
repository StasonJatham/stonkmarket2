"""Shared defaults for scheduled jobs.

Job Naming Convention:
    <domain>_<frequency> - Clear, descriptive names

Job Categories:
    1. NEW SYMBOL PROCESSING (every 15 min)
       - symbol_ingest: Process pending symbols, fetch data, submit AI batch

    2. DAILY PRICE UPDATES (after market close)  
       - prices_daily: Fetch latest prices for all tracked symbols
       - signals_daily: Scan for technical buy signals
       - regime_daily: Detect market conditions

    3. WEEKLY AI ANALYSIS (Sunday morning)
       - ai_personas_weekly: Warren Buffett, Peter Lynch etc. analysis
       - ai_bios_weekly: Swipe-style stock bios

    4. AI BATCH POLLING (every 5 min)
       - ai_batch_poll: Check OpenAI batch results

    5. MONTHLY MAINTENANCE
       - fundamentals_monthly: Refresh company fundamentals
       - quant_monthly: Portfolio optimization

    6. BACKGROUND WORKERS (every 5 min)
       - portfolio_worker: Process portfolio analytics queue
       - cleanup_daily: Remove expired data
       - cache_warmup: Pre-cache popular data
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
    # 1. NEW SYMBOL PROCESSING - When users add new stocks
    # =========================================================================
    "symbol_ingest": (
        "*/15 * * * *",
        "Process new symbols - fetches price history, fundamentals, and queues AI analysis. "
        "Batches symbols added in the last 15 minutes. Idempotent - skips already processed."
    ),
    
    # =========================================================================
    # 2. DAILY PRICE & ANALYSIS - After market close
    # =========================================================================
    "prices_daily": (
        "0 23 * * 1-5",
        "Daily price update - fetches closing prices and updates dip states for all stocks. "
        "Runs Mon-Fri at 11 PM UTC (after US market close)."
    ),
    "signals_daily": (
        "0 22 * * 1-5",
        "Technical signal scanner - finds RSI oversold, MACD crossovers, Bollinger squeezes. "
        "Runs Mon-Fri at 10 PM UTC."
    ),
    "regime_daily": (
        "30 22 * * 1-5",
        "Market regime detection - identifies bull/bear market and volatility conditions. "
        "Runs Mon-Fri at 10:30 PM UTC."
    ),
    "strategy_optimize_nightly": (
        "30 23 * * 1-5",
        "Strategy optimization - runs full backtest with recency weighting, finds best strategy "
        "for each symbol that works NOW. Includes fundamental filters. "
        "Runs Mon-Fri at 11:30 PM UTC (30 min after prices_daily)."
    ),
    
    # =========================================================================
    # 3. WEEKLY AI ANALYSIS - Sunday morning (batch for cost savings)
    # =========================================================================
    "ai_personas_weekly": (
        "0 3 * * 0",
        "AI investor personas (Warren Buffett, Peter Lynch, Cathie Wood, Michael Burry) - "
        "each analyzes all stocks from their investment philosophy. Sunday 3 AM UTC."
    ),
    "ai_bios_weekly": (
        "0 4 * * 0",
        "AI swipe bios - generates fun 'dating profile' style descriptions for stocks. "
        "Sunday 4 AM UTC."
    ),
    
    # =========================================================================
    # 4. AI BATCH POLLING - Check for completed OpenAI batch jobs
    # =========================================================================
    "ai_batch_poll": (
        "*/5 * * * *",
        "OpenAI batch result collector - checks for completed AI jobs and stores results. "
        "Every 5 minutes."
    ),
    
    # =========================================================================
    # 5. MONTHLY MAINTENANCE
    # =========================================================================
    "fundamentals_monthly": (
        "0 2 1 * *",
        "Company fundamentals refresh - updates P/E, EPS, revenue, margins for all stocks. "
        "1st of each month at 2 AM UTC."
    ),
    "quant_monthly": (
        "0 3 1 * *",
        "Quant engine optimization - recalculates portfolio weights and risk models. "
        "1st of each month at 3 AM UTC."
    ),
    
    # =========================================================================
    # 6. BACKGROUND WORKERS & MAINTENANCE
    # =========================================================================
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
    # High priority - time-sensitive market data
    "prices_daily": {"queue": "high", "priority": 9},
    "cache_warmup": {"queue": "high", "priority": 8},
    "ai_batch_poll": {"queue": "high", "priority": 8},
    
    # Default priority - data ingestion & analysis
    "symbol_ingest": {"queue": "default", "priority": 7},
    "signals_daily": {"queue": "default", "priority": 7},
    "regime_daily": {"queue": "default", "priority": 6},
    "strategy_optimize_nightly": {"queue": "batch", "priority": 6},  # Heavy computation
    "fundamentals_monthly": {"queue": "default", "priority": 5},
    "portfolio_worker": {"queue": "default", "priority": 5},
    
    # Batch queue - AI jobs (can run longer)
    "ai_personas_weekly": {"queue": "batch", "priority": 6},
    "ai_bios_weekly": {"queue": "batch", "priority": 4},
    
    # Low priority - maintenance
    "cleanup_daily": {"queue": "low", "priority": 2},
    "quant_monthly": {"queue": "default", "priority": 4},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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
