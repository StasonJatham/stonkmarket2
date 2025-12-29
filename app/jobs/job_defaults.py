"""Shared defaults for scheduled jobs.

Job Naming Convention:
    <domain>_<frequency> - Clear, descriptive names

Job Categories:
    1. REAL-TIME PROCESSING (every 5-15 min)
       - symbol_ingest: Process NEW symbols from queue
       - ai_batch_poll: Check OpenAI batch results
       - portfolio_worker: Process analytics queue
       - cache_warmup: Pre-cache chart data

    2. DAILY MARKET DATA (after market close, Mon-Fri)
       - prices_daily: Fetch closing prices (11 PM)
       - signals_daily: Technical signal scanner (10 PM)
       - regime_daily: Market regime detection (10:30 PM)
       - strategy_nightly: Backtest & optimize (11:30 PM)
       - quant_scoring_daily: Quant metrics (11:45 PM)
       - dipfinder_daily: Dip signals (11:50 PM)

    3. WEEKLY AI ANALYSIS (Sunday morning)
       - ai_personas_weekly: Warren Buffett, Peter Lynch etc.
       - ai_bios_weekly: Swipe-style stock bios

    4. WEEKLY MAINTENANCE (Sunday)
       - data_backfill: Fill ALL data gaps (comprehensive)

    5. MONTHLY MAINTENANCE
       - fundamentals_monthly: Refresh company fundamentals
       - quant_monthly: Portfolio optimization

    6. DAILY CLEANUP
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
    # Order matters: prices → signals → regime → strategy → quant → dipfinder
    # =========================================================================
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
    "prices_daily": (
        "0 23 * * 1-5",
        "Daily price update - fetches closing prices and updates dip states for all stocks. "
        "Runs Mon-Fri at 11 PM UTC (after US market close)."
    ),
    "strategy_nightly": (
        "30 23 * * 1-5",
        "Strategy optimization - runs full backtest with recency weighting, finds best strategy "
        "for each symbol that works NOW. Includes fundamental filters. "
        "Runs Mon-Fri at 11:30 PM UTC (30 min after prices_daily)."
    ),
    "quant_scoring_daily": (
        "45 23 * * 1-5",
        "Quant scoring - computes quantitative metrics (momentum, quality, value, volatility) "
        "for all tracked symbols. Runs Mon-Fri at 11:45 PM UTC."
    ),
    "dipfinder_daily": (
        "50 23 * * 1-5",
        "DipFinder signal refresh - computes dip metrics, scores, and enhanced analysis for all "
        "tracked symbols. Must run after quant_scoring_daily. Runs Mon-Fri at 11:50 PM UTC."
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
    # 4. WEEKLY MAINTENANCE - Data integrity
    # =========================================================================
    "data_backfill": (
        "0 2 * * 0",
        "Comprehensive data backfill - fills ALL data gaps: missing sectors, summaries, "
        "price history, fundamentals, quant scores, dipfinder signals. Sunday 2 AM UTC."
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
    # High priority - time-sensitive market data
    "prices_daily": {"queue": "high", "priority": 9},
    "ai_batch_poll": {"queue": "high", "priority": 8},
    "cache_warmup": {"queue": "high", "priority": 8},
    
    # Default priority - data ingestion & analysis
    "symbol_ingest": {"queue": "default", "priority": 7},
    "signals_daily": {"queue": "default", "priority": 7},
    "regime_daily": {"queue": "default", "priority": 6},
    "quant_scoring_daily": {"queue": "default", "priority": 6},
    "dipfinder_daily": {"queue": "default", "priority": 6},
    "fundamentals_monthly": {"queue": "default", "priority": 5},
    "portfolio_worker": {"queue": "default", "priority": 5},
    "quant_monthly": {"queue": "default", "priority": 4},
    
    # Batch queue - heavy computation / AI jobs
    "strategy_nightly": {"queue": "batch", "priority": 6},
    "ai_personas_weekly": {"queue": "batch", "priority": 6},
    "ai_bios_weekly": {"queue": "batch", "priority": 4},
    "data_backfill": {"queue": "batch", "priority": 3},
    
    # Low priority - maintenance
    "cleanup_daily": {"queue": "low", "priority": 2},
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
