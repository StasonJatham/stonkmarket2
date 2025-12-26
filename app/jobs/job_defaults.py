"""Shared defaults for scheduled jobs."""

from __future__ import annotations

from typing import Iterable, Tuple

from sqlalchemy.dialects.postgresql import insert

from app.database.connection import get_session
from app.database.orm import CronJob

DEFAULT_SCHEDULES: dict[str, Tuple[str, str]] = {
    "initial_data_ingest": ("*/15 * * * *", "Process queued symbols every 15 min"),
    "data_grab": ("0 23 * * 1-5", "Fetch stock data Mon-Fri 11pm"),
    "cache_warmup": ("*/30 * * * *", "Pre-cache chart data every 30 min"),
    "batch_ai_swipe": ("0 3 * * 0", "Generate swipe bios weekly Sunday 3am"),
    "batch_ai_analysis": ("0 4 * * 0", "Generate dip analysis weekly Sunday 4am"),
    "batch_poll": ("*/5 * * * *", "Poll for completed batch jobs every 5 min"),
    "fundamentals_refresh": ("0 2 1 * *", "Refresh stock fundamentals monthly 1st at 2am"),
    "ai_agents_analysis": ("0 5 * * 0", "AI agent analysis weekly Sunday 5am"),
    "ai_agents_batch_submit": ("0 3 * * 0", "Submit AI agent batch job weekly Sunday 3am"),
    "ai_agents_batch_collect": ("0 */4 * * *", "Collect AI agent batch results every 4 hours"),
    "portfolio_analytics_worker": ("*/5 * * * *", "Process queued portfolio analytics jobs"),
    "cleanup": ("0 0 * * *", "Clean up expired data daily midnight"),
}

JOB_PRIORITIES: dict[str, dict[str, int | str]] = {
    "data_grab": {"queue": "high", "priority": 9},
    "cache_warmup": {"queue": "high", "priority": 8},
    "batch_poll": {"queue": "high", "priority": 8},
    "process_new_symbol": {"queue": "default", "priority": 7},
    "process_approved_symbol": {"queue": "default", "priority": 7},
    "dipfinder_run": {"queue": "default", "priority": 6},
    "initial_data_ingest": {"queue": "default", "priority": 6},
    "fundamentals_refresh": {"queue": "default", "priority": 5},
    "ai_agents_analysis": {"queue": "default", "priority": 5},
    "portfolio_analytics_worker": {"queue": "default", "priority": 5},
    "dipfinder_refresh_all": {"queue": "batch", "priority": 4},
    "ai_agents_batch_submit": {"queue": "batch", "priority": 6},
    "ai_agents_batch_collect": {"queue": "batch", "priority": 7},
    "batch_ai_swipe": {"queue": "batch", "priority": 4},
    "batch_ai_analysis": {"queue": "batch", "priority": 4},
    "cleanup": {"queue": "low", "priority": 2},
}


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
                cron_expression=cron_expr,
                config={"description": description},
                is_active=True,
            ).on_conflict_do_nothing(index_elements=["name"])
            await session.execute(stmt)
        await session.commit()
