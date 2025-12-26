"""Celery beat scheduler backed by cronjobs table."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from celery.beat import Scheduler, ScheduleEntry
from celery.schedules import crontab

from app.core.logging import get_logger
from app.jobs.job_defaults import get_job_schedule, get_job_priority, seed_cronjobs
from app.jobs.registry import list_job_names

logger = get_logger("jobs.celery_scheduler")

# Dedicated event loop for Celery Beat async operations
_beat_loop: asyncio.AbstractEventLoop | None = None


def _get_beat_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated event loop for beat scheduler."""
    global _beat_loop
    if _beat_loop is None or _beat_loop.is_closed():
        _beat_loop = asyncio.new_event_loop()
    return _beat_loop


def _run_async(coro: Any) -> Any:
    """Run async coroutine in the dedicated beat loop."""
    loop = _get_beat_loop()
    return loop.run_until_complete(coro)


def _cron_to_crontab(expr: str) -> crontab:
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {expr}")
    minute, hour, day_of_month, month_of_year, day_of_week = parts
    return crontab(
        minute=minute,
        hour=hour,
        day_of_week=day_of_week,
        day_of_month=day_of_month,
        month_of_year=month_of_year,
    )


def _load_cronjobs() -> list[dict[str, Any]]:
    from app.repositories import cronjobs_orm as cron_repo

    async def _fetch():
        return await cron_repo.list_cronjobs()

    return _run_async(_fetch())


class DatabaseScheduler(Scheduler):
    """Celery beat scheduler that reloads cronjobs from the database."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._last_reload = 0.0
        self._reload_interval = int(os.getenv("CELERY_CRON_SYNC_SECONDS", "60"))
        super().__init__(*args, **kwargs)

    def setup_schedule(self) -> None:
        import app.jobs.definitions  # noqa: F401 - register jobs

        job_names = list_job_names()
        try:
            _run_async(seed_cronjobs(job_names))
        except Exception as exc:
            logger.warning(f"Failed to seed cronjobs: {exc}")

        self._reload_schedule()

    def _reload_schedule(self) -> None:
        try:
            jobs = _load_cronjobs()
        except Exception as exc:
            logger.warning(f"Failed to load cronjobs: {exc}")
            return

        schedule: dict[str, ScheduleEntry] = {}
        for job in jobs:
            try:
                schedule_entry = ScheduleEntry(
                    name=job.name,
                    task=f"jobs.{job.name}",
                    schedule=_cron_to_crontab(job.cron),
                    args=(),
                    kwargs={},
                    options=get_job_priority(job.name),
                    last_run_at=self.app.now(),
                    total_run_count=0,
                )
                schedule[job.name] = schedule_entry
            except Exception as exc:
                logger.warning(f"Skipping cronjob {job.name}: {exc}")

        self.schedule = schedule
        self._last_reload = time.monotonic()
        logger.info(f"Loaded {len(schedule)} cronjobs from database")

    def tick(self, *args: Any, **kwargs: Any) -> float:
        if time.monotonic() - self._last_reload > self._reload_interval:
            self._reload_schedule()
        return super().tick(*args, **kwargs)
