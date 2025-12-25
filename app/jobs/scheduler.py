"""Job scheduler using APScheduler with async support."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import settings
from app.core.logging import get_logger

from .registry import get_job, get_all_jobs

logger = get_logger("jobs.scheduler")

# Global scheduler instance
_scheduler: Optional["JobScheduler"] = None


class JobScheduler:
    """Background job scheduler with distributed locking support."""

    def __init__(self):
        self._scheduler = AsyncIOScheduler(
            timezone=settings.scheduler_timezone,
            job_defaults={
                "coalesce": True,  # Combine missed runs into one
                "max_instances": 1,  # Only one instance per job at a time
                "misfire_grace_time": 60 * 5,  # 5 minutes grace period
            },
        )
        self._running = False

    async def start(self) -> None:
        """Start the scheduler and load jobs from database."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        if not settings.scheduler_enabled:
            logger.info("Scheduler disabled via SCHEDULER_ENABLED=false")
            return

        # Load and schedule jobs from database
        await self._load_jobs()

        # Start scheduler
        self._scheduler.start()
        self._running = True
        logger.info("Job scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._scheduler.shutdown(wait=True)
        self._running = False
        logger.info("Job scheduler stopped")

    async def _load_jobs(self) -> None:
        """Load job schedules from database, seeding any missing registered jobs."""
        from app.repositories import cronjobs as cron_repo
        from app.database.connection import execute

        # First, seed any registered jobs that don't exist in DB
        registered_jobs = get_all_jobs()

        # Define default schedules for registered jobs
        default_schedules = {
            "initial_data_ingest": ("*/15 * * * *", "Process queued symbols every 15 min"),
            "data_grab": ("0 23 * * 1-5", "Fetch stock data Mon-Fri 11pm"),
            "cache_warmup": ("*/30 * * * *", "Pre-cache chart data every 30 min"),
            "batch_ai_swipe": ("0 3 * * 0", "Generate swipe bios weekly Sunday 3am"),
            "batch_ai_analysis": (
                "0 4 * * 0",
                "Generate dip analysis weekly Sunday 4am",
            ),
            "batch_poll": ("*/5 * * * *", "Poll for completed batch jobs every 5 min"),
            "fundamentals_refresh": ("0 2 1 * *", "Refresh stock fundamentals monthly 1st at 2am"),
            "ai_agents_analysis": ("0 5 * * 0", "AI agent analysis weekly Sunday 5am"),
            "cleanup": ("0 0 * * *", "Clean up expired data daily midnight"),
        }

        for job_name in registered_jobs:
            cron_expr, description = default_schedules.get(
                job_name, ("0 * * * *", f"Job: {job_name}")
            )
            try:
                await execute(
                    """
                    INSERT INTO cronjobs (name, cron_expression, is_active, config)
                    VALUES ($1, $2, TRUE, $3)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    job_name,
                    cron_expr,
                    f'{{"description": "{description}"}}',
                )
            except Exception as e:
                logger.warning(f"Failed to seed job {job_name}: {e}")

        # Now load and schedule all jobs from database
        jobs = await cron_repo.list_cronjobs()

        for job_config in jobs:
            job_func = get_job(job_config.name)
            if job_func is None:
                logger.warning(f"Unknown job: {job_config.name}")
                continue

            try:
                trigger = CronTrigger.from_crontab(job_config.cron)
                self._scheduler.add_job(
                    self._wrap_job(job_config.name, job_func),
                    trigger=trigger,
                    id=job_config.name,
                    name=job_config.description or job_config.name,
                    replace_existing=True,
                )
                logger.info(f"Scheduled job: {job_config.name} ({job_config.cron})")
            except Exception as e:
                logger.error(f"Failed to schedule job {job_config.name}: {e}")

    def _wrap_job(self, name: str, func: Callable) -> Callable:
        """Wrap job function with locking and logging."""

        async def wrapper():
            await self._execute_job(name, func)

        return wrapper

    async def _execute_job(self, name: str, func: Callable) -> None:
        """Execute a job with distributed locking and logging."""
        from app.cache.distributed_lock import DistributedLock
        from app.repositories import cronjobs as cron_repo

        lock = DistributedLock(f"job:{name}", timeout=60 * 30)  # 30 min max

        try:
            acquired = await lock.acquire()
            if not acquired:
                logger.info(f"Job {name} skipped - already running on another instance")
                return

            logger.info(f"Job {name} started")
            start_time = datetime.now(timezone.utc)

            try:
                # Execute the job (no connection needed anymore)
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()

                duration_ms = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )
                result_msg = str(result) if result else "Completed successfully"

                # Update job stats in database
                await cron_repo.update_job_stats(name, "ok", duration_ms)

                logger.info(f"Job {name} completed in {duration_ms}ms")

            except Exception as e:
                duration_ms = int(
                    (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                )

                # Update job stats with error
                await cron_repo.update_job_stats(name, "error", duration_ms, str(e))

                logger.exception(f"Job {name} failed after {duration_ms}ms")

        finally:
            await lock.release()

    async def run_job_now(self, name: str) -> str:
        """Manually trigger a job execution."""
        job_func = get_job(name)
        if job_func is None:
            raise ValueError(f"Unknown job: {name}")

        await self._execute_job(name, job_func)
        return f"Job {name} executed"

    def get_next_run_time(self, name: str) -> Optional[datetime]:
        """Get next scheduled run time for a job."""
        job = self._scheduler.get_job(name)
        if job:
            return job.next_run_time
        return None

    async def reschedule_job(self, name: str, cron_expression: str) -> bool:
        """Reschedule a job with a new cron expression."""
        job_func = get_job(name)
        if job_func is None:
            logger.warning(f"Cannot reschedule unknown job: {name}")
            return False

        try:
            trigger = CronTrigger.from_crontab(cron_expression)
            
            # Check if job exists
            existing_job = self._scheduler.get_job(name)
            if existing_job:
                # Reschedule existing job
                self._scheduler.reschedule_job(name, trigger=trigger)
                logger.info(f"Rescheduled job: {name} with cron: {cron_expression}")
            else:
                # Add new job
                self._scheduler.add_job(
                    self._wrap_job(name, job_func),
                    trigger=trigger,
                    id=name,
                    name=name,
                    replace_existing=True,
                )
                logger.info(f"Added job: {name} with cron: {cron_expression}")
            return True
        except Exception as e:
            logger.error(f"Failed to reschedule job {name}: {e}")
            return False

    def get_jobs_status(self) -> list:
        """Get status of all scheduled jobs."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_time": job.next_run_time.isoformat()
                    if job.next_run_time
                    else None,
                    "pending": job.pending,
                }
            )
        return jobs


def get_scheduler() -> Optional[JobScheduler]:
    """Get the global scheduler instance."""
    global _scheduler
    return _scheduler


async def start_scheduler() -> JobScheduler:
    """Start the global scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = JobScheduler()
    await _scheduler.start()
    return _scheduler


async def stop_scheduler() -> None:
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler is not None:
        await _scheduler.stop()
        _scheduler = None
