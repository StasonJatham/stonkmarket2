"""Job scheduler using APScheduler with async support."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from croniter import croniter

from app.core.config import settings
from app.core.logging import get_logger
from app.database import get_db_connection, init_db

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

        # Initialize database
        init_db()

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
        """Load job schedules from database."""
        from app.repositories.cronjobs import list_cronjobs

        # Get database connection
        for conn in get_db_connection():
            jobs = list_cronjobs(conn)
            break

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
        from app.repositories.cronjobs import insert_log

        lock = DistributedLock(f"job:{name}", timeout=60 * 30)  # 30 min max

        try:
            acquired = await lock.acquire()
            if not acquired:
                logger.info(f"Job {name} skipped - already running on another instance")
                return

            logger.info(f"Job {name} started")
            start_time = datetime.now(timezone.utc)

            try:
                # Get fresh connection for job execution
                for conn in get_db_connection():
                    if asyncio.iscoroutinefunction(func):
                        result = await func(conn)
                    else:
                        result = func(conn)
                    break

                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                message = str(result) if result else "Completed successfully"

                # Log success
                for conn in get_db_connection():
                    insert_log(conn, name, "ok", f"{message} ({duration:.2f}s)")
                    break

                logger.info(f"Job {name} completed in {duration:.2f}s")

            except Exception as e:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()

                # Log error
                for conn in get_db_connection():
                    insert_log(conn, name, "error", str(e)[:1000])
                    break

                logger.exception(f"Job {name} failed after {duration:.2f}s")

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

    def get_jobs_status(self) -> list:
        """Get status of all scheduled jobs."""
        jobs = []
        for job in self._scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "pending": job.pending,
            })
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
