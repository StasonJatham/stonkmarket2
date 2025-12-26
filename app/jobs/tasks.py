"""Celery tasks for background jobs."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import app.jobs.definitions  # noqa: F401 - register jobs

from app.celery_app import celery_app
from app.core.logging import get_logger
from app.jobs.executor import execute_job
from app.repositories import cronjobs_orm as cron_repo

logger = get_logger("jobs.celery_tasks")


def _run_async(coro: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


async def _execute_job_locked(job_name: str) -> str:
    from app.cache.distributed_lock import DistributedLock

    lock = DistributedLock(f"job:{job_name}", timeout=60 * 30, blocking=False)
    acquired = await lock.acquire()
    if not acquired:
        try:
            await cron_repo.update_job_stats(
                job_name, "skipped", 0, "Already running"
            )
        except Exception as stats_exc:
            logger.warning(f"Failed to update job stats for {job_name}: {stats_exc}")
        return f"Skipped {job_name}: already running"

    start = time.monotonic()
    try:
        message = await execute_job(job_name)
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            await cron_repo.update_job_stats(job_name, "ok", duration_ms)
        except Exception as stats_exc:
            logger.warning(f"Failed to update job stats for {job_name}: {stats_exc}")
        return message
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        try:
            await cron_repo.update_job_stats(job_name, "error", duration_ms, str(exc))
        except Exception as stats_exc:
            logger.warning(f"Failed to update job stats for {job_name}: {stats_exc}")
        logger.exception("Job failed", extra={"job": job_name})
        raise
    finally:
        await lock.release()


def _run_job(job_name: str) -> str:
    return _run_async(_execute_job_locked(job_name))


@celery_app.task(name="jobs.initial_data_ingest")
def initial_data_ingest_task() -> str:
    return _run_job("initial_data_ingest")


@celery_app.task(name="jobs.data_grab")
def data_grab_task() -> str:
    return _run_job("data_grab")


@celery_app.task(name="jobs.cache_warmup")
def cache_warmup_task() -> str:
    return _run_job("cache_warmup")


@celery_app.task(name="jobs.batch_ai_swipe")
def batch_ai_swipe_task() -> str:
    return _run_job("batch_ai_swipe")


@celery_app.task(name="jobs.batch_ai_analysis")
def batch_ai_analysis_task() -> str:
    return _run_job("batch_ai_analysis")


@celery_app.task(name="jobs.batch_poll")
def batch_poll_task() -> str:
    return _run_job("batch_poll")


@celery_app.task(name="jobs.fundamentals_refresh")
def fundamentals_refresh_task() -> str:
    return _run_job("fundamentals_refresh")


@celery_app.task(name="jobs.ai_agents_analysis")
def ai_agents_analysis_task() -> str:
    return _run_job("ai_agents_analysis")


@celery_app.task(name="jobs.ai_agents_batch_submit")
def ai_agents_batch_submit_task() -> str:
    return _run_job("ai_agents_batch_submit")


@celery_app.task(name="jobs.ai_agents_batch_collect")
def ai_agents_batch_collect_task() -> str:
    return _run_job("ai_agents_batch_collect")


@celery_app.task(name="jobs.cleanup")
def cleanup_task() -> str:
    return _run_job("cleanup")


@celery_app.task(name="jobs.portfolio_analytics_worker")
def portfolio_analytics_worker_task() -> str:
    return _run_job("portfolio_analytics_worker")
