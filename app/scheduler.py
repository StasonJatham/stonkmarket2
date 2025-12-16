from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Optional

from croniter import croniter, CroniterBadCronError

from .config import settings
from .db import init_db
from .repositories import cronjobs as cron_repo
from .services.cron_runner import run_job

logger = logging.getLogger("scheduler")


async def _sleep_until_next_minute():
    now = dt.datetime.utcnow()
    nxt = (now + dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
    await asyncio.sleep(max(0.0, (nxt - dt.datetime.utcnow()).total_seconds()))


def _is_due(cron_expr: str, now: dt.datetime) -> bool:
    try:
        return croniter.match(cron_expr, now)
    except CroniterBadCronError:
        logger.error("Invalid cron expression: %s", cron_expr)
        return False


async def _run_due_jobs(now: dt.datetime):
    conn = init_db()
    try:
        jobs = cron_repo.list_cronjobs(conn)
        for job in jobs:
            if not _is_due(job.cron, now):
                continue
            try:
                msg = run_job(conn, job.name)
                cron_repo.insert_log(conn, job.name, "ok", msg)
                logger.info("Ran job %s: %s", job.name, msg)
            except Exception as exc:  # noqa: BLE001
                cron_repo.insert_log(conn, job.name, "error", str(exc))
                logger.exception("Job %s failed", job.name)
    finally:
        conn.close()


async def scheduler_loop():
    # Align to minute boundary, then check every minute (UTC).
    await _sleep_until_next_minute()
    while True:
        now = dt.datetime.utcnow().replace(second=0, microsecond=0)
        await _run_due_jobs(now)
        await _sleep_until_next_minute()


def start_scheduler(app):
    if not settings.scheduler_enabled:
        logger.info("Scheduler disabled via SCHEDULER_ENABLED")
        return
    loop = asyncio.get_event_loop()
    task = loop.create_task(scheduler_loop())
    app.state.scheduler_task = task
    logger.info("Scheduler started")
