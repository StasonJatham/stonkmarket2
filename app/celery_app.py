"""Celery application setup for background jobs."""

from __future__ import annotations

import os

from celery import Celery
from kombu import Queue

from app.core.config import settings
from app.jobs.job_defaults import JOB_PRIORITIES


def _build_task_routes() -> dict[str, dict[str, int | str]]:
    routes: dict[str, dict[str, int | str]] = {}
    for job_name, config in JOB_PRIORITIES.items():
        routes[f"jobs.{job_name}"] = {
            "queue": config["queue"],
            "priority": config["priority"],
        }
    return routes


broker_url = os.getenv("CELERY_BROKER_URL", settings.valkey_url)
result_backend = os.getenv("CELERY_RESULT_BACKEND", broker_url)

celery_app = Celery("stonkmarket", broker=broker_url, backend=result_backend)

celery_app.conf.update(
    timezone=settings.scheduler_timezone,
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_queue="default",
    task_default_priority=5,
    task_queue_max_priority=9,
    task_routes=_build_task_routes(),
    broker_transport_options={
        "visibility_timeout": 60 * 60,
        "priority_steps": list(range(10)),
    },
    task_queues=(
        Queue("high", routing_key="high", max_priority=9),
        Queue("default", routing_key="default", max_priority=9),
        Queue("low", routing_key="low", max_priority=9),
        Queue("batch", routing_key="batch", max_priority=9),
    ),
    beat_scheduler="app.jobs.celery_scheduler:DatabaseScheduler",
    beat_max_loop_interval=30,
)

# Register tasks
celery_app.autodiscover_tasks(["app.jobs"])
