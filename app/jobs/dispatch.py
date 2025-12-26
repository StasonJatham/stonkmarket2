"""Celery job dispatch utilities."""

from __future__ import annotations

from typing import Any

from celery.result import AsyncResult

from app.celery_app import celery_app
from app.core.exceptions import JobError
from app.jobs.registry import get_job


def enqueue_job(name: str) -> str:
    """Submit a registered job to Celery and return the task id."""
    if get_job(name) is None:
        raise JobError(message=f"Unknown job: {name}", error_code="UNKNOWN_JOB")

    result = celery_app.send_task(f"jobs.{name}")
    return result.id


def get_task_status(task_id: str) -> dict[str, Any]:
    """Fetch Celery task status from the result backend."""
    result = AsyncResult(task_id, app=celery_app)
    payload: dict[str, Any] = {
        "task_id": task_id,
        "status": result.status,
    }

    if result.successful():
        payload["result"] = result.result
    elif result.failed():
        payload["error"] = str(result.result)

    if result.traceback:
        payload["traceback"] = result.traceback

    return payload
