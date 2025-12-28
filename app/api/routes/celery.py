"""Celery monitoring routes (Flower proxy with Valkey fallback)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import require_admin
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.services.flower_service import fetch_flower
from app.services import celery_inspect


logger = get_logger("routes.celery")

router = APIRouter(prefix="/celery", tags=["Celery"])


@router.get(
    "/workers",
    summary="List Celery workers",
    description="Proxy to Flower workers endpoint with Celery inspect fallback (admin only).",
)
async def list_workers(
    admin: TokenData = Depends(require_admin),
) -> dict:
    # Try Flower first
    try:
        payload = await fetch_flower("api/workers")
        if isinstance(payload, dict) and payload:
            return payload
    except ExternalServiceError:
        logger.debug("Flower workers endpoint not available, using Celery inspect")

    # Fallback to Celery inspect
    return celery_inspect.list_workers()


@router.get(
    "/tasks",
    summary="List Celery tasks",
    description="Proxy to Flower tasks endpoint (admin only).",
)
async def list_tasks(
    limit: int = Query(100, ge=1, le=1000),
    admin: TokenData = Depends(require_admin),
) -> dict:
    try:
        return await fetch_flower("api/tasks", params={"limit": limit})
    except ExternalServiceError:
        logger.debug("Flower tasks endpoint not available")
        return {}


@router.get(
    "/tasks/{task_id}",
    summary="Get task details",
    description="Proxy to Flower task info endpoint (admin only).",
)
async def get_task_info(
    task_id: str = Path(..., min_length=1, max_length=200),
    admin: TokenData = Depends(require_admin),
) -> dict:
    try:
        return await fetch_flower(f"api/task/info/{task_id}")
    except ExternalServiceError:
        logger.debug("Flower task info endpoint not available")
        return {"task_id": task_id, "state": "UNKNOWN"}


@router.get(
    "/queues",
    summary="List Celery queues",
    description="Proxy to Flower queues endpoint with Valkey fallback (admin only).",
)
async def list_queues(
    admin: TokenData = Depends(require_admin),
) -> dict | list[dict]:
    # Try Flower first
    try:
        payload = await fetch_flower("api/queues")
        if payload:
            return payload
    except ExternalServiceError:
        logger.debug("Flower queues endpoint not available")

    # Try Celery inspect
    queues = celery_inspect.list_queues()
    if queues:
        return queues

    # Fallback to Valkey direct query
    queue_lengths = await celery_inspect.get_queue_lengths_from_valkey()
    return [
        {"name": name, "messages": length, "state": "active"}
        for name, length in queue_lengths.items()
    ]


@router.get(
    "/broker",
    summary="Get broker status",
    description="Proxy to Flower broker endpoint with Valkey fallback (admin only).",
)
async def get_broker_status(
    admin: TokenData = Depends(require_admin),
) -> dict:
    # Try Flower first
    try:
        return await fetch_flower("api/broker")
    except ExternalServiceError:
        logger.debug("Flower broker endpoint not available, using Valkey direct")

    # Fallback to Valkey info
    return await celery_inspect.get_broker_info_from_valkey()


@router.get(
    "/snapshot",
    summary="Get Celery snapshot",
    description="Fetch workers, queues, broker, and recent tasks in a single request (admin only).",
)
async def get_celery_snapshot(
    limit: int = Query(100, ge=1, le=1000),
    admin: TokenData = Depends(require_admin),
) -> dict:
    """Return a single snapshot payload for the Celery monitor UI."""
    workers_result, queues_result, broker_result, tasks_result = await asyncio.gather(
        list_workers(admin=admin),
        list_queues(admin=admin),
        get_broker_status(admin=admin),
        list_tasks(limit=limit, admin=admin),
        return_exceptions=True,
    )

    workers = workers_result if not isinstance(workers_result, Exception) else {}
    queues = queues_result if not isinstance(queues_result, Exception) else []
    broker = broker_result if not isinstance(broker_result, Exception) else {}

    tasks_payload = tasks_result if not isinstance(tasks_result, Exception) else {}
    tasks: list[dict] = []
    if isinstance(tasks_payload, dict):
        tasks = [
            {**info, "uuid": uuid}
            for uuid, info in tasks_payload.items()
        ]
        tasks.sort(key=lambda item: item.get("received", 0) or 0, reverse=True)
    elif isinstance(tasks_payload, list):
        tasks = tasks_payload

    return {
        "workers": workers,
        "queues": queues,
        "broker": broker,
        "tasks": tasks,
    }
