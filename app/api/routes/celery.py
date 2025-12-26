"""Celery monitoring routes (Flower proxy)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import require_admin
from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.core.security import TokenData
from app.services.flower_service import fetch_flower


logger = get_logger("routes.celery")

router = APIRouter(prefix="/celery", tags=["Celery"])


@router.get(
    "/workers",
    summary="List Celery workers",
    description="Proxy to Flower workers endpoint (admin only).",
)
async def list_workers(
    admin: TokenData = Depends(require_admin),
) -> dict:
    return await fetch_flower("api/workers")


@router.get(
    "/tasks",
    summary="List Celery tasks",
    description="Proxy to Flower tasks endpoint (admin only).",
)
async def list_tasks(
    limit: int = Query(100, ge=1, le=1000),
    admin: TokenData = Depends(require_admin),
) -> dict:
    return await fetch_flower("api/tasks", params={"limit": limit})


@router.get(
    "/tasks/{task_id}",
    summary="Get task details",
    description="Proxy to Flower task info endpoint (admin only).",
)
async def get_task_info(
    task_id: str = Path(..., min_length=1, max_length=200),
    admin: TokenData = Depends(require_admin),
) -> dict:
    return await fetch_flower(f"api/task/info/{task_id}")


@router.get(
    "/queues",
    summary="List Celery queues",
    description="Proxy to Flower queues endpoint (admin only).",
)
async def list_queues(
    admin: TokenData = Depends(require_admin),
) -> dict:
    try:
        return await fetch_flower("api/queues")
    except ExternalServiceError:
        # Flower may not support queues endpoint - return empty
        logger.debug("Flower queues endpoint not available")
        return {}


@router.get(
    "/broker",
    summary="Get broker status",
    description="Proxy to Flower broker endpoint (admin only).",
)
async def get_broker_status(
    admin: TokenData = Depends(require_admin),
) -> dict:
    try:
        return await fetch_flower("api/broker")
    except ExternalServiceError:
        # Flower may not support broker endpoint - return empty
        logger.debug("Flower broker endpoint not available")
        return {}
