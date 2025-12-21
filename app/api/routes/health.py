"""Health check endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.cache.client import valkey_healthcheck
from app.core.config import settings
from app.database import db_healthcheck
from app.schemas.common import HealthResponse

router = APIRouter(prefix="/health")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of the API and its dependencies.",
)
async def health_check() -> HealthResponse:
    """
    Perform health check on API and dependencies.

    Returns overall health status and individual service checks.
    """
    checks = {
        "database": await db_healthcheck(),
        "cache": await valkey_healthcheck(),
    }

    # Determine overall status
    if all(checks.values()):
        status = "healthy"
    elif checks.get("database", False):
        status = "degraded"  # DB ok but cache down
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )


@router.get(
    "/ready",
    summary="Readiness check",
    description="Check if the API is ready to accept traffic.",
)
async def readiness_check() -> dict:
    """
    Kubernetes-style readiness probe.

    Returns 200 if ready, useful for load balancer health checks.
    """
    db_ok = db_healthcheck()
    if not db_ok:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not ready",
        )
    return {"status": "ready"}


@router.get(
    "/live",
    summary="Liveness check",
    description="Check if the API process is alive.",
)
async def liveness_check() -> dict:
    """
    Kubernetes-style liveness probe.

    Simple check that the process is running.
    """
    return {"status": "alive"}
