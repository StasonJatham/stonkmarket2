"""Health check endpoints."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from app.cache.client import valkey_healthcheck
from app.core.config import settings
from app.core.logging import get_logger
from app.database.connection import get_pg_pool
from app.schemas.common import HealthResponse


router = APIRouter(prefix="/health")

logger = get_logger("health")


async def db_healthcheck() -> bool:
    """Check PostgreSQL database health."""
    try:
        pool = await get_pg_pool()
        if pool is None:
            return False
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.warning(f"Database healthcheck failed: {e}")
        return False


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
        timestamp=datetime.now(UTC),
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
    db_ok = await db_healthcheck()
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


@router.get(
    "/cache",
    summary="Cache statistics",
    description="Get cache hit/miss statistics for monitoring.",
)
async def cache_stats() -> dict:
    """
    Get cache performance statistics.

    Returns hit rates, error counts, and timing information for each cache prefix.
    Useful for monitoring and debugging cache performance.
    """
    from app.cache.metrics import cache_metrics

    return cache_metrics.get_summary()


@router.get(
    "/db",
    summary="Database health details",
    description="Get detailed database health and connection pool statistics.",
)
async def db_health_details() -> dict:
    """
    Get detailed database health information.

    Returns connection pool stats, database version, and diagnostic info.
    Useful for monitoring pool utilization and identifying connection issues.
    """
    result = {
        "status": "unknown",
        "pool": None,
        "database": None,
        "error": None,
    }

    try:
        pool = await get_pg_pool()
        if pool is None:
            result["status"] = "unhealthy"
            result["error"] = "Connection pool not initialized"
            return result

        # Pool statistics
        result["pool"] = {
            "size": pool.get_size(),
            "free_size": pool.get_idle_size(),
            "used_size": pool.get_size() - pool.get_idle_size(),
            "min_size": pool.get_min_size(),
            "max_size": pool.get_max_size(),
        }

        # Database diagnostics
        async with pool.acquire() as conn:
            # Get PostgreSQL version
            version = await conn.fetchval("SELECT version()")

            # Get current connections
            stats = await conn.fetchrow("""
                SELECT
                    numbackends as active_connections,
                    xact_commit as transactions_committed,
                    xact_rollback as transactions_rolled_back,
                    blks_read as blocks_read,
                    blks_hit as blocks_hit,
                    tup_returned as rows_returned,
                    tup_fetched as rows_fetched
                FROM pg_stat_database
                WHERE datname = current_database()
            """)

            result["database"] = {
                "version": version,
                "active_connections": stats["active_connections"] if stats else None,
                "cache_hit_ratio": (
                    round(stats["blocks_hit"] / (stats["blocks_read"] + stats["blocks_hit"]) * 100, 2)
                    if stats and (stats["blocks_read"] + stats["blocks_hit"]) > 0
                    else None
                ),
            }

        result["status"] = "healthy"

    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        result["status"] = "unhealthy"
        result["error"] = str(e)

    return result
