"""CronJob management routes - strict REST endpoints."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Path, Query, status

from app.api.dependencies import get_db, require_admin
from app.core.exceptions import NotFoundError
from app.core.security import TokenData
from app.jobs import execute_job
from app.repositories import cronjobs as cron_repo
from app.schemas.cronjobs import (
    CronJobLogCreate,
    CronJobLogListResponse,
    CronJobLogResponse,
    CronJobResponse,
    CronJobUpdate,
    JobStatusResponse,
)

router = APIRouter()


def _validate_job_name(name: str = Path(..., min_length=1, max_length=50)) -> str:
    """Validate and normalize job name from path parameter."""
    return name.strip().lower()


@router.get(
    "",
    response_model=List[CronJobResponse],
    summary="List all cron jobs",
    description="Get all configured cron jobs (admin only).",
)
async def list_cronjobs(
    admin: TokenData = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> List[CronJobResponse]:
    """List all cron jobs."""
    jobs = cron_repo.list_cronjobs(conn)
    return [
        CronJobResponse(
            name=j.name,
            cron=j.cron,
            description=j.description,
        )
        for j in jobs
    ]


@router.get(
    "/{name}",
    response_model=CronJobResponse,
    summary="Get cron job",
    description="Get a specific cron job's configuration.",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_cronjob(
    name: str = Depends(_validate_job_name),
    admin: TokenData = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> CronJobResponse:
    """Get a specific cron job."""
    job = cron_repo.get_cronjob(conn, name)
    if job is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )
    return CronJobResponse(
        name=job.name,
        cron=job.cron,
        description=job.description,
    )


@router.put(
    "/{name}",
    response_model=CronJobResponse,
    summary="Update cron job",
    description="Update a cron job's schedule.",
    responses={
        404: {"description": "Job not found"},
    },
)
async def update_cronjob(
    payload: CronJobUpdate,
    name: str = Depends(_validate_job_name),
    admin: TokenData = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> CronJobResponse:
    """Update a cron job's schedule."""
    existing = cron_repo.get_cronjob(conn, name)
    if existing is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )

    updated = cron_repo.upsert_cronjob(conn, name, payload.cron)
    return CronJobResponse(
        name=updated.name,
        cron=updated.cron,
        description=updated.description,
    )


@router.post(
    "/{name}/run",
    response_model=JobStatusResponse,
    summary="Run cron job now",
    description="Manually trigger a cron job execution.",
    responses={
        404: {"description": "Job not found"},
        500: {"description": "Job execution failed"},
    },
)
async def run_cronjob(
    name: str = Depends(_validate_job_name),
    admin: TokenData = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> JobStatusResponse:
    """Manually run a cron job."""
    job = cron_repo.get_cronjob(conn, name)
    if job is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )

    start_time = time.monotonic()

    try:
        message = await execute_job(name, conn)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Log success
        created_at = cron_repo.insert_log(conn, name, "ok", message)

        return JobStatusResponse(
            name=name,
            status="ok",
            message=message,
            duration_ms=duration_ms,
            created_at=datetime.fromisoformat(created_at),
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Log failure
        created_at = cron_repo.insert_log(conn, name, "error", str(exc)[:1000])

        return JobStatusResponse(
            name=name,
            status="error",
            message=str(exc),
            duration_ms=duration_ms,
            created_at=datetime.fromisoformat(created_at),
        )


@router.get(
    "/{name}/logs",
    response_model=List[CronJobLogResponse],
    summary="Get job logs",
    description="Get execution logs for a specific cron job.",
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_job_logs(
    name: str = Depends(_validate_job_name),
    limit: int = Query(default=50, ge=1, le=200, description="Max logs to return"),
    admin: TokenData = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> List[CronJobLogResponse]:
    """Get logs for a specific cron job."""
    job = cron_repo.get_cronjob(conn, name)
    if job is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )

    rows = cron_repo.list_logs(conn, name, limit)
    return [
        CronJobLogResponse(
            name=r[0],
            status=r[1],
            message=r[2],
            created_at=datetime.fromisoformat(r[3]),
        )
        for r in rows
    ]


@router.get(
    "/logs/all",
    response_model=CronJobLogListResponse,
    summary="Get all logs",
    description="Get all cron job logs with pagination.",
)
async def get_all_logs(
    limit: int = Query(default=50, ge=1, le=500, description="Max logs to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    search: Optional[str] = Query(default=None, max_length=100, description="Search filter"),
    status_filter: Optional[str] = Query(
        default=None, alias="status", description="Filter by status"
    ),
    admin: TokenData = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> CronJobLogListResponse:
    """Get all cron job logs with pagination."""
    rows, total = cron_repo.list_all_logs(conn, limit, offset, search, status_filter)
    logs = [
        CronJobLogResponse(
            name=r[0],
            status=r[1],
            message=r[2],
            created_at=datetime.fromisoformat(r[3]),
        )
        for r in rows
    ]
    return CronJobLogListResponse(
        logs=logs,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "/{name}/logs",
    status_code=status.HTTP_201_CREATED,
    response_model=CronJobLogResponse,
    summary="Add job log",
    description="Manually add a log entry for a cron job.",
    responses={
        404: {"description": "Job not found"},
    },
)
async def add_job_log(
    payload: CronJobLogCreate,
    name: str = Depends(_validate_job_name),
    admin: TokenData = Depends(require_admin),
    conn: sqlite3.Connection = Depends(get_db),
) -> CronJobLogResponse:
    """Manually add a log entry for a cron job."""
    job = cron_repo.get_cronjob(conn, name)
    if job is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )

    created_at = cron_repo.insert_log(conn, name, payload.status, payload.message)
    return CronJobLogResponse(
        name=name,
        status=payload.status,
        message=payload.message,
        created_at=datetime.fromisoformat(created_at),
    )
