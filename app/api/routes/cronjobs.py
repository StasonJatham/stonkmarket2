"""CronJob management routes - PostgreSQL async."""

from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import require_admin
from app.core.exceptions import NotFoundError
from app.core.security import TokenData
from app.jobs import execute_job
from app.repositories import cronjobs as cron_repo
from app.schemas.cronjobs import (
    CronJobResponse,
    CronJobUpdate,
    CronJobWithStatsResponse,
    JobStatusResponse,
)

router = APIRouter()


def _validate_job_name(name: str = Path(..., min_length=1, max_length=50)) -> str:
    """Validate and normalize job name from path parameter."""
    return name.strip().lower()


@router.get(
    "/logs/all",
    response_model=dict,
    summary="Get cron job logs",
    description="Get execution logs for all cron jobs (admin only).",
)
async def get_cron_logs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    admin: TokenData = Depends(require_admin),
) -> dict:
    """Get cron job logs - currently returns last run info from jobs."""
    jobs = await cron_repo.list_cronjobs_with_stats()

    # Convert job stats to log-like entries
    logs = []
    for job in jobs:
        if job.last_run:
            log_entry = {
                "name": job.name,
                "status": job.last_status or "unknown",
                "message": job.last_error
                if job.last_status == "error"
                else f"Completed in {job.last_duration_ms}ms",
                "created_at": job.last_run.isoformat() if job.last_run else None,
                "duration_ms": job.last_duration_ms,
            }
            # Apply filters
            if search and search.lower() not in job.name.lower():
                continue
            if status and job.last_status != status:
                continue
            logs.append(log_entry)

    # Sort by created_at descending
    logs.sort(key=lambda x: x["created_at"] or "", reverse=True)

    # Apply pagination
    total = len(logs)
    logs = logs[offset : offset + limit]

    return {"logs": logs, "total": total}


@router.get(
    "",
    response_model=List[CronJobWithStatsResponse],
    summary="List all cron jobs",
    description="Get all configured cron jobs with execution stats (admin only).",
)
async def list_cronjobs(
    admin: TokenData = Depends(require_admin),
) -> List[CronJobWithStatsResponse]:
    """List all cron jobs with stats."""
    jobs = await cron_repo.list_cronjobs_with_stats()
    return [
        CronJobWithStatsResponse(
            name=j.name,
            cron=j.cron,
            description=j.description,
            last_run=j.last_run,
            last_status=j.last_status,
            last_duration_ms=j.last_duration_ms,
            run_count=j.run_count,
            error_count=j.error_count,
            last_error=j.last_error,
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
) -> CronJobResponse:
    """Get a specific cron job."""
    job = await cron_repo.get_cronjob(name)
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
) -> CronJobResponse:
    """Update a cron job's schedule."""
    existing = await cron_repo.get_cronjob(name)
    if existing is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )

    updated = await cron_repo.upsert_cronjob(name, payload.cron)
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
) -> JobStatusResponse:
    """Manually run a cron job."""
    job = await cron_repo.get_cronjob(name)
    if job is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )

    start_time = time.monotonic()

    try:
        message = await execute_job(name)
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Update job stats
        await cron_repo.update_job_stats(name, "ok", duration_ms)

        return JobStatusResponse(
            name=name,
            status="ok",
            message=message,
            duration_ms=duration_ms,
            created_at=datetime.utcnow(),
        )
    except Exception as exc:
        duration_ms = int((time.monotonic() - start_time) * 1000)

        # Update job stats with error
        await cron_repo.update_job_stats(name, "error", duration_ms, str(exc))

        return JobStatusResponse(
            name=name,
            status="error",
            message=str(exc),
            duration_ms=duration_ms,
            created_at=datetime.utcnow(),
        )
