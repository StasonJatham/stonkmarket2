"""CronJob management routes - PostgreSQL async."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Path, Query

from app.api.dependencies import require_admin
from app.core.exceptions import NotFoundError
from app.core.security import TokenData
from app.jobs import enqueue_job, get_task_status
from app.repositories import cronjobs_orm as cron_repo
from app.schemas.cronjobs import (
    CronJobResponse,
    CronJobUpdate,
    CronJobWithStatsResponse,
    JobStatusResponse,
    TaskStatusResponse,
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
            status = job.last_status or "unknown"
            if status == "error":
                message = job.last_error
            elif status == "skipped":
                message = job.last_error or "Skipped"
            elif status == "queued":
                message = "Queued"
            else:
                message = f"Completed in {job.last_duration_ms}ms"

            log_entry = {
                "name": job.name,
                "status": status,
                "message": message,
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
    from app.jobs import reschedule_job

    existing = await cron_repo.get_cronjob(name)
    if existing is None:
        raise NotFoundError(
            message=f"Cron job '{name}' not found",
            details={"name": name},
        )

    updated = await cron_repo.upsert_cronjob(name, payload.cron)

    # Reschedule the running job with new cron expression
    await reschedule_job(name, payload.cron)

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

    try:
        task_id = enqueue_job(name)
        return JobStatusResponse(
            name=name,
            status="queued",
            message="Job enqueued",
            task_id=task_id,
            created_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        return JobStatusResponse(
            name=name,
            status="error",
            message=str(exc),
            created_at=datetime.now(timezone.utc),
        )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get Celery task status",
    description="Fetch Celery task status from the result backend.",
)
async def get_celery_task_status(
    task_id: str = Path(..., min_length=1, max_length=200),
    admin: TokenData = Depends(require_admin),
) -> TaskStatusResponse:
    """Get Celery task status for admin monitoring."""
    status = get_task_status(task_id)
    return TaskStatusResponse(**status)
