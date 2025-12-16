from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ...api.deps import get_db, require_admin
from ...models import (
    CronJobLogPayload,
    CronJobLogResponse,
    CronJobResponse,
    CronJobUpdatePayload,
)
from ...repositories import cronjobs as cron_repo
from ...services.cron_runner import run_job

router = APIRouter(prefix="/cronjobs", tags=["cronjobs"], dependencies=[Depends(require_admin)])


@router.get("", response_model=List[CronJobResponse])
def list_all(conn=Depends(get_db)):
    jobs = cron_repo.list_cronjobs(conn)
    return [CronJobResponse(name=j.name, cron=j.cron, description=j.description) for j in jobs]


@router.put("/{name}", response_model=CronJobResponse)
def update(name: str, payload: CronJobUpdatePayload, conn=Depends(get_db)):
    existing = cron_repo.get_cronjob(conn, name)
    if existing is None:
        raise HTTPException(status_code=404, detail="Cron job not found")
    updated = cron_repo.upsert_cronjob(conn, name, payload.cron)
    return CronJobResponse(name=updated.name, cron=updated.cron, description=updated.description)


@router.post("/{name}/logs", status_code=status.HTTP_204_NO_CONTENT)
def add_log(name: str, payload: CronJobLogPayload, conn=Depends(get_db)):
    job = cron_repo.get_cronjob(conn, name)
    if job is None:
        raise HTTPException(status_code=404, detail="Cron job not found")
    cron_repo.insert_log(conn, name, payload.status, payload.message)
    return None


@router.get("/{name}/logs", response_model=List[CronJobLogResponse])
def get_logs(name: str, limit: int = 50, conn=Depends(get_db)):
    job = cron_repo.get_cronjob(conn, name)
    if job is None:
        raise HTTPException(status_code=404, detail="Cron job not found")
    limit = max(1, min(limit, 200))
    rows = cron_repo.list_logs(conn, name, limit)
    return [CronJobLogResponse(name=r[0], status=r[1], message=r[2], created_at=r[3]) for r in rows]


@router.post("/{name}/run", response_model=CronJobLogResponse)
def run_now(name: str, conn=Depends(get_db)):
    job = cron_repo.get_cronjob(conn, name)
    if job is None:
        raise HTTPException(status_code=404, detail="Cron job not found")
    try:
        message = run_job(conn, name)
        created_at = cron_repo.insert_log(conn, name, "ok", message)
        return CronJobLogResponse(name=name, status="ok", message=message, created_at=created_at)
    except HTTPException:
        raise
    except Exception as exc:
        created_at = cron_repo.insert_log(conn, name, "error", str(exc))
        raise HTTPException(status_code=500, detail="Cron run failed") from exc
