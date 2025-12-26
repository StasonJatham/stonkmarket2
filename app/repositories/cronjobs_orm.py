"""Cron job repository using SQLAlchemy ORM.

This is the modern ORM-based implementation replacing raw SQL in cronjobs.py.

Usage:
    from app.repositories.cronjobs_orm import list_cronjobs, get_cronjob, upsert_cronjob
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.database.connection import get_session
from app.database.orm import CronJob as CronJobORM
from app.core.logging import get_logger

logger = get_logger("repositories.cronjobs_orm")


class CronJobConfig:
    """Cron job configuration."""

    def __init__(self, name: str, cron: str, description: str | None = None):
        self.name = name
        self.cron = cron
        self.description = description

    @classmethod
    def from_orm(cls, job: CronJobORM) -> "CronJobConfig":
        return cls(
            name=job.name,
            cron=job.cron,
            description=job.description,
        )


class CronJobWithStats(CronJobConfig):
    """Cron job with execution stats."""

    def __init__(
        self,
        name: str,
        cron: str,
        description: str | None = None,
        last_run: datetime | None = None,
        last_status: str | None = None,
        last_duration_ms: int | None = None,
        run_count: int = 0,
        error_count: int = 0,
        last_error: str | None = None,
    ):
        super().__init__(name, cron, description)
        self.last_run = last_run
        self.last_status = last_status
        self.last_duration_ms = last_duration_ms
        self.run_count = run_count
        self.error_count = error_count
        self.last_error = last_error

    @classmethod
    def from_orm(cls, job: CronJobORM) -> "CronJobWithStats":
        return cls(
            name=job.name,
            cron=job.cron,
            description=job.description,
            last_run=job.last_run,
            last_status=job.last_status,
            last_duration_ms=job.last_duration_ms,
            run_count=job.run_count or 0,
            error_count=job.error_count or 0,
            last_error=job.last_error,
        )


async def list_cronjobs() -> List[CronJobConfig]:
    """List all active cron jobs."""
    async with get_session() as session:
        result = await session.execute(
            select(CronJobORM)
            .where(CronJobORM.is_active == True)
            .order_by(CronJobORM.name)
        )
        jobs = result.scalars().all()
        return [CronJobConfig.from_orm(job) for job in jobs]


async def list_cronjobs_with_stats() -> List[CronJobWithStats]:
    """List all active cron jobs with execution statistics."""
    async with get_session() as session:
        result = await session.execute(
            select(CronJobORM)
            .where(CronJobORM.is_active == True)
            .order_by(CronJobORM.name)
        )
        jobs = result.scalars().all()
        return [CronJobWithStats.from_orm(job) for job in jobs]


async def get_cronjob(name: str) -> Optional[CronJobConfig]:
    """Get a cron job by name."""
    async with get_session() as session:
        result = await session.execute(
            select(CronJobORM).where(CronJobORM.name == name)
        )
        job = result.scalar_one_or_none()
        return CronJobConfig.from_orm(job) if job else None


async def upsert_cronjob(
    name: str, cron_expr: str, description: str | None = None
) -> CronJobConfig:
    """Create or update a cron job."""
    now = datetime.now(timezone.utc)
    config = {"description": description} if description else None
    
    async with get_session() as session:
        stmt = insert(CronJobORM).values(
            name=name,
            cron_expression=cron_expr,
            config=config,
            is_active=True,
        ).on_conflict_do_update(
            index_elements=["name"],
            set_={
                "cron_expression": cron_expr,
                "updated_at": now,
                **({"config": config} if description else {}),
            },
        )
        await session.execute(stmt)
        await session.commit()
    
    return await get_cronjob(name)  # type: ignore


async def update_job_stats(
    name: str, status: str, duration_ms: int, error: str | None = None
) -> None:
    """Update job execution statistics after a run."""
    now = datetime.now(timezone.utc)
    
    async with get_session() as session:
        result = await session.execute(
            select(CronJobORM).where(CronJobORM.name == name)
        )
        job = result.scalar_one_or_none()
        
        if job:
            job.last_run = now
            job.last_status = status
            job.last_duration_ms = duration_ms
            job.run_count = (job.run_count or 0) + 1
            job.updated_at = now
            
            if status == "ok":
                job.last_error = None
            else:
                job.error_count = (job.error_count or 0) + 1
                job.last_error = error[:1000] if error else None
            
            await session.commit()
