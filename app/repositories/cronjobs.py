"""Cron job repository - PostgreSQL async."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from app.database.connection import fetch_all, fetch_one, execute


class CronJobConfig:
    """Cron job configuration."""

    def __init__(self, name: str, cron: str, description: str | None = None):
        self.name = name
        self.cron = cron
        self.description = description

    @classmethod
    def from_row(cls, row) -> "CronJobConfig":
        return cls(
            name=row["name"],
            cron=row["cron"],
            description=row.get("description"),
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


async def list_cronjobs() -> List[CronJobConfig]:
    """List all active cron jobs."""
    rows = await fetch_all(
        """
        SELECT 
            name, 
            cron_expression as cron, 
            config->>'description' as description 
        FROM cronjobs 
        WHERE is_active = TRUE
        ORDER BY name ASC
        """
    )
    return [CronJobConfig.from_row(row) for row in rows]


async def list_cronjobs_with_stats() -> List[CronJobWithStats]:
    """List all active cron jobs with execution statistics."""
    rows = await fetch_all(
        """
        SELECT 
            name, 
            cron_expression as cron, 
            config->>'description' as description,
            last_run,
            last_status,
            last_duration_ms,
            run_count,
            error_count,
            last_error
        FROM cronjobs 
        WHERE is_active = TRUE
        ORDER BY name ASC
        """
    )
    return [
        CronJobWithStats(
            name=row["name"],
            cron=row["cron"],
            description=row.get("description"),
            last_run=row.get("last_run"),
            last_status=row.get("last_status"),
            last_duration_ms=row.get("last_duration_ms"),
            run_count=row.get("run_count", 0),
            error_count=row.get("error_count", 0),
            last_error=row.get("last_error"),
        )
        for row in rows
    ]


async def get_cronjob(name: str) -> Optional[CronJobConfig]:
    """Get a cron job by name."""
    row = await fetch_one(
        """
        SELECT 
            name, 
            cron_expression as cron, 
            config->>'description' as description 
        FROM cronjobs 
        WHERE name = $1
        """,
        name,
    )
    return CronJobConfig.from_row(row) if row else None


async def upsert_cronjob(name: str, cron: str) -> CronJobConfig:
    """Create or update a cron job."""
    await execute(
        """
        INSERT INTO cronjobs(name, cron_expression, is_active)
        VALUES ($1, $2, TRUE)
        ON CONFLICT(name) DO UPDATE SET
            cron_expression = EXCLUDED.cron_expression,
            updated_at = NOW()
        """,
        name,
        cron,
    )
    return await get_cronjob(name)  # type: ignore


async def update_job_stats(
    name: str, status: str, duration_ms: int, error: str | None = None
) -> None:
    """Update job execution statistics after a run."""
    if status == "ok":
        await execute(
            """
            UPDATE cronjobs SET
                last_run = NOW(),
                last_status = $2,
                last_duration_ms = $3,
                run_count = run_count + 1,
                last_error = NULL,
                updated_at = NOW()
            WHERE name = $1
            """,
            name,
            status,
            duration_ms,
        )
    else:
        await execute(
            """
            UPDATE cronjobs SET
                last_run = NOW(),
                last_status = $2,
                last_duration_ms = $3,
                run_count = run_count + 1,
                error_count = error_count + 1,
                last_error = $4,
                updated_at = NOW()
            WHERE name = $1
            """,
            name,
            status,
            duration_ms,
            error[:1000] if error else None,
        )
