"""CronJob-related schemas."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class CronJobResponse(BaseModel):
    """CronJob response schema."""

    name: str = Field(..., description="Job name")
    cron: str = Field(..., description="Cron expression")
    description: Optional[str] = Field(None, description="Job description")
    next_run: Optional[datetime] = Field(None, description="Next scheduled run time")

    model_config = {"from_attributes": True}


class CronJobUpdate(BaseModel):
    """CronJob update request schema."""

    cron: str = Field(
        ...,
        min_length=9,
        max_length=50,
        description="Cron expression (e.g., '0 6 * * 1-5')",
        examples=["0 6 * * 1-5"],
    )

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        """Validate cron expression."""
        from croniter import croniter, CroniterBadCronError

        v = v.strip()
        try:
            croniter(v)
        except (CroniterBadCronError, ValueError) as e:
            raise ValueError(f"Invalid cron expression: {e}")
        return v


class CronJobLogCreate(BaseModel):
    """CronJob log creation schema."""

    status: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Job status (ok/error)",
        examples=["ok", "error"],
    )
    message: Optional[str] = Field(
        None,
        max_length=10000,
        description="Log message",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate status value."""
        v = v.strip().lower()
        if v not in ("ok", "error", "running", "skipped"):
            raise ValueError("Status must be one of: ok, error, running, skipped")
        return v


class CronJobLogResponse(BaseModel):
    """CronJob log response schema."""

    name: str = Field(..., description="Job name")
    status: str = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Log message")
    created_at: datetime = Field(..., description="Log timestamp")

    model_config = {"from_attributes": True}


class CronJobWithStatsResponse(BaseModel):
    """CronJob response with execution stats."""

    name: str = Field(..., description="Job name")
    cron: str = Field(..., description="Cron expression")
    description: Optional[str] = Field(None, description="Job description")
    last_run: Optional[datetime] = Field(None, description="Last execution time")
    last_status: Optional[str] = Field(None, description="Last execution status")
    last_duration_ms: Optional[int] = Field(
        None, description="Last execution duration in ms"
    )
    run_count: int = Field(default=0, description="Total run count")
    error_count: int = Field(default=0, description="Total error count")
    last_error: Optional[str] = Field(None, description="Last error message")
    next_run: Optional[datetime] = Field(None, description="Next scheduled run time")

    model_config = {"from_attributes": True}


class CronJobLogListResponse(BaseModel):
    """Paginated cron job log list response."""

    logs: List[CronJobLogResponse] = Field(..., description="Log entries")
    total: int = Field(..., description="Total number of logs")
    limit: int = Field(default=50, description="Items per page")
    offset: int = Field(default=0, description="Current offset")


class JobStatusResponse(BaseModel):
    """Job execution status response."""

    name: str = Field(..., description="Job name")
    status: str = Field(..., description="Execution status")
    message: str = Field(..., description="Result message")
    task_id: Optional[str] = Field(None, description="Celery task id")
    duration_ms: Optional[int] = Field(None, description="Execution duration in ms")
    created_at: datetime = Field(..., description="Execution timestamp")


class TaskStatusResponse(BaseModel):
    """Celery task status response."""

    task_id: str = Field(..., description="Celery task id")
    status: str = Field(..., description="Celery task status")
    result: Optional[str] = Field(None, description="Task result (if completed)")
    error: Optional[str] = Field(None, description="Task error (if failed)")
    traceback: Optional[str] = Field(None, description="Task traceback (if failed)")
