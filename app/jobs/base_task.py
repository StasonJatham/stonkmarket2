"""Reliable Celery task base class with retry, DLQ, and observability."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from celery import Task

from app.core.logging import get_logger

logger = get_logger("jobs.base_task")

# Truncation limits for log entries (args can be huge for AI prompts)
# This only affects LOGGING - actual task args are never truncated
LOG_ARGS_LIMIT_RETRY = 500  # Shorter for retries (many entries possible)
LOG_ARGS_LIMIT_DLQ = 2000  # Longer for DLQ (need debug context)
LOG_KWARGS_LIMIT = 1000
LOG_TRACEBACK_LIMIT = 4000


class ReliableTask(Task):
    """Base task class with automatic retries, DLQ logging, and observability.

    Features:
        - Automatic retry with exponential backoff (3 attempts)
        - Dead Letter Queue (DLQ) logging on final failure
        - Structured logging for retries and failures
        - Time limit enforcement from job_defaults

    Usage:
        @celery_app.task(base=ReliableTask, bind=True)
        def my_task(self):
            ...
    """

    # Retry configuration
    autoretry_for = (Exception,)  # Retry on any exception
    max_retries = 3  # Give up after 3 attempts
    retry_backoff = True  # Exponential backoff
    retry_backoff_max = 600  # Max 10 minutes between retries
    retry_jitter = True  # Add randomness to prevent thundering herd

    # Don't auto-acknowledge until task completes (late ack)
    acks_late = True

    # Reject and requeue on worker crash
    reject_on_worker_lost = True

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Log retry attempts with structured context.
        
        NOTE: Args are truncated for logging only. The actual task
        receives full arguments - this just prevents huge log entries.
        """
        logger.warning(
            "Task retrying",
            extra={
                "task_name": self.name,
                "task_id": task_id,
                "attempt": self.request.retries + 1,
                "max_retries": self.max_retries,
                "exception": str(exc),
                "exception_type": type(exc).__name__,
                "args": str(args)[:LOG_ARGS_LIMIT_RETRY],
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Log to DLQ after all retries exhausted.
        
        NOTE: Args/kwargs are truncated for logging only. This prevents
        multi-MB log entries from AI prompts while preserving enough
        context for debugging.
        """
        # DLQ entry - structured log that can be parsed for alerting
        logger.error(
            "DEAD_LETTER_QUEUE: Task failed permanently",
            extra={
                "dlq": True,  # Flag for log aggregation
                "task_name": self.name,
                "task_id": task_id,
                "attempts": self.request.retries + 1,
                "exception": str(exc),
                "exception_type": type(exc).__name__,
                "task_args": str(args)[:LOG_ARGS_LIMIT_DLQ],
                "task_kwargs": str(kwargs)[:LOG_KWARGS_LIMIT],
                "traceback": str(einfo)[:LOG_TRACEBACK_LIMIT] if einfo else None,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Log successful completion (debug level for observability)."""
        logger.debug(
            "Task completed successfully",
            extra={
                "task_name": self.name,
                "task_id": task_id,
                "result": str(retval)[:200] if retval else None,
            },
        )

    def before_start(
        self,
        task_id: str,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Record task start time for duration tracking."""
        self.request._start_time = time.monotonic()
        logger.debug(
            "Task starting",
            extra={
                "task_name": self.name,
                "task_id": task_id,
                "attempt": self.request.retries + 1,
            },
        )

    def after_return(
        self,
        status: str,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Log task completion with duration."""
        start_time = getattr(self.request, "_start_time", None)
        duration_ms = None
        if start_time:
            duration_ms = int((time.monotonic() - start_time) * 1000)

        if status == "SUCCESS":
            logger.info(
                "Task finished",
                extra={
                    "task_name": self.name,
                    "task_id": task_id,
                    "status": status,
                    "duration_ms": duration_ms,
                },
            )
        elif status == "FAILURE":
            logger.error(
                "Task failed",
                extra={
                    "task_name": self.name,
                    "task_id": task_id,
                    "status": status,
                    "duration_ms": duration_ms,
                    "exception": str(einfo) if einfo else None,
                },
            )
