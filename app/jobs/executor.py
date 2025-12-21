"""Job execution with retry and error handling."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.core.exceptions import JobError
from app.core.logging import get_logger
from app.database import get_db_connection

from .registry import get_job

logger = get_logger("jobs.executor")


async def execute_job(
    name: str,
    conn: Optional[sqlite3.Connection] = None,
) -> str:
    """
    Execute a job by name.

    Args:
        name: Job name
        conn: Optional database connection (will create one if not provided)

    Returns:
        Job result message

    Raises:
        JobError: If job execution fails
    """
    job_func = get_job(name)
    if job_func is None:
        raise JobError(message=f"Unknown job: {name}", error_code="UNKNOWN_JOB")

    start_time = time.monotonic()

    try:
        if conn is None:
            for db_conn in get_db_connection():
                if asyncio.iscoroutinefunction(job_func):
                    result = await job_func(db_conn)
                else:
                    result = job_func(db_conn)
                break
        else:
            if asyncio.iscoroutinefunction(job_func):
                result = await job_func(conn)
            else:
                result = job_func(conn)

        duration = time.monotonic() - start_time
        message = str(result) if result else "Completed"
        logger.info(f"Job {name} executed in {duration:.2f}s: {message}")
        return message

    except Exception as e:
        duration = time.monotonic() - start_time
        logger.exception(f"Job {name} failed after {duration:.2f}s")
        raise JobError(
            message=f"Job execution failed: {str(e)}",
            error_code="JOB_EXECUTION_FAILED",
            details={"job_name": name, "duration_seconds": duration},
        ) from e


async def execute_job_with_retry(
    name: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    conn: Optional[sqlite3.Connection] = None,
) -> str:
    """
    Execute a job with exponential backoff retry.

    Args:
        name: Job name
        max_retries: Maximum retry attempts
        retry_delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay after each retry
        conn: Optional database connection

    Returns:
        Job result message

    Raises:
        JobError: If all retries exhausted
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await execute_job(name, conn)
        except JobError as e:
            last_error = e

            if attempt < max_retries:
                delay = retry_delay * (backoff_factor ** attempt)
                logger.warning(
                    f"Job {name} failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"Job {name} failed after {max_retries + 1} attempts")

    raise JobError(
        message=f"Job failed after {max_retries + 1} attempts: {last_error}",
        error_code="JOB_RETRIES_EXHAUSTED",
        details={"job_name": name, "attempts": max_retries + 1},
    )
