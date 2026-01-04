"""Shared utilities for job definitions.

Common helpers and logging functions used across all job modules.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger


if TYPE_CHECKING:
    import pandas as pd

logger = get_logger("jobs.utils")


def log_job_success(job_name: str, message: str, **metrics: Any) -> None:
    """Log a structured job success message with metrics.
    
    Args:
        job_name: Name of the job (e.g., "cache_warmup")
        message: Human-readable summary message
        **metrics: Key-value pairs of metrics to include in structured log
    
    Example:
        log_job_success("cache_warmup", "Warmed 60 chart caches",
            items_warmed=60, symbols_cached=20, duration_ms=1234)
    """
    # Build structured log data
    log_data = {
        "job": job_name,
        "status": "success",
        **metrics,
    }
    
    # Log with structured data for JSON parsing, and human message for text logs
    # The message format is: "job_name completed: message | metrics_json"
    metrics_str = " ".join(f"{k}={v}" for k, v in metrics.items())
    logger.info(f"{job_name} completed: {message} | {metrics_str}", extra={"extra_fields": log_data})


def get_close_column(df: "pd.DataFrame") -> str:
    """Get the best close column name, preferring adjusted close.
    
    Adjusted close accounts for stock splits and dividends, making
    historical price comparisons accurate.
    
    Args:
        df: Price DataFrame with Close and/or Adj Close columns
        
    Returns:
        Column name to use ('Adj Close' if available, else 'Close')
    """
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        return "Adj Close"
    return "Close"


def job_timer() -> float:
    """Start a job timer.
    
    Returns:
        Start time from time.monotonic()
    
    Usage:
        job_start = job_timer()
        # ... do work ...
        duration_ms = elapsed_ms(job_start)
    """
    return time.monotonic()


def elapsed_ms(start: float) -> int:
    """Calculate elapsed time in milliseconds.
    
    Args:
        start: Start time from job_timer() or time.monotonic()
        
    Returns:
        Elapsed time in milliseconds (integer)
    """
    return int((time.monotonic() - start) * 1000)


def elapsed_seconds(start: float) -> float:
    """Calculate elapsed time in seconds.
    
    Args:
        start: Start time from job_timer() or time.monotonic()
        
    Returns:
        Elapsed time in seconds (float, 1 decimal)
    """
    return round(time.monotonic() - start, 1)
