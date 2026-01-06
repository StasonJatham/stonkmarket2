"""
Centralized Data Conversion Helpers.

This module provides safe type conversion utilities used across the codebase.
All data conversion helpers should be imported from here to avoid duplication.

Usage:
    from app.core.data_helpers import safe_float, safe_int, safe_date, latest_value
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any


def _is_na(value: Any) -> bool:
    """Check if value is pandas NA/NaT without requiring pandas import."""
    try:
        import pandas as pd
        return pd.isna(value)
    except ImportError:
        return False


def safe_float(value: Any, default: float | None = None) -> float | None:
    """
    Safely convert value to float.
    
    Handles None, NaN, Inf, pandas NA/NaT, and conversion errors gracefully.
    
    Args:
        value: Any value to convert
        default: Default to return if conversion fails
        
    Returns:
        Float value or default if conversion fails
    """
    if value is None:
        return default
    if _is_na(value):
        return default
    try:
        f = float(value)
        # Check for NaN and Inf
        if f != f or f == float('inf') or f == float('-inf'):
            return default
        return f
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int | None = None) -> int | None:
    """
    Safely convert value to int.
    
    Handles None, pandas NA/NaT, and conversion errors gracefully.
    
    Args:
        value: Any value to convert
        default: Default to return if conversion fails
        
    Returns:
        Int value or default if conversion fails
    """
    if value is None:
        return default
    if _is_na(value):
        return default
    try:
        # Handle float strings like "123.0"
        return int(float(value))
    except (ValueError, TypeError):
        return default


def safe_date(value: Any) -> date | None:
    """
    Safely convert value to date.
    
    Handles datetime, date, ISO strings, and timestamps.
    
    Args:
        value: Any value to convert
        
    Returns:
        Date value or None if conversion fails
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        # Try ISO format first
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace('Z', '+00:00')).date()
        # Try timestamp
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value).date()
    except (ValueError, TypeError, OSError):
        pass
    return None


def latest_value(value: Any) -> float | None:
    """
    Extract the most recent value from dict, list, or scalar.
    
    Handles yfinance's inconsistent data formats where values can be:
    - A scalar (float/int)
    - A dict with date keys {"2024-01-01": 100, "2024-06-01": 110}
    - A list [100, 110, 120]
    
    Args:
        value: Any value (dict, list, or scalar)
        
    Returns:
        Most recent float value or None
    """
    if isinstance(value, dict):
        # Sort keys by date (descending) and get first non-None value
        for key in _sorted_period_keys(list(value.keys())):
            v = value.get(key)
            if v is not None:
                return safe_float(v)
        return None
    if isinstance(value, list):
        # Return first non-None value
        for item in value:
            if item is not None:
                return safe_float(item)
        return None
    return safe_float(value)


def _sorted_period_keys(keys: list[Any]) -> list[Any]:
    """Sort period keys by date descending (most recent first)."""
    def _key(item: Any) -> datetime:
        try:
            return datetime.fromisoformat(str(item))
        except (ValueError, TypeError):
            return datetime.min
    return sorted(keys, key=_key, reverse=True)


async def run_in_executor(func: Any, *args: Any) -> Any:
    """
    Run a blocking function in the default thread pool.
    
    Use this to wrap blocking I/O calls (like yfinance) in async code.
    
    Args:
        func: Blocking function to run
        *args: Arguments to pass to the function
        
    Returns:
        Result of the function call
    """
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


def pct_change(
    current: float | None,
    previous: float | None,
    *,
    as_percent: bool = False,
) -> float | None:
    """
    Compute percentage change between two values.
    
    Args:
        current: Current/new value
        previous: Previous/old value (base for comparison)
        as_percent: If True, multiply by 100 (e.g., 0.05 -> 5.0)
        
    Returns:
        Percentage change as decimal (0.05) or percent (5.0), or None if invalid
        
    Examples:
        >>> pct_change(110, 100)
        0.1
        >>> pct_change(110, 100, as_percent=True)
        10.0
        >>> pct_change(50, 100)
        -0.5
    """
    if current is None or previous is None or previous == 0:
        return None
    result = (current - previous) / abs(previous)
    return result * 100 if as_percent else result


__all__ = [
    "safe_float",
    "safe_int",
    "safe_date",
    "latest_value",
    "run_in_executor",
    "pct_change",
]
