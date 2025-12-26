"""Task tracking helpers for symbol processing."""

from __future__ import annotations

from typing import Optional

from app.cache.cache import Cache

_TASK_CACHE = Cache(prefix="symbol_tasks", default_ttl=3600)


async def store_symbol_task(symbol: str, task_id: str) -> None:
    """Store the latest task id for a symbol."""
    await _TASK_CACHE.set(symbol.upper(), task_id)


async def get_symbol_task(symbol: str) -> Optional[str]:
    """Get the latest task id for a symbol."""
    cached = await _TASK_CACHE.get(symbol.upper())
    if isinstance(cached, str):
        return cached
    return None


async def clear_symbol_task(symbol: str) -> None:
    """Clear the latest task id for a symbol."""
    await _TASK_CACHE.delete(symbol.upper())
