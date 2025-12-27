"""Celery inspect fallback helpers."""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.logging import get_logger


logger = get_logger("services.celery_inspect")


def _get_celery_app():
    """Lazy import to avoid circular imports."""
    from app.celery_app import celery_app

    return celery_app


def list_workers(timeout: float = 1.0) -> Dict[str, Any]:
    """Return worker stats via Celery inspect (fallback when Flower is unavailable)."""
    try:
        celery_app = _get_celery_app()
        inspector = celery_app.control.inspect(timeout=timeout)
        stats = inspector.stats() or {}
        active = inspector.active() or {}
        scheduled = inspector.scheduled() or {}
        reserved = inspector.reserved() or {}

        workers: Dict[str, Any] = {}
        for name, info in stats.items():
            worker = dict(info)
            worker["status"] = "online"
            worker["active"] = active.get(name, [])
            worker["scheduled"] = scheduled.get(name, [])
            worker["reserved"] = reserved.get(name, [])
            workers[name] = worker

        for name, tasks in active.items():
            if name not in workers:
                workers[name] = {"status": "online", "active": tasks}

        return workers
    except Exception as e:
        logger.warning(f"Celery inspect failed: {e}")
        return {}


def list_queues(timeout: float = 1.0) -> List[Dict[str, Any]]:
    """Return active queues via Celery inspect (fallback when Flower is unavailable)."""
    try:
        celery_app = _get_celery_app()
        inspector = celery_app.control.inspect(timeout=timeout)
        active_queues = inspector.active_queues() or {}

        queues: Dict[str, Dict[str, Any]] = {}
        for entries in active_queues.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or entry.get("queue")
                if not name:
                    continue
                queue = queues.setdefault(
                    name, {"name": name, "messages": None, "consumers": 0, "state": "active"}
                )
                consumers = entry.get("consumers")
                if isinstance(consumers, int):
                    queue["consumers"] = (queue.get("consumers") or 0) + consumers
                for key in ("state", "messages"):
                    if entry.get(key) is not None:
                        queue[key] = entry.get(key)

        return list(queues.values())
    except Exception as e:
        logger.warning(f"Celery inspect queues failed: {e}")
        return []


async def get_queue_lengths_from_valkey() -> Dict[str, int]:
    """Get queue lengths directly from Valkey/Redis broker."""
    try:
        from app.cache.client import get_valkey_client
        
        client = await get_valkey_client()
        queue_lengths: Dict[str, int] = {}
        
        # Standard Celery queue names
        queue_names = ["celery", "default", "high", "low", "batch"]
        
        for queue_name in queue_names:
            try:
                length = await client.llen(queue_name)
                if length > 0 or queue_name in ["celery", "default"]:
                    queue_lengths[queue_name] = length
            except Exception:
                pass
        
        return queue_lengths
    except Exception as e:
        logger.warning(f"Failed to get queue lengths from Valkey: {e}")
        return {}


async def get_broker_info_from_valkey() -> Dict[str, Any]:
    """Get broker info directly from Valkey."""
    try:
        from app.cache.client import get_valkey_client
        
        client = await get_valkey_client()
        info = await client.info()
        
        return {
            "transport": "valkey",
            "connected": True,
            "redis_version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "uptime_in_seconds": info.get("uptime_in_seconds", 0),
        }
    except Exception as e:
        logger.warning(f"Failed to get broker info from Valkey: {e}")
        return {"transport": "valkey", "connected": False, "error": str(e)}
