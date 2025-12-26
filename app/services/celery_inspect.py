"""Celery inspect fallback helpers."""

from __future__ import annotations

from typing import Any, Dict, List


def _get_celery_app():
    """Lazy import to avoid circular imports."""
    from app.celery_app import celery_app

    return celery_app


def list_workers(timeout: float = 1.0) -> Dict[str, Any]:
    """Return worker stats via Celery inspect (fallback when Flower is unavailable)."""
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


def list_queues(timeout: float = 1.0) -> List[Dict[str, Any]]:
    """Return active queues via Celery inspect (fallback when Flower is unavailable)."""
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
