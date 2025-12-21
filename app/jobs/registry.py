"""Job registry for mapping job names to functions."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from app.core.logging import get_logger

logger = get_logger("jobs.registry")

# Global job registry
_registry: Dict[str, Callable] = {}


def register_job(name: str) -> Callable:
    """
    Decorator to register a job function.

    Usage:
        @register_job("data_grab")
        def data_grab_job(conn):
            # Job implementation
            ...
    """

    def decorator(func: Callable) -> Callable:
        _registry[name] = func
        logger.debug(f"Registered job: {name}")
        return func

    return decorator


def get_job(name: str) -> Optional[Callable]:
    """Get a registered job function by name."""
    return _registry.get(name)


def get_all_jobs() -> Dict[str, Callable]:
    """Get all registered jobs."""
    return _registry.copy()


def list_job_names() -> List[str]:
    """List all registered job names."""
    return list(_registry.keys())


class JobRegistry:
    """Class-based job registry for more complex scenarios."""

    def __init__(self):
        self._jobs: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a job function."""
        self._jobs[name] = func
        logger.debug(f"Registered job: {name}")

    def get(self, name: str) -> Optional[Callable]:
        """Get a job function by name."""
        return self._jobs.get(name)

    def all(self) -> Dict[str, Callable]:
        """Get all registered jobs."""
        return self._jobs.copy()

    def names(self) -> List[str]:
        """List all job names."""
        return list(self._jobs.keys())
