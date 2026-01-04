"""
System Test Configuration.

Controls strictness levels, container mapping, and error detection patterns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_container_names() -> dict[str, str]:
    """Get container names based on environment.

    If SYSTEM_TEST_USE_DEV=1, use the dev stack container names.
    Otherwise, use the isolated test stack container names.
    """
    use_dev = os.getenv("SYSTEM_TEST_USE_DEV", "0") == "1"

    if use_dev:
        return {
            "api": "stonkmarket-api-dev",
            "worker": "stonkmarket-celery-worker-dev",
            "worker-batch": "stonkmarket-celery-worker-batch-dev",
            "postgres": "stonkmarket-postgres-dev",
            "valkey": "stonkmarket-valkey-dev",
        }
    else:
        return {
            "api": "stonkmarket-api-test",
            "worker": "stonkmarket-celery-worker-test",
            "worker-batch": "stonkmarket-celery-worker-batch-test",
            "postgres": "stonkmarket-postgres-test",
            "valkey": "stonkmarket-valkey-test",
        }


@dataclass
class SystemTestConfig:
    """Configuration for system tests."""

    # Container name mapping
    containers: dict[str, str] = field(default_factory=_get_container_names)

    # Strictness: fail on warnings in addition to errors
    strict_mode: bool = True

    # Error detection patterns (case-insensitive regex)
    error_patterns: list[str] = field(
        default_factory=lambda: [
            r"ERROR",
            r"CRITICAL",
            r"Traceback \(most recent call last\)",
            r"Exception:",
            r"FATAL",
            r"panic:",
            r"KeyError:",
            r"AttributeError:",
            r"TypeError:",
            r"ValueError:",
            r"ImportError:",
            r"ModuleNotFoundError:",
            r"NameError:",
            r"RuntimeError:",
            r"HTTP 5\d{2}",
            r"status_code=5\d{2}",
        ]
    )

    # Warning patterns (only checked in strict mode)
    warning_patterns: list[str] = field(
        default_factory=lambda: [
            r"WARNING",
            r"WARN",
            r"DeprecationWarning",
            r"PendingDeprecationWarning",
        ]
    )

    # Patterns to ignore (false positives, expected behavior)
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            r"DEBUG",
            r"healthcheck",
            r"health check",
            r"TzCache",  # yfinance timezone cache warning (cosmetic)
            r"peewee.*not installed",  # Optional dependency
            r"rate limit",  # Expected rate limiting behavior
            r"404.*fundamentals",  # ETFs don't have fundamentals
            r"INFO",  # Info logs are fine
        ]
    )

    # Timeouts (seconds)
    celery_drain_timeout: float = 30.0
    api_timeout: float = 10.0
    stack_startup_timeout: float = 60.0

    # Log settings
    log_buffer_lines: int = 500

    # API base URLs
    api_base_url: str = "http://localhost:8000"
    frontend_base_url: str = "http://localhost:5173"

    @classmethod
    def from_env(cls) -> "SystemTestConfig":
        """Load config from environment variables."""
        return cls(
            strict_mode=os.getenv("SYSTEM_TEST_STRICT", "1") == "1",
            celery_drain_timeout=float(os.getenv("CELERY_TIMEOUT", "30")),
            api_base_url=os.getenv("API_BASE_URL", "http://localhost:8000"),
            frontend_base_url=os.getenv("FRONTEND_BASE_URL", "http://localhost:5173"),
        )


# Global default config instance
_config: SystemTestConfig | None = None


def get_config() -> SystemTestConfig:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = SystemTestConfig.from_env()
    return _config
