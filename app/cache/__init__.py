"""Valkey (Redis-compatible) cache module."""

from .client import (
    get_valkey_client,
    close_valkey_client,
    valkey_healthcheck,
)
from .cache import (
    Cache,
    cache_key,
    cached,
    invalidate_pattern,
)
from .rate_limit import (
    RateLimiter,
    check_rate_limit,
)
from .distributed_lock import (
    DistributedLock,
    acquire_lock,
)

__all__ = [
    "get_valkey_client",
    "close_valkey_client",
    "valkey_healthcheck",
    "Cache",
    "cache_key",
    "cached",
    "invalidate_pattern",
    "RateLimiter",
    "check_rate_limit",
    "DistributedLock",
    "acquire_lock",
]
