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
from .http_cache import (
    CacheableResponse,
    NotModifiedResponse,
    with_http_cache,
    generate_etag,
    check_if_none_match,
    CachePresets,
)
from .metrics import (
    cache_metrics,
    CacheTimer,
)

__all__ = [
    # Client
    "get_valkey_client",
    "close_valkey_client",
    "valkey_healthcheck",
    # Cache
    "Cache",
    "cache_key",
    "cached",
    "invalidate_pattern",
    # Rate limiting
    "RateLimiter",
    "check_rate_limit",
    # Locking
    "DistributedLock",
    "acquire_lock",
    # HTTP caching
    "CacheableResponse",
    "NotModifiedResponse",
    "with_http_cache",
    "generate_etag",
    "check_if_none_match",
    "CachePresets",
    # Metrics
    "cache_metrics",
    "CacheTimer",
]
