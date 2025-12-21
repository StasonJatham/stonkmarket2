"""Cache utilities with typed helpers and stampede protection."""

from __future__ import annotations

import asyncio
import hashlib
import json
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from app.core.config import settings
from app.core.logging import get_logger

from .client import get_valkey_client

logger = get_logger("cache")

T = TypeVar("T")

# Cache key prefixes for namespacing
CACHE_PREFIX = "stonkmarket"
CACHE_VERSION = "v1"


def cache_key(*parts: Union[str, int, float], prefix: str = "cache") -> str:
    """
    Generate a consistent cache key from parts.

    Usage:
        cache_key("stock", "AAPL", "info") -> "stonkmarket:v1:cache:stock:AAPL:info"
    """
    sanitized = [str(part).replace(":", "_") for part in parts]
    return f"{CACHE_PREFIX}:{CACHE_VERSION}:{prefix}:{':'.join(sanitized)}"


def _serialize(value: Any) -> str:
    """Serialize value to JSON string."""
    return json.dumps(value, default=str)


def _deserialize(value: str) -> Any:
    """Deserialize JSON string to value."""
    return json.loads(value)


class Cache:
    """Typed cache wrapper with common patterns."""

    def __init__(self, prefix: str = "cache", default_ttl: int = None):
        self.prefix = prefix
        self.default_ttl = default_ttl or settings.cache_default_ttl

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        client = await get_valkey_client()
        full_key = cache_key(key, prefix=self.prefix)
        try:
            value = await client.get(full_key)
            if value is not None:
                logger.debug(f"Cache hit: {full_key}")
                return _deserialize(value)
            logger.debug(f"Cache miss: {full_key}")
            return None
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """Set value in cache with TTL."""
        client = await get_valkey_client()
        full_key = cache_key(key, prefix=self.prefix)
        try:
            serialized = _serialize(value)
            await client.set(full_key, serialized, ex=ttl or self.default_ttl)
            logger.debug(f"Cache set: {full_key}, TTL: {ttl or self.default_ttl}s")
            return True
        except Exception as e:
            logger.warning(f"Cache set failed: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        client = await get_valkey_client()
        full_key = cache_key(key, prefix=self.prefix)
        try:
            await client.delete(full_key)
            logger.debug(f"Cache delete: {full_key}")
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed: {e}")
            return False

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """Get from cache or compute and cache value (cache-aside pattern)."""
        value = await self.get(key)
        if value is not None:
            return value

        # Compute value
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        if value is not None:
            await self.set(key, value, ttl)

        return value

    async def get_or_set_with_lock(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        lock_timeout: int = 10,
    ) -> Any:
        """
        Get from cache or compute with stampede protection.

        Uses a lock to prevent multiple concurrent computations (single-flight pattern).
        """
        value = await self.get(key)
        if value is not None:
            return value

        # Try to acquire lock for computation
        from .distributed_lock import DistributedLock

        lock_key = f"{key}:lock"
        lock = DistributedLock(lock_key, timeout=lock_timeout)

        acquired = await lock.acquire()
        if acquired:
            try:
                # Double-check cache after acquiring lock
                value = await self.get(key)
                if value is not None:
                    return value

                # Compute value
                if asyncio.iscoroutinefunction(factory):
                    value = await factory()
                else:
                    value = factory()

                if value is not None:
                    await self.set(key, value, ttl)

                return value
            finally:
                await lock.release()
        else:
            # Wait for lock holder to compute, then retry get
            await asyncio.sleep(0.5)
            value = await self.get(key)
            if value is not None:
                return value
            # Lock timed out or still computing, compute anyway
            if asyncio.iscoroutinefunction(factory):
                return await factory()
            return factory()


async def invalidate_pattern(pattern: str) -> int:
    """Invalidate all keys matching a pattern. Use carefully in production."""
    client = await get_valkey_client()
    full_pattern = f"{CACHE_PREFIX}:{CACHE_VERSION}:*{pattern}*"
    try:
        keys = []
        async for key in client.scan_iter(match=full_pattern, count=100):
            keys.append(key)
        if keys:
            await client.delete(*keys)
            logger.info(f"Invalidated {len(keys)} keys matching: {pattern}")
        return len(keys)
    except Exception as e:
        logger.warning(f"Pattern invalidation failed: {e}")
        return 0


def cached(
    key_prefix: str,
    ttl: int = None,
    key_builder: Optional[Callable[..., str]] = None,
    with_lock: bool = False,
):
    """
    Decorator for caching function results.

    Usage:
        @cached("stock_info", ttl=300)
        async def get_stock_info(symbol: str) -> dict:
            ...

        @cached("ranking", key_builder=lambda: "all")
        async def compute_ranking() -> list:
            ...
    """
    cache = Cache(prefix=key_prefix, default_ttl=ttl or settings.cache_default_ttl)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Build cache key
            if key_builder:
                key = key_builder(*args, **kwargs)
            else:
                # Auto-generate key from function args
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key = hashlib.md5(":".join(key_parts).encode()).hexdigest()

            # Get or compute
            if with_lock:
                return await cache.get_or_set_with_lock(
                    key,
                    lambda: func(*args, **kwargs),
                    ttl,
                )
            else:
                return await cache.get_or_set(
                    key,
                    lambda: func(*args, **kwargs),
                    ttl,
                )

        return wrapper

    return decorator
