"""Distributed locking using Valkey."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from app.core.logging import get_logger

from .client import get_valkey_client


logger = get_logger("cache.lock")

LOCK_PREFIX = "stonkmarket:lock"


class DistributedLock:
    """
    Distributed lock implementation using Valkey.

    Uses the Redlock algorithm pattern for safety.
    """

    def __init__(
        self,
        name: str,
        timeout: int = 30,
        blocking: bool = True,
        blocking_timeout: float | None = None,
    ):
        """
        Initialize distributed lock.

        Args:
            name: Lock name (will be prefixed)
            timeout: Lock expiration in seconds (auto-release if holder dies)
            blocking: Whether to block waiting for lock
            blocking_timeout: Max time to wait for lock (None = wait forever)
        """
        self.name = name
        self.key = f"{LOCK_PREFIX}:{name}"
        self.timeout = timeout
        self.blocking = blocking
        self.blocking_timeout = blocking_timeout
        self.token = str(uuid.uuid4())
        self._acquired = False

    async def acquire(self) -> bool:
        """
        Acquire the lock.

        Returns True if lock was acquired, False otherwise.
        """
        client = await get_valkey_client()
        start_time = time.monotonic()

        while True:
            # Try to set lock with NX (only if not exists)
            acquired = await client.set(
                self.key,
                self.token,
                ex=self.timeout,
                nx=True,
            )

            if acquired:
                self._acquired = True
                logger.debug(f"Lock acquired: {self.name}")
                return True

            if not self.blocking:
                return False

            # Check timeout
            if self.blocking_timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.blocking_timeout:
                    logger.debug(f"Lock acquisition timeout: {self.name}")
                    return False

            # Wait before retry
            await asyncio.sleep(0.1)

    async def release(self) -> bool:
        """
        Release the lock.

        Only releases if we still hold it (token matches).
        """
        if not self._acquired:
            return False

        client = await get_valkey_client()

        # Use Lua script for atomic check-and-delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        try:
            result = await client.eval(lua_script, 1, self.key, self.token)
            self._acquired = False
            if result:
                logger.debug(f"Lock released: {self.name}")
                return True
            else:
                logger.warning(f"Lock release failed (token mismatch): {self.name}")
                return False
        except Exception as e:
            logger.error(f"Lock release error: {e}")
            self._acquired = False
            return False

    async def extend(self, additional_time: int = None) -> bool:
        """Extend lock timeout if we still hold it."""
        if not self._acquired:
            return False

        client = await get_valkey_client()
        new_timeout = additional_time or self.timeout

        # Use Lua script for atomic check-and-expire
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("expire", KEYS[1], ARGV[2])
        else
            return 0
        end
        """

        try:
            result = await client.eval(lua_script, 1, self.key, self.token, new_timeout)
            if result:
                logger.debug(f"Lock extended: {self.name} by {new_timeout}s")
                return True
            return False
        except Exception as e:
            logger.error(f"Lock extend error: {e}")
            return False

    async def __aenter__(self) -> DistributedLock:
        if not await self.acquire():
            raise RuntimeError(f"Failed to acquire lock: {self.name}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release()


@asynccontextmanager
async def acquire_lock(
    name: str,
    timeout: int = 30,
    blocking: bool = True,
    blocking_timeout: float | None = None,
):
    """
    Context manager for distributed lock.

    Usage:
        async with acquire_lock("my_job"):
            # Critical section
            ...
    """
    lock = DistributedLock(
        name,
        timeout=timeout,
        blocking=blocking,
        blocking_timeout=blocking_timeout,
    )
    try:
        acquired = await lock.acquire()
        if not acquired:
            raise RuntimeError(f"Failed to acquire lock: {name}")
        yield lock
    finally:
        await lock.release()


async def is_locked(name: str) -> bool:
    """Check if a lock is currently held."""
    client = await get_valkey_client()
    key = f"{LOCK_PREFIX}:{name}"
    return await client.exists(key) > 0
