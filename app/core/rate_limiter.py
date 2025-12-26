"""Global rate limiter for external API calls."""

from __future__ import annotations

import asyncio
import threading
import time
from functools import wraps

from app.core.logging import get_logger


logger = get_logger("core.rate_limiter")


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Thread-safe and async-compatible.
    """

    def __init__(
        self,
        name: str,
        calls_per_second: float = 2.0,
        burst_size: int = 5,
    ):
        """
        Initialize rate limiter.
        
        Args:
            name: Identifier for logging
            calls_per_second: Sustained rate limit
            burst_size: Maximum burst of calls allowed
        """
        self.name = name
        self.calls_per_second = calls_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()
        self._async_lock: asyncio.Lock | None = None

    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(
            self.burst_size,
            self.tokens + elapsed * self.calls_per_second
        )
        self.last_update = now

    def acquire_sync(self, timeout: float = 30.0) -> bool:
        """
        Acquire a token synchronously, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait for a token
            
        Returns:
            True if token acquired, False if timeout
        """
        start = time.monotonic()

        while True:
            with self._lock:
                self._refill()

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

                # Calculate wait time
                wait_time = (1.0 - self.tokens) / self.calls_per_second

            if time.monotonic() - start + wait_time > timeout:
                logger.warning(f"Rate limiter {self.name} timeout after {timeout}s")
                return False

            logger.debug(f"Rate limiter {self.name} waiting {wait_time:.2f}s")
            time.sleep(min(wait_time, 0.5))  # Sleep in chunks to allow timeout

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire a token asynchronously, waiting if necessary.
        
        Args:
            timeout: Maximum time to wait for a token
            
        Returns:
            True if token acquired, False if timeout
        """
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        start = time.monotonic()

        while True:
            async with self._async_lock:
                with self._lock:
                    self._refill()

                    if self.tokens >= 1.0:
                        self.tokens -= 1.0
                        return True

                    wait_time = (1.0 - self.tokens) / self.calls_per_second

            if time.monotonic() - start + wait_time > timeout:
                logger.warning(f"Rate limiter {self.name} timeout after {timeout}s")
                return False

            logger.debug(f"Rate limiter {self.name} waiting {wait_time:.2f}s")
            await asyncio.sleep(min(wait_time, 0.5))

    def status(self) -> dict:
        """Get current rate limiter status."""
        with self._lock:
            self._refill()
            return {
                "name": self.name,
                "tokens_available": self.tokens,
                "burst_size": self.burst_size,
                "calls_per_second": self.calls_per_second,
            }


# Global rate limiters
_limiters: dict[str, RateLimiter] = {}


def get_rate_limiter(
    name: str,
    calls_per_second: float = 2.0,
    burst_size: int = 5,
) -> RateLimiter:
    """
    Get or create a named rate limiter.
    
    Args:
        name: Unique name for the limiter
        calls_per_second: Rate limit (only used on creation)
        burst_size: Burst size (only used on creation)
        
    Returns:
        RateLimiter instance
    """
    if name not in _limiters:
        _limiters[name] = RateLimiter(name, calls_per_second, burst_size)
        logger.info(f"Created rate limiter '{name}': {calls_per_second}/s, burst={burst_size}")
    return _limiters[name]


# Pre-configured rate limiters for common APIs
YFINANCE_LIMITER = "yfinance"


def get_yfinance_limiter() -> RateLimiter:
    """
    Get the yfinance rate limiter.
    
    Conservative settings: 2 calls/sec, burst of 5.
    Yahoo Finance doesn't publish official limits but is known to rate limit.
    """
    return get_rate_limiter(YFINANCE_LIMITER, calls_per_second=2.0, burst_size=5)


def rate_limited(limiter_name: str):
    """
    Decorator for sync functions to apply rate limiting.
    
    Usage:
        @rate_limited("yfinance")
        def fetch_stock_data(symbol: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = get_rate_limiter(limiter_name)
            if not limiter.acquire_sync():
                raise RuntimeError(f"Rate limit timeout for {limiter_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def async_rate_limited(limiter_name: str):
    """
    Decorator for async functions to apply rate limiting.
    
    Usage:
        @async_rate_limited("yfinance")
        async def fetch_stock_data(symbol: str):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            limiter = get_rate_limiter(limiter_name)
            if not await limiter.acquire():
                raise RuntimeError(f"Rate limit timeout for {limiter_name}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
