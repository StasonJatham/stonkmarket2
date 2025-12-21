"""Rate limiting using Valkey."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from app.core.config import settings
from app.core.exceptions import RateLimitError
from app.core.logging import get_logger

from .client import get_valkey_client

logger = get_logger("cache.rate_limit")

RATE_LIMIT_PREFIX = "stonkmarket:rate_limit"


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int
    reset_at: float
    limit: int


class RateLimiter:
    """
    Token bucket rate limiter using Valkey.

    Uses sliding window log algorithm for accurate rate limiting.
    """

    def __init__(self, key_prefix: str, limit: int, window: int):
        """
        Initialize rate limiter.

        Args:
            key_prefix: Prefix for rate limit keys
            limit: Maximum requests allowed in window
            window: Time window in seconds
        """
        self.key_prefix = key_prefix
        self.limit = limit
        self.window = window

    def _get_key(self, identifier: str) -> str:
        """Generate rate limit key for identifier."""
        return f"{RATE_LIMIT_PREFIX}:{self.key_prefix}:{identifier}"

    async def check(self, identifier: str) -> RateLimitResult:
        """
        Check rate limit for identifier.

        Args:
            identifier: Unique identifier (e.g., IP address, user ID)

        Returns:
            RateLimitResult with allowed status and remaining quota
        """
        client = await get_valkey_client()
        key = self._get_key(identifier)
        now = time.time()
        window_start = now - self.window

        # Use Lua script for atomic operation
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window = tonumber(ARGV[4])

        -- Remove old entries outside the window
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)

        -- Count current requests in window
        local count = redis.call('ZCARD', key)

        if count < limit then
            -- Add new request
            redis.call('ZADD', key, now, now .. ':' .. math.random())
            redis.call('EXPIRE', key, window)
            return {1, limit - count - 1, now + window}
        else
            -- Get oldest entry for reset time
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local reset_at = oldest[2] and (tonumber(oldest[2]) + window) or (now + window)
            return {0, 0, reset_at}
        end
        """

        try:
            result = await client.eval(
                lua_script, 1, key, now, window_start, self.limit, self.window
            )
            allowed, remaining, reset_at = result
            return RateLimitResult(
                allowed=bool(allowed),
                remaining=int(remaining),
                reset_at=float(reset_at),
                limit=self.limit,
            )
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open on error to not block legitimate traffic
            return RateLimitResult(
                allowed=True,
                remaining=self.limit,
                reset_at=now + self.window,
                limit=self.limit,
            )

    async def is_allowed(self, identifier: str) -> bool:
        """Simple check if request is allowed."""
        result = await self.check(identifier)
        return result.allowed


def parse_rate_limit(rate_string: str) -> Tuple[int, int]:
    """
    Parse rate limit string like "100/minute" or "10/second".

    Returns (limit, window_in_seconds)
    """
    parts = rate_string.lower().split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid rate limit format: {rate_string}")

    limit = int(parts[0])
    unit = parts[1].strip()

    windows = {
        "second": 1,
        "sec": 1,
        "s": 1,
        "minute": 60,
        "min": 60,
        "m": 60,
        "hour": 3600,
        "hr": 3600,
        "h": 3600,
        "day": 86400,
        "d": 86400,
    }

    if unit not in windows:
        raise ValueError(f"Unknown time unit: {unit}")

    return limit, windows[unit]


# Pre-configured rate limiters
def get_auth_rate_limiter() -> RateLimiter:
    """Get rate limiter for auth endpoints."""
    limit, window = parse_rate_limit(settings.rate_limit_auth)
    return RateLimiter("auth", limit, window)


def get_api_rate_limiter(authenticated: bool = False) -> RateLimiter:
    """Get rate limiter for API endpoints.
    
    Args:
        authenticated: If True, use higher limits for authenticated users
    """
    if authenticated:
        limit, window = parse_rate_limit(settings.rate_limit_api_authenticated)
        return RateLimiter("api_auth", limit, window)
    else:
        limit, window = parse_rate_limit(settings.rate_limit_api_anonymous)
        return RateLimiter("api_anon", limit, window)


async def check_rate_limit(
    identifier: str,
    limiter: Optional[RateLimiter] = None,
    key_prefix: str = "api",
) -> RateLimitResult:
    """
    Check rate limit and raise exception if exceeded.

    Args:
        identifier: Request identifier (IP, user ID, etc.)
        limiter: Optional custom rate limiter
        key_prefix: Key prefix if using default limiter

    Raises:
        RateLimitError: If rate limit exceeded
    """
    if not settings.rate_limit_enabled:
        return RateLimitResult(allowed=True, remaining=999, reset_at=0, limit=999)

    if limiter is None:
        if key_prefix == "auth":
            limiter = get_auth_rate_limiter()
        else:
            limiter = get_api_rate_limiter()

    result = await limiter.check(identifier)

    if not result.allowed:
        raise RateLimitError(
            message=f"Rate limit exceeded. Try again in {int(result.reset_at - time.time())} seconds.",
            details={
                "limit": result.limit,
                "reset_at": result.reset_at,
            },
        )

    return result
