"""HTTP caching utilities for FastAPI responses.

Provides ETag generation, Cache-Control headers, and conditional request handling.

Usage:
    from app.cache.http_cache import CacheableResponse, with_http_cache

    @router.get("/ranking")
    async def get_ranking():
        data = await fetch_ranking()
        return CacheableResponse(
            data,
            max_age=60,
            stale_while_revalidate=300,
        )

    # Or with decorator for automatic ETag handling:
    @router.get("/chart/{symbol}")
    @with_http_cache(max_age=3600)
    async def get_chart(symbol: str):
        return await fetch_chart(symbol)
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any, TypeVar

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.core.logging import get_logger


logger = get_logger("cache.http")

T = TypeVar("T")


def generate_etag(data: Any) -> str:
    """Generate a weak ETag from data.
    
    Uses a hash of the JSON-serialized data for consistent ETags.
    """
    if isinstance(data, (dict, list)):
        content = json.dumps(data, sort_keys=True, default=str)
    else:
        content = str(data)

    hash_val = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:16]
    return f'W/"{hash_val}"'


def generate_date_etag(date: datetime) -> str:
    """Generate ETag from a date/version.
    
    Useful for data that changes on known schedules (e.g., daily market data).
    """
    return f'W/"{date.strftime("%Y%m%d%H")}"'


class CacheableResponse(JSONResponse):
    """JSONResponse with Cache-Control and ETag headers.
    
    Args:
        content: Response data
        max_age: Cache duration in seconds (default: 60)
        stale_while_revalidate: Allow stale content for this many seconds (default: 0)
        stale_if_error: Allow stale content on error for this many seconds (default: 0)
        private: Whether this is private (user-specific) data (default: False)
        etag: Custom ETag (auto-generated if None)
        last_modified: Last modification datetime for the data
    """

    def __init__(
        self,
        content: Any,
        *,
        max_age: int = 60,
        stale_while_revalidate: int = 0,
        stale_if_error: int = 0,
        private: bool = False,
        etag: str | None = None,
        last_modified: datetime | None = None,
        status_code: int = 200,
        headers: dict | None = None,
        **kwargs,
    ):
        super().__init__(content=content, status_code=status_code, headers=headers, **kwargs)

        # Build Cache-Control header
        cache_parts = []
        cache_parts.append("private" if private else "public")
        cache_parts.append(f"max-age={max_age}")

        if stale_while_revalidate > 0:
            cache_parts.append(f"stale-while-revalidate={stale_while_revalidate}")

        if stale_if_error > 0:
            cache_parts.append(f"stale-if-error={stale_if_error}")

        self.headers["Cache-Control"] = ", ".join(cache_parts)

        # Generate and set ETag
        if etag is None:
            etag = generate_etag(content)
        self.headers["ETag"] = etag

        # Set Last-Modified if provided
        if last_modified:
            self.headers["Last-Modified"] = last_modified.strftime(
                "%a, %d %b %Y %H:%M:%S GMT"
            )


class NotModifiedResponse(Response):
    """304 Not Modified response for conditional requests."""

    def __init__(self, etag: str, headers: dict | None = None):
        super().__init__(
            content=None,
            status_code=304,
            headers=headers,
        )
        self.headers["ETag"] = etag


def check_if_none_match(request: Request, etag: str) -> bool:
    """Check if request's If-None-Match header matches the ETag.
    
    Returns True if the client's cached version is still valid.
    """
    if_none_match = request.headers.get("If-None-Match")
    if not if_none_match:
        return False

    # Handle multiple ETags (comma-separated)
    client_etags = [e.strip().strip('"').lstrip("W/").strip('"')
                    for e in if_none_match.split(",")]
    server_etag = etag.strip('"').lstrip("W/").strip('"')

    return server_etag in client_etags


def with_http_cache(
    max_age: int = 60,
    stale_while_revalidate: int = 0,
    stale_if_error: int = 0,
    private: bool = False,
):
    """Decorator to add HTTP caching to an endpoint.
    
    Automatically handles:
    - ETag generation from response data
    - If-None-Match conditional requests (304 responses)
    - Cache-Control headers
    
    Usage:
        @router.get("/data")
        @with_http_cache(max_age=3600, stale_while_revalidate=86400)
        async def get_data(request: Request):
            return {"data": "value"}
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs) -> T | NotModifiedResponse:
            # Call the original function
            result = await func(*args, request=request, **kwargs)

            # If already a Response, just add headers
            if isinstance(result, Response):
                if "Cache-Control" not in result.headers:
                    cache_parts = ["private" if private else "public", f"max-age={max_age}"]
                    if stale_while_revalidate:
                        cache_parts.append(f"stale-while-revalidate={stale_while_revalidate}")
                    result.headers["Cache-Control"] = ", ".join(cache_parts)
                return result

            # Generate ETag from result
            etag = generate_etag(result)

            # Check If-None-Match for conditional request
            if check_if_none_match(request, etag):
                logger.debug(f"304 Not Modified: {request.url.path}")
                return NotModifiedResponse(etag=etag)

            # Return with cache headers
            return CacheableResponse(
                content=result,
                max_age=max_age,
                stale_while_revalidate=stale_while_revalidate,
                stale_if_error=stale_if_error,
                private=private,
                etag=etag,
            )

        return wrapper
    return decorator


# Preset cache configurations for common patterns
class CachePresets:
    """Common cache configurations for different data types."""

    # Stock ranking - updates after cron job, but check often
    RANKING = {
        "max_age": 60,
        "stale_while_revalidate": 300,
        "stale_if_error": 3600,
    }

    # Chart data - stable for the day, long cache
    CHART = {
        "max_age": 3600,
        "stale_while_revalidate": 86400,
        "stale_if_error": 86400,
    }

    # Stock info - rarely changes
    STOCK_INFO = {
        "max_age": 3600,
        "stale_while_revalidate": 86400,
    }

    # Symbol list - changes on add/remove
    SYMBOLS = {
        "max_age": 3600,
        "stale_while_revalidate": 86400,
    }

    # Benchmarks - very stable configuration
    BENCHMARKS = {
        "max_age": 86400,
        "stale_while_revalidate": 86400 * 7,
    }

    # Settings - always fresh for admin
    NO_CACHE = {
        "max_age": 0,
        "private": True,
    }
