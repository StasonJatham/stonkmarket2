"""Valkey client connection management."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("cache.client")

# Global connection pool
_pool: Optional[ConnectionPool] = None
_client: Optional[Redis] = None


async def init_valkey_pool() -> ConnectionPool:
    """Initialize Valkey connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            settings.valkey_url,
            max_connections=settings.valkey_max_connections,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info(
            "Valkey connection pool initialized", extra={"url": settings.valkey_url}
        )
    return _pool


async def get_valkey_client() -> Redis:
    """Get Valkey client instance."""
    global _client
    if _client is None:
        pool = await init_valkey_pool()
        _client = Redis(connection_pool=pool)
    return _client


async def close_valkey_client() -> None:
    """Close Valkey client and connection pool."""
    global _client, _pool
    if _client is not None:
        await _client.close()
        _client = None
    if _pool is not None:
        await _pool.disconnect()
        _pool = None
    logger.info("Valkey connection pool closed")


async def valkey_healthcheck() -> bool:
    """Check Valkey connection health."""
    try:
        client = await get_valkey_client()
        result = await asyncio.wait_for(client.ping(), timeout=5.0)
        return result is True or result == "PONG"
    except Exception as e:
        logger.warning(f"Valkey healthcheck failed: {e}")
        return False


@asynccontextmanager
async def valkey_connection():
    """Context manager for Valkey connection."""
    client = await get_valkey_client()
    try:
        yield client
    except redis.RedisError as e:
        logger.error(f"Valkey operation failed: {e}")
        raise
