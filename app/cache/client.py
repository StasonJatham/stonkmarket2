"""Valkey client connection management."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import ConnectionPool, Redis

from app.core.config import settings
from app.core.logging import get_logger


logger = get_logger("cache.client")

# Global connection pool - per event loop
_pools: dict[int, ConnectionPool] = {}
_clients: dict[int, Redis] = {}


def _get_loop_id() -> int:
    """Get current event loop id for tracking client per loop."""
    try:
        loop = asyncio.get_running_loop()
        return id(loop)
    except RuntimeError:
        return 0


async def init_valkey_pool() -> ConnectionPool:
    """Initialize Valkey connection pool for current event loop."""
    loop_id = _get_loop_id()
    if loop_id not in _pools:
        _pools[loop_id] = ConnectionPool.from_url(
            settings.valkey_url,
            max_connections=settings.valkey_max_connections,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
            health_check_interval=30,
        )
        logger.info(
            "Valkey connection pool initialized", 
            extra={"url": settings.valkey_url, "loop_id": loop_id}
        )
    return _pools[loop_id]


async def get_valkey_client() -> Redis:
    """Get Valkey client instance for current event loop."""
    loop_id = _get_loop_id()
    if loop_id not in _clients:
        pool = await init_valkey_pool()
        _clients[loop_id] = Redis(connection_pool=pool)
    return _clients[loop_id]


async def close_valkey_client() -> None:
    """Close Valkey client and connection pool for current event loop."""
    loop_id = _get_loop_id()
    if loop_id in _clients:
        await _clients[loop_id].aclose()
        del _clients[loop_id]
    if loop_id in _pools:
        await _pools[loop_id].disconnect()
        del _pools[loop_id]
    logger.info("Valkey connection pool closed", extra={"loop_id": loop_id})


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
