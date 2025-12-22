"""PostgreSQL database connection management with asyncpg pooling."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Any

import asyncpg
from asyncpg import Pool, Connection, Record

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("database")

# Global connection pool (PostgreSQL)
_pool: Optional[Pool] = None


async def init_pg_pool() -> Pool:
    """Initialize PostgreSQL connection pool."""
    global _pool

    if _pool is not None:
        return _pool

    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            command_timeout=60,
            server_settings={
                "application_name": "stonkmarket",
                "timezone": "UTC",
            },
        )
        logger.info("PostgreSQL pool initialized")
        return _pool
    except Exception as e:
        logger.warning(f"PostgreSQL pool init failed: {e}")
        raise


async def get_pg_pool() -> Optional[Pool]:
    """Get PostgreSQL connection pool, initializing if necessary."""
    global _pool
    if _pool is None:
        await init_pg_pool()
    return _pool


@asynccontextmanager
async def get_pg_connection() -> AsyncIterator[Connection]:
    """Get a PostgreSQL connection for async operations."""
    pool = await get_pg_pool()
    if pool is None:
        raise RuntimeError("PostgreSQL pool not available")
    async with pool.acquire() as conn:
        yield conn


async def close_pg_pool() -> None:
    """Close PostgreSQL connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")


# Async query helpers for PostgreSQL
async def fetch_one(query: str, *args) -> Optional[Record]:
    """Execute a query and fetch one result."""
    async with get_pg_connection() as conn:
        return await conn.fetchrow(query, *args)


async def fetch_all(query: str, *args) -> list[Record]:
    """Execute a query and fetch all results."""
    async with get_pg_connection() as conn:
        return await conn.fetch(query, *args)


async def fetch_val(query: str, *args) -> Any:
    """Execute a query and fetch a single value."""
    async with get_pg_connection() as conn:
        return await conn.fetchval(query, *args)


async def execute(query: str, *args) -> str:
    """Execute a query without returning results."""
    async with get_pg_connection() as conn:
        return await conn.execute(query, *args)


async def execute_many(query: str, args: list) -> None:
    """Execute a query with multiple sets of parameters."""
    async with get_pg_connection() as conn:
        await conn.executemany(query, args)
