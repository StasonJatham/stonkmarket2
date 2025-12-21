"""PostgreSQL database connection management with asyncpg pooling."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Optional, Any

import asyncpg
from asyncpg import Pool, Connection, Record

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("database")

# Global connection pool
_pool: Optional[Pool] = None


async def init_db() -> Pool:
    """Initialize the database connection pool."""
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
        
        # Bootstrap defaults
        async with _pool.acquire() as conn:
            await _bootstrap_defaults(conn)
        
        logger.info("Database pool initialized", extra={
            "min_size": settings.db_pool_min_size,
            "max_size": settings.db_pool_max_size,
        })
        
        return _pool
        
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


async def _bootstrap_defaults(conn: Connection) -> None:
    """Bootstrap default data if not exists."""
    from app.core.security import hash_password
    
    now = datetime.utcnow()
    
    # Check if admin user exists
    admin_exists = await conn.fetchval(
        "SELECT EXISTS(SELECT 1 FROM auth_user WHERE username = $1)",
        settings.default_admin_user
    )
    
    if not admin_exists:
        await conn.execute(
            """
            INSERT INTO auth_user (username, password_hash, created_at, updated_at)
            VALUES ($1, $2, $3, $3)
            ON CONFLICT (username) DO NOTHING
            """,
            settings.default_admin_user,
            hash_password(settings.default_admin_password),
            now,
        )
        logger.info(f"Created default admin user: {settings.default_admin_user}")
    
    # Insert default symbols
    for symbol in settings.default_symbols:
        await conn.execute(
            """
            INSERT INTO symbols (symbol, name, added_at, updated_at)
            VALUES ($1, $1, $2, $2)
            ON CONFLICT (symbol) DO NOTHING
            """,
            symbol.upper(),
            now,
        )


async def get_pool() -> Pool:
    """Get the database connection pool, initializing if necessary."""
    global _pool
    
    if _pool is None:
        await init_db()
    
    return _pool


async def get_db_connection() -> AsyncIterator[Connection]:
    """
    Async dependency for getting a database connection.
    
    Usage:
        @router.get("/items")
        async def get_items(conn: Connection = Depends(get_db_connection)):
            ...
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def get_db() -> AsyncIterator[Connection]:
    """
    Async context manager for getting a database connection.
    
    Usage:
        async with get_db() as conn:
            await conn.fetch("SELECT ...")
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def transaction(conn: Connection) -> AsyncIterator[Connection]:
    """
    Async context manager for database transactions.
    
    Usage:
        async with transaction(conn) as tx:
            await tx.execute("INSERT ...")
            await tx.execute("UPDATE ...")
        # Auto-commits on success, rolls back on exception
    """
    async with conn.transaction():
        yield conn


async def close_db() -> None:
    """Close the database connection pool."""
    global _pool
    
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database pool closed")


async def db_healthcheck() -> bool:
    """Check database health."""
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchval("SELECT 1")
            return result == 1
    except Exception as e:
        logger.warning(f"Database healthcheck failed: {e}")
        return False


# ============================================================================
# Query Helpers
# ============================================================================

async def fetch_one(query: str, *args) -> Optional[Record]:
    """Execute a query and fetch one result."""
    async with get_db() as conn:
        return await conn.fetchrow(query, *args)


async def fetch_all(query: str, *args) -> list[Record]:
    """Execute a query and fetch all results."""
    async with get_db() as conn:
        return await conn.fetch(query, *args)


async def fetch_val(query: str, *args) -> Any:
    """Execute a query and fetch a single value."""
    async with get_db() as conn:
        return await conn.fetchval(query, *args)


async def execute(query: str, *args) -> str:
    """Execute a query without returning results."""
    async with get_db() as conn:
        return await conn.execute(query, *args)


async def execute_many(query: str, args: list) -> None:
    """Execute a query with multiple sets of parameters."""
    async with get_db() as conn:
        await conn.executemany(query, args)


# ============================================================================
# Record to Dict Conversion
# ============================================================================

def record_to_dict(record: Optional[Record]) -> Optional[dict]:
    """Convert an asyncpg Record to a dictionary."""
    if record is None:
        return None
    return dict(record)


def records_to_dicts(records: list[Record]) -> list[dict]:
    """Convert a list of asyncpg Records to dictionaries."""
    return [dict(r) for r in records]


# ============================================================================
# Sync Wrapper for Legacy Code
# ============================================================================

def run_sync(coro):
    """Run an async coroutine synchronously (for legacy code migration)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an async context, can't use run_until_complete
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(coro)
