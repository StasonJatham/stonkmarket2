"""PostgreSQL database connection management with asyncpg pooling and SQLAlchemy ORM.

This module provides two interfaces:
1. Raw asyncpg pool (legacy) - for existing code using raw SQL
2. SQLAlchemy async sessions - for new ORM-based code

The SQLAlchemy engine uses the same connection pool settings and provides
full async support via the asyncpg driver.

Usage (SQLAlchemy ORM):
    from app.database.connection import get_session
    from app.database.orm import Symbol
    
    async with get_session() as session:
        symbol = await session.get(Symbol, "AAPL")
        symbol.name = "Apple Inc."
        await session.commit()

Usage (Raw asyncpg - legacy):
    from app.database.connection import fetch_one, execute
    
    row = await fetch_one("SELECT * FROM symbols WHERE symbol = $1", "AAPL")
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg


# =============================================================================
# DATABASE URL UTILITIES
# =============================================================================

def get_async_database_url(url: str) -> str:
    """Convert database URL to SQLAlchemy async format.
    
    Converts: postgresql://user:pass@host:port/db
    To:       postgresql+asyncpg://user:pass@host:port/db
    
    This is a shared utility used by both connection.py and alembic/env.py
    to ensure consistent URL handling.
    """
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url
from asyncpg import Connection, Pool, Record
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger


logger = get_logger("database")

# =============================================================================
# LEGACY ASYNCPG POOL (for existing raw SQL code)
# =============================================================================

# Global connection pool (PostgreSQL)
_pool: Pool | None = None


async def init_pg_pool() -> Pool:
    """Initialize PostgreSQL connection pool."""
    global _pool

    if _pool is not None:
        return _pool

    async def _init_connection(conn: Connection) -> None:
        """Initialize connection with JSON codec for JSONB columns."""
        await conn.set_type_codec(
            'jsonb',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )
        await conn.set_type_codec(
            'json',
            encoder=json.dumps,
            decoder=json.loads,
            schema='pg_catalog'
        )

    try:
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            command_timeout=60,
            init=_init_connection,
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


async def get_pg_pool() -> Pool | None:
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
async def fetch_one(query: str, *args) -> Record | None:
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


@asynccontextmanager
async def transaction() -> AsyncIterator[Connection]:
    """Execute multiple queries in a transaction.
    
    Usage:
        async with transaction() as conn:
            await conn.execute("INSERT INTO ...", ...)
            await conn.execute("UPDATE ...", ...)
            # Auto-commits on success, rolls back on exception
    
    This ensures atomicity for multi-step operations that must succeed
    or fail together. Use this instead of multiple fetch_one/execute calls
    when operations depend on each other.
    """
    pool = await get_pg_pool()
    if pool is None:
        raise RuntimeError("PostgreSQL pool not available")
    async with pool.acquire() as conn, conn.transaction():
        yield conn


# =============================================================================
# SQLALCHEMY ASYNC ENGINE & SESSIONS
# =============================================================================

# Global SQLAlchemy engine
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_sqlalchemy_url() -> str:
    """Get SQLAlchemy async database URL from settings."""
    return get_async_database_url(settings.database_url)


async def init_sqlalchemy_engine() -> AsyncEngine:
    """Initialize SQLAlchemy async engine."""
    global _engine, _session_factory

    if _engine is not None:
        return _engine

    try:
        _engine = create_async_engine(
            _get_sqlalchemy_url(),
            pool_size=settings.db_pool_min_size,
            max_overflow=settings.db_pool_max_size - settings.db_pool_min_size,
            pool_pre_ping=True,  # Check connection health
            pool_recycle=3600,  # Recycle connections after 1 hour (prevents stale connections)
            echo=False,  # Set True for SQL logging during debug
        )

        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,  # Don't expire objects after commit
            autoflush=False,  # Manual flush control
        )

        logger.info("SQLAlchemy async engine initialized")
        return _engine
    except Exception as e:
        logger.error(f"SQLAlchemy engine init failed: {e}")
        raise


async def get_engine() -> AsyncEngine:
    """Get SQLAlchemy async engine, initializing if necessary."""
    global _engine
    if _engine is None:
        await init_sqlalchemy_engine()
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Get an async SQLAlchemy session.
    
    Usage:
        async with get_session() as session:
            result = await session.execute(select(Symbol))
            symbols = result.scalars().all()
    """
    global _session_factory

    if _session_factory is None:
        await init_sqlalchemy_engine()

    async with _session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def close_sqlalchemy_engine() -> None:
    """Close SQLAlchemy async engine."""
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("SQLAlchemy engine closed")


# =============================================================================
# UNIFIED LIFECYCLE MANAGEMENT
# =============================================================================


async def init_database() -> None:
    """Initialize all database connections (asyncpg pool + SQLAlchemy engine)."""
    await init_pg_pool()
    await init_sqlalchemy_engine()


async def close_database() -> None:
    """Close all database connections."""
    await close_pg_pool()
    await close_sqlalchemy_engine()
