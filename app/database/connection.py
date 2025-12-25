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

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional, Any

import asyncpg
from asyncpg import Pool, Connection, Record
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("database")

# =============================================================================
# LEGACY ASYNCPG POOL (for existing raw SQL code)
# =============================================================================

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


# =============================================================================
# SQLALCHEMY ASYNC ENGINE & SESSIONS
# =============================================================================

# Global SQLAlchemy engine
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker[AsyncSession]] = None


def _get_sqlalchemy_url() -> str:
    """Convert DATABASE_URL to SQLAlchemy async format.
    
    Converts: postgresql://user:pass@host:port/db
    To:       postgresql+asyncpg://user:pass@host:port/db
    """
    url = settings.database_url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://", 1)
    return url


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
