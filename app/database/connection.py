"""PostgreSQL database connection management with asyncpg pooling.

Also provides SQLite sync connections for backward compatibility during migration.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Any

import asyncpg
from asyncpg import Pool, Connection, Record

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("database")

# Global connection pool (PostgreSQL)
_pool: Optional[Pool] = None

# SQLite database path for legacy/sync access
_sqlite_path: Optional[Path] = None


# ============================================================================
# SQLite Sync Connections (Legacy - for backward compatibility)
# ============================================================================

def init_sqlite_db(db_path: str = None) -> None:
    """Initialize SQLite database for sync access."""
    global _sqlite_path
    
    path = db_path or settings.sqlite_path or "/data/dips.sqlite"
    _sqlite_path = Path(path)
    _sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create tables if needed
    with sqlite3.connect(_sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        _init_sqlite_tables(conn)
    
    logger.info(f"SQLite database initialized: {_sqlite_path}")


def _init_sqlite_tables(conn: sqlite3.Connection) -> None:
    """Create SQLite tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS symbols (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            min_dip_pct REAL DEFAULT 0.1,
            min_days INTEGER DEFAULT 2,
            added_at TEXT,
            updated_at TEXT,
            created_at TEXT,
            sector TEXT,
            market_cap REAL
        );
        
        CREATE TABLE IF NOT EXISTS dip_state (
            symbol TEXT PRIMARY KEY,
            ref_high REAL,
            days_below INTEGER DEFAULT 0,
            last_price REAL,
            updated_at TEXT,
            FOREIGN KEY (symbol) REFERENCES symbols(symbol)
        );
        
        CREATE TABLE IF NOT EXISTS dip_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            current_price REAL,
            ath_price REAL,
            dip_percentage REAL,
            recorded_at TEXT
        );
        
        CREATE TABLE IF NOT EXISTS cronjobs (
            name TEXT PRIMARY KEY,
            cron TEXT NOT NULL,
            description TEXT
        );
        
        CREATE TABLE IF NOT EXISTS cronjob_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT,
            created_at TEXT
        );
        
        CREATE TABLE IF NOT EXISTS auth_user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            mfa_secret TEXT,
            mfa_enabled INTEGER DEFAULT 0,
            mfa_backup_codes TEXT,
            created_at TEXT,
            updated_at TEXT,
            is_admin INTEGER DEFAULT 1
        );
        
        CREATE TABLE IF NOT EXISTS stock_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            title TEXT,
            reason TEXT,
            submitted_by TEXT,
            status TEXT DEFAULT 'pending',
            vote_count INTEGER DEFAULT 0,
            rejection_reason TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        
        CREATE TABLE IF NOT EXISTS suggestion_votes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            suggestion_id INTEGER NOT NULL,
            voter_id TEXT NOT NULL,
            vote INTEGER NOT NULL,
            created_at TEXT,
            UNIQUE(suggestion_id, voter_id)
        );
        
        CREATE TABLE IF NOT EXISTS user_api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            user_id TEXT,
            is_premium INTEGER DEFAULT 0,
            vote_weight INTEGER DEFAULT 1,
            created_at TEXT,
            expires_at TEXT,
            last_used_at TEXT,
            usage_count INTEGER DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS secure_api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_name TEXT UNIQUE NOT NULL,
            encrypted_key TEXT NOT NULL,
            key_hint TEXT,
            created_at TEXT,
            updated_at TEXT,
            created_by TEXT
        );
        
        -- Insert default cronjobs
        INSERT OR IGNORE INTO cronjobs (name, cron, description) 
        VALUES ('data_grab', '0 */4 * * *', 'Fetch stock prices every 4 hours');
        INSERT OR IGNORE INTO cronjobs (name, cron, description)
        VALUES ('analysis', '30 */4 * * *', 'Run analysis 30 min after data grab');
    """)
    conn.commit()
    
    # Ensure default admin exists
    from app.core.security import hash_password
    cur = conn.execute("SELECT 1 FROM auth_user WHERE username = ?", (settings.default_admin_user,))
    if cur.fetchone() is None:
        now = datetime.utcnow().isoformat()
        conn.execute(
            "INSERT INTO auth_user (username, password_hash, created_at, updated_at, is_admin) VALUES (?, ?, ?, ?, 1)",
            (settings.default_admin_user, hash_password(settings.default_admin_password), now, now)
        )
        conn.commit()
        logger.info(f"Created default admin user: {settings.default_admin_user}")


@contextmanager
def get_db_connection() -> Iterator[sqlite3.Connection]:
    """Get a SQLite database connection (sync, for legacy repos)."""
    global _sqlite_path
    
    if _sqlite_path is None:
        init_sqlite_db()
    
    conn = sqlite3.connect(_sqlite_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Initialize database (sync wrapper for backwards compatibility)."""
    init_sqlite_db()


def close_db() -> None:
    """Close database (sync, no-op for SQLite)."""
    pass


# Alias for backwards compatibility
get_db = get_db_connection


async def db_healthcheck() -> bool:
    """Check database health."""
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.warning(f"Database healthcheck failed: {e}")
        return False


# ============================================================================
# PostgreSQL Async Connections (for new async code like DipFinder)
# ============================================================================

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
        logger.warning(f"PostgreSQL pool init failed (may not be configured): {e}")
        return None


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
