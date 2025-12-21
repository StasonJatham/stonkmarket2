"""Database connection management with pooling."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Iterator, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("database")

# SQLite DDL
DDL = """
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT PRIMARY KEY,
    min_dip_pct REAL NOT NULL,
    min_days INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dip_state (
    symbol TEXT PRIMARY KEY,
    ref_high REAL NOT NULL,
    days_below INTEGER NOT NULL,
    last_price REAL NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(symbol) REFERENCES symbols(symbol) ON DELETE CASCADE
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
    created_at TEXT NOT NULL,
    FOREIGN KEY(name) REFERENCES cronjobs(name) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS auth_user (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Stock suggestions from users
CREATE TABLE IF NOT EXISTS stock_suggestions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',
    vote_count INTEGER NOT NULL DEFAULT 0,
    name TEXT,
    sector TEXT,
    industry TEXT,
    summary TEXT,
    last_price REAL,
    price_90d_ago REAL,
    price_change_90d REAL,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    fetched_at TEXT,
    removed_at TEXT
);

-- Votes for suggestions (for deduplication)
CREATE TABLE IF NOT EXISTS suggestion_votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    suggestion_id INTEGER NOT NULL,
    voter_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(suggestion_id, voter_hash),
    FOREIGN KEY(suggestion_id) REFERENCES stock_suggestions(id) ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_cronjob_logs_created_at ON cronjob_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cronjob_logs_name ON cronjob_logs(name);
CREATE INDEX IF NOT EXISTS idx_dip_state_updated_at ON dip_state(updated_at);
CREATE INDEX IF NOT EXISTS idx_suggestions_status ON stock_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_suggestions_vote_count ON stock_suggestions(vote_count DESC);
CREATE INDEX IF NOT EXISTS idx_suggestions_symbol ON stock_suggestions(symbol);
CREATE INDEX IF NOT EXISTS idx_suggestion_votes_suggestion ON suggestion_votes(suggestion_id);
"""


class ConnectionPool:
    """Simple SQLite connection pool for thread-safety."""

    def __init__(self, db_path: str, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._pool: Queue[sqlite3.Connection] = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._initialized = False

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        resolved = Path(self.db_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            resolved.as_posix(),
            check_same_thread=False,
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row

        # Configure connection for performance and safety
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache

        return conn

    def initialize(self) -> None:
        """Initialize the pool and database schema."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Create and initialize one connection
            conn = self._create_connection()
            conn.executescript(DDL)
            self._bootstrap_defaults(conn)
            conn.commit()

            # Put it in the pool
            self._pool.put(conn)

            # Pre-fill pool
            for _ in range(self.pool_size - 1):
                try:
                    self._pool.put_nowait(self._create_connection())
                except Exception as e:
                    logger.warning(f"Failed to pre-fill connection pool: {e}")
                    break

            self._initialized = True
            logger.info(f"Database initialized: {self.db_path}")

    def _bootstrap_defaults(self, conn: sqlite3.Connection) -> None:
        """Bootstrap default data."""
        from app.core.security import hash_password

        now = datetime.utcnow().isoformat()

        # Insert default symbols
        conn.executemany(
            """
            INSERT OR IGNORE INTO symbols(symbol, min_dip_pct, min_days, created_at)
            VALUES (?, ?, ?, ?)
            """,
            [
                (sym.upper(), settings.default_min_dip_pct, settings.default_min_days, now)
                for sym in settings.default_symbols
            ],
        )

        # Insert default cron jobs
        conn.executemany(
            """
            INSERT OR IGNORE INTO cronjobs(name, cron, description)
            VALUES (?, ?, ?)
            """,
            [
                ("data_grab", "0 6 * * 1-5", "Download fresh quotes"),
                ("analysis", "30 6 * * 1-5", "Run dip ranking"),
            ],
        )

        # Insert default admin user
        conn.execute(
            "INSERT OR IGNORE INTO auth_user(username, password_hash, updated_at) VALUES (?, ?, ?)",
            (settings.default_admin_user, hash_password(settings.default_admin_password), now),
        )

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        if not self._initialized:
            self.initialize()

        try:
            return self._pool.get(timeout=10.0)
        except Empty:
            # Pool exhausted, create a new connection
            logger.warning("Connection pool exhausted, creating new connection")
            return self._create_connection()

    def release_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        try:
            self._pool.put_nowait(conn)
        except Exception:
            # Pool full, close connection
            try:
                conn.close()
            except Exception:
                pass

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Exception:
                pass
        self._initialized = False
        logger.info("All database connections closed")


# Global connection pool
_pool: Optional[ConnectionPool] = None


def init_db(db_path: Optional[str] = None) -> ConnectionPool:
    """Initialize the database and return the connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(db_path or settings.db_path)
        _pool.initialize()
    return _pool


def get_db_connection() -> Iterator[sqlite3.Connection]:
    """
    Dependency for getting a database connection.

    Usage:
        @router.get("/items")
        def get_items(conn: sqlite3.Connection = Depends(get_db_connection)):
            ...
    """
    global _pool
    if _pool is None:
        init_db()

    conn = _pool.get_connection()
    try:
        yield conn
    finally:
        _pool.release_connection(conn)


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """
    Context manager for getting a database connection.

    Usage:
        with get_db() as conn:
            conn.execute("SELECT ...")
    """
    global _pool
    if _pool is None:
        init_db()

    conn = _pool.get_connection()
    try:
        yield conn
    finally:
        _pool.release_connection(conn)


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """
    Context manager for database transactions.

    Usage:
        with transaction(conn) as tx:
            tx.execute("INSERT ...")
            tx.execute("UPDATE ...")
        # Auto-commits on success, rolls back on exception
    """
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def close_db() -> None:
    """Close all database connections."""
    global _pool
    if _pool is not None:
        _pool.close_all()
        _pool = None


def db_healthcheck() -> bool:
    """Check database health."""
    try:
        global _pool
        if _pool is None:
            return False
        conn = _pool.get_connection()
        try:
            conn.execute("SELECT 1").fetchone()
            return True
        finally:
            _pool.release_connection(conn)
    except Exception as e:
        logger.warning(f"Database healthcheck failed: {e}")
        return False
