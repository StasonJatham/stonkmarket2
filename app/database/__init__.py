"""Database module with PostgreSQL connection pooling and SQLAlchemy ORM.

Provides two interfaces:
1. Legacy asyncpg helpers (fetch_one, fetch_all, execute) - for existing raw SQL code
2. SQLAlchemy ORM (get_session, Base, models) - for new ORM-based code

New code should use SQLAlchemy ORM with get_session().
"""

# asyncpg interface (raw SQL - legacy, being migrated to ORM)
from .connection import (
    close_database,
    close_pg_pool,
    close_sqlalchemy_engine,
    execute,
    execute_many,
    fetch_all,
    fetch_one,
    fetch_val,
    get_async_database_url,
    get_engine,
    get_pg_connection,
    get_pg_pool,
    get_session,
    init_database,
    init_pg_pool,
    # SQLAlchemy interface
    init_sqlalchemy_engine,
    transaction,
)

# SQLAlchemy ORM models
from .orm import (
    # AI
    AiAgentAnalysis,
    AnalysisVersion,
    # API & Batch
    ApiUsage,
    # Auth & Security
    AuthUser,
    Base,
    BatchJob,
    BatchTaskError,
    # Scheduler
    CronJob,
    DataVersion,
    DipAIAnalysis,
    DipfinderConfig,
    DipfinderHistory,
    DipfinderSignal,
    DipHistory,
    DipState,
    # Voting
    DipVote,
    # DipFinder
    PriceHistory,
    SecureApiKey,
    StockFundamentals,
    # Suggestions
    StockSuggestion,
    SuggestionVote,
    # Symbols & Dips
    Symbol,
    SymbolIngestQueue,
    SymbolSearchLog,
    # Search & Versioning
    SymbolSearchResult,
    UserApiKey,
    YfinanceInfoCache,
)


__all__ = [
    # asyncpg helpers
    "init_pg_pool",
    "close_pg_pool",
    "get_pg_pool",
    "get_pg_connection",
    "fetch_one",
    "fetch_all",
    "fetch_val",
    "execute",
    "execute_many",
    "transaction",
    "get_async_database_url",
    # SQLAlchemy
    "init_sqlalchemy_engine",
    "close_sqlalchemy_engine",
    "get_session",
    "get_engine",
    "init_database",
    "close_database",
    "Base",
    # ORM models
    "AuthUser",
    "SecureApiKey",
    "UserApiKey",
    "Symbol",
    "DipState",
    "DipHistory",
    "StockSuggestion",
    "SuggestionVote",
    "DipVote",
    "DipAIAnalysis",
    "ApiUsage",
    "BatchJob",
    "BatchTaskError",
    "CronJob",
    "PriceHistory",
    "DipfinderSignal",
    "DipfinderConfig",
    "DipfinderHistory",
    "YfinanceInfoCache",
    "AiAgentAnalysis",
    "StockFundamentals",
    "SymbolSearchResult",
    "SymbolSearchLog",
    "DataVersion",
    "AnalysisVersion",
    "SymbolIngestQueue",
]
