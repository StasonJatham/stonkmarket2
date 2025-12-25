"""Database module with PostgreSQL connection pooling and SQLAlchemy ORM.

Provides two interfaces:
1. Legacy asyncpg helpers (fetch_one, fetch_all, execute) - for existing raw SQL code
2. SQLAlchemy ORM (get_session, Base, models) - for new ORM-based code

New code should use SQLAlchemy ORM with get_session().
"""

# Legacy asyncpg interface (raw SQL)
from .connection import (
    init_pg_pool,
    close_pg_pool,
    get_pg_pool,
    get_pg_connection,
    fetch_one,
    fetch_all,
    fetch_val,
    execute,
    execute_many,
    transaction,  # Transaction wrapper for multi-step operations
    get_async_database_url,  # Shared URL conversion utility
    # SQLAlchemy interface
    init_sqlalchemy_engine,
    close_sqlalchemy_engine,
    get_session,
    get_engine,
    init_database,
    close_database,
)

# Legacy dataclass models - DEPRECATED
# Import directly from repositories instead or use ORM models from .orm
# Keeping this import for backwards compatibility but it will emit a warning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from .models import (
        DipState as _DipStateDataclass,
        SymbolConfig as _SymbolConfigDataclass,
        CronJobConfig as _CronJobConfigDataclass,
        AuthUser as _AuthUserDataclass,
        StockSuggestion as _StockSuggestionDataclass,
        SuggestionVote as _SuggestionVoteDataclass,
    )

# Re-export with deprecation notice
DipState = _DipStateDataclass
SymbolConfig = _SymbolConfigDataclass
CronJobConfig = _CronJobConfigDataclass
AuthUser = _AuthUserDataclass
StockSuggestion = _StockSuggestionDataclass
SuggestionVote = _SuggestionVoteDataclass

# SQLAlchemy ORM models (preferred for new code)
from .orm import (
    Base,
    # Auth & Security
    AuthUser as AuthUserORM,
    SecureApiKey,
    UserApiKey,
    # Symbols & Dips
    Symbol,
    DipState as DipStateORM,
    DipHistory,
    # Suggestions
    StockSuggestion as StockSuggestionORM,
    SuggestionVote as SuggestionVoteORM,
    # Voting
    DipVote,
    DipAIAnalysis,
    # API & Batch
    ApiUsage,
    BatchJob,
    BatchTaskError,
    # Scheduler
    CronJob,
    # DipFinder
    PriceHistory,
    DipfinderSignal,
    DipfinderConfig,
    DipfinderHistory,
    YfinanceInfoCache,
    # AI
    AiAgentAnalysis,
    StockFundamentals,
    # Search & Versioning
    SymbolSearchResult,
    SymbolSearchLog,
    DataVersion,
    AnalysisVersion,
    SymbolIngestQueue,
)

__all__ = [
    # Legacy asyncpg
    "init_pg_pool",
    "close_pg_pool",
    "get_pg_pool",
    "get_pg_connection",
    "fetch_one",
    "fetch_all",
    "fetch_val",
    "execute",
    "execute_many",
    # SQLAlchemy
    "init_sqlalchemy_engine",
    "close_sqlalchemy_engine",
    "get_session",
    "get_engine",
    "init_database",
    "close_database",
    "Base",
    # Legacy dataclass models
    "DipState",
    "SymbolConfig",
    "CronJobConfig",
    "AuthUser",
    "StockSuggestion",
    "SuggestionVote",
    # ORM models
    "AuthUserORM",
    "SecureApiKey",
    "UserApiKey",
    "Symbol",
    "DipStateORM",
    "DipHistory",
    "StockSuggestionORM",
    "SuggestionVoteORM",
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
    "YFinanceInfoCache",
    "AIAgentAnalysis",
    "StockFundamentals",
    "SymbolSearchResult",
    "SymbolSearchLog",
    "DataVersion",
    "AnalysisVersion",
    "SymbolIngestQueue",
]
