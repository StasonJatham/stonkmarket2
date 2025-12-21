"""Pydantic schemas for API request/response validation."""

from .auth import (
    LoginRequest,
    LoginResponse,
    UserResponse,
    PasswordChangeRequest,
)
from .symbols import (
    SymbolCreate,
    SymbolUpdate,
    SymbolResponse,
)
from .dips import (
    DipStateResponse,
    RankingEntry,
    ChartPoint,
    StockInfo,
)
from .cronjobs import (
    CronJobResponse,
    CronJobUpdate,
    CronJobLogCreate,
    CronJobLogResponse,
    CronJobLogListResponse,
)
from .common import (
    ErrorResponse,
    HealthResponse,
    PaginationParams,
)
from .suggestions import (
    SuggestionStatus,
    SuggestionCreate,
    SuggestionVote,
    SuggestionResponse,
    SuggestionListResponse,
    SuggestionAdminAction,
    TopSuggestion,
)

__all__ = [
    # Auth
    "LoginRequest",
    "LoginResponse",
    "UserResponse",
    "PasswordChangeRequest",
    # Symbols
    "SymbolCreate",
    "SymbolUpdate",
    "SymbolResponse",
    # Dips
    "DipStateResponse",
    "RankingEntry",
    "ChartPoint",
    "StockInfo",
    # CronJobs
    "CronJobResponse",
    "CronJobUpdate",
    "CronJobLogCreate",
    "CronJobLogResponse",
    "CronJobLogListResponse",
    # Suggestions
    "SuggestionStatus",
    "SuggestionCreate",
    "SuggestionVote",
    "SuggestionResponse",
    "SuggestionListResponse",
    "SuggestionAdminAction",
    "TopSuggestion",
    # Common
    "ErrorResponse",
    "HealthResponse",
    "PaginationParams",
]
