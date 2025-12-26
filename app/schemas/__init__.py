"""Pydantic schemas for API request/response validation."""

from .auth import (
    LoginRequest,
    LoginResponse,
    PasswordChangeRequest,
    UserResponse,
)
from .common import (
    ErrorResponse,
    HealthResponse,
    PaginationParams,
)
from .cronjobs import (
    CronJobLogCreate,
    CronJobLogListResponse,
    CronJobLogResponse,
    CronJobResponse,
    CronJobUpdate,
    TaskStatusResponse,
)
from .dips import (
    ChartPoint,
    DipStateResponse,
    RankingEntry,
    StockInfo,
)
from .portfolio import (
    HoldingInput,
    HoldingResponse,
    PortfolioAnalyticsJobResponse,
    PortfolioAnalyticsRequest,
    PortfolioAnalyticsResponse,
    PortfolioCreateRequest,
    PortfolioDetailResponse,
    PortfolioResponse,
    PortfolioUpdateRequest,
    TransactionInput,
    TransactionResponse,
)
from .quant_engine import (
    AuditBlockResponse,
    EngineOutputResponse,
    GenerateRecommendationsRequest,
    RecommendationRowResponse,
    TuningResultResponse,
    ValidationResultResponse,
)
from .suggestions import (
    SuggestionAdminAction,
    SuggestionApprove,
    SuggestionCreate,
    SuggestionListResponse,
    SuggestionReject,
    SuggestionResponse,
    SuggestionStatus,
    SuggestionVote,
    TopSuggestion,
)
from .symbols import (
    SymbolCreate,
    SymbolResponse,
    SymbolUpdate,
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
    "TaskStatusResponse",
    # Suggestions
    "SuggestionStatus",
    "SuggestionCreate",
    "SuggestionVote",
    "SuggestionResponse",
    "SuggestionListResponse",
    "SuggestionAdminAction",
    "SuggestionApprove",
    "SuggestionReject",
    "TopSuggestion",
    # Portfolio
    "PortfolioCreateRequest",
    "PortfolioUpdateRequest",
    "PortfolioResponse",
    "PortfolioDetailResponse",
    "HoldingInput",
    "HoldingResponse",
    "TransactionInput",
    "TransactionResponse",
    "PortfolioAnalyticsRequest",
    "PortfolioAnalyticsResponse",
    "PortfolioAnalyticsJobResponse",
    # Quant Engine
    "GenerateRecommendationsRequest",
    "EngineOutputResponse",
    "RecommendationRowResponse",
    "AuditBlockResponse",
    "ValidationResultResponse",
    "TuningResultResponse",
    # Common
    "ErrorResponse",
    "HealthResponse",
    "PaginationParams",
]
