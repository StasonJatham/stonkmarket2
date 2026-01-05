"""Admin settings schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    id: str = Field(..., description="Unique identifier for the benchmark")
    symbol: str = Field(..., description="Yahoo Finance symbol for the benchmark")
    name: str = Field(..., description="Display name for the benchmark")
    description: str | None = Field(None, description="Description of the benchmark")


class SectorETFConfig(BaseModel):
    """Sector ETF configuration."""

    sector: str = Field(..., description="Sector name (e.g., 'Technology', 'Healthcare')")
    symbol: str = Field(..., description="ETF symbol (e.g., 'XLK', 'XLV')")
    name: str = Field(..., description="Display name for the ETF")


class AppSettingsResponse(BaseModel):
    """Application settings response (read-only from env/config)."""

    app_name: str
    app_version: str
    environment: str
    debug: bool
    default_min_dip_pct: float
    default_min_days: int
    history_days: int
    chart_days: int
    vote_cooldown_days: int
    auto_approve_enabled: bool
    auto_approve_votes: int
    auto_approve_unique_voters: int
    auto_approve_min_age_hours: int
    rate_limit_enabled: bool
    rate_limit_auth: str
    rate_limit_api_anonymous: str
    rate_limit_api_authenticated: str
    scheduler_timezone: str
    external_api_timeout: int
    external_api_retries: int
    # Logo.dev settings
    logo_dev_public_key_configured: bool = Field(
        default=False, description="Whether Logo.dev public key is configured"
    )
    logo_cache_days: int = Field(
        default=90, description="Days to cache logos before refetching"
    )


class RuntimeSettingsResponse(BaseModel):
    """Runtime settings response (can be changed at runtime)."""

    signal_threshold_strong_buy: float = Field(
        default=80.0, ge=0, le=100, description="Score threshold for strong buy signal"
    )
    signal_threshold_buy: float = Field(
        default=60.0, ge=0, le=100, description="Score threshold for buy signal"
    )
    signal_threshold_hold: float = Field(
        default=40.0, ge=0, le=100, description="Score threshold for hold signal"
    )
    ai_enrichment_enabled: bool = Field(
        default=False, description="Enable AI enrichment features"
    )
    ai_batch_size: int = Field(
        default=5, ge=0, le=50, description="Batch size for AI processing (0 = all stocks)"
    )
    ai_model: str = Field(
        default="gpt-5-mini", description="AI model to use for enrichment"
    )
    suggestion_cleanup_days: int = Field(
        default=30, ge=1, le=365, description="Days to keep rejected suggestions"
    )
    auto_approve_votes: int = Field(
        default=10, ge=1, le=1000, description="Votes needed for auto-approval"
    )
    # Cache TTL settings (in seconds, 0 = no caching / real-time)
    cache_ttl_symbols: int = Field(
        default=0, ge=0, le=3600, description="Cache TTL for symbols list (seconds, 0=disabled)"
    )
    cache_ttl_suggestions: int = Field(
        default=0, ge=0, le=3600, description="Cache TTL for suggestions (seconds, 0=disabled)"
    )
    cache_ttl_ai_content: int = Field(
        default=0, ge=0, le=7200, description="Cache TTL for AI content (seconds, 0=disabled)"
    )
    cache_ttl_ranking: int = Field(
        default=0, ge=0, le=7200, description="Cache TTL for rankings/dashboard (seconds, 0=disabled)"
    )
    cache_ttl_charts: int = Field(
        default=0, ge=0, le=7200, description="Cache TTL for price charts (seconds, 0=disabled)"
    )
    benchmarks: list[BenchmarkConfig] = Field(
        default_factory=list, description="Configured benchmark indices"
    )
    sector_etfs: list[SectorETFConfig] = Field(
        default_factory=list, description="Sector ETF mappings for comparison"
    )
    # Trading/Backtest Configuration
    trading_initial_capital: float = Field(
        default=50000.0, ge=1000, le=10_000_000, description="Initial capital for backtesting (€)"
    )
    trading_flat_cost_per_trade: float = Field(
        default=1.0, ge=0, le=100, description="Flat cost per trade (€)"
    )
    trading_slippage_bps: float = Field(
        default=5.0, ge=0, le=100, description="Slippage in basis points per side"
    )
    trading_stop_loss_pct: float = Field(
        default=15.0, ge=1, le=50, description="Stop loss percentage"
    )
    trading_take_profit_pct: float = Field(
        default=30.0, ge=5, le=100, description="Take profit percentage"
    )
    trading_max_holding_days: int = Field(
        default=60, ge=5, le=365, description="Maximum holding period in days (60d optimal for capital efficiency)"
    )
    trading_min_trades_required: int = Field(
        default=30, ge=10, le=200, description="Minimum trades for statistical significance"
    )
    trading_walk_forward_folds: int = Field(
        default=5, ge=2, le=10, description="Number of walk-forward folds for validation"
    )
    trading_train_ratio: float = Field(
        default=0.70, ge=0.50, le=0.90, description="Train/test split ratio"
    )


class RuntimeSettingsUpdate(BaseModel):
    """Runtime settings update request."""

    signal_threshold_strong_buy: float | None = Field(default=None, ge=0, le=100)
    signal_threshold_buy: float | None = Field(default=None, ge=0, le=100)
    signal_threshold_hold: float | None = Field(default=None, ge=0, le=100)
    ai_enrichment_enabled: bool | None = None
    ai_batch_size: int | None = Field(default=None, ge=0, le=50)
    ai_model: str | None = None
    suggestion_cleanup_days: int | None = Field(default=None, ge=1, le=365)
    auto_approve_votes: int | None = Field(default=None, ge=1, le=1000)
    # Cache TTL settings (0 = disabled)
    cache_ttl_symbols: int | None = Field(default=None, ge=0, le=3600)
    cache_ttl_suggestions: int | None = Field(default=None, ge=0, le=3600)
    cache_ttl_ai_content: int | None = Field(default=None, ge=0, le=7200)
    cache_ttl_ranking: int | None = Field(default=None, ge=0, le=7200)
    cache_ttl_charts: int | None = Field(default=None, ge=0, le=7200)
    benchmarks: list[BenchmarkConfig] | None = None
    sector_etfs: list[SectorETFConfig] | None = None
    # Trading/Backtest Configuration
    trading_initial_capital: float | None = Field(default=None, ge=1000, le=10_000_000)
    trading_flat_cost_per_trade: float | None = Field(default=None, ge=0, le=100)
    trading_slippage_bps: float | None = Field(default=None, ge=0, le=100)
    trading_stop_loss_pct: float | None = Field(default=None, ge=1, le=50)
    trading_take_profit_pct: float | None = Field(default=None, ge=5, le=100)
    trading_max_holding_days: int | None = Field(default=None, ge=5, le=365)
    trading_min_trades_required: int | None = Field(default=None, ge=10, le=200)
    trading_walk_forward_folds: int | None = Field(default=None, ge=2, le=10)
    trading_train_ratio: float | None = Field(default=None, ge=0.50, le=0.90)


class CronJobSummary(BaseModel):
    """Summary of a cron job."""

    name: str
    cron: str
    description: str | None = None
    last_run: str | None = None
    last_status: str | None = None


class SystemStatusResponse(BaseModel):
    """Complete system status response."""

    app_settings: AppSettingsResponse
    runtime_settings: RuntimeSettingsResponse
    cronjobs: list[CronJobSummary]
    openai_configured: bool
    logo_dev_configured: bool
    total_symbols: int
    pending_suggestions: int


class BatchJobResponse(BaseModel):
    """Response for a single batch job."""

    id: int
    batch_id: str
    job_type: str = Field(..., description="Type of batch job: 'rating', 'bio', etc.")
    status: str = Field(
        ...,
        description="Job status: pending, validating, in_progress, finalizing, completed, failed, expired, cancelled"
    )
    total_requests: int = Field(default=0, description="Total items in batch")
    completed_requests: int = Field(default=0, description="Successfully completed items")
    failed_requests: int = Field(default=0, description="Failed items")
    estimated_cost_usd: float | None = Field(default=None, description="Estimated cost in USD")
    actual_cost_usd: float | None = Field(default=None, description="Actual cost in USD")
    created_at: str | None = None
    completed_at: str | None = None

    @property
    def progress_pct(self) -> float:
        """Calculate progress percentage."""
        if self.total_requests == 0:
            return 0
        return round((self.completed_requests / self.total_requests) * 100, 1)


class BatchJobListResponse(BaseModel):
    """Response for batch job list."""

    jobs: list[BatchJobResponse]
    total: int
    active_count: int = Field(default=0, description="Number of actively processing jobs")
    limit: int = Field(default=20, description="Items per page")
    offset: int = Field(default=0, description="Pagination offset")


# Settings Change History schemas


class SettingsChangeHistoryItem(BaseModel):
    """Single settings change history entry."""

    id: int
    setting_type: str = Field(..., description="Type of setting (runtime, cronjob, api_key)")
    setting_key: str = Field(..., description="The specific setting key that changed")
    old_value: Any | None = Field(None, description="Previous value")
    new_value: Any | None = Field(None, description="New value")
    changed_by: int | None = Field(None, description="User ID who made the change")
    changed_by_username: str | None = Field(None, description="Username who made the change")
    change_reason: str | None = Field(None, description="Reason for the change")
    reverted: bool = Field(default=False, description="Whether this change has been reverted")
    reverted_at: datetime | None = Field(None, description="When the change was reverted")
    reverted_by: int | None = Field(None, description="User ID who reverted the change")
    created_at: datetime = Field(..., description="When the change was made")


class SettingsChangeHistoryResponse(BaseModel):
    """Response for settings change history list."""

    changes: list[SettingsChangeHistoryItem]
    total: int
    limit: int
    offset: int


class RevertChangeRequest(BaseModel):
    """Request to revert a settings change."""

    reason: str | None = Field(None, description="Optional reason for reverting")


class RevertChangeResponse(BaseModel):
    """Response after reverting a settings change."""

    success: bool
    message: str
    reverted_setting_type: str
    reverted_setting_key: str
    restored_value: Any | None = None
