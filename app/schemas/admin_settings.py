"""Admin settings schemas."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    id: str = Field(..., description="Unique identifier for the benchmark")
    symbol: str = Field(..., description="Yahoo Finance symbol for the benchmark")
    name: str = Field(..., description="Display name for the benchmark")
    description: Optional[str] = Field(None, description="Description of the benchmark")


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
    scheduler_enabled: bool
    scheduler_timezone: str
    external_api_timeout: int
    external_api_retries: int


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
        default="gpt-4o-mini", description="AI model to use for enrichment"
    )
    suggestion_cleanup_days: int = Field(
        default=30, ge=1, le=365, description="Days to keep rejected suggestions"
    )
    benchmarks: List[BenchmarkConfig] = Field(
        default_factory=list, description="Configured benchmark indices"
    )


class RuntimeSettingsUpdate(BaseModel):
    """Runtime settings update request."""

    signal_threshold_strong_buy: Optional[float] = Field(default=None, ge=0, le=100)
    signal_threshold_buy: Optional[float] = Field(default=None, ge=0, le=100)
    signal_threshold_hold: Optional[float] = Field(default=None, ge=0, le=100)
    ai_enrichment_enabled: Optional[bool] = None
    ai_batch_size: Optional[int] = Field(default=None, ge=0, le=50)
    ai_model: Optional[str] = None
    suggestion_cleanup_days: Optional[int] = Field(default=None, ge=1, le=365)
    benchmarks: Optional[List[BenchmarkConfig]] = None


class CronJobSummary(BaseModel):
    """Summary of a cron job."""

    name: str
    cron: str
    description: Optional[str] = None
    last_run: Optional[str] = None
    last_status: Optional[str] = None


class SystemStatusResponse(BaseModel):
    """Complete system status response."""

    app_settings: AppSettingsResponse
    runtime_settings: RuntimeSettingsResponse
    cronjobs: List[CronJobSummary]
    openai_configured: bool
    total_symbols: int
    pending_suggestions: int
