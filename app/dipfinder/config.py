"""DipFinder configuration with thresholds and settings.

All thresholds are loaded from environment or settings, with sensible defaults.
Per-symbol overrides can be stored in the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class DipFinderSettings(BaseSettings):
    """DipFinder settings from environment."""

    model_config = ConfigDict(extra="ignore")

    # Available windows for dip calculation
    dipfinder_windows: list[int] = Field(
        default=[7, 30, 100, 365],
        description="Available windows for dip calculation (days)",
    )

    # Dip thresholds
    dipfinder_min_dip_abs: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Minimum absolute dip percentage (10%)",
    )
    dipfinder_min_persist_days: int = Field(
        default=2, ge=1, le=30, description="Minimum days dip must persist"
    )
    dipfinder_dip_percentile_threshold: float = Field(
        default=0.80,
        ge=0.50,
        le=0.99,
        description="Top percentile for significant dip (top 20%)",
    )
    dipfinder_dip_vs_typical_threshold: float = Field(
        default=1.5, ge=1.0, le=5.0, description="Multiplier vs typical dip size"
    )

    # Market context thresholds
    dipfinder_market_dip_threshold: float = Field(
        default=0.06, ge=0.01, le=0.20, description="Market dip threshold (6%)"
    )
    dipfinder_excess_dip_stock_specific: float = Field(
        default=0.04,
        ge=0.01,
        le=0.20,
        description="Excess dip for stock-specific classification (4%)",
    )
    dipfinder_excess_dip_market: float = Field(
        default=0.03,
        ge=0.01,
        le=0.10,
        description="Max excess for pure market dip (3%)",
    )

    # Quality and stability gates
    dipfinder_quality_gate: float = Field(
        default=60.0, ge=0, le=100, description="Minimum quality score for alert"
    )
    dipfinder_stability_gate: float = Field(
        default=60.0, ge=0, le=100, description="Minimum stability score for alert"
    )

    # Alert thresholds
    dipfinder_alert_good: float = Field(
        default=70.0, ge=50, le=100, description="Minimum final score for 'good' alert"
    )
    dipfinder_alert_strong: float = Field(
        default=80.0,
        ge=60,
        le=100,
        description="Minimum final score for 'strong' alert",
    )

    # Score weights
    dipfinder_weight_dip: float = Field(
        default=0.45, ge=0, le=1, description="Weight for dip score in final"
    )
    dipfinder_weight_quality: float = Field(
        default=0.30, ge=0, le=1, description="Weight for quality score in final"
    )
    dipfinder_weight_stability: float = Field(
        default=0.25, ge=0, le=1, description="Weight for stability score in final"
    )

    # Domain-specific scoring
    dipfinder_domain_scoring_enabled: bool = Field(
        default=True,
        description="Enable domain-specific scoring (banks, REITs, ETFs, etc.)",
    )
    dipfinder_domain_scoring_log_enabled: bool = Field(
        default=False,
        description="Log domain classification and scoring details",
    )

    # Default benchmark
    dipfinder_default_benchmark: str = Field(
        default="SPY", description="Default benchmark ticker"
    )

    # Cache TTLs (seconds)
    dipfinder_price_cache_ttl: int = Field(
        default=3600, ge=300, description="Price data cache TTL (1 hour)"
    )
    dipfinder_info_cache_ttl: int = Field(
        default=86400, ge=3600, description="yfinance info cache TTL (24 hours)"
    )
    dipfinder_signal_cache_ttl: int = Field(
        default=1800, ge=300, description="Computed signal cache TTL (30 min)"
    )

    # Rate limiting for yfinance
    dipfinder_yf_batch_size: int = Field(
        default=10, ge=1, le=50, description="Max symbols per yfinance batch request"
    )
    dipfinder_yf_batch_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Delay between yfinance batches (seconds)",
    )
    dipfinder_yf_info_delay: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Delay between yfinance info requests (seconds)",
    )

    # History settings
    dipfinder_history_years: int = Field(
        default=5, ge=1, le=10, description="Years of price history to fetch"
    )


@dataclass
class DipFinderConfig:
    """Complete DipFinder configuration.

    This can be loaded from settings or overridden per-symbol from the database.
    """

    # Windows
    windows: list[int] = field(default_factory=lambda: [7, 30, 100, 365])

    # Dip thresholds
    min_dip_abs: float = 0.10
    min_persist_days: int = 2
    dip_percentile_threshold: float = 0.80
    dip_vs_typical_threshold: float = 1.5

    # Market context
    market_dip_threshold: float = 0.06
    excess_dip_stock_specific: float = 0.04
    excess_dip_market: float = 0.03

    # Gates
    quality_gate: float = 60.0
    stability_gate: float = 60.0

    # Alert thresholds
    alert_good: float = 70.0
    alert_strong: float = 80.0

    # Weights (must sum to 1.0)
    weight_dip: float = 0.45
    weight_quality: float = 0.30
    weight_stability: float = 0.25

    # Default benchmark
    default_benchmark: str = "SPY"

    # Cache TTLs
    price_cache_ttl: int = 3600
    info_cache_ttl: int = 86400
    signal_cache_ttl: int = 1800

    # Rate limiting
    yf_batch_size: int = 10
    yf_batch_delay: float = 1.0
    yf_info_delay: float = 0.5

    # History
    history_years: int = 5

    # Domain-specific scoring
    domain_scoring_enabled: bool = True
    domain_scoring_log_enabled: bool = False

    @classmethod
    def from_settings(
        cls, settings: DipFinderSettings | None = None
    ) -> DipFinderConfig:
        """Create config from settings."""
        if settings is None:
            settings = DipFinderSettings()

        return cls(
            windows=settings.dipfinder_windows,
            min_dip_abs=settings.dipfinder_min_dip_abs,
            min_persist_days=settings.dipfinder_min_persist_days,
            dip_percentile_threshold=settings.dipfinder_dip_percentile_threshold,
            dip_vs_typical_threshold=settings.dipfinder_dip_vs_typical_threshold,
            market_dip_threshold=settings.dipfinder_market_dip_threshold,
            excess_dip_stock_specific=settings.dipfinder_excess_dip_stock_specific,
            excess_dip_market=settings.dipfinder_excess_dip_market,
            quality_gate=settings.dipfinder_quality_gate,
            stability_gate=settings.dipfinder_stability_gate,
            alert_good=settings.dipfinder_alert_good,
            alert_strong=settings.dipfinder_alert_strong,
            weight_dip=settings.dipfinder_weight_dip,
            weight_quality=settings.dipfinder_weight_quality,
            weight_stability=settings.dipfinder_weight_stability,
            default_benchmark=settings.dipfinder_default_benchmark,
            price_cache_ttl=settings.dipfinder_price_cache_ttl,
            info_cache_ttl=settings.dipfinder_info_cache_ttl,
            signal_cache_ttl=settings.dipfinder_signal_cache_ttl,
            yf_batch_size=settings.dipfinder_yf_batch_size,
            yf_batch_delay=settings.dipfinder_yf_batch_delay,
            yf_info_delay=settings.dipfinder_yf_info_delay,
            history_years=settings.dipfinder_history_years,
            domain_scoring_enabled=settings.dipfinder_domain_scoring_enabled,
            domain_scoring_log_enabled=settings.dipfinder_domain_scoring_log_enabled,
        )

    def with_overrides(
        self,
        min_dip_abs: float | None = None,
        min_persist_days: int | None = None,
        dip_percentile_threshold: float | None = None,
        dip_vs_typical_threshold: float | None = None,
        quality_gate: float | None = None,
        stability_gate: float | None = None,
    ) -> DipFinderConfig:
        """Return a new config with optional overrides applied."""
        from dataclasses import replace

        overrides = {}
        if min_dip_abs is not None:
            overrides["min_dip_abs"] = min_dip_abs
        if min_persist_days is not None:
            overrides["min_persist_days"] = min_persist_days
        if dip_percentile_threshold is not None:
            overrides["dip_percentile_threshold"] = dip_percentile_threshold
        if dip_vs_typical_threshold is not None:
            overrides["dip_vs_typical_threshold"] = dip_vs_typical_threshold
        if quality_gate is not None:
            overrides["quality_gate"] = quality_gate
        if stability_gate is not None:
            overrides["stability_gate"] = stability_gate

        return replace(self, **overrides) if overrides else self


@lru_cache(maxsize=1)
def get_dipfinder_config() -> DipFinderConfig:
    """Get cached DipFinder configuration from settings."""
    return DipFinderConfig.from_settings()
