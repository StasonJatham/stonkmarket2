"""
Core services for the unified V3 quant engine.

This module provides shared services used by both the scoring system
and the backtester:
- TechnicalService: All indicator calculations (RSI, SMA, MACD, etc.)
- RegimeService: Market regime detection (Bull/Bear/Crash)
- DomainService: Domain/sector classification with scoring weights

These services ensure consistency between live scoring and backtesting.
"""

from app.quant_engine.core.technical_service import (
    TechnicalService,
    TechnicalSnapshot,
    IndicatorConfig,
    get_technical_service,
)
from app.quant_engine.core.regime_service import (
    RegimeService,
    RegimeState,
    RegimeConfig,
    StrategyConfig,
    MarketRegime,
    StrategyMode,
    REGIME_STRATEGY_CONFIGS,
    get_regime_service,
)
from app.quant_engine.core.domain_service import (
    DomainService,
    DomainAnalysisResult,
    Sector,
    get_domain_service,
    normalize_sector,
    domain_to_sector,
)

# Re-export Domain from dipfinder for convenience
from app.quant_engine.dipfinder.domain import Domain, DomainClassification

# Config and limits
from app.quant_engine.core.config import QuantLimits, QUANT_LIMITS

# Domain analysis
from app.quant_engine.core.domain_analysis import (
    DomainAnalysis,
    DomainMetrics,
    perform_domain_analysis,
    domain_analysis_to_dict,
)

# Indicators
from app.quant_engine.core.indicators import (
    compute_indicators,
    prepare_price_dataframe,
)

__all__ = [
    # Technical Service
    "TechnicalService",
    "TechnicalSnapshot", 
    "IndicatorConfig",
    "get_technical_service",
    # Regime Service
    "RegimeService",
    "RegimeState",
    "RegimeConfig",
    "StrategyConfig",
    "MarketRegime",
    "StrategyMode",
    "REGIME_STRATEGY_CONFIGS",
    "get_regime_service",
    # Domain Service
    "DomainService",
    "DomainAnalysisResult",
    "Domain",
    "DomainClassification",
    "Sector",
    "get_domain_service",
    "normalize_sector",
    "domain_to_sector",
    # Config
    "QuantLimits",
    "QUANT_LIMITS",
    # Domain Analysis
    "DomainAnalysis",
    "DomainMetrics",
    "perform_domain_analysis",
    "domain_analysis_to_dict",
    # Indicators
    "compute_indicators",
    "prepare_price_dataframe",
]
