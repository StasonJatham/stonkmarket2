"""
Quantitative Portfolio Engine V3 - Unified Architecture
========================================================

A non-predictive, risk-based portfolio optimization system with unified services.

This engine DOES NOT forecast future returns. Instead, it provides:
- Risk diagnostics (decomposition, tail risk, diversification metrics)
- Risk-based portfolio optimization (Risk Parity, Min Variance, CVaR, HRP)
- Technical signal scanning with per-stock optimization
- Unified scoring via ScoringOrchestrator
- Regime-aware strategy selection

Design Philosophy
-----------------
1. NO return forecasting - we don't predict prices
2. Risk-based allocation - optimize for risk, not expected returns  
3. Robust optimization - methods that don't require return estimates
4. Backtested signals - only use signals with proven historical edge
5. SINGLE SOURCE OF TRUTH - TechnicalService, RegimeService, DomainScoring

Modules
-------
- core: Shared services (TechnicalService, RegimeService)
- scoring: Unified scoring system (ScoringOrchestrator)
- analytics: Portfolio risk diagnostics and analysis
- risk_optimizer: Risk-based portfolio optimization methods
- signals: Technical signal scanner with per-stock optimization
- backtest: Advanced backtesting with regime awareness
"""

from __future__ import annotations

__version__ = "3.0.0"

# =============================================================================
# CORE SERVICES - The foundation of V3
# =============================================================================

from app.quant_engine.core import (
    # Technical Analysis
    TechnicalService,
    TechnicalSnapshot,
    IndicatorConfig,
    get_technical_service,
    # Regime Detection
    RegimeService,
    RegimeState,
    RegimeConfig,
    StrategyConfig,
    MarketRegime,
    StrategyMode,
    REGIME_STRATEGY_CONFIGS,
    get_regime_service,
)

# =============================================================================
# SCORING - Unified entry point for stock analysis
# =============================================================================

from app.quant_engine.scoring import (
    ScoringOrchestrator,
    get_scoring_orchestrator,
    StockAnalysisDashboard,
    ScoreComponents,
    EntryAnalysis,
    RiskAssessment,
)

# =============================================================================
# ANALYTICS - Portfolio risk diagnostics
# =============================================================================

from app.quant_engine.risk.analytics import (
    analyze_portfolio,
    compute_correlation_analysis,
    compute_covariance_matrix,
    compute_diversification_metrics,
    compute_risk_decomposition,
    compute_tail_risk,
    CorrelationAnalysis,
    detect_regime,
    DiversificationMetrics,
    PortfolioAnalytics,
    RiskDecomposition,
    TailRiskAnalysis,
    translate_for_user,
)

# =============================================================================
# RISK OPTIMIZER - Data Types Only
# =============================================================================

from app.quant_engine.risk.optimizer import (
    AllocationRecommendation,
    RiskOptimizationConstraints,
    RiskOptimizationMethod,
    RiskOptimizationResult,
)

# =============================================================================
# SIGNALS - Technical signal scanning
# =============================================================================

from app.quant_engine.signals.scanner import (
    OptimizedSignal,
    scan_all_stocks,
    ScanResult,
    StockOpportunity,
)

# =============================================================================
# DOMAIN ANALYSIS - Sector-specific metrics
# =============================================================================

# Sector and normalize_sector come from core.domain_service (single source of truth)
from app.quant_engine.core import Sector, normalize_sector

# Domain analysis functions for sector-specific technical analysis
from app.quant_engine.core.domain_analysis import (
    DomainAnalysis,
    DomainMetrics,
    perform_domain_analysis,
    domain_analysis_to_dict,
)


__all__ = [
    "__version__",
    # Core Services - Technical
    "TechnicalService",
    "TechnicalSnapshot",
    "IndicatorConfig",
    "get_technical_service",
    # Core Services - Regime
    "RegimeService",
    "RegimeState",
    "RegimeConfig",
    "StrategyConfig",
    "MarketRegime",
    "StrategyMode",
    "REGIME_STRATEGY_CONFIGS",
    "get_regime_service",
    # Scoring
    "ScoringOrchestrator",
    "get_scoring_orchestrator",
    "StockAnalysisDashboard",
    "ScoreComponents",
    "EntryAnalysis",
    "RiskAssessment",
    # Analytics
    "analyze_portfolio",
    "compute_correlation_analysis",
    "compute_covariance_matrix",
    "compute_diversification_metrics",
    "compute_risk_decomposition",
    "compute_tail_risk",
    "CorrelationAnalysis",
    "detect_regime",
    "DiversificationMetrics",
    "PortfolioAnalytics",
    "RiskDecomposition",
    "TailRiskAnalysis",
    "translate_for_user",
    # Risk Optimizer
    "AllocationRecommendation",
    "RiskOptimizationConstraints",
    "RiskOptimizationMethod",
    "RiskOptimizationResult",
    # Signals
    "OptimizedSignal",
    "scan_all_stocks",
    "ScanResult",
    "StockOpportunity",
    # Domain Analysis
    "DomainAnalysis",
    "DomainMetrics",
    "perform_domain_analysis",
    "domain_analysis_to_dict",
    "normalize_sector",
    "Sector",
]
