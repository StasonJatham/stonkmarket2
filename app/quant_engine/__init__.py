"""
Quantitative Portfolio Engine V2
================================

A non-predictive, risk-based portfolio optimization system.

This engine DOES NOT forecast future returns. Instead, it provides:
- Risk diagnostics (decomposition, tail risk, diversification metrics)
- Risk-based portfolio optimization (Risk Parity, Min Variance, CVaR, HRP)
- Technical signal scanning with per-stock optimization
- User-friendly translation of complex analytics

Design Philosophy
-----------------
1. NO return forecasting - we don't predict prices
2. Risk-based allocation - optimize for risk, not expected returns  
3. Robust optimization - methods that don't require return estimates
4. Backtested signals - only use signals with proven historical edge
5. User-friendly output - translate quant jargon to plain English

Modules
-------
- analytics: Portfolio risk diagnostics and analysis
- risk_optimizer: Risk-based portfolio optimization methods
- signals: Technical signal scanner with per-stock optimization

Optimization Methods
--------------------
- RISK_PARITY: Equal risk contribution from each asset
- MIN_VARIANCE: Minimize total portfolio volatility
- MAX_DIVERSIFICATION: Maximize diversification ratio
- CVAR: Minimize Conditional VaR (expected shortfall)
- HRP: Hierarchical Risk Parity (López de Prado)
"""

from __future__ import annotations

__version__ = "2.0.0"

# Analytics
from app.quant_engine.analytics import (
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
    RegimeState,
    RiskDecomposition,
    TailRiskAnalysis,
    translate_for_user,
)

# Risk Optimizer
from app.quant_engine.risk_optimizer import (
    AllocationRecommendation,
    generate_allocation_recommendation,
    optimize_cvar,
    optimize_hrp,
    optimize_max_diversification,
    optimize_min_variance,
    optimize_portfolio_risk_based,
    optimize_risk_parity,
    RiskOptimizationConstraints,
    RiskOptimizationMethod,
    RiskOptimizationResult,
)

# Signals
from app.quant_engine.signals import (
    OptimizedSignal,
    scan_all_stocks,
    ScanResult,
    StockOpportunity,
)


__all__ = [
    "__version__",
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
    "RegimeState",
    "RiskDecomposition",
    "TailRiskAnalysis",
    "translate_for_user",
    # Risk Optimizer
    "AllocationRecommendation",
    "generate_allocation_recommendation",
    "optimize_cvar",
    "optimize_hrp",
    "optimize_max_diversification",
    "optimize_min_variance",
    "optimize_portfolio_risk_based",
    "optimize_risk_parity",
    "RiskOptimizationConstraints",
    "RiskOptimizationMethod",
    "RiskOptimizationResult",
    # Signals
    "OptimizedSignal",
    "scan_all_stocks",
    "ScanResult",
    "StockOpportunity",
]
