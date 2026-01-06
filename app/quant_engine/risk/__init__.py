"""
Risk Module - Portfolio risk analysis and optimization.

This module provides:
- Portfolio analytics (correlation, tail risk, diversification)
- Risk-based optimization (Risk Parity, Min Variance, CVaR, HRP)
- Risk highlights for portfolio holdings
"""

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
from app.quant_engine.risk.optimizer import (
    AllocationRecommendation,
    RiskOptimizationConstraints,
    RiskOptimizationMethod,
    RiskOptimizationResult,
)
from app.quant_engine.risk.highlights import (
    build_portfolio_risk_highlights,
    RiskHighlight,
)

__all__ = [
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
    # Optimizer
    "AllocationRecommendation",
    "RiskOptimizationConstraints",
    "RiskOptimizationMethod",
    "RiskOptimizationResult",
    # Highlights
    "build_portfolio_risk_highlights",
    "RiskHighlight",
]
