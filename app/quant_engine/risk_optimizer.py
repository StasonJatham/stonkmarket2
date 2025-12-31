"""
Risk-Based Portfolio Optimization - Data Types.

This module contains data types for portfolio optimization.
Actual optimization is now handled by skfolio in app/portfolio/service.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RiskOptimizationMethod(str, Enum):
    """Available risk-based optimization methods."""
    
    RISK_PARITY = "risk_parity"
    MIN_VARIANCE = "min_variance"
    MAX_DIVERSIFICATION = "max_diversification"
    CVAR = "cvar"
    HIERARCHICAL_RISK_PARITY = "hrp"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class RiskOptimizationConstraints:
    """Constraints for portfolio optimization."""
    
    min_weight: float = 0.01  # Minimum 1% per position
    max_weight: float = 0.40  # Maximum 40% per position
    max_turnover: float | None = None  # Maximum weight change (optional)
    
    # Position limits per asset (optional)
    position_limits: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class RiskOptimizationResult:
    """Result from portfolio optimization."""
    
    method: str
    weights: dict[str, float]
    
    # Risk metrics of optimal portfolio
    portfolio_volatility: float
    portfolio_var_95: float
    diversification_ratio: float
    
    # Optimization diagnostics
    converged: bool
    iterations: int
    objective_value: float
    
    # Comparison to current (if provided)
    current_weights: dict[str, float] | None = None
    weight_changes: dict[str, float] | None = None
    turnover: float = 0.0
    
    # Risk changes
    vol_change_pct: float = 0.0
    
    # Optimization quality indicators (new)
    optimization_quality: str = "optimal"  # "optimal", "degraded", "fallback"
    quality_reason: str = ""  # Explanation if not optimal


@dataclass
class AllocationRecommendation:
    """User-facing allocation recommendation."""
    
    # What to do
    recommendations: list[dict]  # [{"symbol": "AAPL", "action": "BUY", "amount_eur": 500, ...}]
    
    # Current vs optimal
    current_portfolio: dict[str, float]
    optimal_portfolio: dict[str, float]
    
    # Risk improvement
    current_risk: dict
    optimal_risk: dict
    risk_improvement_summary: str
    
    # Confidence and explanation
    confidence: str  # HIGH, MEDIUM, LOW
    explanation: str
    warnings: list[str]
