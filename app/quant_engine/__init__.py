"""
Quantitative Portfolio Engine
=============================

Research-grade portfolio decision engine implementing:
- Factor-based expected return models (Ridge/Lasso ensemble)
- Statistical DipScore (factor-residual z-score, informational only)
- PCA-based risk model with MCR computation
- Incremental mean-variance optimization with transaction costs
- Walk-forward hyperparameter tuning with OOS validation
- EUR base currency, long-only, €1/trade fixed costs

The engine NEVER generates direct orders from dip signals.
All recommendations flow from the optimizer's Δw* solution.

Modules
-------
- features: Feature engineering from price/factor data
- alpha_models: Ridge/Lasso ensemble with uncertainty quantification
- dip: Statistical DipScore (factor-residual z-score)
- regimes: Trend/volatility regime detection
- risk: PCA factor model for covariance estimation
- optimizer: Incremental QP with constraints and cost modeling
- walk_forward: Walk-forward validation harness
- tuner: Automated hyperparameter optimization
- persistence: Database models and artifact storage
- service: Main orchestration service
- schemas: API request/response schemas

Non-Negotiable Rules
--------------------
1. Every decision from explicit mathematics (no ad-hoc triggers)
2. Every assumption statistically testable and falsifiable
3. Every signal validated out-of-sample (walk-forward)
4. Dip logic only affects μ_hat or uncertainty, never orders
5. Long-only, no leverage, EUR base currency
6. Monthly inflows (€1,000–€1,500) part of optimization
7. Transaction costs: fixed €1 per trade modeled explicitly
"""

from __future__ import annotations

__version__ = "1.0.0"

# Core types
from app.quant_engine.types import (
    QuantConfig,
    AlphaResult,
    DipArtifacts,
    RegimeState,
    RiskModel,
    OptimizationResult,
    RecommendationRow,
    EngineOutput,
    HyperparameterLog,
    WalkForwardFold,
    WalkForwardResult,
    ConstraintStatus,
    SolverStatus,
    RegimeTrend,
    RegimeVolatility,
    MomentumCondition,
    ActionType,
)

# Services
from app.quant_engine.service import QuantEngineService, get_default_config

# Alpha models
from app.quant_engine.alpha_models import AlphaModelEnsemble

# Risk models
from app.quant_engine.risk import fit_pca_risk_model

# Optimizer
from app.quant_engine.optimizer import optimize_portfolio

# Walk-forward
from app.quant_engine.walk_forward import WalkForwardValidator

# Tuner
from app.quant_engine.tuner import HyperparameterTuner, HyperparameterGrid

__all__ = [
    "__version__",
    # Types
    "QuantConfig",
    "AlphaResult",
    "DipArtifacts",
    "RegimeState",
    "RiskModel",
    "OptimizationResult",
    "RecommendationRow",
    "EngineOutput",
    "HyperparameterLog",
    "WalkForwardFold",
    "WalkForwardResult",
    "ConstraintStatus",
    "SolverStatus",
    "RegimeTrend",
    "RegimeVolatility",
    "MomentumCondition",
    "ActionType",
    # Services
    "QuantEngineService",
    "get_default_config",
    # Alpha
    "AlphaModelEnsemble",
    # Risk
    "fit_pca_risk_model",
    # Optimizer
    "optimize_portfolio",
    # Validation
    "WalkForwardValidator",
    # Tuning
    "HyperparameterTuner",
    "HyperparameterGrid",
]
