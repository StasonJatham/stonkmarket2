"""
Pydantic schemas for the Quantitative Portfolio Engine API.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RecommendationRowResponse(BaseModel):
    """Single recommendation from the quant engine."""

    model_config = ConfigDict(from_attributes=True)

    ticker: str = Field(..., description="Asset ticker symbol")
    name: str | None = Field(None, description="Asset name")
    action: str = Field(..., description="BUY, SELL, or HOLD")
    notional_eur: float = Field(..., description="Trade amount in EUR")
    delta_weight: float = Field(..., description="Weight change from current")
    target_weight: float = Field(..., description="Target weight after trade")
    mu_hat: float = Field(..., description="Expected return estimate")
    uncertainty: float = Field(..., description="Forecast uncertainty")
    risk_contribution: float = Field(..., description="Contribution to portfolio risk")
    dip_score: float | None = Field(None, description="Statistical DipScore (z-score)")
    dip_bucket: str | None = Field(None, description="DipScore bucket")
    marginal_utility: float = Field(..., description="Marginal utility of position")
    # Legacy fields for backward compatibility
    legacy_dip_pct: float | None = Field(None, description="Legacy dip percentage")
    legacy_days_in_dip: int | None = Field(None, description="Legacy days in dip")
    legacy_domain_score: float | None = Field(None, description="Legacy domain score")


class AuditBlockResponse(BaseModel):
    """Audit information for transparency and reproducibility."""

    timestamp: datetime = Field(..., description="Recommendation timestamp")
    config_hash: int = Field(..., description="Hash of configuration used")
    mu_hat_summary: dict = Field(..., description="Summary stats of expected returns")
    risk_model_summary: dict = Field(..., description="Risk model info")
    optimizer_status: str = Field(..., description="Solver status")
    constraint_binding: list[str] = Field(..., description="Assets at max weight")
    turnover_realized: float = Field(..., description="Actual turnover")
    regime_state: str = Field(..., description="Market regime (e.g., bull_low)")
    dip_stats: dict | None = Field(None, description="DipScore statistics")
    error_message: str | None = Field(None, description="Error message if any")


class EngineOutputResponse(BaseModel):
    """Complete response from the quant engine."""

    recommendations: list[RecommendationRowResponse] = Field(
        ..., description="Ranked recommendations"
    )
    as_of_date: datetime = Field(..., description="Date of recommendations")
    portfolio_value_eur: float = Field(..., description="Portfolio value in EUR")
    inflow_eur: float = Field(..., description="Monthly inflow in EUR")
    total_trades: int = Field(..., description="Number of trades")
    total_transaction_cost_eur: float = Field(..., description="Total transaction costs")
    expected_portfolio_return: float = Field(..., description="Expected portfolio return")
    expected_portfolio_risk: float = Field(..., description="Expected portfolio volatility")
    audit: AuditBlockResponse = Field(..., description="Audit block")


class GenerateRecommendationsRequest(BaseModel):
    """Request to generate portfolio recommendations."""

    portfolio_value_eur: float = Field(..., ge=0, description="Current portfolio value in EUR")
    inflow_eur: float = Field(1000.0, ge=0, description="Monthly inflow in EUR")
    current_weights: dict[str, float] | None = Field(
        None, description="Current portfolio weights by ticker"
    )
    force_retrain: bool = Field(False, description="Force model retraining")


class ValidationResultResponse(BaseModel):
    """Response from walk-forward validation."""

    n_folds: int = Field(..., description="Number of walk-forward folds")
    aggregate_sharpe: float = Field(..., description="Aggregate Sharpe ratio")
    aggregate_return: float = Field(..., description="Aggregate return")
    aggregate_volatility: float = Field(..., description="Aggregate volatility")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    baseline_sharpe: dict[str, float] = Field(..., description="Baseline Sharpe ratios")
    hit_rate: float = Field(..., description="Direction prediction accuracy")
    total_turnover: float = Field(..., description="Total turnover across folds")


class TuningResultResponse(BaseModel):
    """Response from hyperparameter tuning."""

    best_params: dict = Field(..., description="Best hyperparameters found")
    best_score: float = Field(..., description="Best objective score")
    n_evaluations: int = Field(..., description="Number of parameter combinations evaluated")
    tuning_time_seconds: float = Field(..., description="Time taken for tuning")
