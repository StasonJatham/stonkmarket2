"""
Quantitative Portfolio Engine API routes.

Provides endpoints for generating recommendations from the quant engine.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import require_user
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import TokenData
from app.core.logging import get_logger
from app.repositories import auth_user_orm as auth_repo
from app.repositories import portfolios_orm as portfolios_repo
from app.repositories import price_history_orm as price_history_repo
from app.quant_engine import (
    QuantEngineService,
    get_default_config,
    QuantConfig,
)
from app.schemas.quant_engine import (
    GenerateRecommendationsRequest,
    EngineOutputResponse,
    RecommendationRowResponse,
    AuditBlockResponse,
    ValidationResultResponse,
    TuningResultResponse,
)

router = APIRouter(prefix="/portfolios", tags=["Quant Engine"])
logger = get_logger("routes.quant_engine")


async def _get_user_id(user: TokenData) -> int:
    """Get user ID from token data."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    return record.id


async def _fetch_prices_for_symbols(
    symbols: list[str],
    lookback_days: int = 400,
) -> pd.DataFrame:
    """
    Fetch price history for multiple symbols.
    
    Returns a DataFrame with dates as index and symbols as columns.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    
    price_dfs = {}
    for symbol in symbols:
        df = await price_history_repo.get_prices_as_dataframe(
            symbol, start_date, end_date
        )
        if df is not None and "Close" in df.columns:
            price_dfs[symbol] = df["Close"]
    
    if not price_dfs:
        return pd.DataFrame()
    
    # Combine into single DataFrame
    prices = pd.DataFrame(price_dfs)
    prices = prices.dropna(how="all")
    prices = prices.ffill()  # Forward fill missing
    
    return prices


async def _fetch_market_benchmark(lookback_days: int = 400) -> pd.Series | None:
    """Fetch market benchmark prices (SPY)."""
    end_date = date.today()
    start_date = end_date - timedelta(days=lookback_days)
    
    df = await price_history_repo.get_prices_as_dataframe(
        "SPY", start_date, end_date
    )
    if df is not None and "Close" in df.columns:
        return df["Close"]
    return None


def _holdings_to_weights(
    holdings: list[dict[str, Any]],
    portfolio_value_eur: float,
    current_prices: pd.Series,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert holdings to weight array.
    
    Returns (weights, assets) tuple.
    """
    if portfolio_value_eur <= 0:
        return np.array([]), []
    
    assets = []
    weights = []
    
    for h in holdings:
        symbol = h["symbol"]
        quantity = float(h.get("quantity", 0))
        
        if symbol in current_prices.index:
            price = current_prices[symbol]
            value = quantity * price
            weight = value / portfolio_value_eur
            assets.append(symbol)
            weights.append(weight)
    
    if not assets:
        return np.array([]), []
    
    # Normalize weights
    weight_arr = np.array(weights)
    total = weight_arr.sum()
    if total > 0:
        weight_arr = weight_arr / total
    
    return weight_arr, assets


def _engine_output_to_response(output: Any) -> EngineOutputResponse:
    """Convert EngineOutput dataclass to Pydantic response."""
    recommendations = []
    for r in output.recommendations:
        rec_dict = r.to_dict() if hasattr(r, "to_dict") else vars(r)
        recommendations.append(RecommendationRowResponse(
            ticker=rec_dict.get("ticker", ""),
            action=rec_dict.get("action", "HOLD"),
            notional_eur=rec_dict.get("notional_eur", 0.0),
            delta_weight=rec_dict.get("delta_weight", 0.0),
            target_weight=rec_dict.get("target_weight", 0.0),
            mu_hat=rec_dict.get("mu_hat", 0.0),
            uncertainty=rec_dict.get("mu_hat_uncertainty", {}).get("oos_rmse", 0.0)
            if isinstance(rec_dict.get("mu_hat_uncertainty"), dict)
            else 0.0,
            risk_contribution=rec_dict.get("risk", {}).get("mcr", 0.0)
            if isinstance(rec_dict.get("risk"), dict)
            else 0.0,
            dip_score=rec_dict.get("dip", {}).get("dip_score")
            if isinstance(rec_dict.get("dip"), dict)
            else None,
            dip_bucket=rec_dict.get("dip", {}).get("bucket")
            if isinstance(rec_dict.get("dip"), dict)
            else None,
            marginal_utility=rec_dict.get("delta_utility_net", 0.0),
        ))
    
    # Build audit block
    audit_dict = output.audit.to_dict() if hasattr(output.audit, "to_dict") else {}
    audit = AuditBlockResponse(
        timestamp=datetime.now(),
        config_hash=hash(str(audit_dict.get("hyperparams", {}))),
        mu_hat_summary={
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        },
        risk_model_summary=audit_dict.get("risk_model", {}),
        optimizer_status=output.solver_status.value
        if hasattr(output.solver_status, "value")
        else str(output.solver_status),
        constraint_binding=audit_dict.get("constraint_binding", []),
        turnover_realized=audit_dict.get("turnover", 0.0),
        regime_state=audit_dict.get("regime", "neutral_medium"),
        dip_stats=audit_dict.get("dip_stats"),
        error_message=None,
    )
    
    return EngineOutputResponse(
        recommendations=recommendations,
        as_of_date=output.as_of
        if isinstance(output.as_of, datetime)
        else datetime.combine(output.as_of, datetime.min.time()),
        portfolio_value_eur=output.portfolio_value_eur,
        inflow_eur=output.inflow_eur,
        total_trades=len([r for r in recommendations if r.action != "HOLD"]),
        total_transaction_cost_eur=output.total_transaction_cost_eur,
        expected_portfolio_return=output.expected_portfolio_return,
        expected_portfolio_risk=output.expected_portfolio_risk,
        audit=audit,
    )


@router.post(
    "/{portfolio_id}/recommendations",
    response_model=EngineOutputResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Quant Engine Recommendations",
    description="""
    Generate portfolio recommendations using the quantitative engine.
    
    The engine uses:
    - Ridge/Lasso ensemble alpha model with OOS validation
    - PCA-based factor risk model
    - Incremental mean-variance optimization with transaction costs
    - DipScore adjustments (informational only, never generates orders)
    
    Returns ranked recommendations with full audit trail.
    """,
)
async def generate_recommendations(
    portfolio_id: int,
    payload: GenerateRecommendationsRequest,
    user: TokenData = Depends(require_user),
) -> EngineOutputResponse:
    """Generate quant engine recommendations for a portfolio."""
    user_id = await _get_user_id(user)
    
    # Verify portfolio ownership
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    
    # Get holdings
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")
    
    # Extract symbols from holdings
    symbols = [h["symbol"] for h in holdings]
    
    # Fetch price history
    prices = await _fetch_prices_for_symbols(symbols)
    if prices.empty:
        raise ValidationError(
            message="No price history available for portfolio symbols"
        )
    
    # Fetch market benchmark
    market_prices = await _fetch_market_benchmark()
    
    # Calculate current weights
    if prices.columns.tolist():
        current_prices = prices.iloc[-1]
        w_current, assets = _holdings_to_weights(
            holdings, payload.portfolio_value_eur, current_prices
        )
    else:
        w_current = np.array([])
        assets = []
    
    if len(assets) == 0:
        raise ValidationError(
            message="Unable to compute weights for portfolio holdings"
        )
    
    # Filter prices to only assets we have weights for
    prices = prices[assets]
    
    # Use provided weights if available
    if payload.current_weights:
        w_current = np.array([
            payload.current_weights.get(a, 0.0) for a in assets
        ])
        # Normalize
        total = w_current.sum()
        if total > 0:
            w_current = w_current / total
    
    # Initialize engine with default config
    config = get_default_config()
    engine = QuantEngineService(config=config)
    
    # Generate recommendations
    output = engine.generate_recommendations(
        prices=prices,
        w_current=w_current,
        portfolio_value_eur=payload.portfolio_value_eur,
        inflow_eur=payload.inflow_eur,
        market_prices=market_prices,
        retrain=payload.force_retrain,
    )
    
    return _engine_output_to_response(output)


@router.post(
    "/{portfolio_id}/quant/validate",
    response_model=ValidationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Walk-Forward Validation",
    description="""
    Run walk-forward validation on the quant engine for a portfolio.
    
    This validates the model using historical data with proper train/test splits.
    """,
)
async def validate_walk_forward(
    portfolio_id: int,
    n_folds: int = Query(5, ge=2, le=20),
    user: TokenData = Depends(require_user),
) -> ValidationResultResponse:
    """Run walk-forward validation for a portfolio."""
    user_id = await _get_user_id(user)
    
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")
    
    symbols = [h["symbol"] for h in holdings]
    
    # Fetch longer history for walk-forward
    prices = await _fetch_prices_for_symbols(symbols, lookback_days=1500)
    if prices.empty:
        raise ValidationError(
            message="Insufficient price history for validation"
        )
    
    market_prices = await _fetch_market_benchmark(lookback_days=1500)
    
    config = get_default_config()
    engine = QuantEngineService(config=config)
    
    result = engine.validate_walk_forward(
        prices=prices,
        market_prices=market_prices,
        n_folds=n_folds,
    )
    
    return ValidationResultResponse(
        n_folds=len(result.folds),
        aggregate_sharpe=result.aggregate_sharpe,
        aggregate_return=result.aggregate_return,
        aggregate_volatility=result.aggregate_volatility,
        max_drawdown=result.aggregate_max_drawdown,
        baseline_sharpe=result.baseline_sharpe,
        hit_rate=result.hit_rate,
        total_turnover=result.total_turnover,
    )


@router.post(
    "/{portfolio_id}/quant/tune",
    response_model=TuningResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Tune Hyperparameters",
    description="""
    Tune quant engine hyperparameters using nested walk-forward validation.
    
    This is a computationally expensive operation.
    """,
)
async def tune_hyperparameters(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> TuningResultResponse:
    """Tune hyperparameters for a portfolio."""
    import time
    
    user_id = await _get_user_id(user)
    
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")
    
    symbols = [h["symbol"] for h in holdings]
    
    prices = await _fetch_prices_for_symbols(symbols, lookback_days=1500)
    if prices.empty:
        raise ValidationError(
            message="Insufficient price history for tuning"
        )
    
    market_prices = await _fetch_market_benchmark(lookback_days=1500)
    
    start_time = time.time()
    
    config = get_default_config()
    engine = QuantEngineService(config=config)
    
    best_params, best_score, logs = engine.tune_hyperparameters(
        prices=prices,
        market_prices=market_prices,
    )
    
    elapsed = time.time() - start_time
    
    return TuningResultResponse(
        best_params=best_params,
        best_score=best_score,
        n_evaluations=len(logs),
        tuning_time_seconds=elapsed,
    )
