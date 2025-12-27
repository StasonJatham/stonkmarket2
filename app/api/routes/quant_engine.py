"""
Quantitative Portfolio Engine V2 API Routes.

Risk-based portfolio optimization - NO return forecasting.

Provides endpoints for:
- Portfolio risk analytics (risk decomposition, tail risk, diversification)
- Risk-based allocation recommendations (Risk Parity, CVaR, HRP, etc.)
- Technical signal scanning with per-stock optimization
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import require_user
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.quant_engine import (
    analyze_portfolio,
    generate_allocation_recommendation,
    RiskOptimizationMethod,
    scan_all_stocks,
    translate_for_user,
)
from app.repositories import auth_user_orm as auth_repo
from app.repositories import portfolios_orm as portfolios_repo
from app.repositories import price_history_orm as price_history_repo
from app.schemas.quant_engine import (
    SignalResultResponse,
    SignalScanResponse,
    StockSignalResponse,
)


if TYPE_CHECKING:
    from app.core.security import TokenData


router = APIRouter(prefix="/portfolios", tags=["Quant Engine"])
logger = get_logger("routes.quant_engine")


# =============================================================================
# Helper Functions
# =============================================================================


async def _get_user_id(user: TokenData) -> int:
    """Get user ID from token data."""
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    return record.id


async def _fetch_prices_for_symbols(
    symbols: list[str],
    lookback_days: int = 1260,  # 5 years
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
    prices = prices.ffill()

    return prices


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from prices."""
    return prices.pct_change().dropna()


def _safe_float(value: float | None, default: float = 0.0) -> float:
    """Convert a float to JSON-safe value (handle NaN/inf)."""
    import math
    if value is None or math.isnan(value) or math.isinf(value):
        return default
    return float(value)


# =============================================================================
# Portfolio Analytics Endpoint
# =============================================================================


@router.get(
    "/{portfolio_id}/analytics",
    status_code=status.HTTP_200_OK,
    summary="Portfolio Risk Analytics",
    description="""
    Get comprehensive risk analytics for a portfolio.
    
    This endpoint provides non-predictive risk diagnostics:
    - Risk decomposition (which positions contribute most to risk)
    - Tail risk analysis (VaR, CVaR, max drawdown)
    - Diversification metrics (effective N, diversification ratio)
    - Market regime detection
    - Correlation analysis
    
    All analytics are translated into user-friendly language.
    """,
)
async def get_portfolio_analytics(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> dict[str, Any]:
    """Get risk analytics for a portfolio."""
    user_id = await _get_user_id(user)

    # Verify portfolio ownership
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    # Get holdings
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]

    # Fetch price history
    prices = await _fetch_prices_for_symbols(symbols)
    if prices.empty or len(prices) < 60:
        raise ValidationError(
            message="Insufficient price history for analysis (need at least 60 days)"
        )

    # Compute returns
    returns = _compute_returns(prices)

    # Calculate current weights from holdings
    current_prices = prices.iloc[-1]
    weights = {}
    total_value = 0.0
    
    for h in holdings:
        symbol = h["symbol"]
        quantity = float(h.get("quantity", 0))
        if symbol in current_prices.index:
            value = quantity * current_prices[symbol]
            weights[symbol] = value
            total_value += value
    
    if total_value > 0:
        weights = {s: v / total_value for s, v in weights.items()}
    else:
        # Equal weight fallback
        weights = {s: 1.0 / len(symbols) for s in symbols}

    # Fetch SPY returns for market regime detection
    spy_prices = await _fetch_prices_for_symbols(["SPY"])
    market_returns = None
    if not spy_prices.empty and "SPY" in spy_prices.columns:
        market_returns = spy_prices["SPY"].pct_change().dropna()

    # Run analytics
    analytics = analyze_portfolio(
        holdings=weights,
        returns=returns,
        market_returns=market_returns,
        total_value=total_value,
    )

    # Translate to user-friendly format
    user_friendly = translate_for_user(analytics)

    return {
        "portfolio_id": portfolio_id,
        "analyzed_at": datetime.now().isoformat(),
        "total_value_eur": _safe_float(total_value),
        "n_positions": analytics.n_positions,
        "risk_score": analytics.overall_risk_score,
        "summary": user_friendly["summary"],
        "risk": user_friendly["risk"],
        "diversification": user_friendly["diversification"],
        "market": user_friendly["market"],
        "insights": user_friendly["insights"],
        "action_items": user_friendly["action_items"],
        # Raw analytics for power users
        "raw": {
            "portfolio_volatility": _safe_float(analytics.risk_decomposition.portfolio_volatility),
            "var_95": _safe_float(analytics.tail_risk.var_95_daily),
            "cvar_95": _safe_float(analytics.tail_risk.cvar_95_daily),
            "max_drawdown": _safe_float(analytics.tail_risk.max_drawdown),
            "effective_n": _safe_float(analytics.diversification.effective_n),
            "diversification_ratio": _safe_float(analytics.diversification.diversification_ratio),
            "regime": analytics.regime.regime,
            "risk_contributions": {
                s: _safe_float(v) 
                for s, v in analytics.risk_decomposition.risk_contribution_pct.items()
            },
        },
    }


# =============================================================================
# Allocation Recommendation Endpoint
# =============================================================================


@router.post(
    "/{portfolio_id}/allocate",
    status_code=status.HTTP_200_OK,
    summary="Get Allocation Recommendation",
    description="""
    Get a risk-based allocation recommendation for new investment.
    
    This answers: "Where should my next €X go?"
    
    Uses risk-based optimization (no return forecasting):
    - RISK_PARITY: Equal risk contribution from each position
    - MIN_VARIANCE: Minimize total portfolio volatility  
    - MAX_DIVERSIFICATION: Maximize diversification ratio
    - CVAR: Minimize expected loss in worst scenarios
    - HRP: Hierarchical Risk Parity (correlation-based clustering)
    
    Returns specific trade recommendations with rationale.
    """,
)
async def get_allocation_recommendation(
    portfolio_id: int,
    inflow_eur: float = Query(1000.0, ge=50, le=100000, description="Amount to invest"),
    method: str = Query(
        "risk_parity",
        description="Optimization method: risk_parity, min_variance, max_diversification, cvar, hrp"
    ),
    user: TokenData = Depends(require_user),
) -> dict[str, Any]:
    """Get allocation recommendation for a portfolio."""
    user_id = await _get_user_id(user)

    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]

    prices = await _fetch_prices_for_symbols(symbols)
    if prices.empty or len(prices) < 60:
        raise ValidationError(
            message="Insufficient price history for optimization (need at least 60 days)"
        )

    returns = _compute_returns(prices)

    # Calculate current weights and portfolio value
    current_prices = prices.iloc[-1]
    weights = {}
    total_value = 0.0
    
    for h in holdings:
        symbol = h["symbol"]
        quantity = float(h.get("quantity", 0))
        if symbol in current_prices.index:
            value = quantity * current_prices[symbol]
            weights[symbol] = value
            total_value += value
    
    if total_value > 0:
        weights = {s: v / total_value for s, v in weights.items()}
    else:
        weights = {s: 1.0 / len(symbols) for s in symbols}

    # Parse optimization method
    method_map = {
        "risk_parity": RiskOptimizationMethod.RISK_PARITY,
        "min_variance": RiskOptimizationMethod.MIN_VARIANCE,
        "max_diversification": RiskOptimizationMethod.MAX_DIVERSIFICATION,
        "cvar": RiskOptimizationMethod.CVAR,
        "hrp": RiskOptimizationMethod.HIERARCHICAL_RISK_PARITY,
        "equal_weight": RiskOptimizationMethod.EQUAL_WEIGHT,
    }
    opt_method = method_map.get(method.lower(), RiskOptimizationMethod.RISK_PARITY)

    # Generate recommendation
    recommendation = generate_allocation_recommendation(
        returns=returns,
        symbols=list(returns.columns),
        current_weights=weights,
        inflow_eur=inflow_eur,
        portfolio_value_eur=total_value,
        method=opt_method,
    )

    return {
        "portfolio_id": portfolio_id,
        "method": recommendation.method if hasattr(recommendation, 'method') else method,
        "inflow_eur": inflow_eur,
        "portfolio_value_eur": _safe_float(total_value),
        "confidence": recommendation.confidence,
        "explanation": recommendation.explanation,
        "risk_improvement": recommendation.risk_improvement_summary,
        "current_risk": {
            "volatility": _safe_float(recommendation.current_risk.get("volatility", 0)),
            "label": recommendation.current_risk.get("volatility_label", "Unknown"),
        },
        "optimal_risk": {
            "volatility": _safe_float(recommendation.optimal_risk.get("volatility", 0)),
            "label": recommendation.optimal_risk.get("volatility_label", "Unknown"),
            "diversification_ratio": _safe_float(
                recommendation.optimal_risk.get("diversification_ratio", 0)
            ),
        },
        "trades": recommendation.recommendations,
        "current_weights": {s: _safe_float(v) for s, v in recommendation.current_portfolio.items()},
        "target_weights": {s: _safe_float(v) for s, v in recommendation.optimal_portfolio.items()},
        "warnings": recommendation.warnings,
    }


# =============================================================================
# Global Recommendations Router (No Auth Required)
# =============================================================================

global_router = APIRouter(prefix="/recommendations", tags=["Quant Engine"])


# ============================================================================
# Signal Scanner Endpoint (Global - No Auth Required)
# ============================================================================


@global_router.get(
    "/signals",
    response_model=SignalScanResponse,
    summary="Scan stocks for buy opportunities",
    description="""
    Scan all tracked stocks for technical buy signals with per-stock optimization.
    
    This endpoint answers:
    - "Which stock should I buy next?"
    - "What's the optimal holding period after this signal?"
    - "Which dip is a statistical overreaction?"
    
    It tests 14 technical signals (RSI, Z-score, MACD, Bollinger Bands, SMA, etc.)
    for each stock, optimizes the threshold and holding period parameters,
    and ranks stocks by their current buy opportunity score.
    
    Each signal includes:
    - optimal_threshold: Best signal threshold for this specific stock
    - optimal_holding_days: Best holding period after signal triggers
    - win_rate: Historical success rate at optimal parameters
    - avg_return_pct: Average return when buying on this signal
    """,
)
async def scan_signals() -> SignalScanResponse:
    """Scan all stocks for technical buy signals with per-stock optimization."""
    from app.repositories import symbols_orm as symbols_repo
    
    holding_days_options = [5, 10, 20, 40, 60]
    
    symbols = await symbols_repo.list_symbols()
    
    if not symbols:
        return SignalScanResponse(
            scanned_at=datetime.now(),
            holding_days_tested=holding_days_options,
            stocks=[],
            top_opportunities=[],
            n_active_signals=0,
        )
    
    # Build symbol list (exclude benchmarks)
    symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]
    symbol_names = {s.symbol: s.name or s.symbol for s in symbols}
    
    # Fetch price data
    prices_df = await _fetch_prices_for_symbols(symbol_list, lookback_days=1260)
    
    if prices_df.empty:
        return SignalScanResponse(
            scanned_at=datetime.now(),
            holding_days_tested=holding_days_options,
            stocks=[],
            top_opportunities=[],
            n_active_signals=0,
        )
    
    # Convert DataFrame to dict of Series
    price_data = {col: prices_df[col].dropna() for col in prices_df.columns}
    
    # Run signal scanner with optimization
    opportunities = scan_all_stocks(price_data, symbol_names, holding_days_options)
    
    # Convert to response
    stocks = []
    total_active_signals = 0
    
    for opp in opportunities:
        signal_responses = [
            SignalResultResponse(
                name=sig.name,
                description=sig.description,
                value=sig.current_value,
                is_buy_signal=sig.is_buy_signal,
                strength=sig.signal_strength,
                optimal_threshold=sig.optimal_threshold,
                optimal_holding_days=sig.optimal_holding_days,
                win_rate=sig.win_rate,
                avg_return_pct=sig.avg_return_pct,
                max_return_pct=sig.max_return_pct,
                min_return_pct=sig.min_return_pct,
                n_signals=sig.n_signals,
                improvement_pct=sig.improvement_pct,
            )
            for sig in opp.signals
        ]
        
        active_signal_responses = [
            SignalResultResponse(
                name=sig.name,
                description=sig.description,
                value=sig.current_value,
                is_buy_signal=sig.is_buy_signal,
                strength=sig.signal_strength,
                optimal_threshold=sig.optimal_threshold,
                optimal_holding_days=sig.optimal_holding_days,
                win_rate=sig.win_rate,
                avg_return_pct=sig.avg_return_pct,
                max_return_pct=sig.max_return_pct,
                min_return_pct=sig.min_return_pct,
                n_signals=sig.n_signals,
                improvement_pct=sig.improvement_pct,
            )
            for sig in opp.active_signals
        ]
        
        total_active_signals += len(opp.active_signals)
        
        stocks.append(StockSignalResponse(
            symbol=opp.symbol,
            name=opp.name,
            buy_score=opp.buy_score,
            opportunity_type=opp.opportunity_type,
            opportunity_reason=opp.opportunity_reason,
            current_price=opp.current_price,
            price_vs_52w_high_pct=opp.price_vs_52w_high_pct,
            price_vs_52w_low_pct=opp.price_vs_52w_low_pct,
            zscore_20d=opp.zscore_20d,
            zscore_60d=opp.zscore_60d,
            rsi_14=opp.rsi_14,
            best_signal_name=opp.best_signal_name,
            best_holding_days=opp.best_holding_days,
            best_expected_return=opp.best_expected_return,
            signals=signal_responses,
            active_buy_signals=active_signal_responses,
        ))
    
    top_opportunities = [s.symbol for s in stocks[:3] if s.buy_score > 0]
    
    return SignalScanResponse(
        scanned_at=datetime.now(),
        holding_days_tested=holding_days_options,
        stocks=stocks,
        top_opportunities=top_opportunities,
        n_active_signals=total_active_signals,
    )


# ============================================================================
# Global Analytics Endpoint (No Auth - for Dashboard)
# ============================================================================


@global_router.get(
    "/market-analysis",
    summary="Global Market Analysis",
    description="""
    Get market-wide risk analysis for all tracked symbols.
    
    Provides:
    - Current market regime (bull/bear, high/low volatility)
    - Average correlation across assets
    - Diversification opportunities
    - Sector concentration warnings
    """,
)
async def get_market_analysis() -> dict[str, Any]:
    """Get global market analysis."""
    from app.repositories import symbols_orm as symbols_repo
    from app.quant_engine.analytics import detect_regime, compute_correlation_analysis
    
    symbols = await symbols_repo.list_symbols()
    
    if not symbols:
        return {
            "analyzed_at": datetime.now().isoformat(),
            "status": "no_data",
            "message": "No symbols tracked",
        }
    
    symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]
    
    prices = await _fetch_prices_for_symbols(symbol_list, lookback_days=400)
    
    if prices.empty or len(prices) < 60:
        return {
            "analyzed_at": datetime.now().isoformat(),
            "status": "insufficient_data",
            "message": "Insufficient price history",
        }
    
    returns = _compute_returns(prices)
    
    # Detect regime
    regime = detect_regime(returns)
    
    # Compute correlation analysis
    corr_analysis = compute_correlation_analysis(returns)
    
    return {
        "analyzed_at": datetime.now().isoformat(),
        "n_symbols": len(returns.columns),
        "regime": {
            "current": regime.regime,
            "trend": regime.trend,
            "volatility": regime.volatility,
            "description": regime.description,
            "recommendation": regime.risk_budget_recommendation,
        },
        "correlations": {
            "average": _safe_float(corr_analysis.average_correlation),
            "n_clusters": corr_analysis.n_clusters,
            "clusters": corr_analysis.clusters,
            "stress_correlation": _safe_float(corr_analysis.stress_correlation),
        },
        "insights": [
            f"Market is in a {regime.regime} regime",
            f"Average correlation: {corr_analysis.average_correlation:.0%}",
            f"Found {corr_analysis.n_clusters} correlation clusters",
        ],
    }
