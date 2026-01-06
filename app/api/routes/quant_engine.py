"""
Quantitative Portfolio Engine V2 API Routes.

Risk-based portfolio optimization - NO return forecasting.

Provides endpoints for:
- Portfolio risk analytics (risk decomposition, tail risk, diversification)
- Risk-based allocation recommendations (Risk Parity, CVaR, HRP, etc.)
- Technical signal scanning with per-stock optimization
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.api.dependencies import require_user
from app.cache.cache import Cache
from app.core.data_helpers import safe_float
from app.core.exceptions import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.quant_engine import (
    analyze_portfolio,
    scan_all_stocks,
    translate_for_user,
    perform_domain_analysis,
    domain_analysis_to_dict,
)
from app.quant_engine.risk.highlights import build_portfolio_risk_highlights
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

# Short-lived cache for price data (5 min) to avoid re-fetching across related calls
_price_cache = Cache(prefix="stonkmarket:v1:quant_prices")


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
    lookback_days: int = 1260,  # 5 years default
    use_max_history: bool = False,  # NEW: fetch ALL available data
) -> pd.DataFrame:
    """
    Fetch price history for multiple symbols in parallel.

    Returns a DataFrame with dates as index and symbols as columns.
    Falls back to yfinance if data not in database.
    Uses short-lived cache to avoid re-fetching across related calls.
    
    Args:
        symbols: List of stock tickers
        lookback_days: Number of days to look back (default 5 years)
        use_max_history: If True, fetch ALL available history (overrides lookback_days)
    """
    from app.services.data_providers.yfinance_service import get_yfinance_service
    import hashlib
    import yfinance as yf
    
    end_date = date.today()
    
    # For max history, use 30 years as a reasonable maximum
    if use_max_history:
        lookback_days = 7500  # ~30 years
    
    start_date = end_date - timedelta(days=lookback_days)
    
    # Check cache first - key based on symbols + date range
    symbols_sorted = sorted(symbols)
    cache_suffix = "max" if use_max_history else str(lookback_days)
    cache_key = f"{hashlib.md5(','.join(symbols_sorted).encode()).hexdigest()}:{cache_suffix}:{end_date}"
    cached = await _price_cache.get(cache_key)
    if cached is not None:
        logger.debug(f"Price cache hit for {len(symbols)} symbols")
        # Reconstruct DataFrame from cached data
        df = pd.DataFrame(
            data=cached["data"],
            index=pd.to_datetime(cached["index"]),
            columns=cached["columns"],
        )
        return df

    # For max history, use yfinance period="max" directly
    if use_max_history:
        logger.info(f"Fetching MAX history for {len(symbols)} symbols")
        price_dfs = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="max")
                if not df.empty and "Close" in df.columns:
                    price_dfs[symbol] = df["Close"]
                    logger.debug(f"Fetched {len(df)} days of history for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch max history for {symbol}: {e}")
        
        if not price_dfs:
            return pd.DataFrame()
        
        # Combine into single DataFrame
        prices = pd.DataFrame(price_dfs)
        prices = prices.dropna(how="all")
        prices = prices.ffill()
        
        # Cache for 5 minutes
        cache_data = {
            "index": [str(d) for d in prices.index.tolist()],
            "columns": prices.columns.tolist(),
            "data": prices.values.tolist(),
        }
        await _price_cache.set(cache_key, cache_data, ttl=300)
        
        return prices

    # Fetch all symbols from DB in parallel
    async def fetch_symbol(symbol: str) -> tuple[str, pd.Series | None]:
        df = await price_history_repo.get_prices_as_dataframe(
            symbol, start_date, end_date
        )
        if df is not None and "Close" in df.columns and len(df) >= 20:
            return symbol, df["Close"]
        return symbol, None
    
    results = await asyncio.gather(*[fetch_symbol(s) for s in symbols])
    
    price_dfs = {}
    missing_symbols = []
    for symbol, series in results:
        if series is not None:
            price_dfs[symbol] = series
        else:
            missing_symbols.append(symbol)
    
    # Fetch missing symbols from yfinance (common for sector ETFs)
    if missing_symbols:
        from app.services.prices import get_price_service
        price_service = get_price_service()
        yf_results = await price_service.get_prices_batch(
            missing_symbols, start_date, end_date
        )
        for symbol, df in yf_results.items():
            if df is not None and not df.empty and "Close" in df.columns:
                price_dfs[symbol] = df["Close"]

    if not price_dfs:
        return pd.DataFrame()

    # Combine into single DataFrame
    prices = pd.DataFrame(price_dfs)
    prices = prices.dropna(how="all")
    prices = prices.ffill()
    
    # Cache for 5 minutes - covers typical page load scenario
    # Convert index to string for JSON serialization
    cache_data = {
        "index": [str(d) for d in prices.index.tolist()],
        "columns": prices.columns.tolist(),
        "data": prices.values.tolist(),
    }
    await _price_cache.set(cache_key, cache_data, ttl=300)

    return prices


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from prices."""
    return prices.pct_change().dropna()


async def _get_symbol_prices(symbol: str, days: int = 365) -> pd.DataFrame | None:
    """Fetch price history for a single symbol as DataFrame with Close column."""
    from app.services.prices import get_price_service
    
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # Try database first
    df = await price_history_repo.get_prices_as_dataframe(symbol, start_date, end_date)
    if df is not None and "Close" in df.columns and len(df) >= 20:
        return df
    
    # Fallback to PriceService (DB first, yfinance fallback)
    price_service = get_price_service()
    df = await price_service.get_prices(symbol, start_date, end_date)
    if df is not None and not df.empty and "Close" in df.columns:
        return df
    
    return None


async def _get_symbol_max_history(symbol: str) -> pd.DataFrame | None:
    """
    Fetch ALL available price history for a single symbol.
    
    Uses yfinance period="max" to get maximum available history.
    Useful for long-term backtesting and crash testing.
    
    Returns:
        DataFrame with OHLCV data, or None if fetch failed
    """
    import yfinance as yf
    
    logger.info(f"Fetching max history for {symbol}")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")
        
        if df.empty:
            logger.warning(f"No history found for {symbol}")
            return None
        
        logger.info(f"Fetched {len(df)} days of history for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching max history for {symbol}: {e}")
        return None



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

    risk_highlights: list[dict[str, Any]] = []
    try:
        risk_highlights = await build_portfolio_risk_highlights(symbols)
    except Exception as exc:
        logger.debug(f"Risk highlights failed for portfolio {portfolio_id}: {exc}")

    return {
        "portfolio_id": portfolio_id,
        "analyzed_at": datetime.now().isoformat(),
        "total_value_eur": safe_float(total_value, 0.0),
        "n_positions": analytics.n_positions,
        "risk_score": analytics.overall_risk_score,
        "summary": user_friendly["summary"],
        "risk": user_friendly["risk"],
        "diversification": user_friendly["diversification"],
        "market": user_friendly["market"],
        "insights": user_friendly["insights"],
        "action_items": user_friendly["action_items"],
        "risk_highlights": risk_highlights,
        # Raw analytics for power users
        "raw": {
            "portfolio_volatility": safe_float(analytics.risk_decomposition.portfolio_volatility, 0.0),
            "var_95": safe_float(analytics.tail_risk.var_95_daily, 0.0),
            "cvar_95": safe_float(analytics.tail_risk.cvar_95_daily, 0.0),
            "max_drawdown": safe_float(analytics.tail_risk.max_drawdown, 0.0),
            "effective_n": safe_float(analytics.diversification.effective_n, 0.0),
            "diversification_ratio": safe_float(analytics.diversification.diversification_ratio, 0.0),
            "regime": analytics.regime.regime,
            "risk_contributions": {
                s: safe_float(v, 0.0) 
                for s, v in analytics.risk_decomposition.risk_contribution_pct.items()
            },
        },
    }


# =============================================================================
# Allocation Recommendation Endpoint (Using skfolio)
# =============================================================================


@router.post(
    "/{portfolio_id}/allocate",
    status_code=status.HTTP_200_OK,
    summary="Get Allocation Recommendation",
    description="""
    Get a risk-based allocation recommendation for new investment using skfolio.
    
    This answers: "Where should my next €X go?"
    
    Uses professional portfolio optimization with skfolio:
    - RISK_PARITY: Equal risk contribution from each position
    - MIN_CVAR: Minimize expected loss in worst scenarios
    - MAX_DIVERSIFICATION: Maximize diversification ratio
    - EQUAL_WEIGHT: Simple 1/N allocation
    
    Features:
    - Walk-forward cross-validation to select best method
    - Pre-selection to remove highly correlated assets
    - EUR-based trade recommendations
    """,
)
async def get_allocation_recommendation(
    portfolio_id: int,
    inflow_eur: float = Query(1000.0, ge=50, le=100000, description="Amount to invest"),
    method: str = Query(
        "",
        description="Optimization method: risk_parity, min_cvar, max_diversification, equal_weight. Leave empty to auto-select best."
    ),
    user: TokenData = Depends(require_user),
) -> dict[str, Any]:
    """Get allocation recommendation using skfolio optimization."""
    from app.portfolio.service import PortfolioContext, run_skfolio
    
    user_id = await _get_user_id(user)

    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    symbols = [h["symbol"] for h in holdings]

    # Fetch price history in parallel
    async def fetch_with_symbol(symbol: str) -> tuple[str, pd.DataFrame | None]:
        df = await _get_symbol_prices(symbol, days=365)
        return symbol, df
    
    price_results = await asyncio.gather(*[fetch_with_symbol(s) for s in symbols])
    prices_by_symbol: dict[str, pd.DataFrame] = {
        symbol: df for symbol, df in price_results if df is not None and not df.empty
    }
    
    if len(prices_by_symbol) < 2:
        raise ValidationError(message="Need at least 2 assets with price history")

    # Build combined prices DataFrame for returns calculation
    combined = pd.DataFrame({s: df["Close"] for s, df in prices_by_symbol.items()}).dropna()
    if combined.empty or len(combined) < 60:
        raise ValidationError(message="Insufficient price history (need 60+ days)")
    
    returns = combined.pct_change().dropna()

    # Calculate portfolio values for context
    current_prices = combined.iloc[-1]
    portfolio_values_dict = {}
    total_portfolio_value = 0.0
    for h in holdings:
        symbol = h["symbol"]
        qty = float(h.get("quantity", 0))
        if symbol in current_prices.index:
            val = qty * current_prices[symbol]
            portfolio_values_dict[symbol] = val
            total_portfolio_value += val
    
    # Build portfolio values series (daily total value)
    portfolio_values = (combined * pd.Series({
        h["symbol"]: float(h.get("quantity", 0)) for h in holdings
    })).sum(axis=1)

    # Build PortfolioContext
    context = PortfolioContext(
        portfolio_id=portfolio_id,
        portfolio=portfolio,
        holdings=holdings,
        prices_by_symbol=prices_by_symbol,
        portfolio_values=portfolio_values,
        returns=returns.mean(axis=1),  # Portfolio returns (simplified)
        benchmark_returns=None,
    )

    # Run skfolio optimization
    # 'auto' or empty = cross-validate and pick best method
    opt_method = None if method in ("", "auto") else method
    result = run_skfolio(
        context=context,
        method=opt_method,
        inflow_amount=inflow_eur,
    )

    if result.get("status") == "error":
        raise ValidationError(message=result.get("warnings", ["Optimization failed"])[0])

    data = result.get("data", {})
    
    # Method explanations
    method_explanations = {
        "min_cvar": "Minimizes expected loss in worst 5% of scenarios (CVaR)",
        "risk_parity": "Allocates so each position contributes equally to risk",
        "max_diversification": "Maximizes the diversification ratio",
        "equal_weight": "Simple 1/N equal weight across all assets",
        "inverse_volatility": "Weights inversely proportional to volatility (fallback)",
    }
    best_method = data.get("best_method", "risk_parity")
    
    # Volatility labels
    def vol_label(vol: float) -> str:
        if vol < 0.10:
            return "Low"
        elif vol < 0.20:
            return "Moderate"
        elif vol < 0.30:
            return "High"
        return "Very High"
    
    current_vol = data.get("current_volatility", 0)
    target_vol = data.get("expected_volatility", 0)

    return {
        "portfolio_id": portfolio_id,
        "method": best_method,
        "inflow_eur": inflow_eur,
        "portfolio_value_eur": safe_float(data.get("portfolio_value", 0), 0.0),
        "confidence": data.get("confidence", "MEDIUM"),
        "explanation": method_explanations.get(best_method, "Risk-based optimization"),
        "risk_improvement": data.get("risk_improvement", ""),
        "current_risk": {
            "volatility": safe_float(current_vol, 0.0),
            "label": vol_label(current_vol),
        },
        "optimal_risk": {
            "volatility": safe_float(target_vol, 0.0),
            "label": vol_label(target_vol),
            "diversification_ratio": safe_float(data.get("sharpe", 0) + 1, 0.0),  # Approximation
        },
        "trades": data.get("trades", []),
        "current_weights": {s: safe_float(v * 100, 0.0) for s, v in data.get("current_weights", {}).items()},
        "target_weights": {s: safe_float(v * 100, 0.0) for s, v in data.get("weights", {}).items()},
        "cv_results": data.get("cv_results", {}),
        "warnings": result.get("warnings", []),
    }


# =============================================================================
# Global Recommendations Router (No Auth Required)
# =============================================================================

global_router = APIRouter(prefix="/recommendations", tags=["Quant Engine"])


# ============================================================================
# Main Recommendations Endpoint (Global - for Landing/Dashboard)
# ============================================================================

from app.schemas.quant_engine import (
    QuantAuditBlock,
    QuantEngineResponse,
    QuantRecommendation,
)


@global_router.get(
    "",
    response_model=QuantEngineResponse,
    summary="Get stock recommendations",
    description="""
    Get ranked stock recommendations combining signal analysis with dip metrics.
    
    This is the main endpoint for the landing page and dashboard.
    Returns stocks ranked by buy opportunity, combining:
    - Technical signal analysis (RSI, MACD, Bollinger, etc.)
    - Dip metrics (percentage from ATH, days in dip)
    - AI analysis snippets (if available)
    
    Each stock includes:
    - action: BUY, SELL, or HOLD recommendation
    - mu_hat: Expected return estimate
    - dip_score: Z-score based dip signal
    - marginal_utility: Ranking score
    - AI summary/rating when available
    """,
)
async def get_recommendations(
    inflow_eur: float = Query(
        1000.0,
        ge=100,
        le=1000000,
        description="Notional investment amount in EUR"
    ),
    limit: int = Query(
        40,
        ge=1,
        le=100,
        description="Maximum recommendations to return"
    ),
) -> QuantEngineResponse:
    """
    Get ranked stock recommendations for landing page and dashboard.
    
    PERFORMANCE OPTIMIZED: Uses pre-computed data from background jobs.
    No expensive signal optimization is done per-request.
    """
    import asyncio
    from datetime import date
    
    from app.repositories import dip_state_orm as dip_state_repo
    from app.repositories import dip_votes_orm as dip_votes_repo
    from app.repositories import symbols_orm as symbols_repo
    from app.repositories import dipfinder_orm as dipfinder_repo
    from sqlalchemy import select
    from app.database.connection import get_session
    from app.database.orm import StrategySignal, QuantScore
    from app.cache.cache import Cache
    from app.schemas.quant_engine import EvidenceBlockResponse
    
    # Check cache first (5 minute TTL)
    cache = Cache(prefix="recommendations", default_ttl=300)
    cache_key = f"recs_{inflow_eur}_{limit}"
    cached = await cache.get(cache_key)
    if cached:
        # Reconstruct response from cached dict
        return QuantEngineResponse(**cached)
    
    # Parallel fetch base data sources
    symbols_task = symbols_repo.list_symbols()
    dip_states_task = dip_state_repo.get_all_dip_states()
    ai_analyses_task = dip_votes_repo.get_all_ai_analyses()
    
    symbols, dip_states, ai_analyses = await asyncio.gather(
        symbols_task, dip_states_task, ai_analyses_task
    )
    
    if not symbols:
        return QuantEngineResponse(
            recommendations=[],
            as_of_date=date.today().isoformat(),
            portfolio_value_eur=0.0,
            inflow_eur=inflow_eur,
            total_trades=0,
            total_transaction_cost_eur=0.0,
            expected_portfolio_return=0.0,
            expected_portfolio_risk=0.0,
            audit=QuantAuditBlock(
                timestamp=datetime.now().isoformat(),
                config_hash=0,
                optimizer_status="no_data",
                regime_state="unknown",
            ),
        )
    
    # Build lookup maps
    dip_state_map = {d.symbol: d for d in dip_states}
    ai_map = {a.symbol: a for a in ai_analyses}
    
    # Get pre-computed strategy signals and quant scores from database
    strategy_map: dict[str, Any] = {}
    quant_score_map: dict[str, Any] = {}
    try:
        async with get_session() as session:
            result = await session.execute(select(StrategySignal))
            strategy_signals = result.scalars().all()
            for sig in strategy_signals:
                strategy_map[sig.symbol] = sig
            
            # Load quant scores (APUS + DOUS)
            result = await session.execute(select(QuantScore))
            quant_scores = result.scalars().all()
            for qs in quant_scores:
                quant_score_map[qs.symbol] = qs
    except Exception as e:
        logger.warning(f"Failed to fetch strategy signals or quant scores: {e}")
    
    # Build recommendations from pre-computed data
    symbol_names = {s.symbol: s.name or s.symbol for s in symbols}
    symbol_sector_map = {s.symbol: s.sector for s in symbols}
    symbol_mcap_map = {s.symbol: s.market_cap for s in symbols}
    symbol_list = [s.symbol for s in symbols if s.symbol not in ("SPY", "^GSPC", "URTH")]

    dipfinder_signals = await dipfinder_repo.get_latest_signals_for_tickers(symbol_list)
    dipfinder_map = {}
    for sig in (dipfinder_signals or []):
        if sig.ticker not in dipfinder_map:
            dipfinder_map[sig.ticker] = sig
    
    recommendations: list[QuantRecommendation] = []
    
    for symbol in symbol_list:
        dip = dip_state_map.get(symbol)
        ai = ai_map.get(symbol)
        dipfinder = dipfinder_map.get(symbol)
        strategy = strategy_map.get(symbol)
        quant_score = quant_score_map.get(symbol)
        name = symbol_names.get(symbol, symbol)
        sector = symbol_sector_map.get(symbol)
        market_cap = symbol_mcap_map.get(symbol)
        
        # Calculate metrics from dip state (already in DB)
        dip_pct = float(dip.dip_percentage) if dip and dip.dip_percentage is not None else None
        current_price = float(dip.current_price) if dip and dip.current_price else 0
        opportunity_type = dip.opportunity_type if dip and dip.opportunity_type else "NONE"
        # Extreme Value Analysis (EVA) fields
        is_tail_event = dip.is_tail_event if dip else False
        return_period_years = float(dip.return_period_years) if dip and dip.return_period_years else None
        regime_dip_percentile = float(dip.regime_dip_percentile) if dip and dip.regime_dip_percentile else None
        days_in_dip = None
        if dip and dip.dip_start_date:
            days_in_dip = (date.today() - dip.dip_start_date).days
        
        # Note: DipState doesn't have change_1d, would need to compute from price history
        change_percent = None
        
        # Extract strategy signal data (pre-computed, NOT computed per-request)
        strategy_beats_bh = False
        strategy_beats_spy = False
        strategy_signal_type = None
        strategy_win_rate_val = None
        strategy_vs_bh_pct = None
        strategy_total_return_pct = None
        expected_recovery_days = None
        win_rate = None
        strategy_name = None
        strategy_recent_trades = None
        strategy_comparison = None
        
        strategy_n_trades = None
        if strategy:
            try:
                # StrategySignal has flat columns, not nested dicts
                strategy_signal_type = strategy.signal_type
                strategy_vs_bh_pct = float(strategy.vs_buy_hold_pct) if strategy.vs_buy_hold_pct else None
                strategy_total_return_pct = float(strategy.total_return_pct) if strategy.total_return_pct else None
                strategy_beats_bh = strategy.beats_buy_hold
                strategy_beats_spy = getattr(strategy, 'beats_spy', False) or False
                strategy_win_rate_val = float(strategy.win_rate) if strategy.win_rate else None
                win_rate = float(strategy.win_rate) / 100 if strategy.win_rate else None  # Convert from pct
                strategy_n_trades = strategy.n_trades if strategy.n_trades else 0
                expected_recovery_days = getattr(strategy, 'typical_recovery_days', None)
                strategy_name = getattr(strategy, 'strategy_name', None)
                strategy_recent_trades = getattr(strategy, 'recent_trades', None)
                strategy_comparison = getattr(strategy, 'strategy_comparison', None)
            except Exception:
                pass
        
        # Get typical dip from dipfinder (pre-computed)
        typical_dip_pct = None
        dip_vs_typical = None
        dip_score = None
        is_unusual_dip = False
        
        if dipfinder:
            if dipfinder.dip_vs_typical is not None:
                dip_vs_typical = float(dipfinder.dip_vs_typical)
                is_unusual_dip = dip_vs_typical >= 1.5
            if dipfinder.dip_score is not None:
                dip_score = float(dipfinder.dip_score)
            if dipfinder.stability_factors:
                stability = dipfinder.stability_factors
                if isinstance(stability, str):
                    import json
                    try:
                        stability = json.loads(stability)
                    except (json.JSONDecodeError, TypeError):
                        stability = {}
                if isinstance(stability, dict) and stability.get("typical_dip_365"):
                    typical_dip_pct = float(stability["typical_dip_365"]) * 100
        
        # Determine action - MUST be backed by proven math (strategy signals)
        # NO recommendations based on "dip + AI" alone anymore
        action = "HOLD"
        buy_score = 0.0
        
        # Minimum trades for statistical validity
        MIN_TRADES = 5
        has_enough_trades = strategy_n_trades is not None and strategy_n_trades >= MIN_TRADES
        
        # QUALITY GATE: Only BUY if strategy beats BOTH buy-and-hold AND SPY benchmark
        # AND has enough trades for statistical validity
        # Must beat SPY because otherwise just buy SPY index fund
        beats_benchmarks = strategy_beats_bh and strategy_beats_spy
        if beats_benchmarks and strategy_signal_type == "BUY" and has_enough_trades:
            action = "BUY"
            buy_score = 80.0
        # Note: We no longer give BUY based on dip_pct or win_rate alone - that's not math-backed
        
        # Simple expected return estimate - ONLY from proven strategies
        mu_hat = 0.0
        if beats_benchmarks and strategy_vs_bh_pct:
            mu_hat = strategy_vs_bh_pct / 100
        # Note: No longer estimating returns from dip_pct alone - that's not proven
        
        # AI analysis (informational only, not for ranking)
        ai_summary = ai.rating_reasoning if ai else None
        ai_rating = ai.ai_rating if ai else None
        
        # Calculate best chance score - MATH-BACKED ONLY
        # Start at 0, only add points for proven factors
        best_chance_score = 0.0
        best_chance_reasons = []
        
        # Primary factor: Strategy beats BOTH buy-and-hold AND SPY (proven)
        if beats_benchmarks:
            best_chance_score += 40
            best_chance_reasons.append("Beats B&H + SPY")
            if strategy_vs_bh_pct and strategy_vs_bh_pct > 10:
                best_chance_score += min(strategy_vs_bh_pct / 2, 20)
                best_chance_reasons.append(f"+{strategy_vs_bh_pct:.0f}% vs B&H")
        
        # Secondary factor: Win rate > 55% (proven)
        if strategy_win_rate_val and strategy_win_rate_val >= 55:
            best_chance_score += 15
            best_chance_reasons.append(f"{strategy_win_rate_val:.0f}% win rate")
        
        # Tertiary factor: Unusual dip (math-computed)
        if is_unusual_dip and dip_vs_typical and dip_vs_typical >= 1.5:
            best_chance_score += 15
            best_chance_reasons.append(f"{dip_vs_typical:.1f}x typical dip")
        elif dip_pct and dip_pct > 20:
            # Significant dip but not compared to typical - small boost
            best_chance_score += 5
            best_chance_reasons.append(f"{dip_pct:.0f}% dip")
        
        # AI is informational only - minor boost for confirmation
        if ai_rating in ("strong_buy", "buy") and best_chance_score > 30:
            best_chance_score += 5
            best_chance_reasons.append(f"AI: {ai_rating}")
        
        # NEW: Override best_chance_score with quant_score if available
        # The quant_score is computed using stationary bootstrap and deflated Sharpe
        quant_mode = None
        quant_score_a = None
        quant_score_b = None
        quant_gate_pass = False
        evidence_response = None
        
        if quant_score:
            # Use the pre-computed quant score as the primary ranking metric
            best_chance_score = float(quant_score.best_score)
            quant_mode = quant_score.mode
            quant_score_a = float(quant_score.score_a) if quant_score.score_a else None
            quant_score_b = float(quant_score.score_b) if quant_score.score_b else None
            quant_gate_pass = quant_score.gate_pass
            
            # Build evidence block from stored data
            if quant_score.evidence:
                evidence_response = EvidenceBlockResponse(**quant_score.evidence)
            else:
                # Build from individual columns if JSONB is missing
                evidence_response = EvidenceBlockResponse(
                    p_outperf=float(quant_score.p_outperf or 0),
                    ci_low=float(quant_score.ci_low or 0),
                    ci_high=float(quant_score.ci_high or 0),
                    dsr=float(quant_score.dsr or 0),
                    median_edge=float(quant_score.median_edge or 0),
                    edge_vs_stock=float(quant_score.edge_vs_stock or 0),
                    edge_vs_spy=float(quant_score.edge_vs_spy or 0),
                    worst_regime_edge=float(quant_score.worst_regime_edge or 0),
                    cvar_5=float(quant_score.cvar_5 or 0),
                    fund_mom=float(quant_score.fund_mom or 0),
                    val_z=float(quant_score.val_z or 0),
                    event_risk=quant_score.event_risk or False,
                    p_recovery=float(quant_score.p_recovery or 0),
                    expected_value=float(quant_score.expected_value or 0),
                    sector_relative=float(quant_score.sector_relative or 0),
                )
            
            # Update best_chance_reasons with quant mode
            if quant_gate_pass:
                best_chance_reasons = [f"Mode A: {best_chance_score:.0f}"]
                if evidence_response and evidence_response.p_outperf >= 0.75:
                    best_chance_reasons.append(f"P(edge>0)={evidence_response.p_outperf:.0%}")
            else:
                best_chance_reasons = [f"Mode B: {best_chance_score:.0f}"]
                if evidence_response and evidence_response.p_recovery > 0.5:
                    best_chance_reasons.append(f"P(rec)={evidence_response.p_recovery:.0%}")
            
            # Upgrade action based on quant gate
            if quant_gate_pass and quant_score.best_score >= 70:
                action = "BUY"
                buy_score = float(quant_score.best_score)
        
        best_chance_score = max(0, min(100, best_chance_score))
        
        # Opportunity rating - must be consistent with action
        # If action is HOLD, opportunity_rating should not be "buy" or "strong_buy"
        # This prevents confusing users with "Strong Buy" rating but "HOLD" action
        if action == "HOLD":
            # For HOLD action, cap rating at "hold" regardless of score
            # The score still shows opportunity quality for informational purposes
            opportunity_rating = "hold"
        elif best_chance_score >= 75:
            opportunity_rating = "strong_buy"
        elif best_chance_score >= 60:
            opportunity_rating = "buy"
        elif best_chance_score >= 40:
            opportunity_rating = "hold"
        else:
            opportunity_rating = "avoid"
        
        recommendations.append(QuantRecommendation(
            ticker=symbol,
            name=name,
            action=action,
            notional_eur=inflow_eur / len(symbol_list) if action == "BUY" else 0,
            delta_weight=0.01 if action == "BUY" else 0,
            target_weight=0.05 if action == "BUY" else 0,
            last_price=current_price if current_price else None,
            change_percent=change_percent,
            market_cap=float(market_cap) if market_cap else None,
            sector=sector,
            mu_hat=mu_hat,
            uncertainty=1.0 - (buy_score / 100),
            risk_contribution=0.0,
            opportunity_score=best_chance_score,
            opportunity_rating=opportunity_rating,
            expected_recovery_days=expected_recovery_days,
            typical_dip_pct=typical_dip_pct,
            dip_vs_typical=dip_vs_typical,
            is_unusual_dip=is_unusual_dip,
            opportunity_type=opportunity_type,
            is_tail_event=is_tail_event,
            return_period_years=return_period_years,
            regime_dip_percentile=regime_dip_percentile,
            win_rate=win_rate,
            dip_score=dip_score,
            dip_bucket=_dip_bucket(dip_pct) if dip_pct else None,
            marginal_utility=buy_score / 100,
            legacy_dip_pct=dip_pct,
            legacy_days_in_dip=days_in_dip,
            ai_summary=ai_summary,
            ai_rating=ai_rating,
            strategy_beats_bh=strategy_beats_bh,
            strategy_signal=strategy_signal_type,
            strategy_win_rate=strategy_win_rate_val,
            strategy_vs_bh_pct=strategy_vs_bh_pct,
            strategy_total_return_pct=strategy_total_return_pct,
            strategy_name=strategy_name,
            strategy_recent_trades=strategy_recent_trades,
            strategy_comparison=strategy_comparison,
            best_chance_score=best_chance_score,
            best_chance_reason=" • ".join(best_chance_reasons[:3]) if best_chance_reasons else None,
            # APUS + DOUS Dual-Mode Scoring
            quant_mode=quant_mode,
            quant_score_a=quant_score_a,
            quant_score_b=quant_score_b,
            quant_gate_pass=quant_gate_pass,
            quant_evidence=evidence_response,
        ))
    
    # Sort by best_chance_score
    recommendations.sort(key=lambda r: r.best_chance_score, reverse=True)
    recommendations = recommendations[:limit]
    
    buy_count = sum(1 for r in recommendations if r.action == "BUY")
    high_conviction_count = sum(1 for r in recommendations if r.best_chance_score >= 60)
    avg_mu_hat = sum(r.mu_hat for r in recommendations) / len(recommendations) if recommendations else 0
    
    # Generate market message based on opportunity quality
    certified_count = sum(1 for r in recommendations if r.quant_gate_pass)
    top_score = recommendations[0].best_chance_score if recommendations else 0
    
    # Simplified market message - only show when nothing actionable
    market_message = None
    if certified_count > 0:
        if certified_count >= 3:
            market_message = None  # Plenty of opportunities, no message needed
        else:
            market_message = f"{certified_count} certified opportunity(ies) meet all quality criteria."
    elif top_score > 0:
        market_message = f"No actionable opportunities today. Top score: {top_score:.0f}/100."
    # If all stocks are HOLD or DIP_ENTRY, show nothing - the cards are self-explanatory
    
    # Legacy fallback for when quant_scores are not yet computed
    if not quant_score_map and buy_count == 0:
        market_message = "No BUY signals today. Quant scoring job has not run yet."
    
    response = QuantEngineResponse(
        recommendations=recommendations,
        as_of_date=date.today().isoformat(),
        portfolio_value_eur=10000.0,
        inflow_eur=inflow_eur,
        total_trades=buy_count,
        total_transaction_cost_eur=buy_count * 5.0,
        expected_portfolio_return=avg_mu_hat,
        expected_portfolio_risk=0.15,
        audit=QuantAuditBlock(
            timestamp=datetime.now().isoformat(),
            config_hash=hash(f"{inflow_eur}-{limit}"),
            optimizer_status="success",
            regime_state="unknown",
        ),
        market_message=market_message,
    )
    
    # Cache the response
    await cache.set(cache_key, response.model_dump())
    
    return response


# ============================================================================
# Hero Stock Endpoint (Rotating featured stock for landing page)
# ============================================================================


class HeroStockResponse(BaseModel):
    """Hero stock for landing page display."""
    symbol: str
    company_name: str | None = None
    action: str
    mu_hat: float
    dip_pct: float | None = None
    days_in_dip: int | None = None
    best_chance_score: float = 0.0
    best_chance_reason: str | None = None
    ai_summary: str | None = None
    ai_rating: str | None = None
    cache_expires_at: str  # ISO timestamp when this hero expires


# Cache for hero stock rotation
_hero_cache = Cache(prefix="hero_stock", default_ttl=900)  # 15 minutes


@global_router.get(
    "/hero",
    response_model=HeroStockResponse,
    summary="Get rotating hero stock",
    description="""
    Get the current featured stock for landing page hero section.
    
    This endpoint returns a rotating stock from the top recommendations,
    cached for 15 minutes. Each rotation picks a different stock from
    the top 5 BUY recommendations to provide variety.
    """,
)
async def get_hero_stock() -> HeroStockResponse:
    """Get rotating hero stock for landing page."""
    import hashlib
    from datetime import datetime, timezone
    
    # Check cache first
    cached = await _hero_cache.get("current")
    if cached:
        return HeroStockResponse(**cached)
    
    # Get recommendations (uses its own cache)
    recommendations_response = await get_recommendations(inflow_eur=1000.0, limit=20)
    recommendations = recommendations_response.recommendations
    
    # Filter to BUY recommendations with best_chance_score > 50
    buy_recs = [
        r for r in recommendations 
        if r.action == "BUY" and r.best_chance_score >= 50
    ]
    
    if not buy_recs:
        # Fallback to any BUY, or first recommendation
        buy_recs = [r for r in recommendations if r.action == "BUY"]
        if not buy_recs:
            buy_recs = recommendations[:5] if recommendations else []
    
    if not buy_recs:
        # No recommendations at all - return a sensible error state
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No recommendations available for hero display"
        )
    
    # Rotate among top 5: Use time-based hash to pick a consistent one per 15-min window
    now = datetime.now(timezone.utc)
    window_id = int(now.timestamp()) // 900  # 15-minute windows
    hash_input = f"hero-{window_id}"
    index = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % min(len(buy_recs), 5)
    
    hero_rec = buy_recs[index]
    
    # Calculate when this hero expires (next 15-minute boundary)
    next_window = (window_id + 1) * 900
    expires_at = datetime.fromtimestamp(next_window, tz=timezone.utc)
    
    # Get company name from stock info
    from app.repositories import symbols_orm as symbols_repo
    symbol_data = await symbols_repo.get_symbol(hero_rec.ticker)
    company_name = symbol_data.name if symbol_data else None
    
    response = HeroStockResponse(
        symbol=hero_rec.ticker,
        company_name=company_name,
        action=hero_rec.action,
        mu_hat=hero_rec.mu_hat,
        dip_pct=hero_rec.legacy_dip_pct,
        days_in_dip=hero_rec.legacy_days_in_dip,
        best_chance_score=hero_rec.best_chance_score,
        best_chance_reason=hero_rec.best_chance_reason,
        ai_summary=hero_rec.ai_summary,
        ai_rating=hero_rec.ai_rating,
        cache_expires_at=expires_at.isoformat(),
    )
    
    # Cache for remaining time in this window
    remaining_seconds = next_window - int(now.timestamp())
    if remaining_seconds > 0:
        await _hero_cache.set("current", response.model_dump(), ttl=remaining_seconds)
    
    return response


def _dip_bucket(dip_pct: float | None) -> str | None:
    """Classify dip percentage into buckets."""
    if dip_pct is None:
        return None
    if dip_pct < 10:
        return "shallow"
    if dip_pct < 20:
        return "moderate"
    if dip_pct < 35:
        return "deep"
    if dip_pct < 50:
        return "very_deep"
    return "extreme"


# ============================================================================
# Signal Triggers Endpoint (Historical Buy Signals for Chart Markers)
# ============================================================================


class SignalTriggerResponse(BaseModel):
    """A historical signal trigger for chart display."""
    date: str
    signal_name: str
    price: float
    win_rate: float
    avg_return_pct: float
    holding_days: int
    drawdown_pct: float = 0.0  # The threshold that triggered (for drawdown signals)
    signal_type: str = "entry"  # "entry" for buy signals, "exit" for sell signals


class SignalTriggersResponse(BaseModel):
    """Response containing historical signal triggers with benchmark comparison."""
    symbol: str
    signal_name: str | None = None  # The signal being tracked
    triggers: list[SignalTriggerResponse]
    
    # Benchmark comparison - how the signal performed vs simply holding
    buy_hold_return_pct: float = 0.0  # Return from buying and holding over the lookback period
    signal_return_pct: float = 0.0  # Aggregate return from signal-based trading
    edge_vs_buy_hold_pct: float = 0.0  # Signal return - buy hold return (the "alpha")
    n_trades: int = 0  # Number of signal triggers in the period
    beats_buy_hold: bool = False  # Whether signal strategy outperformed buy-and-hold
    actual_win_rate: float = 0.0  # Actual win rate from visible trades (not backtest estimate)


@global_router.get(
    "/{symbol}/signal-triggers",
    response_model=SignalTriggersResponse,
    summary="Get historical signal triggers for chart markers",
    description="""
    Get historical buy signal trigger points for a stock.
    
    Returns triggers for the TOP/BEST signal for this stock (highest expected value).
    These are dates when buy signals were triggered based on PAST DATA ONLY.
    There is no look-ahead bias - each trigger uses only data available at that time.
    
    Use these to overlay buy markers on price charts.
    Results are pre-computed nightly for tracked symbols.
    """,
)
async def get_signal_triggers(
    symbol: str,
    lookback_days: int = Query(default=365, ge=30, le=730, description="Days to look back"),
) -> SignalTriggersResponse:
    """Get historical signal triggers for chart markers with benchmark comparison."""
    from app.repositories import quant_precomputed_orm as quant_repo
    
    symbol = symbol.strip().upper()
    
    # Check precomputed cache first (only for default 365-day lookback)
    if lookback_days == 365:
        precomputed = await quant_repo.get_precomputed(symbol)
        if precomputed and precomputed.signal_triggers:
            cached = precomputed.signal_triggers
            return SignalTriggersResponse(
                symbol=symbol,
                signal_name=cached.get("signal_name"),
                triggers=[
                    SignalTriggerResponse(
                        date=t["date"],
                        signal_name=t["signal_name"],
                        price=t["price"],
                        win_rate=t["win_rate"],
                        avg_return_pct=t["avg_return_pct"],
                        holding_days=t.get("holding_days", 20),
                        drawdown_pct=t.get("drawdown_pct", 0.0),
                        signal_type=t.get("signal_type", "entry"),
                    )
                    for t in cached.get("triggers", [])
                ],
                buy_hold_return_pct=cached.get("buy_hold_return_pct", 0.0),
                signal_return_pct=cached.get("signal_return_pct", 0.0),
                edge_vs_buy_hold_pct=cached.get("edge_vs_buy_hold_pct", 0.0),
                n_trades=cached.get("n_trades", 0),
                beats_buy_hold=cached.get("edge_vs_buy_hold_pct", 0.0) > 0,
                actual_win_rate=cached.get("triggers", [{}])[0].get("win_rate", 0.0) if cached.get("triggers") else 0.0,
            )
    
    # Fallback to computing inline for non-default lookback
    from app.quant_engine.signals.scanner import get_historical_triggers
    
    # Fetch price data - use same 5-year window as the signals scanner for consistent optimization
    prices_df = await _fetch_prices_for_symbols([symbol], lookback_days=1260)
    
    if prices_df.empty or symbol not in prices_df.columns:
        return SignalTriggersResponse(symbol=symbol, signal_name=None, triggers=[])
    
    # Convert to dict format expected by signals module
    close_prices = prices_df[symbol].dropna()
    price_data = {
        "close": close_prices,
    }
    
    # Get historical triggers for the BEST signal
    triggers = get_historical_triggers(price_data, lookback_days=lookback_days)
    
    # Get the signal name from the triggers (they're all from the same signal now)
    signal_name = triggers[0].signal_name if triggers else None
    
    # Calculate benchmark comparison
    # Buy-and-hold return over the lookback period
    buy_hold_return_pct = 0.0
    signal_return_pct = 0.0
    edge_vs_buy_hold_pct = 0.0
    
    # Count entry triggers only (not exits)
    entry_triggers = [t for t in triggers if t.signal_type == "entry"]
    n_trades = len(entry_triggers)
    
    if len(close_prices) >= lookback_days and lookback_days > 0:
        lookback_slice = close_prices.iloc[-lookback_days:]
        if len(lookback_slice) >= 2:
            first_price = lookback_slice.iloc[0]
            last_price = lookback_slice.iloc[-1]
            buy_hold_return_pct = ((last_price / first_price) - 1) * 100
    
    # Signal aggregate return - sum of all actual trade returns (exits show individual trade returns)
    # This represents total return from signal-based trading
    exit_triggers = [t for t in triggers if t.signal_type == "exit"]
    if exit_triggers:
        # Sum the actual returns from all completed trades
        signal_return_pct = sum(t.avg_return_pct for t in exit_triggers)
        edge_vs_buy_hold_pct = signal_return_pct - buy_hold_return_pct
    
    # Get actual win rate from the triggers (now properly calculated)
    actual_win_rate = entry_triggers[0].win_rate if entry_triggers else 0.0
    
    # Strategy beats buy-and-hold if it has positive edge
    beats_bh = edge_vs_buy_hold_pct > 0
    
    return SignalTriggersResponse(
        symbol=symbol,
        signal_name=signal_name,
        triggers=[
            SignalTriggerResponse(
                date=t.date,
                signal_name=t.signal_name,
                price=t.price,
                win_rate=t.win_rate,
                avg_return_pct=t.avg_return_pct,
                holding_days=t.holding_days,
                drawdown_pct=t.drawdown_pct,
                signal_type=t.signal_type,
            )
            for t in triggers
        ],
        buy_hold_return_pct=round(buy_hold_return_pct, 2),
        signal_return_pct=round(signal_return_pct, 2),
        edge_vs_buy_hold_pct=round(edge_vs_buy_hold_pct, 2),
        n_trades=n_trades,
        beats_buy_hold=beats_bh,
        actual_win_rate=round(actual_win_rate, 4),
    )


# ============================================================================
# Signal Backtest Endpoint (Compare Signal Buys vs Buy-and-Hold)
# ============================================================================


class SignalTradeResponse(BaseModel):
    """A single trade in the backtest."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    signal_name: str
    won: bool


class SignalBacktestResponse(BaseModel):
    """Backtest comparison: signal buys vs buy-and-hold."""
    symbol: str
    period_start: str
    period_end: str
    
    # Signal strategy results
    signal_name: str
    n_trades: int
    win_rate: float
    total_return_pct: float
    avg_return_per_trade: float
    holding_days_per_trade: int
    trades: list[SignalTradeResponse]
    
    # Buy-and-hold comparison
    buy_hold_return_pct: float
    buy_hold_start_price: float
    buy_hold_end_price: float
    
    # Edge metrics
    signal_edge_pct: float  # signal return - buy&hold return
    signal_outperformed: bool


@global_router.get(
    "/{symbol}/signal-backtest",
    response_model=SignalBacktestResponse,
    summary="Backtest signal strategy vs buy-and-hold",
    description="""
    Compare the performance of buying on signal triggers vs simple buy-and-hold.
    
    Returns pre-computed results from the nightly quant analysis job.
    Results are refreshed daily after market close.
    """,
)
async def get_signal_backtest(
    symbol: str,
    lookback_days: int = Query(default=730, ge=90, le=1825, description="Days to backtest"),
) -> SignalBacktestResponse:
    """Get pre-computed signal backtest results."""
    from app.repositories import quant_precomputed_orm as quant_repo
    
    symbol = symbol.strip().upper()
    
    # Get pre-computed results from database
    cached = await quant_repo.get_precomputed(symbol)
    
    if cached and cached.backtest_signal_name:
        return SignalBacktestResponse(
            symbol=symbol,
            period_start=str(cached.data_start) if cached.data_start else "",
            period_end=str(cached.data_end) if cached.data_end else "",
            signal_name=cached.backtest_signal_name or "Unknown",
            n_trades=cached.backtest_n_trades or 0,
            win_rate=float(cached.backtest_win_rate) if cached.backtest_win_rate else 0.0,
            total_return_pct=float(cached.backtest_total_return_pct) if cached.backtest_total_return_pct else 0.0,
            avg_return_per_trade=float(cached.backtest_avg_return_per_trade) if cached.backtest_avg_return_per_trade else 0.0,
            holding_days_per_trade=cached.backtest_holding_days or 0,
            trades=[],  # Historical trades not cached for performance
            buy_hold_return_pct=float(cached.backtest_buy_hold_return_pct) if cached.backtest_buy_hold_return_pct else 0.0,
            buy_hold_start_price=0.0,  # Not cached
            buy_hold_end_price=0.0,  # Not cached
            signal_edge_pct=float(cached.backtest_edge_pct) if cached.backtest_edge_pct else 0.0,
            signal_outperformed=cached.backtest_outperformed,
        )
    
    # No cached data - return empty response (job will populate it)
    raise HTTPException(
        status_code=404,
        detail=f"No backtest data for {symbol}. Data is computed nightly after market close."
    )


# ============================================================================
# Full Trade Engine Endpoints (Entry + Exit Optimization)
# ============================================================================


class ExitStrategyResponse(BaseModel):
    """Exit strategy details."""
    name: str
    threshold: float
    description: str = ""


class TradeCycleResponse(BaseModel):
    """A complete trade cycle."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    entry_signal: str
    exit_signal: str


class FullTradeResponse(BaseModel):
    """Complete trade analysis with optimized entry AND exit, benchmarked vs SPY."""
    symbol: str
    entry_signal_name: str
    entry_threshold: float
    entry_description: str = ""
    exit_strategy_name: str
    exit_threshold: float
    exit_description: str = ""
    n_complete_trades: int
    win_rate: float
    avg_return_pct: float
    total_return_pct: float
    max_return_pct: float
    max_drawdown_pct: float
    avg_holding_days: float
    sharpe_ratio: float = 0.0
    # Benchmark comparison
    buy_hold_return_pct: float
    spy_return_pct: float = 0.0
    edge_vs_buy_hold_pct: float
    edge_vs_spy_pct: float = 0.0
    beats_both_benchmarks: bool = False
    # Exit timing analysis
    exit_predictability: float = 0.0  # How consistent is exit timing?
    upside_captured_pct: float = 0.0  # What % of potential gain was captured?
    trades: list[TradeCycleResponse]
    current_buy_signal: bool
    current_sell_signal: bool
    days_since_last_signal: int = 0


class CombinedSignalResponse(BaseModel):
    """Signal combination result."""
    name: str
    component_signals: list[str]
    logic: str
    win_rate: float
    avg_return_pct: float
    n_signals: int
    improvement_vs_best_single: float


class DipAnalysisResponse(BaseModel):
    """Analysis of whether a dip is overreaction or real decline (falling knife)."""
    symbol: str
    current_drawdown_pct: float
    typical_dip_pct: float
    max_historical_dip_pct: float
    dip_zscore: float = 0.0  # How many std devs from typical
    is_unusually_deep: bool
    deviation_from_typical: float
    # Technical analysis
    technical_score: float = 0.0  # -1 (bearish) to +1 (bullish)
    trend_broken: bool = False  # Below SMA 200?
    volume_confirmation: bool = False  # High volume = capitulation
    momentum_divergence: bool = False  # Price down but RSI up
    # Classification
    dip_type: str  # OVERREACTION, NORMAL_VOLATILITY, FUNDAMENTAL_DECLINE
    confidence: float
    action: str  # STRONG_BUY, BUY, WAIT, AVOID
    reasoning: str
    # Probability estimates
    recovery_probability: float = 0.5
    expected_return_if_buy: float = 0.0
    expected_loss_if_knife: float = 0.0


class CurrentSignalsResponse(BaseModel):
    """Current real-time buy and sell signals."""
    symbol: str
    buy_signals: list[dict]
    sell_signals: list[dict]
    overall_action: str
    reasoning: str


@global_router.get(
    "/{symbol}/full-trade",
    response_model=FullTradeResponse,
    summary="Get optimized full trade strategy",
    description="""
    Get the best complete trade strategy for a stock: optimized entry signal + exit signal.
    
    Returns pre-computed results from the nightly quant analysis job.
    Results are refreshed daily after market close.
    """,
)
async def get_full_trade_strategy(
    symbol: str,
) -> FullTradeResponse:
    """Get pre-computed full trade strategy results."""
    from app.repositories import quant_precomputed_orm as quant_repo
    
    symbol = symbol.strip().upper()
    
    # Get pre-computed results from database
    cached = await quant_repo.get_precomputed(symbol)
    
    if cached and cached.trade_entry_signal:
        return FullTradeResponse(
            symbol=symbol,
            entry_signal_name=cached.trade_entry_signal or "Unknown",
            entry_threshold=float(cached.trade_entry_threshold) if cached.trade_entry_threshold else 0.0,
            entry_description="",
            exit_strategy_name=cached.trade_exit_signal or "Unknown",
            exit_threshold=float(cached.trade_exit_threshold) if cached.trade_exit_threshold else 0.0,
            exit_description="",
            n_complete_trades=cached.trade_n_trades or 0,
            win_rate=float(cached.trade_win_rate) if cached.trade_win_rate else 0.0,
            avg_return_pct=float(cached.trade_avg_return_pct) if cached.trade_avg_return_pct else 0.0,
            total_return_pct=float(cached.trade_total_return_pct) if cached.trade_total_return_pct else 0.0,
            max_return_pct=0.0,  # Not cached
            max_drawdown_pct=0.0,  # Not cached
            avg_holding_days=0.0,  # Not cached
            sharpe_ratio=float(cached.trade_sharpe_ratio) if cached.trade_sharpe_ratio else 0.0,
            buy_hold_return_pct=float(cached.trade_buy_hold_return_pct) if cached.trade_buy_hold_return_pct else 0.0,
            spy_return_pct=float(cached.trade_spy_return_pct) if cached.trade_spy_return_pct else 0.0,
            edge_vs_buy_hold_pct=(float(cached.trade_total_return_pct) - float(cached.trade_buy_hold_return_pct)) if cached.trade_total_return_pct and cached.trade_buy_hold_return_pct else 0.0,
            edge_vs_spy_pct=(float(cached.trade_total_return_pct) - float(cached.trade_spy_return_pct)) if cached.trade_total_return_pct and cached.trade_spy_return_pct else 0.0,
            beats_both_benchmarks=cached.trade_beats_both,
            exit_predictability=0.0,  # Not cached
            upside_captured_pct=0.0,  # Not cached
            trades=[],  # Historical trades not cached for performance
            current_buy_signal=False,
            current_sell_signal=False,
            days_since_last_signal=0,
        )
    
    # No cached data
    raise HTTPException(
        status_code=404,
        detail=f"No trade strategy data for {symbol}. Data is computed nightly after market close."
    )


@global_router.get(
    "/{symbol}/signal-combinations",
    response_model=list[CombinedSignalResponse],
    summary="Test signal combinations",
    description="""
    Get pre-computed signal combination results.
    
    Returns pre-computed results from the nightly quant analysis job.
    Results are refreshed daily after market close.
    """,
)
async def get_signal_combinations(
    symbol: str,
) -> list[CombinedSignalResponse]:
    """Get pre-computed signal combinations for a stock."""
    from app.repositories import quant_precomputed_orm as quant_repo
    
    symbol = symbol.strip().upper()
    
    # Get pre-computed results from database
    cached = await quant_repo.get_precomputed(symbol)
    
    if cached and cached.signal_combinations:
        results = []
        for combo in cached.signal_combinations:
            results.append(CombinedSignalResponse(
                name=combo.get("name", "Unknown"),
                component_signals=combo.get("component_signals", []),
                logic=combo.get("logic", ""),
                win_rate=combo.get("win_rate", 0.0),
                avg_return_pct=combo.get("avg_return_pct", 0.0),
                n_signals=combo.get("n_signals", 0),
                improvement_vs_best_single=combo.get("improvement_vs_best_single", 0.0),
            ))
        return results
    
    return []


@global_router.get(
    "/{symbol}/dip-analysis",
    response_model=DipAnalysisResponse,
    summary="Analyze if dip is overreaction or falling knife",
    description="""
    Get pre-computed dip analysis results.
    
    Returns pre-computed results from the nightly quant analysis job.
    Results are refreshed daily after market close.
    """,
)
async def get_dip_analysis(
    symbol: str,
) -> DipAnalysisResponse:
    """Get pre-computed dip analysis results."""
    from app.repositories import quant_precomputed_orm as quant_repo
    
    symbol = symbol.strip().upper()
    
    # Get pre-computed results from database
    cached = await quant_repo.get_precomputed(symbol)
    
    if cached and cached.dip_type:
        return DipAnalysisResponse(
            symbol=symbol,
            current_drawdown_pct=float(cached.dip_current_drawdown_pct) if cached.dip_current_drawdown_pct else 0.0,
            typical_dip_pct=float(cached.dip_typical_pct) if cached.dip_typical_pct else 0.0,
            max_historical_dip_pct=float(cached.dip_max_historical_pct) if cached.dip_max_historical_pct else 0.0,
            dip_zscore=float(cached.dip_zscore) if cached.dip_zscore else 0.0,
            is_unusually_deep=abs(float(cached.dip_zscore)) > 1.5 if cached.dip_zscore else False,
            deviation_from_typical=(float(cached.dip_current_drawdown_pct) - float(cached.dip_typical_pct)) if cached.dip_current_drawdown_pct and cached.dip_typical_pct else 0.0,
            technical_score=0.0,  # Not cached
            trend_broken=False,  # Not cached
            volume_confirmation=False,  # Not cached
            momentum_divergence=False,  # Not cached
            dip_type=cached.dip_type or "NORMAL_VOLATILITY",
            confidence=float(cached.dip_confidence) if cached.dip_confidence else 0.5,
            action=cached.dip_action or "WAIT",
            reasoning=cached.dip_reasoning or "",
            recovery_probability=float(cached.dip_recovery_probability) if cached.dip_recovery_probability else 0.5,
            expected_return_if_buy=0.0,  # Not cached
            expected_loss_if_knife=0.0,  # Not cached
        )
    
    # No cached data
    raise HTTPException(
        status_code=404,
        detail=f"No dip analysis data for {symbol}. Data is computed nightly after market close."
    )


@global_router.get(
    "/{symbol}/current-signals",
    response_model=CurrentSignalsResponse,
    summary="Get current real-time buy and sell signals",
    description="""
    Get pre-computed current signals for a stock.
    
    Returns pre-computed results from the nightly quant analysis job.
    Results are refreshed daily after market close.
    
    The overall_action is adjusted to be consistent with:
    - The quant scoring system (mode: HOLD/DOWNTREND)
    - The strategy signal (signal_type: not BUY)
    """,
)
async def get_current_signals(
    symbol: str,
) -> CurrentSignalsResponse:
    """Get pre-computed current signals, adjusted for quant score consistency."""
    from app.repositories import quant_precomputed_orm as quant_repo
    from app.repositories.quant_scores_orm import get_quant_score
    from sqlalchemy import select
    from app.database.orm import StrategySignal
    from app.database.connection import get_session
    
    symbol = symbol.strip().upper()
    
    # Get pre-computed results from database
    cached = await quant_repo.get_precomputed(symbol)
    
    # Also get the quant score to check mode
    quant_score = await get_quant_score(symbol)
    
    # Also check strategy signal - if it says HOLD/WAIT, respect that
    strategy_signal_type = None
    async with get_session() as session:
        result = await session.execute(
            select(StrategySignal.signal_type).where(StrategySignal.symbol == symbol)
        )
        row = result.scalar_one_or_none()
        if row:
            strategy_signal_type = row
    
    if cached and cached.current_signals:
        signals = cached.current_signals
        overall_action = signals.get("overall_action", "HOLD")
        reasoning = signals.get("reasoning", "")
        
        should_hold = False
        hold_reason = ""
        
        # Check 1: quant mode is HOLD or DOWNTREND
        if quant_score and quant_score.mode in ("HOLD", "DOWNTREND"):
            should_hold = True
            hold_reason = f"Quant mode: {quant_score.mode}"
        
        # Check 2: strategy signal is not BUY (e.g., HOLD, WAIT, WATCH)
        if strategy_signal_type and strategy_signal_type not in ("BUY",):
            should_hold = True
            hold_reason = hold_reason or f"Strategy: {strategy_signal_type}"
        
        # Consistency adjustment: if quant/strategy says hold, cap the action
        if should_hold and overall_action in ("STRONG_BUY", "BUY", "WEAK_BUY"):
            original_action = overall_action
            overall_action = "HOLD"
            reasoning = f"Technical: {original_action}, but {hold_reason}"
        
        return CurrentSignalsResponse(
            symbol=symbol,
            buy_signals=signals.get("buy_signals", []),
            sell_signals=signals.get("sell_signals", []),
            overall_action=overall_action,
            reasoning=reasoning,
        )
    
    # No cached data - return empty
    return CurrentSignalsResponse(
        symbol=symbol,
        buy_signals=[],
        sell_signals=[],
        overall_action="HOLD",
        reasoning="No cached signals. Data is computed nightly.",
    )


# ============================================================================
# Signal Scanner Endpoint (Global - No Auth Required)
# ============================================================================


@global_router.get(
    "/signals",
    response_model=SignalScanResponse,
    summary="Scan stocks for buy opportunities",
    description="""
    Get cached technical buy signal scan results.
    
    This endpoint returns pre-computed results from the daily signal scanner job.
    Results are refreshed nightly after market close (10 PM UTC).
    
    Returns:
    - Top stocks ranked by buy opportunity score
    - Active buy signals for each stock
    - Optimal holding periods based on backtesting
    """,
)
async def scan_signals() -> SignalScanResponse:
    """Get cached signal scan results from the daily job."""
    from app.cache.cache import Cache
    
    cache = Cache(prefix="signals", default_ttl=86400)
    
    # Try to get cached results
    cached = await cache.get("daily_scan")
    
    if cached:
        # Build response from cache
        stocks = []
        for opp in cached.get("opportunities", []):
            stocks.append(StockSignalResponse(
                symbol=opp["symbol"],
                name=opp.get("name", opp["symbol"]),
                buy_score=opp.get("buy_score", 0),
                opportunity_type=opp.get("opportunity_type"),
                opportunity_reason=opp.get("opportunity_reason"),
                current_price=opp.get("current_price"),
                price_vs_52w_high_pct=opp.get("price_vs_52w_high_pct"),
                price_vs_52w_low_pct=opp.get("price_vs_52w_low_pct"),
                zscore_20d=opp.get("zscore_20d"),
                zscore_60d=opp.get("zscore_60d"),
                rsi_14=opp.get("rsi_14"),
                best_signal_name=opp.get("best_signal_name"),
                best_holding_days=opp.get("best_holding_days"),
                best_expected_return=opp.get("best_expected_return"),
                signals=[],  # Full signals not cached for performance
                active_buy_signals=[],
            ))
        
        scanned_at_str = cached.get("scanned_at", str(datetime.now().date()))
        try:
            scanned_at = datetime.fromisoformat(scanned_at_str)
        except (ValueError, TypeError):
            scanned_at = datetime.now()
        
        total_active = sum(opp.get("n_active_signals", 0) for opp in cached.get("opportunities", []))
        
        return SignalScanResponse(
            scanned_at=scanned_at,
            holding_days_tested=holding_days_options,
            stocks=stocks,
            top_opportunities=[s.symbol for s in stocks[:3] if s.buy_score > 0],
            n_active_signals=total_active,
        )
    
    # No cache - return empty (job will populate it)
    return SignalScanResponse(
        scanned_at=datetime.now(),
        holding_days_tested=holding_days_options,
        stocks=[],
        top_opportunities=[],
        n_active_signals=0,
    )


# ============================================================================
# Global Analytics Endpoint (No Auth - for Dashboard)
# ============================================================================


@global_router.get(
    "/market-analysis",
    summary="Global Market Analysis",
    description="""
    Get market-wide risk analysis for all tracked symbols.
    
    Returns pre-computed results from the hourly market analysis job.
    Results are refreshed every hour.
    """,
)
async def get_market_analysis() -> dict[str, Any]:
    """Get cached global market analysis."""
    from app.cache.cache import Cache
    
    cache = Cache(prefix="market_analysis", default_ttl=3600)
    
    # Try to get cached results
    cached = await cache.get("global")
    
    if cached:
        return cached
    
    # No cache - return empty (job will populate it)
    return {
        "analyzed_at": datetime.now().isoformat(),
        "status": "computing",
        "message": "Market analysis is computed hourly. Please try again later.",
    }


# =============================================================================
# Backtest V2 Endpoint (Regime-Adaptive Strategy System)
# =============================================================================


class BacktestV2RequestParams(BaseModel):
    """Request parameters for V2 backtest."""
    
    initial_capital: float = 10_000.0
    monthly_contribution: float = 1_000.0
    use_max_history: bool = True
    run_crash_testing: bool = True
    run_alpha_gauntlet: bool = True
    use_meta_rule: bool = True


class RegimeInfoResponse(BaseModel):
    """Current market regime information."""
    
    regime: str
    strategy_mode: str
    drawdown_pct: float
    volatility_regime: str
    description: str


class TradeMarkerV2Response(BaseModel):
    """Trade marker for V2 backtest."""
    
    timestamp: str
    price: float
    type: str
    shares: float
    value: float
    reason: str
    regime: str | None = None
    pnl_pct: float | None = None


class ScenarioResultV2Response(BaseModel):
    """Result of a single scenario simulation."""
    
    scenario: str
    final_value: float
    total_invested: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    n_trades: int
    win_rate: float
    avg_trade_return: float


class CrashTestV2Response(BaseModel):
    """Crash test result."""
    
    crash_name: str
    peak_to_trough_pct: float
    recovery_days: int | None = None
    accumulation_shares: float = 0.0
    avg_buy_price: float = 0.0


class BacktestV2Response(BaseModel):
    """Complete V2 backtest response."""
    
    symbol: str
    period_start: str
    period_end: str
    period_years: float
    
    # Current regime
    current_regime: RegimeInfoResponse
    
    # Best scenario
    best_scenario: str | None = None
    best_return_pct: float = 0.0
    
    # Comparisons
    strategy_vs_bh: float = 0.0
    dca_vs_lump_sum: float = 0.0
    dip_vs_regular_dca: float = 0.0
    
    # Scenario results
    scenarios: dict[str, ScenarioResultV2Response] = {}
    
    # Crash tests
    crash_tests: list[CrashTestV2Response] = []
    
    # Gauntlet
    gauntlet_verdict: str | None = None
    gauntlet_message: str | None = None
    gauntlet_score: float = 0.0
    
    # META Rule stats
    meta_rule_stats: dict[str, Any] = {}
    
    # Trade markers (for charting)
    trade_markers: list[TradeMarkerV2Response] = []
    
    # Regime breakdown
    regime_days: dict[str, int] = {}


@global_router.get(
    "/{symbol}/backtest-v2",
    response_model=BacktestV2Response,
    summary="V2 Regime-Adaptive Backtest",
    description="""
    Run V2 backtest with regime-adaptive strategy system.
    
    This endpoint uses the new V2 engine which provides:
    - Regime detection (Bull/Bear/Crash/Recovery)
    - META Rule fundamental checks for bear market buys
    - DCA and scale-in portfolio simulation
    - Comparison against buy-and-hold and SPY
    - Crash testing (2008, 2020, 2022)
    
    Uses MAXIMUM available price history for robust analysis.
    """,
)
async def get_backtest(
    symbol: str,
    initial_capital: float = Query(default=10_000.0, ge=1000, le=1_000_000),
    monthly_contribution: float = Query(default=1_000.0, ge=0, le=100_000),
    use_max_history: bool = Query(default=True, description="Use all available history"),
) -> BacktestV2Response:
    """Run V2 backtest with regime-adaptive strategies."""
    from app.quant_engine.backtest import (
        BacktestV2Service,
        BacktestV2Config,
    )
    
    symbol = symbol.strip().upper()
    
    # Fetch price history
    if use_max_history:
        prices = await _get_symbol_max_history(symbol)
    else:
        prices = await _get_symbol_prices(symbol, days=1260)
    
    if prices is None or prices.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No price data available for {symbol}",
        )
    
    # Configure V2 service
    config = BacktestV2Config(
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        run_crash_testing=True,
        run_alpha_gauntlet=True,
        use_meta_rule=True,
    )
    
    # Run backtest
    service = BacktestV2Service(config)
    
    try:
        result = await service.run_full_backtest(
            symbol=symbol,
            prices=prices,
            dip_signals=None,  # Will be computed by service
        )
    except Exception as e:
        logger.error(f"V2 backtest failed for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Backtest failed: {str(e)}",
        )
    
    # Convert to response
    regime_info = RegimeInfoResponse(
        regime=result.current_regime.regime.value,
        strategy_mode=result.current_regime.strategy_mode.value,
        drawdown_pct=result.current_regime.drawdown_pct,
        volatility_regime=result.current_regime.volatility_regime,
        description=result.current_regime.description,
    )
    
    # Build scenario results
    scenarios = {}
    if result.simulation:
        for scenario, scenario_result in result.simulation.scenarios.items():
            scenarios[scenario.value] = ScenarioResultV2Response(
                scenario=scenario.value,
                final_value=scenario_result.final_value,
                total_invested=scenario_result.total_invested,
                total_return_pct=scenario_result.total_return_pct,
                annualized_return_pct=scenario_result.annualized_return_pct,
                max_drawdown_pct=scenario_result.max_drawdown_pct,
                sharpe_ratio=scenario_result.sharpe_ratio,
                calmar_ratio=scenario_result.calmar_ratio,
                n_trades=scenario_result.n_trades,
                win_rate=scenario_result.win_rate,
                avg_trade_return=scenario_result.avg_trade_return,
            )
    
    # Build crash test results
    crash_tests = []
    for ct in result.crash_tests:
        crash_tests.append(CrashTestV2Response(
            crash_name=ct.crash_name,
            peak_to_trough_pct=ct.drawdown.peak_to_trough_pct if ct.drawdown else 0,
            recovery_days=ct.recovery.days_to_recover if ct.recovery else None,
            accumulation_shares=ct.accumulation.shares_accumulated if ct.accumulation else 0,
            avg_buy_price=ct.accumulation.avg_buy_price if ct.accumulation else 0,
        ))
    
    # Build trade markers
    trade_markers = []
    for marker in result.trade_markers:
        trade_markers.append(TradeMarkerV2Response(
            timestamp=str(marker.get("timestamp", "")),
            price=float(marker.get("price", 0)),
            type=str(marker.get("type", "")),
            shares=float(marker.get("shares", 0)),
            value=float(marker.get("value", 0)),
            reason=str(marker.get("reason", "")),
            regime=marker.get("regime"),
            pnl_pct=marker.get("pnl_pct"),
        ))
    
    # Gauntlet info
    gauntlet_verdict = None
    gauntlet_message = None
    gauntlet_score = 0.0
    if result.gauntlet:
        gauntlet_verdict = result.gauntlet.verdict.value
        gauntlet_message = result.gauntlet.message
        gauntlet_score = result.gauntlet.overall_score
    
    # Regime days
    regime_days = {r.value: count for r, count in result.regime_days.items()}
    
    return BacktestV2Response(
        symbol=symbol,
        period_start=str(result.period_start.date()),
        period_end=str(result.period_end.date()),
        period_years=result.period_years,
        current_regime=regime_info,
        best_scenario=result.best_scenario.value if result.best_scenario else None,
        best_return_pct=result.best_return_pct,
        strategy_vs_bh=result.strategy_vs_bh,
        dca_vs_lump_sum=result.dca_vs_lump_sum,
        dip_vs_regular_dca=result.dip_vs_regular_dca,
        scenarios=scenarios,
        crash_tests=crash_tests,
        gauntlet_verdict=gauntlet_verdict,
        gauntlet_message=gauntlet_message,
        gauntlet_score=gauntlet_score,
        meta_rule_stats=result.meta_rule_stats,
        trade_markers=trade_markers,
        regime_days=regime_days,
    )


# =============================================================================
# Deep Value Analysis Endpoint
# =============================================================================


class IntrinsicValueEstimateResponse(BaseModel):
    """Individual intrinsic value estimate."""
    method: str
    value: float
    confidence: float
    reasoning: str


class QualityFactorResponse(BaseModel):
    """Quality factor assessment."""
    value: float | None
    signal: str


class DeepValueResponse(BaseModel):
    """Deep value analysis response."""
    symbol: str
    current_price: float
    
    # Intrinsic value
    intrinsic_value: float
    intrinsic_value_method: str
    upside_pct: float
    value_status: str
    all_estimates: list[IntrinsicValueEstimateResponse]
    
    # Quality assessment
    quality_score: float
    quality_tier: str
    quality_factors: dict[str, QualityFactorResponse]
    
    # Market context
    market_regime: str
    regime_context: str
    
    # Alert info
    priority: str
    alert_reason: str
    action_recommendation: str


@router.get(
    "/deep-value/{symbol}",
    response_model=DeepValueResponse,
    summary="Deep Value Analysis",
    description="Analyze a stock for deep value opportunity - combines intrinsic value "
                "calculation with quality assessment and market regime awareness.",
)
async def get_deep_value_analysis(
    symbol: str,
    user: "TokenData" = Depends(require_user),
) -> DeepValueResponse:
    """
    Get deep value analysis for a symbol.
    
    Combines:
    - Multiple intrinsic value methods (analyst, PEG, Graham, DCF)
    - Quality score based on fundamentals
    - Market regime detection
    - Alert priority and action recommendation
    """
    from app.services.deep_value_service import DeepValueService
    import yfinance as yf
    
    symbol = symbol.upper()
    
    try:
        # Get stock data
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Symbol {symbol} not found"
            )
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not current_price:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not get current price for {symbol}"
            )
        
        # Build fundamentals dict
        fundamentals = {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'profit_margin': info.get('profitMargins'),
            'gross_margin': info.get('grossMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'free_cash_flow': info.get('freeCashflow'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'target_mean_price': info.get('targetMeanPrice'),
            'num_analyst_opinions': info.get('numberOfAnalystOpinions'),
        }
        
        # Get SPY data for regime detection
        spy_ticker = yf.Ticker('SPY')
        spy_hist = spy_ticker.history(period='1y')
        spy_prices = spy_hist['Close'] if not spy_hist.empty else None
        
        # Generate analysis
        service = DeepValueService()
        alert = service.generate_alert(symbol, current_price, fundamentals, spy_prices)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not calculate intrinsic value for {symbol}"
            )
        
        # Convert quality factors
        quality_factors = {
            k: QualityFactorResponse(
                value=v.get('value') if isinstance(v, dict) else None,
                signal=v.get('signal', '') if isinstance(v, dict) else str(v)
            )
            for k, v in alert.quality_factors.items()
        }
        
        return DeepValueResponse(
            symbol=symbol,
            current_price=current_price,
            intrinsic_value=alert.intrinsic_value,
            intrinsic_value_method=alert.intrinsic_value_method.value,
            upside_pct=alert.upside_pct,
            value_status=alert.value_status.value,
            all_estimates=[
                IntrinsicValueEstimateResponse(
                    method=e.method.value,
                    value=e.value,
                    confidence=e.confidence,
                    reasoning=e.reasoning
                )
                for e in alert.all_estimates
            ],
            quality_score=alert.quality_score,
            quality_tier=alert.quality_tier.value,
            quality_factors=quality_factors,
            market_regime=alert.market_regime,
            regime_context=alert.regime_context,
            priority=alert.priority.value,
            alert_reason=alert.alert_reason,
            action_recommendation=alert.action_recommendation,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deep value analysis failed for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )
