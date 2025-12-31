"""Portfolio analytics service with optional quant tool adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.cache.cache import Cache
from app.core.logging import get_logger
from app.dipfinder.service import DatabasePriceProvider
from app.repositories import portfolios_orm as portfolios_repo


logger = get_logger("portfolio.service")


@dataclass
class PortfolioContext:
    portfolio_id: int
    portfolio: dict[str, Any]
    holdings: list[dict[str, Any]]
    prices_by_symbol: dict[str, pd.DataFrame]
    portfolio_values: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series | None


_analytics_cache = Cache(prefix="portfolio_analytics", default_ttl=1800)

HEAVY_TOOLS = {
    "vectorbt",
    "mlfinlab",
    "prophet",
    "arch",
    "gluonts",
    "pyflux",
    "lppls",
    "eiten",
    "fbprophet",
}

PARAM_TOOLS = {"vectorbt", "pandas_ta", "pandas_talib", "talipp", "mlfinlab", "alphalens"}

TOOL_TTLS = {
    "vectorbt": 6 * 3600,
    "mlfinlab": 6 * 3600,
    "prophet": 6 * 3600,
    "fbprophet": 6 * 3600,
    "arch": 3 * 3600,
    "skfolio": 1800,  # 30 min cache for skfolio (runs synchronously now)
    "gluonts": 12 * 3600,
    "pyflux": 12 * 3600,
    "lppls": 24 * 3600,
    "eiten": 12 * 3600,
    "quantstats": 1800,
    "pyfolio": 1800,
    "pandas_ta": 1800,
    "pandas_talib": 1800,
    "talipp": 1800,
    "alphalens": 3600,
    "finquant": 3600,
}


def _normalize_tool_name(tool: str) -> str:
    return tool.strip().lower()


def _hash_params(params: dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:10]


def _analytics_cache_key(
    portfolio_id: int,
    tool: str,
    window: str | None,
    start_date: date | None,
    end_date: date | None,
    params: dict[str, Any],
) -> str:
    return f"{portfolio_id}:{tool}:{window}:{start_date}:{end_date}:{_hash_params(params)}"


def _tool_ttl(tool: str) -> int:
    return TOOL_TTLS.get(tool, 1800)


def _tool_params(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    return params if tool in PARAM_TOOLS else {}


def is_cached_result_stale(tool: str, generated_at: datetime | None) -> bool:
    """Check if a stored result is older than the tool TTL."""
    if generated_at is None:
        return False
    now = datetime.now(UTC)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    return (now - generated_at).total_seconds() > _tool_ttl(tool)


def split_tools(tools: list[str]) -> tuple[list[str], list[str]]:
    """Split tools into lightweight and heavy."""
    seen = set()
    normalized: list[str] = []
    for tool in tools:
        key = _normalize_tool_name(tool)
        if key and key not in seen:
            seen.add(key)
            normalized.append(key)
    light = [tool for tool in normalized if tool not in HEAVY_TOOLS]
    heavy = [tool for tool in normalized if tool in HEAVY_TOOLS]
    return light, heavy


async def get_cached_tool_result(
    portfolio_id: int,
    *,
    tool: str,
    window: str | None,
    start_date: date | None,
    end_date: date | None,
    params: dict[str, Any],
) -> dict[str, Any] | None:
    """Return cached or stored analytics result for a tool."""
    tool_params = _tool_params(tool, params)
    cache_key = _analytics_cache_key(
        portfolio_id,
        tool,
        window,
        start_date,
        end_date,
        tool_params,
    )
    cached = await _analytics_cache.get(cache_key)
    if cached is not None:
        result = dict(cached)
        result["source"] = "cache"
        return result

    latest = await portfolios_repo.get_latest_analytics(
        portfolio_id,
        tool=tool,
        window=window,
        params=tool_params,
    )
    if latest:
        payload = dict(latest["payload"])
        payload["source"] = "db"
        payload["generated_at"] = latest.get("created_at")
        await _analytics_cache.set(cache_key, latest["payload"], ttl=_tool_ttl(tool))
        return payload

    return None


async def invalidate_portfolio_analytics_cache(portfolio_id: int) -> int:
    """Invalidate cached analytics for a portfolio."""
    return await _analytics_cache.invalidate_pattern(f"{portfolio_id}:*")


def _compute_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()


def _compute_portfolio_values(
    holdings: list[dict[str, Any]],
    prices_by_symbol: dict[str, pd.DataFrame],
) -> pd.Series:
    frames = []
    for holding in holdings:
        symbol = holding["symbol"]
        qty = float(holding["quantity"])
        df = prices_by_symbol.get(symbol)
        if df is None or df.empty:
            continue
        series = df["Close"].astype(float) * qty
        frames.append(series.rename(symbol))

    if not frames:
        return pd.Series(dtype=float)

    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.ffill()
    portfolio_values = combined.sum(axis=1)
    return portfolio_values


async def _fetch_prices(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    provider = DatabasePriceProvider()
    return await provider.get_prices_batch(symbols, start_date, end_date)


async def build_portfolio_context(
    portfolio_id: int,
    *,
    user_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    benchmark: str | None = None,
) -> PortfolioContext:
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise ValueError("Portfolio not found")

    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValueError("Portfolio has no holdings")
    
    # Default benchmark to SPY for beta calculation
    benchmark = benchmark or "SPY"

    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365))

    symbols = [h["symbol"] for h in holdings]
    prices_by_symbol = await _fetch_prices(symbols, start_date, end_date)

    portfolio_values = _compute_portfolio_values(
        holdings,
        prices_by_symbol,
    )

    returns = _compute_returns(portfolio_values)

    benchmark_returns = None
    if benchmark:
        benchmark_prices = await _fetch_prices([benchmark], start_date, end_date)
        bench_df = benchmark_prices.get(benchmark)
        if bench_df is not None and not bench_df.empty:
            benchmark_returns = _compute_returns(bench_df["Close"].astype(float))

    return PortfolioContext(
        portfolio_id=portfolio_id,
        portfolio=portfolio,
        holdings=holdings,
        prices_by_symbol=prices_by_symbol,
        portfolio_values=portfolio_values,
        returns=returns,
        benchmark_returns=benchmark_returns,
    )


def _basic_performance_metrics(returns: pd.Series) -> dict[str, Any]:
    if returns.empty:
        return {}
    daily_mean = returns.mean()
    daily_vol = returns.std(ddof=0)
    sharpe = float(daily_mean / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0.0
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    years = max(len(returns) / 252, 1e-6)
    cagr = float(cumulative.iloc[-1] ** (1 / years) - 1)
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "volatility": float(daily_vol * np.sqrt(252)),
        "max_drawdown": max_drawdown,
    }


def _tool_result(tool: str, status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": status,
        "data": data,
        "warnings": warnings or [],
    }


def run_quantstats(context: PortfolioContext) -> dict[str, Any]:
    tool = "quantstats"
    if context.returns.empty:
        return _tool_result(tool, "error", {}, ["No return series available"])
    try:
        import quantstats as qs  # type: ignore

        data = {
            "cagr": float(qs.stats.cagr(context.returns)),
            "sharpe": float(qs.stats.sharpe(context.returns)),
            "sortino": float(qs.stats.sortino(context.returns)),
            "volatility": float(qs.stats.volatility(context.returns)),
            "max_drawdown": float(qs.stats.max_drawdown(context.returns)),
        }
        if context.benchmark_returns is not None and not context.benchmark_returns.empty:
            # Use greeks() to get beta and alpha
            greeks = qs.stats.greeks(context.returns, context.benchmark_returns)
            data["beta"] = float(greeks["beta"])
            data["alpha"] = float(greeks["alpha"])
        return _tool_result(tool, "ok", data)
    except Exception as exc:
        logger.info(f"quantstats unavailable, using fallback: {exc}")
        data = _basic_performance_metrics(context.returns)
        return _tool_result(tool, "partial", data, ["quantstats not available, used fallback metrics"])


def run_pyfolio(context: PortfolioContext) -> dict[str, Any]:
    tool = "pyfolio"
    if context.returns.empty:
        return _tool_result(tool, "error", {}, ["No return series available"])
    try:
        import pyfolio as pf  # type: ignore

        data = {
            "cagr": float(pf.timeseries.cagr(context.returns)),
            "sharpe": float(pf.timeseries.sharpe_ratio(context.returns)),
            "max_drawdown": float(pf.timeseries.max_drawdown(context.returns)),
        }
        return _tool_result(tool, "ok", data)
    except Exception:
        data = _basic_performance_metrics(context.returns)
        return _tool_result(tool, "partial", data, ["pyfolio not available, used fallback metrics"])


def run_skfolio(
    context: PortfolioContext,
    method: str | None = None,
    inflow_amount: float | None = None,
) -> dict[str, Any]:
    """
    Portfolio optimization using skfolio with proper pre-selection and cross-validation.
    
    Args:
        context: Portfolio context with holdings and price data
        method: Optional optimization method override (min_cvar, risk_parity, max_diversification, equal_weight)
        inflow_amount: Optional investment amount - if provided, calculates EUR-based trades
    
    Features:
    - Pre-selection: SelectComplete + DropCorrelated to clean universe
    - Multiple optimization methods: MinCVaR, RiskBudgeting, MaxDiversification
    - Walk-forward cross-validation for out-of-sample evaluation
    - Model comparison to select best method
    """
    tool = "skfolio"
    if context.returns.empty:
        return _tool_result(tool, "error", {}, ["No return series available"])

    # Build returns DataFrame from price history
    returns_df = pd.DataFrame({
        s: _compute_returns(df["Close"]) 
        for s, df in context.prices_by_symbol.items() 
        if df is not None and not df.empty
    })
    returns_df = returns_df.dropna(how="all")
    
    if returns_df.empty:
        return _tool_result(tool, "error", {}, ["No price history for optimization"])
    
    if len(returns_df.columns) < 2:
        return _tool_result(tool, "error", {}, ["Need at least 2 assets for optimization"])
    
    # Minimum 60 days of data for meaningful optimization
    if len(returns_df) < 60:
        return _tool_result(tool, "error", {}, [f"Need at least 60 days of data, got {len(returns_df)}"])

    try:
        from sklearn.pipeline import Pipeline
        from sklearn import set_config
        from skfolio import RiskMeasure
        from skfolio.optimization import MeanRisk, RiskBudgeting, MaximumDiversification, ObjectiveFunction
        from skfolio.pre_selection import DropCorrelated
        from skfolio.model_selection import WalkForward, cross_val_predict
        from skfolio.population import Population
        
        set_config(transform_output='pandas')
        
        warnings: list[str] = []
        dropped_assets: list[str] = []
        
        # --- Step 0: Clean NaN values from returns data ---
        # skfolio requires complete data with no missing values
        original_assets = set(returns_df.columns.tolist())
        
        # First, drop columns (stocks) with >20% missing data
        nan_threshold = 0.2
        nan_pct = returns_df.isna().mean()
        good_columns = nan_pct[nan_pct <= nan_threshold].index.tolist()
        bad_columns = nan_pct[nan_pct > nan_threshold].index.tolist()
        
        if bad_columns:
            warnings.append(f"Dropped {len(bad_columns)} assets with >20% missing data: {', '.join(bad_columns[:3])}")
            dropped_assets.extend(bad_columns)
        
        clean_returns = returns_df[good_columns].copy()
        
        # Then, drop rows with any remaining NaN (forward-fill first to minimize loss)
        clean_returns = clean_returns.ffill().bfill()
        
        # If still has NaN (edge case), drop those rows
        if clean_returns.isna().any().any():
            before_len = len(clean_returns)
            clean_returns = clean_returns.dropna()
            if len(clean_returns) < before_len:
                warnings.append(f"Dropped {before_len - len(clean_returns)} rows with missing data")
        
        # Validate we still have enough data
        if len(clean_returns) < 60:
            return _tool_result(tool, "error", {}, [f"After cleaning NaN, only {len(clean_returns)} rows remain (need 60)"])
        if len(clean_returns.columns) < 2:
            return _tool_result(tool, "error", {}, ["After cleaning, fewer than 2 assets remain"])
        
        # --- Step 1: Pre-selection to remove highly correlated assets ---
        pre_selection_info = _run_pre_selection(clean_returns)
        clean_returns = pre_selection_info["clean_returns"]
        pre_dropped = pre_selection_info["dropped"]
        dropped_assets.extend(pre_dropped)
        
        if len(clean_returns.columns) < 2:
            # If too few assets after pre-selection, use pre-cleaned data
            clean_returns = returns_df[good_columns].ffill().bfill().dropna()
            dropped_assets = list(set(original_assets) - set(clean_returns.columns.tolist()))
            warnings.append("Pre-selection dropped too many assets, using cleaned universe")
        
        if pre_dropped:
            warnings.append(f"Pre-selection removed {len(pre_dropped)} correlated assets: {', '.join(pre_dropped[:5])}")
        
        # --- Step 2: Define optimization models ---
        models = {
            "min_cvar": MeanRisk(
                objective_function=ObjectiveFunction.MINIMIZE_RISK,
                risk_measure=RiskMeasure.CVAR,
                min_weights=0.01,  # At least 1% per position
                max_weights=0.40,  # Max 40% single position
            ),
            "risk_parity": RiskBudgeting(
                risk_measure=RiskMeasure.CVAR,
                min_weights=0.01,
                max_weights=0.40,
            ),
            "max_diversification": MaximumDiversification(
                min_weights=0.01,
                max_weights=0.40,
            ),
        }
        
        # Map common method names to internal names
        method_aliases = {
            "cvar": "min_cvar",
            "min_variance": "min_cvar",  # CVaR is similar risk measure
            "hrp": "risk_parity",        # HRP approximated by risk budgeting
            "equal_weight": None,        # Special case: equal weight
        }
        
        # If method is specified and valid, use it directly (skip CV)
        requested_method = method.lower() if method else None
        if requested_method:
            requested_method = method_aliases.get(requested_method, requested_method)
        
        use_equal_weight = requested_method == "equal_weight" or method == "equal_weight"
        skip_cv = requested_method in models or use_equal_weight
        
        cv_results: dict[str, dict[str, Any]] = {}
        best_model_name = requested_method if requested_method in models else "min_cvar"
        best_sharpe = float("-inf")
        
        # --- Step 3: Walk-forward cross-validation (skip if method specified) ---
        if not skip_cv:
            # Use ~80% train, ~20% test rolling windows
            n_days = len(clean_returns)
            train_size = max(60, int(n_days * 0.6))  # At least 60 days training
            test_size = max(20, int(n_days * 0.2))   # At least 20 days testing
            
            cv = WalkForward(train_size=train_size, test_size=test_size)
            
            for name, model in models.items():
                try:
                    pred = cross_val_predict(model, clean_returns, cv=cv)
                    sharpe = float(pred.sharpe_ratio) if hasattr(pred, 'sharpe_ratio') else 0.0
                    returns_arr = np.asarray(pred)
                    volatility = float(np.std(returns_arr) * np.sqrt(252)) if len(returns_arr) > 0 else 0.0
                    cvar = float(np.percentile(returns_arr, 5)) if len(returns_arr) > 0 else 0.0
                    
                    cv_results[name] = {
                        "sharpe": sharpe,
                        "volatility": volatility,
                        "cvar_95": cvar,
                    }
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_model_name = name
                except Exception as e:
                    logger.warning(f"CV failed for {name}: {e}")
                    cv_results[name] = {"error": str(e)}
        
        # --- Step 4: Fit best model on full data to get final weights ---
        if use_equal_weight:
            # Equal weight: simple 1/N allocation
            n_assets = len(clean_returns.columns)
            weights = {s: 1.0 / n_assets for s in clean_returns.columns}
            best_model_name = "equal_weight"
        else:
            best_model = models[best_model_name]
            best_model.fit(clean_returns)
            weights = dict(zip(clean_returns.columns.tolist(), best_model.weights_.tolist()))
        
        # Add zero weights for dropped assets
        for asset in dropped_assets:
            weights[asset] = 0.0
        
        # --- Step 5: Compute portfolio metrics ---
        mean_returns = clean_returns.mean()
        cov = clean_returns.cov()
        weight_arr = np.array([weights.get(s, 0) for s in clean_returns.columns])
        
        portfolio_return = float(mean_returns.values.dot(weight_arr) * 252)
        portfolio_vol = float(np.sqrt(weight_arr.T.dot(cov.values).dot(weight_arr)) * np.sqrt(252))
        opt_sharpe = float(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0.0
        
        # --- Step 6: Generate rebalance trades ---
        current_weights = _get_current_weights(context.holdings, context.prices_by_symbol)
        portfolio_value = _get_portfolio_value(context.holdings, context.prices_by_symbol)
        
        # Compute trades with EUR amounts if inflow specified
        trades = _compute_trades_with_amounts(
            current_weights=current_weights,
            target_weights=weights,
            portfolio_value=portfolio_value,
            inflow_amount=inflow_amount,
        )
        
        # Current portfolio risk
        current_arr = np.array([current_weights.get(s, 0) for s in clean_returns.columns])
        if current_arr.sum() > 0:
            current_arr = current_arr / current_arr.sum()
        else:
            current_arr = np.ones(len(clean_returns.columns)) / len(clean_returns.columns)
        current_vol = float(np.sqrt(current_arr.T.dot(cov.values).dot(current_arr)) * np.sqrt(252))
        
        # Risk change summary
        vol_change_pct = ((portfolio_vol - current_vol) / current_vol * 100) if current_vol > 0 else 0.0
        if vol_change_pct < -5:
            risk_summary = f"Risk reduced by {abs(vol_change_pct):.0f}%"
        elif vol_change_pct > 5:
            risk_summary = f"Risk increased by {vol_change_pct:.0f}%"
        else:
            risk_summary = "Risk approximately unchanged"
        
        data = {
            "weights": weights,
            "expected_return": portfolio_return,
            "expected_volatility": portfolio_vol,
            "sharpe": opt_sharpe,
            "best_method": best_model_name,
            "cv_results": cv_results,
            "dropped_assets": dropped_assets,
            "trades": trades,
            "n_assets": len([w for w in weights.values() if w > 0.001]),
            # Additional fields for allocation UI compatibility
            "current_weights": current_weights,
            "portfolio_value": portfolio_value,
            "inflow_amount": inflow_amount,
            "current_volatility": current_vol,
            "risk_improvement": risk_summary,
            "confidence": "HIGH" if opt_sharpe > 0.5 else "MEDIUM" if opt_sharpe > 0 else "LOW",
        }
        
        return _tool_result(tool, "ok", data, warnings if warnings else None)
        
    except ImportError as e:
        logger.warning(f"skfolio import failed: {e}")
        # Fall back to basic optimization
        return _run_fallback_optimizer(context, returns_df)
    except Exception as e:
        logger.exception(f"skfolio optimization failed: {e}")
        return _tool_result(tool, "error", {}, [f"Optimization failed: {str(e)}"])


def _run_pre_selection(returns_df: pd.DataFrame, corr_threshold: float = 0.90) -> dict[str, Any]:
    """Run pre-selection to remove incomplete and highly correlated assets."""
    try:
        from sklearn import set_config
        from skfolio.pre_selection import DropCorrelated
        
        set_config(transform_output='pandas')
        
        # Track which assets get dropped
        original_assets = set(returns_df.columns.tolist())
        
        # Drop highly correlated assets
        drop_corr = DropCorrelated(threshold=corr_threshold)
        clean_returns = drop_corr.fit_transform(returns_df)
        
        remaining_assets = set(clean_returns.columns.tolist())
        dropped = list(original_assets - remaining_assets)
        
        return {
            "clean_returns": clean_returns,
            "dropped": dropped,
            "reason": f"Correlation threshold {corr_threshold}",
        }
    except Exception as e:
        logger.warning(f"Pre-selection failed, using full universe: {e}")
        return {
            "clean_returns": returns_df,
            "dropped": [],
            "reason": f"Pre-selection failed: {e}",
        }


def _get_current_weights(holdings: list[dict], prices_by_symbol: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Calculate current portfolio weights from holdings and prices."""
    values = {}
    for h in holdings:
        symbol = h["symbol"]
        qty = float(h.get("quantity") or 0)
        df = prices_by_symbol.get(symbol)
        if df is not None and not df.empty:
            price = float(df["Close"].iloc[-1])
            values[symbol] = qty * price
    
    total = sum(values.values())
    if total <= 0:
        return {s: 0.0 for s in values}
    
    return {s: v / total for s, v in values.items()}


def _compute_rebalance_trades(current_weights: dict[str, float], target_weights: dict[str, float]) -> list[dict[str, Any]]:
    """Compute trades needed to rebalance from current to target weights."""
    trades = []
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())
    
    for symbol in all_symbols:
        current = current_weights.get(symbol, 0.0)
        target = target_weights.get(symbol, 0.0)
        delta = target - current
        
        if abs(delta) > 0.005:  # Only show trades > 0.5% weight change
            trades.append({
                "symbol": symbol,
                "current_weight": round(current * 100, 2),
                "target_weight": round(target * 100, 2),
                "delta_weight": round(delta * 100, 2),
                "action": "buy" if delta > 0 else "sell",
            })
    
    # Sort by absolute delta descending
    trades.sort(key=lambda t: abs(t["delta_weight"]), reverse=True)
    return trades


def _get_portfolio_value(holdings: list[dict], prices_by_symbol: dict[str, pd.DataFrame]) -> float:
    """Calculate total portfolio value from holdings and current prices."""
    total = 0.0
    for h in holdings:
        symbol = h["symbol"]
        qty = float(h.get("quantity") or 0)
        df = prices_by_symbol.get(symbol)
        if df is not None and not df.empty:
            price = float(df["Close"].iloc[-1])
            total += qty * price
    return total


def _compute_trades_with_amounts(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    portfolio_value: float,
    inflow_amount: float | None = None,
) -> list[dict[str, Any]]:
    """
    Compute trades needed to rebalance from current to target weights.
    
    If inflow_amount is provided, calculates trade amounts in currency.
    """
    trades = []
    all_symbols = set(current_weights.keys()) | set(target_weights.keys())
    
    # Total value after inflow
    total_value = portfolio_value + (inflow_amount or 0)
    
    for symbol in all_symbols:
        current_pct = current_weights.get(symbol, 0.0)
        target_pct = target_weights.get(symbol, 0.0)
        delta_pct = target_pct - current_pct
        
        # Skip tiny changes
        if abs(delta_pct) <= 0.005:
            continue
        
        # Calculate amounts
        current_value = current_pct * portfolio_value
        target_value = target_pct * total_value
        trade_value = target_value - current_value
        
        # Determine action
        if trade_value > 0:
            action = "BUY"
            reason = "New position for diversification" if current_pct == 0 else "Increase position to improve risk balance"
        else:
            action = "SELL"
            reason = "Close position" if target_pct == 0 else "Reduce overweight position"
        
        trades.append({
            "symbol": symbol,
            "action": action,
            "amount_eur": round(abs(trade_value), 2),
            "current_weight_pct": round(current_pct * 100, 2),
            "target_weight_pct": round(target_pct * 100, 2),
            "delta_weight_pct": round(delta_pct * 100, 2),
            "reason": reason,
        })
    
    # Sort by absolute trade amount descending
    trades.sort(key=lambda t: t["amount_eur"], reverse=True)
    return trades


def _run_fallback_optimizer(context: PortfolioContext, returns_df: pd.DataFrame) -> dict[str, Any]:
    """Fallback optimizer using inverse-volatility weighting (simpler, more robust)."""
    tool = "skfolio"
    warnings = ["skfolio not available, using inverse-volatility fallback"]
    
    # Inverse volatility weighting (risk parity lite)
    vols = returns_df.std()
    inv_vols = 1 / vols.replace(0, np.inf)
    weights = inv_vols / inv_vols.sum()
    
    weight_map = {symbol: float(w) for symbol, w in weights.items()}
    
    mean_returns = returns_df.mean()
    cov = returns_df.cov()
    weight_arr = weights.values
    
    portfolio_return = float(mean_returns.values.dot(weight_arr) * 252)
    portfolio_vol = float(np.sqrt(weight_arr.T.dot(cov.values).dot(weight_arr)) * np.sqrt(252))
    sharpe = float(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0.0
    
    data = {
        "weights": weight_map,
        "expected_return": portfolio_return,
        "expected_volatility": portfolio_vol,
        "sharpe": sharpe,
        "best_method": "inverse_volatility",
        "cv_results": {},
        "dropped_assets": [],
        "trades": [],
        "n_assets": len(weight_map),
    }
    
    return _tool_result(tool, "partial", data, warnings)


def run_arch(context: PortfolioContext) -> dict[str, Any]:
    tool = "arch"
    if context.returns.empty:
        return _tool_result(tool, "error", {}, ["No return series available"])
    returns = context.returns * 100
    try:
        from arch import arch_model  # type: ignore

        model = arch_model(returns, p=1, q=1, vol="Garch", dist="normal")
        result = model.fit(disp="off")
        forecast = result.forecast(horizon=5)
        vol_forecast = float(np.sqrt(forecast.variance.iloc[-1].mean()) / 100)
        data = {"volatility_forecast": vol_forecast, "model": "GARCH(1,1)"}
        return _tool_result(tool, "ok", data)
    except Exception as exc:
        logger.info(f"arch unavailable, using rolling volatility: {exc}")
        if len(returns) >= 20:
            rolling = returns.rolling(20).std().iloc[-1]
            if pd.isna(rolling):
                rolling = returns.std()
            vol = float(rolling / 100)
        else:
            vol = float(returns.std() / 100)
        data = {"volatility_forecast": vol, "model": "rolling"}
        return _tool_result(tool, "partial", data, ["arch not available, used rolling volatility"])


def run_prophet(context: PortfolioContext) -> dict[str, Any]:
    tool = "prophet"
    if context.portfolio_values.empty:
        return _tool_result(tool, "error", {}, ["No portfolio value series available"])
    try:
        from prophet import Prophet  # type: ignore

        df = pd.DataFrame({
            "ds": pd.to_datetime(context.portfolio_values.index),
            "y": context.portfolio_values.values,
        })
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        last = forecast.iloc[-1]
        data = {
            "forecast": float(last["yhat"]),
            "lower": float(last["yhat_lower"]),
            "upper": float(last["yhat_upper"]),
            "horizon_days": 30,
        }
        return _tool_result(tool, "ok", data)
    except Exception as exc:
        logger.info(f"prophet unavailable, using linear trend: {exc}")
        series = context.portfolio_values
        x = np.arange(len(series))
        if len(x) < 2:
            return _tool_result(tool, "error", {}, ["Not enough data for forecast"])
        coeffs = np.polyfit(x, series.values, 1)
        forecast = float(coeffs[0] * (x[-1] + 30) + coeffs[1])
        data = {"forecast": forecast, "horizon_days": 30}
        return _tool_result(tool, "partial", data, ["prophet not available, used linear trend"])


def run_vectorbt(context: PortfolioContext, params: dict[str, Any] | None = None) -> dict[str, Any]:
    tool = "vectorbt"
    params = params or {}
    symbol = params.get("symbol")
    if not symbol:
        return _tool_result(tool, "error", {}, ["Missing symbol in params"])

    df = context.prices_by_symbol.get(symbol.upper())
    if df is None or df.empty:
        return _tool_result(tool, "error", {}, ["No price series for symbol"])

    prices = df["Close"].astype(float)
    fast = int(params.get("fast", 20))
    slow = int(params.get("slow", 50))

    try:
        import vectorbt as vbt  # type: ignore

        fast_ma = vbt.MA.run(prices, window=fast)
        slow_ma = vbt.MA.run(prices, window=slow)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        pf = vbt.Portfolio.from_signals(prices, entries, exits)
        data = {
            "total_return": float(pf.total_return()),
            "sharpe": float(pf.sharpe_ratio()),
            "max_drawdown": float(pf.max_drawdown()),
            "trades": int(pf.trades.count()),
        }
        return _tool_result(tool, "ok", data)
    except Exception:
        # Fallback SMA cross
        fast_ma = prices.rolling(fast).mean()
        slow_ma = prices.rolling(slow).mean()
        signal = (fast_ma > slow_ma).astype(int)
        returns = prices.pct_change().fillna(0)
        strat_returns = returns * signal.shift(1).fillna(0)
        data = _basic_performance_metrics(strat_returns)
        data["strategy"] = f"sma_{fast}_{slow}"
        return _tool_result(tool, "partial", data, ["vectorbt not available, used simple SMA backtest"])


def run_pandas_ta(context: PortfolioContext, params: dict[str, Any] | None = None) -> dict[str, Any]:
    tool = "pandas_ta"
    params = params or {}
    symbol = params.get("symbol")
    if not symbol:
        return _tool_result(tool, "error", {}, ["Missing symbol in params"])

    df = context.prices_by_symbol.get(symbol.upper())
    if df is None or df.empty:
        return _tool_result(tool, "error", {}, ["No price series for symbol"])

    prices = df["Close"].astype(float)
    try:
        import pandas_ta as ta  # type: ignore

        rsi = ta.rsi(prices, length=14).iloc[-1]
        macd = ta.macd(prices).iloc[-1].to_dict()
        bb = ta.bbands(prices).iloc[-1].to_dict()
        data = {"rsi": float(rsi), "macd": macd, "bbands": bb}
        return _tool_result(tool, "ok", data)
    except Exception:
        rsi = _compute_rsi(prices, 14)
        macd = _compute_macd(prices)
        bb = _compute_bbands(prices)
        data = {"rsi": rsi, "macd": macd, "bbands": bb}
        return _tool_result(tool, "partial", data, ["pandas_ta not available, used fallback indicators"])


def run_talipp(context: PortfolioContext, params: dict[str, Any] | None = None) -> dict[str, Any]:
    tool = "talipp"
    params = params or {}
    symbol = params.get("symbol")
    if not symbol:
        return _tool_result(tool, "error", {}, ["Missing symbol in params"])

    df = context.prices_by_symbol.get(symbol.upper())
    if df is None or df.empty:
        return _tool_result(tool, "error", {}, ["No price series for symbol"])

    prices = df["Close"].astype(float)
    try:
        from talipp.indicators import RSI  # type: ignore

        rsi = RSI(prices.tolist(), period=14)[-1]
        return _tool_result(tool, "ok", {"rsi": float(rsi)})
    except Exception:
        rsi = _compute_rsi(prices, 14)
        return _tool_result(tool, "partial", {"rsi": rsi}, ["talipp not available, used fallback RSI"])


def run_mlfinlab(context: PortfolioContext, params: dict[str, Any] | None = None) -> dict[str, Any]:
    tool = "mlfinlab"
    params = params or {}
    features = params.get("features")
    target = params.get("target")
    if features is None or target is None:
        return _tool_result(tool, "error", {}, ["Missing features/target in params"])

    try:
        import mlfinlab  # type: ignore  # noqa: F401
        warnings = ["mlfinlab detected but custom feature pipeline required; using simple correlation"]
    except Exception:
        warnings = ["mlfinlab not available, used simple correlation"]

    df = pd.DataFrame(features)
    target_series = pd.Series(target)
    if len(df) != len(target_series):
        return _tool_result(tool, "error", {}, ["Feature/target length mismatch"])
    correlations = df.apply(lambda col: col.corr(target_series)).sort_values(ascending=False)
    data = {"feature_importance": correlations.fillna(0).to_dict()}
    return _tool_result(tool, "partial", data, warnings)


def run_alphalens(context: PortfolioContext, params: dict[str, Any] | None = None) -> dict[str, Any]:
    tool = "alphalens"
    params = params or {}
    factor = params.get("factor")
    forward_returns = params.get("forward_returns")
    if factor is None:
        return _tool_result(tool, "error", {}, ["Missing factor data in params"])
    if forward_returns is None:
        return _tool_result(tool, "error", {}, ["Missing forward_returns in params"])
    try:
        import alphalens  # type: ignore  # noqa: F401
        warnings = ["alphalens detected but full factor pipeline required; using simple IC"]
    except Exception:
        warnings = ["alphalens not available, used simple IC"]

    factor_series = pd.Series(factor)
    forward_series = pd.Series(forward_returns)
    if len(factor_series) != len(forward_series):
        return _tool_result(tool, "error", {}, ["Factor/forward_returns length mismatch"])
    data = {"information_coefficient": float(factor_series.corr(forward_series))}
    return _tool_result(tool, "partial", data, warnings)


def run_finquant(context: PortfolioContext) -> dict[str, Any]:
    tool = "finquant"
    try:
        import finquant  # type: ignore  # noqa: F401
        return _tool_result(tool, "partial", {}, ["finquant available but no adapter configured"])
    except Exception:
        return _tool_result(tool, "error", {}, ["finquant not installed"])


def run_gluonts(context: PortfolioContext) -> dict[str, Any]:
    tool = "gluonts"
    try:
        import gluonts  # type: ignore  # noqa: F401
        return _tool_result(tool, "partial", {}, ["gluonts available but no adapter configured"])
    except Exception:
        return _tool_result(tool, "error", {}, ["gluonts not installed"])


def run_pyflux(context: PortfolioContext) -> dict[str, Any]:
    tool = "pyflux"
    try:
        import pyflux  # type: ignore  # noqa: F401
        return _tool_result(tool, "partial", {}, ["pyflux available but no adapter configured"])
    except Exception:
        return _tool_result(tool, "error", {}, ["pyflux not installed"])


def run_lppls(context: PortfolioContext) -> dict[str, Any]:
    tool = "lppls"
    try:
        import lppls  # type: ignore  # noqa: F401
        return _tool_result(tool, "partial", {}, ["lppls available but no adapter configured"])
    except Exception:
        return _tool_result(tool, "error", {}, ["lppls not installed"])


def run_eiten(context: PortfolioContext) -> dict[str, Any]:
    tool = "eiten"
    try:
        import eiten  # type: ignore  # noqa: F401
        return _tool_result(tool, "partial", {}, ["eiten available but no adapter configured"])
    except Exception:
        return _tool_result(tool, "error", {}, ["eiten not installed"])


def _compute_rsi(series: pd.Series, period: int) -> float:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = (-delta.clip(upper=0)).rolling(period).mean()
    rs = up / down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _compute_macd(series: pd.Series) -> dict[str, float]:
    fast = series.ewm(span=12, adjust=False).mean()
    slow = series.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return {
        "macd": float(macd.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "hist": float(hist.iloc[-1]),
    }


def _compute_bbands(series: pd.Series) -> dict[str, float]:
    sma = series.rolling(20).mean()
    std = series.rolling(20).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return {
        "upper": float(upper.iloc[-1]),
        "middle": float(sma.iloc[-1]),
        "lower": float(lower.iloc[-1]),
    }


TOOL_HANDLERS = {
    "quantstats": run_quantstats,
    "pyfolio": run_pyfolio,
    "skfolio": run_skfolio,
    "arch": run_arch,
    "prophet": run_prophet,
    "fbprophet": run_prophet,
    "vectorbt": run_vectorbt,
    "pandas_ta": run_pandas_ta,
    "pandas_talib": run_pandas_ta,
    "talipp": run_talipp,
    "mlfinlab": run_mlfinlab,
    "alphalens": run_alphalens,
    "finquant": run_finquant,
    "gluonts": run_gluonts,
    "pyflux": run_pyflux,
    "lppls": run_lppls,
    "eiten": run_eiten,
}


async def run_portfolio_tools(
    portfolio_id: int,
    *,
    user_id: int,
    tools: list[str],
    window: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    benchmark: str | None = None,
    params: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> list[dict[str, Any]]:
    """Run portfolio tools and cache/store results."""
    context = await build_portfolio_context(
        portfolio_id,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        benchmark=benchmark,
    )

    results = []
    params = params or {}
    for tool in tools:
        tool_key = _normalize_tool_name(tool)
        handler = TOOL_HANDLERS.get(tool_key)
        if not handler:
            results.append(_tool_result(tool_key, "error", {}, ["Unknown tool"]))
            continue

        tool_params = _tool_params(tool_key, params)
        cache_key = _analytics_cache_key(
            portfolio_id,
            tool_key,
            window,
            start_date,
            end_date,
            tool_params,
        )
        cached = None if force_refresh else await _analytics_cache.get(cache_key)
        if cached is not None:
            cached_result = dict(cached)
            cached_result["source"] = "cache"
            results.append(cached_result)
            continue

        try:
            if tool_key in PARAM_TOOLS:
                result = handler(context, tool_params)
            else:
                result = handler(context)
        except Exception as exc:
            result = _tool_result(tool_key, "error", {}, [f"Tool failed: {exc}"])

        await _analytics_cache.set(cache_key, result, ttl=_tool_ttl(tool_key))
        await portfolios_repo.save_portfolio_analytics(
            portfolio_id,
            tool=tool_key,
            payload=result,
            status=result.get("status", "ok"),
            as_of_date=end_date or date.today(),
            window=window,
            params=tool_params,
        )
        response_result = dict(result)
        response_result["source"] = "computed"
        results.append(response_result)

    return results
