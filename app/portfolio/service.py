"""Portfolio analytics service with optional quant tool adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

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
    benchmark_returns: Optional[pd.Series]


_analytics_cache = Cache(prefix="portfolio_analytics", default_ttl=1800)

HEAVY_TOOLS = {
    "vectorbt",
    "mlfinlab",
    "prophet",
    "arch",
    "skfolio",
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
    "skfolio": 6 * 3600,
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
    window: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
    params: dict[str, Any],
) -> str:
    return f"{portfolio_id}:{tool}:{window}:{start_date}:{end_date}:{_hash_params(params)}"


def _tool_ttl(tool: str) -> int:
    return TOOL_TTLS.get(tool, 1800)


def _tool_params(tool: str, params: dict[str, Any]) -> dict[str, Any]:
    return params if tool in PARAM_TOOLS else {}


def is_cached_result_stale(tool: str, generated_at: Optional[datetime]) -> bool:
    """Check if a stored result is older than the tool TTL."""
    if generated_at is None:
        return False
    now = datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
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
    window: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
    params: dict[str, Any],
) -> Optional[dict[str, Any]]:
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
    cash_balance: float,
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
    if cash_balance:
        portfolio_values = portfolio_values + float(cash_balance)
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
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    benchmark: Optional[str] = None,
) -> PortfolioContext:
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise ValueError("Portfolio not found")

    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValueError("Portfolio has no holdings")

    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=365))

    symbols = [h["symbol"] for h in holdings]
    prices_by_symbol = await _fetch_prices(symbols, start_date, end_date)

    portfolio_values = _compute_portfolio_values(
        holdings,
        prices_by_symbol,
        float(portfolio.get("cash_balance") or 0),
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


def _tool_result(tool: str, status: str, data: dict[str, Any], warnings: Optional[list[str]] = None) -> dict[str, Any]:
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
        if context.benchmark_returns is not None:
            data["beta"] = float(qs.stats.beta(context.returns, context.benchmark_returns))
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


def run_skfolio(context: PortfolioContext) -> dict[str, Any]:
    tool = "skfolio"
    if context.returns.empty:
        return _tool_result(tool, "error", {}, ["No return series available"])

    returns_df = pd.DataFrame({s: _compute_returns(df["Close"]) for s, df in context.prices_by_symbol.items() if df is not None and not df.empty})
    returns_df = returns_df.dropna(how="all")
    if returns_df.empty:
        return _tool_result(tool, "error", {}, ["No price history for optimization"])

    try:
        import skfolio  # type: ignore  # noqa: F401
        warnings = ["skfolio detected but optimizer not configured; using fallback optimizer"]
    except Exception:
        warnings = ["skfolio not available, used fallback optimizer"]

    mean_returns = returns_df.mean()
    cov = returns_df.cov()
    inv_cov = np.linalg.pinv(cov.values)
    raw_weights = inv_cov.dot(mean_returns.values)
    raw_weights = np.maximum(raw_weights, 0)
    if raw_weights.sum() == 0:
        weights = np.repeat(1 / len(mean_returns), len(mean_returns))
    else:
        weights = raw_weights / raw_weights.sum()

    weight_map = {symbol: float(weight) for symbol, weight in zip(mean_returns.index.tolist(), weights)}
    portfolio_return = float(mean_returns.values.dot(weights) * 252)
    portfolio_vol = float(np.sqrt(weights.T.dot(cov.values).dot(weights)) * np.sqrt(252))
    sharpe = float(portfolio_return / portfolio_vol) if portfolio_vol > 0 else 0.0
    data = {
        "weights": weight_map,
        "expected_return": portfolio_return,
        "expected_volatility": portfolio_vol,
        "sharpe": sharpe,
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


def run_vectorbt(context: PortfolioContext, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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


def run_pandas_ta(context: PortfolioContext, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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


def run_talipp(context: PortfolioContext, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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


def run_mlfinlab(context: PortfolioContext, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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


def run_alphalens(context: PortfolioContext, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
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
    window: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    benchmark: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
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
