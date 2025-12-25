"""
Unified yfinance service - single source of truth for all Yahoo Finance data.

This module centralizes ALL yfinance API calls to:
1. Enforce rate limiting consistently
2. Maximize cache hit rate across the app
3. Save ALL search results to local DB for future instant lookups
4. Minimize duplicate yfinance calls

Usage:
    from app.services.yfinance import (
        get_ticker_info,
        get_price_history,
        search_tickers,
        get_fundamentals_data,
    )
    
    # Get stock info (uses cache, rate limited)
    info = await get_ticker_info("AAPL")
    
    # Search for symbols (saves results to DB)
    results = await search_tickers("apple")
    
    # Get price history
    prices = await get_price_history("AAPL", period="1y")
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone, timedelta, date
from typing import Any, Optional, Literal

import yfinance as yf

from app.core.logging import get_logger
from app.core.rate_limiter import get_yfinance_limiter
from app.database.connection import fetch_one, fetch_all, execute

logger = get_logger("services.yfinance")

# Thread pool for blocking yfinance calls
_executor = ThreadPoolExecutor(max_workers=4)

# In-memory cache with TTL (for very frequent calls within same request cycle)
_INFO_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_PRICE_CACHE: dict[str, tuple[float, list[dict]]] = {}

# Cache TTLs
INFO_CACHE_TTL = 300  # 5 minutes for in-memory
PRICE_CACHE_TTL = 60  # 1 minute for prices


# =============================================================================
# Type Helpers
# =============================================================================


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return None
    try:
        f = float(value)
        if f != f or f == float('inf') or f == float('-inf'):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Safely convert value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def is_etf_or_index(symbol: str, quote_type: Optional[str] = None) -> bool:
    """
    Check if symbol is an ETF, index, or fund.
    
    Uses quote_type from yfinance when available, otherwise infers from symbol pattern.
    This is dynamic - any ETF/fund added by users will be properly detected.
    """
    # Index symbols start with ^
    if symbol.startswith("^"):
        return True
    
    # Check quote type if provided
    if quote_type:
        quote_type_upper = quote_type.upper()
        return quote_type_upper in ("ETF", "INDEX", "MUTUALFUND", "TRUST")


# =============================================================================
# Core yfinance API Calls (Sync - run in thread pool)
# =============================================================================


def _fetch_ticker_info_sync(symbol: str) -> Optional[dict[str, Any]]:
    """
    Fetch complete ticker info from yfinance (blocking).
    
    This is the single source of truth for ticker data.
    Returns a normalized dict with all available fields.
    """
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for {symbol}")
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        
        if not info or not info.get("symbol"):
            return None
        
        quote_type = (info.get("quoteType") or "EQUITY").upper()
        # Dynamic ETF detection from quote_type
        is_etf = is_etf_or_index(symbol, quote_type)
        
        # Get IPO year from first trade date
        ipo_year = None
        first_trade_ms = info.get("firstTradeDateMilliseconds")
        if first_trade_ms:
            ipo_year = datetime.fromtimestamp(first_trade_ms / 1000, tz=timezone.utc).year
        
        # Normalize dividend yield (yfinance returns as decimal e.g. 0.17 = 0.17%)
        raw_div_yield = info.get("dividendYield")
        dividend_yield = raw_div_yield / 100 if raw_div_yield else None
        
        return {
            # Identity
            "symbol": symbol.upper(),
            "name": info.get("shortName") or info.get("longName"),
            "quote_type": quote_type,
            "exchange": info.get("exchange"),
            "currency": info.get("currency"),
            "website": info.get("website"),
            "sector": None if is_etf else info.get("sector"),
            "industry": None if is_etf else info.get("industry"),
            "summary": info.get("longBusinessSummary"),
            "ipo_year": ipo_year,
            "is_etf": is_etf,  # Include ETF flag for downstream use
            
            # Price
            "current_price": _safe_float(info.get("regularMarketPrice") or info.get("previousClose")),
            "previous_close": _safe_float(info.get("previousClose")),
            "fifty_two_week_high": _safe_float(info.get("fiftyTwoWeekHigh")),
            "fifty_two_week_low": _safe_float(info.get("fiftyTwoWeekLow")),
            "fifty_day_average": _safe_float(info.get("fiftyDayAverage")),
            "two_hundred_day_average": _safe_float(info.get("twoHundredDayAverage")),
            
            # Market
            "market_cap": _safe_int(info.get("totalAssets") if is_etf else info.get("marketCap")),
            "avg_volume": _safe_int(info.get("averageVolume")),
            "volume": _safe_int(info.get("volume")),
            
            # Valuation (skip for ETFs)
            "pe_ratio": None if is_etf else _safe_float(info.get("trailingPE")),
            "forward_pe": None if is_etf else _safe_float(info.get("forwardPE")),
            "peg_ratio": None if is_etf else _safe_float(info.get("trailingPegRatio")),
            "price_to_book": None if is_etf else _safe_float(info.get("priceToBook")),
            "price_to_sales": None if is_etf else _safe_float(info.get("priceToSalesTrailing12Months")),
            "enterprise_value": None if is_etf else _safe_int(info.get("enterpriseValue")),
            "ev_to_ebitda": None if is_etf else _safe_float(info.get("enterpriseToEbitda")),
            "ev_to_revenue": None if is_etf else _safe_float(info.get("enterpriseToRevenue")),
            
            # Profitability
            "profit_margin": None if is_etf else _safe_float(info.get("profitMargins")),
            "operating_margin": None if is_etf else _safe_float(info.get("operatingMargins")),
            "gross_margin": None if is_etf else _safe_float(info.get("grossMargins")),
            "ebitda_margin": None if is_etf else _safe_float(info.get("ebitdaMargins")),
            "return_on_equity": None if is_etf else _safe_float(info.get("returnOnEquity")),
            "return_on_assets": None if is_etf else _safe_float(info.get("returnOnAssets")),
            
            # Financial Health
            "debt_to_equity": None if is_etf else _safe_float(info.get("debtToEquity")),
            "current_ratio": None if is_etf else _safe_float(info.get("currentRatio")),
            "quick_ratio": None if is_etf else _safe_float(info.get("quickRatio")),
            "total_cash": None if is_etf else _safe_int(info.get("totalCash")),
            "total_debt": None if is_etf else _safe_int(info.get("totalDebt")),
            "free_cash_flow": None if is_etf else _safe_int(info.get("freeCashflow")),
            "operating_cash_flow": None if is_etf else _safe_int(info.get("operatingCashflow")),
            
            # Per Share
            "book_value": None if is_etf else _safe_float(info.get("bookValue")),
            "eps_trailing": None if is_etf else _safe_float(info.get("trailingEps")),
            "eps_forward": None if is_etf else _safe_float(info.get("forwardEps")),
            "revenue_per_share": None if is_etf else _safe_float(info.get("revenuePerShare")),
            "dividend_yield": dividend_yield,
            
            # Growth
            "revenue_growth": None if is_etf else _safe_float(info.get("revenueGrowth")),
            "earnings_growth": None if is_etf else _safe_float(info.get("earningsGrowth")),
            "earnings_quarterly_growth": None if is_etf else _safe_float(info.get("earningsQuarterlyGrowth")),
            
            # Shares
            "shares_outstanding": None if is_etf else _safe_int(info.get("sharesOutstanding")),
            "float_shares": None if is_etf else _safe_int(info.get("floatShares")),
            "held_percent_insiders": None if is_etf else _safe_float(info.get("heldPercentInsiders")),
            "held_percent_institutions": None if is_etf else _safe_float(info.get("heldPercentInstitutions")),
            "short_ratio": None if is_etf else _safe_float(info.get("shortRatio")),
            "short_percent_of_float": None if is_etf else _safe_float(info.get("shortPercentOfFloat")),
            
            # Risk
            "beta": _safe_float(info.get("beta")),
            
            # Analyst
            "recommendation": None if is_etf else info.get("recommendationKey"),
            "recommendation_mean": None if is_etf else _safe_float(info.get("recommendationMean")),
            "num_analyst_opinions": None if is_etf else _safe_int(info.get("numberOfAnalystOpinions")),
            "target_high_price": None if is_etf else _safe_float(info.get("targetHighPrice")),
            "target_low_price": None if is_etf else _safe_float(info.get("targetLowPrice")),
            "target_mean_price": None if is_etf else _safe_float(info.get("targetMeanPrice")),
            "target_median_price": None if is_etf else _safe_float(info.get("targetMedianPrice")),
            
            # Revenue
            "revenue": None if is_etf else _safe_int(info.get("totalRevenue")),
            "ebitda": None if is_etf else _safe_int(info.get("ebitda")),
            "net_income": None if is_etf else _safe_int(info.get("netIncomeToCommon")),
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch ticker info for {symbol}: {e}")
        return None


def _fetch_price_history_sync(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
) -> Optional[list[dict]]:
    """Fetch price history from yfinance (blocking)."""
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for price history: {symbol}")
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            return None
        
        prices = []
        for idx, row in hist.iterrows():
            prices.append({
                "date": idx.strftime("%Y-%m-%d"),
                "open": round(row["Open"], 2),
                "high": round(row["High"], 2),
                "low": round(row["Low"], 2),
                "close": round(row["Close"], 2),
                "volume": int(row["Volume"]),
            })
        
        return prices
        
    except Exception as e:
        logger.error(f"Failed to fetch price history for {symbol}: {e}")
        return None


def _search_tickers_sync(query: str, max_results: int = 10) -> list[dict[str, Any]]:
    """
    Search yfinance for symbols matching query (blocking).
    
    Returns list of matches with symbol, name, exchange, quote_type.
    """
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit timeout for search: {query}")
        return []
    
    try:
        search = yf.Search(
            query,
            max_results=max_results,
            news_count=0,
            enable_fuzzy_query=True,
        )
        
        results = []
        if hasattr(search, 'quotes') and search.quotes:
            for quote in search.quotes[:max_results]:
                quote_type = (quote.get("quoteType") or "").upper()
                if quote_type not in ("EQUITY", "ETF", "INDEX"):
                    continue
                
                results.append({
                    "symbol": quote.get("symbol", ""),
                    "name": quote.get("shortname") or quote.get("longname") or quote.get("symbol"),
                    "exchange": quote.get("exchange", ""),
                    "quote_type": quote_type,
                    "sector": quote.get("sector"),
                    "industry": quote.get("industry"),
                    "score": quote.get("score", 0),
                })
        
        return results
        
    except Exception as e:
        logger.error(f"yfinance search failed for '{query}': {e}")
        return []


# =============================================================================
# Async Public API
# =============================================================================


async def get_ticker_info(
    symbol: str,
    use_cache: bool = True,
) -> Optional[dict[str, Any]]:
    """
    Get complete ticker info for a symbol.
    
    Uses in-memory cache first, then fetches from yfinance.
    
    Args:
        symbol: Stock symbol
        use_cache: Whether to use in-memory cache
        
    Returns:
        Dict with all ticker info fields, or None if invalid
    """
    symbol = symbol.strip().upper()
    
    # Check in-memory cache
    if use_cache:
        cached = _INFO_CACHE.get(symbol)
        if cached and time.time() - cached[0] < INFO_CACHE_TTL:
            return cached[1]
    
    # Fetch from yfinance
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(_executor, _fetch_ticker_info_sync, symbol)
    
    if info:
        _INFO_CACHE[symbol] = (time.time(), info)
    
    return info


async def get_price_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    use_cache: bool = True,
) -> Optional[list[dict]]:
    """
    Get price history for a symbol.
    
    Args:
        symbol: Stock symbol
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        use_cache: Whether to use in-memory cache
        
    Returns:
        List of OHLCV dicts
    """
    symbol = symbol.strip().upper()
    cache_key = f"{symbol}:{period}:{interval}"
    
    # Check cache
    if use_cache:
        cached = _PRICE_CACHE.get(cache_key)
        if cached and time.time() - cached[0] < PRICE_CACHE_TTL:
            return cached[1]
    
    # Fetch
    loop = asyncio.get_event_loop()
    prices = await loop.run_in_executor(
        _executor, _fetch_price_history_sync, symbol, period, interval
    )
    
    if prices:
        _PRICE_CACHE[cache_key] = (time.time(), prices)
    
    return prices


async def search_tickers(
    query: str,
    max_results: int = 10,
    save_to_db: bool = True,
) -> list[dict[str, Any]]:
    """
    Search for tickers matching query.
    
    IMPORTANT: This function saves ALL search results to the database
    so future searches can use local data for instant results.
    
    Search strategy:
    1. Search local symbols table first (instant)
    2. Check search cache for previous API results
    3. Query yfinance API if needed
    4. Save ALL results to local DB for future queries
    
    Args:
        query: Search query (symbol or company name)
        max_results: Maximum results to return
        save_to_db: Whether to save results to DB (default True)
        
    Returns:
        List of matching symbols with metadata
    """
    if len(query.strip()) < 2:
        return []
    
    normalized = query.strip().upper()
    results = []
    seen_symbols = set()
    
    # 1. Search local symbols table first (fast)
    local_results = await _search_local_db(normalized, max_results)
    for r in local_results:
        if r["symbol"] not in seen_symbols:
            seen_symbols.add(r["symbol"])
            r["source"] = "local"
            results.append(r)
    
    # If we have enough results from local, return early
    if len(results) >= max_results:
        return results[:max_results]
    
    # 2. Check search cache
    cached = await _get_search_cache(normalized)
    if cached:
        for r in cached:
            if r["symbol"] not in seen_symbols:
                seen_symbols.add(r["symbol"])
                r["source"] = "cache"
                results.append(r)
        return results[:max_results]
    
    # 3. Query yfinance API
    loop = asyncio.get_event_loop()
    api_results = await loop.run_in_executor(
        _executor, _search_tickers_sync, query, max_results
    )
    
    # 4. Save to cache and optionally to symbols table
    if api_results:
        await _save_search_cache(normalized, api_results)
        
        if save_to_db:
            # Save search results to symbol_search_results table for future local lookups
            await _save_search_results_to_db(api_results)
        
        for r in api_results:
            if r["symbol"] not in seen_symbols:
                seen_symbols.add(r["symbol"])
                r["source"] = "api"
                results.append(r)
    
    return results[:max_results]


async def validate_symbol(symbol: str) -> Optional[dict[str, Any]]:
    """
    Validate a symbol exists and get basic info.
    
    Checks local DB first, then yfinance.
    
    Returns:
        Dict with symbol info if valid, None otherwise
    """
    symbol = symbol.strip().upper()
    
    # Check local first
    row = await fetch_one(
        """
        SELECT symbol, name, sector, symbol_type, market_cap
        FROM symbols 
        WHERE symbol = $1 AND is_active = TRUE
        """,
        symbol,
    )
    
    if row:
        return {
            "symbol": row["symbol"],
            "name": row["name"],
            "sector": row["sector"],
            "quote_type": row["symbol_type"] or "EQUITY",
            "market_cap": float(row["market_cap"]) if row["market_cap"] else None,
            "valid": True,
            "source": "local",
        }
    
    # Fall back to yfinance
    info = await get_ticker_info(symbol)
    if info and info.get("current_price"):
        return {
            "symbol": info["symbol"],
            "name": info["name"],
            "sector": info["sector"],
            "quote_type": info["quote_type"],
            "market_cap": info["market_cap"],
            "valid": True,
            "source": "api",
        }
    
    return None


async def get_fundamentals_data(symbol: str) -> Optional[dict[str, Any]]:
    """
    Get fundamentals data for a symbol.
    
    This is the complete fundamentals dict from ticker info,
    filtered to just fundamental metrics.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dict with fundamental metrics
    """
    info = await get_ticker_info(symbol)
    if not info:
        return None
    
    # Return subset focused on fundamentals
    return {
        "symbol": info["symbol"],
        "name": info["name"],
        # Valuation
        "pe_ratio": info["pe_ratio"],
        "forward_pe": info["forward_pe"],
        "peg_ratio": info["peg_ratio"],
        "price_to_book": info["price_to_book"],
        "price_to_sales": info["price_to_sales"],
        "ev_to_ebitda": info["ev_to_ebitda"],
        # Profitability
        "profit_margin": info["profit_margin"],
        "operating_margin": info["operating_margin"],
        "gross_margin": info["gross_margin"],
        "return_on_equity": info["return_on_equity"],
        "return_on_assets": info["return_on_assets"],
        # Health
        "debt_to_equity": info["debt_to_equity"],
        "current_ratio": info["current_ratio"],
        "free_cash_flow": info["free_cash_flow"],
        # Growth
        "revenue_growth": info["revenue_growth"],
        "earnings_growth": info["earnings_growth"],
        # Analyst
        "recommendation": info["recommendation"],
        "target_mean_price": info["target_mean_price"],
        "num_analyst_opinions": info["num_analyst_opinions"],
        # Risk
        "beta": info["beta"],
        "short_percent_of_float": info["short_percent_of_float"],
    }


# =============================================================================
# Database Helpers
# =============================================================================


async def _search_local_db(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search local symbols table and search results."""
    rows = await fetch_all(
        """
        SELECT 
            symbol, name, sector, symbol_type, market_cap, pe_ratio,
            CASE 
                WHEN UPPER(symbol) = $1 THEN 100
                WHEN UPPER(symbol) LIKE $1 || '%' THEN 90
                WHEN UPPER(name) LIKE '%' || $1 || '%' THEN 70
                ELSE 50
            END as score
        FROM symbols
        WHERE is_active = TRUE
          AND (UPPER(symbol) LIKE $1 || '%' OR UPPER(name) LIKE '%' || $1 || '%')
        
        UNION ALL
        
        SELECT 
            symbol, name, sector, quote_type as symbol_type, market_cap, NULL as pe_ratio,
            CASE 
                WHEN UPPER(symbol) = $1 THEN 85
                WHEN UPPER(symbol) LIKE $1 || '%' THEN 75
                WHEN UPPER(name) LIKE '%' || $1 || '%' THEN 60
                ELSE 40
            END as score
        FROM symbol_search_results
        WHERE symbol NOT IN (SELECT symbol FROM symbols WHERE is_active = TRUE)
          AND (UPPER(symbol) LIKE $1 || '%' OR UPPER(name) LIKE '%' || $1 || '%')
          AND expires_at > NOW()
        
        ORDER BY score DESC, market_cap DESC NULLS LAST
        LIMIT $2
        """,
        query,
        limit,
    )
    
    return [
        {
            "symbol": r["symbol"],
            "name": r["name"],
            "sector": r["sector"],
            "quote_type": (r["symbol_type"] or "EQUITY").upper(),
            "market_cap": float(r["market_cap"]) if r["market_cap"] else None,
            "pe_ratio": float(r["pe_ratio"]) if r["pe_ratio"] else None,
            "score": r["score"],
        }
        for r in rows
    ]


async def _get_search_cache(query: str) -> Optional[list[dict]]:
    """Get cached search results."""
    import json
    row = await fetch_one(
        """
        SELECT results FROM symbol_search_cache
        WHERE query = $1 AND expires_at > NOW()
        """,
        query,
    )
    
    if row and row["results"]:
        results = row["results"]
        if isinstance(results, str):
            results = json.loads(results)
        return results
    
    return None


async def _save_search_cache(query: str, results: list[dict]) -> None:
    """Cache search results."""
    import json
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    
    try:
        await execute(
            """
            INSERT INTO symbol_search_cache (query, results, expires_at)
            VALUES ($1, $2::jsonb, $3)
            ON CONFLICT (query) DO UPDATE SET
                results = EXCLUDED.results,
                expires_at = EXCLUDED.expires_at,
                updated_at = NOW()
            """,
            query,
            json.dumps(results),
            expires_at,
        )
    except Exception as e:
        logger.warning(f"Failed to cache search results: {e}")


async def _save_search_results_to_db(results: list[dict]) -> None:
    """
    Save search results to symbol_search_results table.
    
    This enables future searches to find these symbols locally.
    """
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    
    for r in results:
        if not r.get("symbol"):
            continue
        
        try:
            await execute(
                """
                INSERT INTO symbol_search_results (
                    symbol, name, sector, industry, exchange, 
                    quote_type, market_cap, expires_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol) DO UPDATE SET
                    name = COALESCE(EXCLUDED.name, symbol_search_results.name),
                    sector = COALESCE(EXCLUDED.sector, symbol_search_results.sector),
                    industry = COALESCE(EXCLUDED.industry, symbol_search_results.industry),
                    exchange = COALESCE(EXCLUDED.exchange, symbol_search_results.exchange),
                    quote_type = COALESCE(EXCLUDED.quote_type, symbol_search_results.quote_type),
                    market_cap = COALESCE(EXCLUDED.market_cap, symbol_search_results.market_cap),
                    expires_at = GREATEST(EXCLUDED.expires_at, symbol_search_results.expires_at),
                    updated_at = NOW()
                """,
                r["symbol"],
                r.get("name"),
                r.get("sector"),
                r.get("industry"),
                r.get("exchange"),
                r.get("quote_type"),
                r.get("market_cap"),
                expires_at,
            )
        except Exception as e:
            logger.debug(f"Failed to save search result {r.get('symbol')}: {e}")


# =============================================================================
# Cache Management
# =============================================================================


def clear_info_cache(symbol: Optional[str] = None) -> None:
    """Clear in-memory info cache."""
    if symbol:
        _INFO_CACHE.pop(symbol.upper(), None)
    else:
        _INFO_CACHE.clear()


def clear_price_cache(symbol: Optional[str] = None) -> None:
    """Clear in-memory price cache."""
    if symbol:
        keys_to_remove = [k for k in _PRICE_CACHE if k.startswith(symbol.upper() + ":")]
        for k in keys_to_remove:
            _PRICE_CACHE.pop(k, None)
    else:
        _PRICE_CACHE.clear()
