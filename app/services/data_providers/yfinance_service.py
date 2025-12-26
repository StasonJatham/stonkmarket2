"""
Unified YFinance Service - Single source of truth for ALL Yahoo Finance data.

This module centralizes ALL yfinance API calls to:
1. Enforce rate limiting consistently via single entry point
2. Maximize cache hit rate with 3-tier caching (memory → Valkey → DB)
3. Track data versions for change detection (prices, fundamentals, calendar)
4. Save ALL search results to DB for future local-first lookups
5. Eliminate duplicate yfinance calls across the codebase

Architecture:
- Single ThreadPoolExecutor for all blocking yfinance calls
- Central rate limiter from app.core.rate_limiter
- Version hashing for change-driven job triggers

Usage:
    from app.services.data_providers import get_yfinance_service
    
    service = get_yfinance_service()
    
    # Get stock info (3-tier cache, rate limited)
    info = await service.get_ticker_info("AAPL")
    
    # Get price history with version tracking
    prices, version = await service.get_price_history("AAPL", period="1y")
    
    # Search symbols (saves to DB)
    results = await service.search_tickers("apple")
    
    # Get calendar events
    calendar = await service.get_calendar("AAPL")
    
    # Check if data changed
    changed = await service.has_data_changed("AAPL", "fundamentals", old_hash)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Literal

import pandas as pd
import yfinance as yf

from sqlalchemy import select, or_
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.core.rate_limiter import get_yfinance_limiter
from app.cache.cache import Cache
from app.database.connection import get_session
from app.database.orm import (
    YfinanceInfoCache,
    SymbolSearchResult,
    DataVersion as DataVersionORM,
    Symbol,
)

logger = get_logger("data_providers.yfinance")

# Single shared executor for ALL yfinance calls
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yfinance")

# In-memory cache (L1) - very short TTL for request coalescing
_MEMORY_CACHE: dict[str, tuple[float, Any]] = {}
MEMORY_CACHE_TTL = 60  # 1 minute

# Valkey cache TTLs (L2)
VALKEY_INFO_TTL = 300  # 5 minutes
VALKEY_PRICE_TTL = 60  # 1 minute
VALKEY_CALENDAR_TTL = 3600  # 1 hour

# DB cache TTLs (L3)
DB_SEARCH_TTL_DAYS = 30
DB_INFO_TTL_HOURS = 24


@dataclass
class DataVersion:
    """Version info for change detection."""
    hash: str
    timestamp: datetime
    source: Literal["prices", "fundamentals", "calendar"]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


TickerInfoStatus = Literal["cached", "fetched", "rate_limited", "not_found", "error"]


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data for version tracking."""
    if data is None:
        return ""
    if isinstance(data, pd.DataFrame):
        # For DataFrames, hash the most recent rows to detect meaningful changes.
        tail = data.tail(5).copy()
        tail.reset_index(inplace=True)
        content = json.dumps(
            {
                "last_index": str(data.index[-1]) if len(data) > 0 else "",
                "tail": tail.to_dict(orient="records"),
            },
            sort_keys=True,
            default=str,
        )
    elif isinstance(data, dict):
        # Sort keys for consistent hashing
        content = json.dumps(data, sort_keys=True, default=str)
    else:
        content = str(data)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


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


def _is_etf_or_index(symbol: str, quote_type: Optional[str] = None) -> bool:
    """Check if symbol is ETF, index, or fund."""
    if symbol.startswith("^"):
        return True
    if quote_type:
        return quote_type.upper() in ("ETF", "INDEX", "MUTUALFUND", "TRUST")
    return False


class YFinanceService:
    """
    Unified service for ALL Yahoo Finance data access.
    
    Features:
    - 3-tier caching: Memory (60s) → Valkey (5min) → DB (hours/days)
    - Central rate limiting via get_yfinance_limiter()
    - Version tracking for change detection
    - Full search result persistence
    """
    
    def __init__(self):
        self._info_cache = Cache(prefix="yf_info", default_ttl=VALKEY_INFO_TTL)
        self._price_cache = Cache(prefix="yf_price", default_ttl=VALKEY_PRICE_TTL)
        self._calendar_cache = Cache(prefix="yf_calendar", default_ttl=VALKEY_CALENDAR_TTL)
        self._limiter = get_yfinance_limiter()
    
    # =========================================================================
    # Memory Cache Helpers (L1)
    # =========================================================================
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get from L1 memory cache if not expired."""
        if key in _MEMORY_CACHE:
            ts, data = _MEMORY_CACHE[key]
            if time.time() - ts < MEMORY_CACHE_TTL:
                return data
            del _MEMORY_CACHE[key]
        return None
    
    def _set_in_memory(self, key: str, data: Any) -> None:
        """Set in L1 memory cache."""
        _MEMORY_CACHE[key] = (time.time(), data)
        # Prune old entries periodically
        if len(_MEMORY_CACHE) > 500:
            now = time.time()
            to_delete = [k for k, (ts, _) in _MEMORY_CACHE.items() if now - ts > MEMORY_CACHE_TTL]
            for k in to_delete:
                del _MEMORY_CACHE[k]
    
    # =========================================================================
    # Core yfinance API Calls (Sync, run in thread pool)
    # =========================================================================
    
    def _fetch_ticker_info_with_status_sync(
        self,
        symbol: str,
    ) -> tuple[Optional[dict[str, Any]], TickerInfoStatus]:
        """Fetch complete ticker info from yfinance (blocking) with status."""
        if not self._limiter.acquire_sync():
            logger.warning(f"Rate limit timeout for {symbol}")
            return None, "rate_limited"

        try:
            def _ts_to_date(value: Any) -> Optional[str]:
                """Convert timestamp to ISO date string for JSON serialization."""
                if value is None:
                    return None
                try:
                    ts = float(value)
                    if ts > 1e12:  # ms -> s
                        ts = ts / 1000.0
                    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
                except (ValueError, TypeError, OSError):
                    return None

            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            if not info or not info.get("symbol"):
                return None, "not_found"

            quote_type = (info.get("quoteType") or "EQUITY").upper()
            is_etf = _is_etf_or_index(symbol, quote_type)

            # Get IPO year from first trade date
            ipo_year = None
            first_trade_ms = info.get("firstTradeDateMilliseconds")
            if first_trade_ms:
                ipo_year = datetime.fromtimestamp(first_trade_ms / 1000, tz=timezone.utc).year

            # Normalize dividend yield
            raw_div_yield = info.get("dividendYield")
            dividend_yield = raw_div_yield / 100 if raw_div_yield else None

            # Earnings metadata
            most_recent_quarter = _ts_to_date(info.get("mostRecentQuarter"))
            earnings_timestamp = _ts_to_date(info.get("earningsTimestamp"))

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
                "is_etf": is_etf,

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
                "payout_ratio": None if is_etf else _safe_float(info.get("payoutRatio")),
                "shares_outstanding": None if is_etf else _safe_int(info.get("sharesOutstanding")),
                "float_shares": None if is_etf else _safe_int(info.get("floatShares")),

                # Growth
                "revenue_growth": None if is_etf else _safe_float(info.get("revenueGrowth")),
                "earnings_growth": None if is_etf else _safe_float(info.get("earningsGrowth")),
                "earnings_quarterly_growth": None if is_etf else _safe_float(info.get("earningsQuarterlyGrowth")),

                # Analyst
                "recommendation": info.get("recommendationKey"),
                "recommendation_mean": _safe_float(info.get("recommendationMean")),
                "target_mean_price": _safe_float(info.get("targetMeanPrice")),
                "target_high_price": _safe_float(info.get("targetHighPrice")),
                "target_low_price": _safe_float(info.get("targetLowPrice")),
                "num_analyst_opinions": _safe_int(info.get("numberOfAnalystOpinions")),

                # Earnings (last reported)
                "earnings_date": earnings_timestamp or most_recent_quarter,
                "most_recent_quarter": most_recent_quarter,

                # Revenue & Earnings
                "revenue": None if is_etf else _safe_int(info.get("totalRevenue")),
                "ebitda": None if is_etf else _safe_int(info.get("ebitda")),
                "net_income": None if is_etf else _safe_int(info.get("netIncomeToCommon") or info.get("netIncome")),

                # Risk
                "beta": _safe_float(info.get("beta")),
                "short_ratio": _safe_float(info.get("shortRatio")),
                "short_percent_of_float": _safe_float(info.get("shortPercentOfFloat")),
                "held_percent_insiders": _safe_float(info.get("heldPercentInsiders")),
                "held_percent_institutions": _safe_float(info.get("heldPercentInstitutions")),

                # Metadata
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }, "fetched"
        except Exception as e:
            logger.warning(f"yfinance ticker info failed for {symbol}: {e}")
            return None, "error"

    def _fetch_ticker_info_sync(self, symbol: str) -> Optional[dict[str, Any]]:
        """Fetch complete ticker info from yfinance (blocking)."""
        data, status = self._fetch_ticker_info_with_status_sync(symbol)
        return data if status == "fetched" else None
    
    def _fetch_price_history_sync(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Fetch price history from yfinance (blocking)."""
        if not self._limiter.acquire_sync():
            logger.warning(f"Rate limit timeout for price history: {symbol}")
            return None
        
        try:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                timeout=30,
            )
            
            if df.empty:
                return None
            
            # Handle MultiIndex columns (newer yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                ticker_upper = symbol.upper()
                try:
                    if ticker_upper in df.columns.get_level_values(1):
                        df = df.xs(ticker_upper, axis=1, level=1)
                    else:
                        df.columns = df.columns.droplevel(1)
                except Exception:
                    pass
            
            return df
        except Exception as e:
            logger.warning(f"yfinance price history failed for {symbol}: {e}")
            return None
    
    def _fetch_price_history_batch_sync(
        self,
        symbols: list[str],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch price history (blocking)."""
        if not self._limiter.acquire_sync():
            logger.warning(f"Rate limit timeout for batch price history")
            return {}
        
        try:
            if len(symbols) == 1:
                df = yf.download(
                    symbols[0],
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    timeout=30,
                )
                if df.empty:
                    return {}
                # Normalize
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        df.columns = df.columns.droplevel(1)
                    except Exception:
                        pass
                return {symbols[0]: df}
            
            df = yf.download(
                " ".join(symbols),
                start=start,
                end=end,
                auto_adjust=True,
                group_by="ticker",
                progress=False,
                timeout=30,
            )
            
            if df.empty:
                return {}
            
            results = {}
            for symbol in symbols:
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        if symbol in df.columns.get_level_values(0):
                            ticker_df = df[symbol]
                        elif symbol.upper() in df.columns.get_level_values(1):
                            ticker_df = df.xs(symbol.upper(), axis=1, level=1)
                        else:
                            continue
                    else:
                        ticker_df = df
                    
                    if not ticker_df.empty:
                        results[symbol] = ticker_df
                except Exception as e:
                    logger.debug(f"Failed to extract {symbol} from batch: {e}")
            
            return results
        except Exception as e:
            logger.warning(f"yfinance batch price history failed: {e}")
            return {}
    
    def _search_tickers_sync(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search for tickers (blocking)."""
        if not self._limiter.acquire_sync():
            logger.warning(f"Rate limit timeout for search: {query}")
            return []
        
        try:
            results: list[dict[str, Any]] = []

            # Prefer yfinance Search when available
            if hasattr(yf, "Search"):
                try:
                    search = yf.Search(query, max_results=max_results)
                    quotes = getattr(search, "quotes", None) or []
                    for q in quotes:
                        symbol = q.get("symbol") or q.get("ticker")
                        if not symbol:
                            continue
                        quote_type = (q.get("quoteType") or "EQUITY").upper()
                        results.append(
                            {
                                "symbol": symbol,
                                "name": q.get("shortname") or q.get("shortName") or q.get("longname") or q.get("longName"),
                                "exchange": q.get("exchange") or q.get("exchDisp"),
                                "quote_type": quote_type,
                                "sector": None if _is_etf_or_index(symbol, quote_type) else q.get("sector"),
                                "industry": q.get("industry"),
                                "market_cap": _safe_int(q.get("marketCap")),
                                "relevance_score": _safe_float(q.get("score") or q.get("relevance")),
                                "confidence_score": _safe_float(q.get("confidenceScore")),
                            }
                        )
                    if results:
                        return results[:max_results]
                except Exception as e:
                    logger.debug(f"yfinance Search failed for {query}: {e}")

            # Fallback: direct ticker lookup
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            if info and info.get("symbol"):
                quote_type = (info.get("quoteType") or "EQUITY").upper()
                results.append(
                    {
                        "symbol": info["symbol"],
                        "name": info.get("shortName") or info.get("longName"),
                        "exchange": info.get("exchange"),
                        "quote_type": quote_type,
                        "sector": None if _is_etf_or_index(info["symbol"], quote_type) else info.get("sector"),
                        "industry": info.get("industry"),
                        "market_cap": _safe_int(info.get("marketCap")),
                        "relevance_score": None,
                        "confidence_score": None,
                    }
                )

            return results[:max_results]
        except Exception as e:
            logger.debug(f"yfinance search failed for {query}: {e}")
            return []
    
    def _fetch_calendar_sync(self, symbol: str) -> Optional[dict[str, Any]]:
        """Fetch earnings/dividend calendar (blocking)."""
        if not self._limiter.acquire_sync():
            logger.warning(f"Rate limit timeout for calendar: {symbol}")
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is None:
                return None
            
            result = {"symbol": symbol.upper()}
            
            if isinstance(calendar, dict):
                earnings_dates = calendar.get("Earnings Date", [])
                if earnings_dates and len(earnings_dates) > 0:
                    result["next_earnings_date"] = earnings_dates[0]
                result["earnings_estimate_high"] = calendar.get("Earnings High")
                result["earnings_estimate_low"] = calendar.get("Earnings Low")
                result["earnings_estimate_avg"] = calendar.get("Earnings Average")
                result["dividend_date"] = calendar.get("Dividend Date")
                result["ex_dividend_date"] = calendar.get("Ex-Dividend Date")
            elif hasattr(calendar, "to_dict"):
                # DataFrame format
                cal_dict = calendar.to_dict()
                result.update(cal_dict)
            
            result["fetched_at"] = datetime.now(timezone.utc).isoformat()
            return result
        except Exception as e:
            logger.debug(f"yfinance calendar failed for {symbol}: {e}")
            return None
    
    def _fetch_financials_sync(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Fetch financial statements from yfinance (blocking).
        
        Returns quarterly and annual:
        - Income statement (revenue, net income, interest income/expense, etc.)
        - Balance sheet (assets, liabilities, equity, loans, deposits, etc.)
        - Cash flow (operating, investing, financing, depreciation, etc.)
        
        Data is normalized to dicts with metric names as keys and most recent value.
        """
        if not self._limiter.acquire_sync():
            logger.warning(f"Rate limit timeout for financials: {symbol}")
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            
            result = {
                "symbol": symbol.upper(),
                "quarterly": {},
                "annual": {},
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            
            # Helper to extract most recent values from DataFrame
            def extract_latest(df: pd.DataFrame) -> dict[str, Optional[float]]:
                if df is None or df.empty:
                    return {}
                extracted = {}
                for metric in df.index:
                    try:
                        val = df.loc[metric].iloc[0]  # Most recent quarter/year
                        if pd.notna(val):
                            extracted[str(metric)] = float(val)
                    except (KeyError, IndexError, ValueError, TypeError):
                        pass
                return extracted
            
            # Quarterly statements
            try:
                result["quarterly"]["income_statement"] = extract_latest(ticker.quarterly_income_stmt)
            except Exception as e:
                logger.debug(f"quarterly_income_stmt failed for {symbol}: {e}")
                result["quarterly"]["income_statement"] = {}
            
            try:
                result["quarterly"]["balance_sheet"] = extract_latest(ticker.quarterly_balance_sheet)
            except Exception as e:
                logger.debug(f"quarterly_balance_sheet failed for {symbol}: {e}")
                result["quarterly"]["balance_sheet"] = {}
            
            try:
                result["quarterly"]["cash_flow"] = extract_latest(ticker.quarterly_cashflow)
            except Exception as e:
                logger.debug(f"quarterly_cashflow failed for {symbol}: {e}")
                result["quarterly"]["cash_flow"] = {}
            
            # Annual statements
            try:
                result["annual"]["income_statement"] = extract_latest(ticker.income_stmt)
            except Exception as e:
                logger.debug(f"income_stmt failed for {symbol}: {e}")
                result["annual"]["income_statement"] = {}
            
            try:
                result["annual"]["balance_sheet"] = extract_latest(ticker.balance_sheet)
            except Exception as e:
                logger.debug(f"balance_sheet failed for {symbol}: {e}")
                result["annual"]["balance_sheet"] = {}
            
            try:
                result["annual"]["cash_flow"] = extract_latest(ticker.cashflow)
            except Exception as e:
                logger.debug(f"cashflow failed for {symbol}: {e}")
                result["annual"]["cash_flow"] = {}
            
            # Log what we found for debugging
            q_income_count = len(result["quarterly"]["income_statement"])
            q_balance_count = len(result["quarterly"]["balance_sheet"])
            q_cash_count = len(result["quarterly"]["cash_flow"])
            logger.debug(
                f"Financials for {symbol}: quarterly income={q_income_count}, "
                f"balance={q_balance_count}, cash_flow={q_cash_count}"
            )
            
            return result
        except Exception as e:
            logger.warning(f"yfinance financials failed for {symbol}: {e}")
            return None
    
    # =========================================================================
    # Async Public API
    # =========================================================================
    
    async def get_ticker_info(
        self,
        symbol: str,
        skip_cache: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Get ticker info with 3-tier caching.
        
        Cache hierarchy:
        1. Memory cache (60s) - for request coalescing
        2. Valkey cache (5min) - distributed cache
        3. DB cache (24h) - persistent fallback
        """
        symbol = symbol.upper()
        cache_key = f"info:{symbol}"
        
        if not skip_cache:
            # L1: Memory
            mem_data = self._get_from_memory(cache_key)
            if mem_data is not None:
                logger.debug(f"Cache hit L1 (memory) for {symbol}")
                return mem_data
            
            # L2: Valkey
            valkey_data = await self._info_cache.get(cache_key)
            if valkey_data is not None:
                logger.debug(f"Cache hit L2 (valkey) for {symbol}")
                self._set_in_memory(cache_key, valkey_data)
                return valkey_data
            
            # L3: DB cache
            async with get_session() as session:
                result = await session.execute(
                    select(YfinanceInfoCache.data).where(
                        YfinanceInfoCache.symbol == symbol,
                        YfinanceInfoCache.expires_at > datetime.now(timezone.utc),
                    )
                )
                db_data = result.scalar_one_or_none()
            
            if db_data:
                logger.debug(f"Cache hit L3 (db) for {symbol}")
                data = db_data
                # Defensive: parse JSON if data is string (legacy or codec issue)
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Invalid JSON in DB cache for {symbol}")
                        data = None
                if data:
                    self._set_in_memory(cache_key, data)
                    await self._info_cache.set(cache_key, data)
                    return data
        
        # Cache miss - fetch from yfinance
        logger.debug(f"Cache miss for {symbol}, fetching from yfinance")
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_executor, self._fetch_ticker_info_sync, symbol)
        
        if data:
            # Store in all cache levels
            self._set_in_memory(cache_key, data)
            await self._info_cache.set(cache_key, data)
            
            # DB cache with 24h expiry
            expires = datetime.now(timezone.utc) + timedelta(hours=24)
            async with get_session() as session:
                stmt = insert(YfinanceInfoCache).values(
                    symbol=symbol,
                    data=data,
                    expires_at=expires,
                    fetched_at=datetime.now(timezone.utc),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol"],
                    set_={
                        "data": stmt.excluded.data,
                        "expires_at": stmt.excluded.expires_at,
                        "fetched_at": stmt.excluded.fetched_at,
                    },
                )
                await session.execute(stmt)
                await session.commit()
        
        return data

    async def get_ticker_info_with_status(
        self,
        symbol: str,
        skip_cache: bool = False,
    ) -> tuple[Optional[dict[str, Any]], TickerInfoStatus]:
        """
        Get ticker info with cache-aware status.

        Returns:
            Tuple of (data, status) where status is one of:
            cached, fetched, rate_limited, not_found, error.
        """
        symbol = symbol.upper()
        cache_key = f"info:{symbol}"

        if not skip_cache:
            mem_data = self._get_from_memory(cache_key)
            if mem_data is not None:
                logger.debug(f"Cache hit L1 (memory) for {symbol}")
                return mem_data, "cached"

            valkey_data = await self._info_cache.get(cache_key)
            if valkey_data is not None:
                logger.debug(f"Cache hit L2 (valkey) for {symbol}")
                self._set_in_memory(cache_key, valkey_data)
                return valkey_data, "cached"

            async with get_session() as session:
                result = await session.execute(
                    select(YfinanceInfoCache.data).where(
                        YfinanceInfoCache.symbol == symbol,
                        YfinanceInfoCache.expires_at > datetime.now(timezone.utc),
                    )
                )
                db_data = result.scalar_one_or_none()

            if db_data:
                logger.debug(f"Cache hit L3 (db) for {symbol}")
                data = db_data
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Invalid JSON in DB cache for {symbol}")
                        data = None
                if data:
                    self._set_in_memory(cache_key, data)
                    await self._info_cache.set(cache_key, data)
                    return data, "cached"

        logger.debug(f"Cache miss for {symbol}, fetching from yfinance")
        loop = asyncio.get_event_loop()
        data, status = await loop.run_in_executor(
            _executor, self._fetch_ticker_info_with_status_sync, symbol
        )

        if status == "fetched" and data:
            self._set_in_memory(cache_key, data)
            await self._info_cache.set(cache_key, data)

            expires = datetime.now(timezone.utc) + timedelta(hours=24)
            async with get_session() as session:
                stmt = insert(YfinanceInfoCache).values(
                    symbol=symbol,
                    data=data,
                    expires_at=expires,
                    fetched_at=datetime.now(timezone.utc),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["symbol"],
                    set_={
                        "data": stmt.excluded.data,
                        "expires_at": stmt.excluded.expires_at,
                        "fetched_at": stmt.excluded.fetched_at,
                    },
                )
                await session.execute(stmt)
                await session.commit()

        return data, status
    
    async def get_financials(
        self,
        symbol: str,
        skip_cache: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Get financial statements with caching.
        
        Returns quarterly and annual income statement, balance sheet, and cash flow.
        Useful for domain-specific analysis:
        - Banks: Net Interest Income, Interest Expense, Provision for Credit Losses
        - REITs: Depreciation (for FFO calculation)
        - Insurers: Loss Adjustment Expense, Net Policyholder Benefits
        - All: Revenue, Net Income, Operating Cash Flow, Capital Expenditures
        
        Cache hierarchy:
        1. Memory cache (60s)
        2. Valkey cache (5min)
        """
        symbol = symbol.upper()
        cache_key = f"financials:{symbol}"
        
        if not skip_cache:
            # L1: Memory
            mem_data = self._get_from_memory(cache_key)
            if mem_data is not None:
                logger.debug(f"Financials cache hit L1 (memory) for {symbol}")
                return mem_data
            
            # L2: Valkey
            valkey_data = await self._info_cache.get(cache_key)
            if valkey_data is not None:
                logger.debug(f"Financials cache hit L2 (valkey) for {symbol}")
                self._set_in_memory(cache_key, valkey_data)
                return valkey_data
        
        # Cache miss - fetch from yfinance
        logger.debug(f"Financials cache miss for {symbol}, fetching from yfinance")
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_executor, self._fetch_financials_sync, symbol)
        
        if data:
            # Store in cache
            self._set_in_memory(cache_key, data)
            await self._info_cache.set(cache_key, data)
        
        return data
    
    async def get_ticker_info_with_financials(
        self,
        symbol: str,
        skip_cache: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Get ticker info enriched with financial statement data.
        
        Combines get_ticker_info() with get_financials() for comprehensive analysis.
        The 'financials' key is added to the standard info dict.
        """
        # Fetch both in parallel
        info_task = asyncio.create_task(self.get_ticker_info(symbol, skip_cache=skip_cache))
        financials_task = asyncio.create_task(self.get_financials(symbol, skip_cache=skip_cache))
        
        info, financials = await asyncio.gather(info_task, financials_task)
        
        if info is None:
            return None
        
        # Merge financials into info
        if financials:
            info["financials"] = financials
        
        return info
    
    async def get_price_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> tuple[Optional[pd.DataFrame], Optional[DataVersion]]:
        """
        Get price history with version tracking.
        
        Returns:
            Tuple of (DataFrame, DataVersion) for change detection
        """
        symbol = symbol.upper()
        
        loop = asyncio.get_event_loop()
        
        if start_date and end_date:
            # Specific date range - use batch method
            results = await loop.run_in_executor(
                _executor,
                self._fetch_price_history_batch_sync,
                [symbol],
                start_date.isoformat(),
                end_date.isoformat(),
            )
            df = results.get(symbol)
        else:
            df = await loop.run_in_executor(
                _executor,
                self._fetch_price_history_sync,
                symbol,
                period,
                interval,
            )
        
        if df is None or df.empty:
            return None, None
        
        # Compute version hash
        last_date = df.index[-1] if len(df) > 0 else None
        last_close = float(df["Close"].iloc[-1]) if len(df) > 0 else None
        
        version = DataVersion(
            hash=_compute_hash(df),
            timestamp=datetime.now(timezone.utc),
            source="prices",
            metadata={
                "last_date": str(last_date.date()) if last_date else None,
                "last_close": last_close,
                "row_count": len(df),
            },
        )
        
        return df, version
    
    async def get_price_history_batch(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, tuple[pd.DataFrame, DataVersion]]:
        """
        Batch fetch price history for multiple symbols.
        
        More efficient than individual calls due to single yfinance request.
        """
        symbols = [s.upper() for s in symbols]
        
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(
            _executor,
            self._fetch_price_history_batch_sync,
            symbols,
            start_date.isoformat(),
            end_date.isoformat(),
        )
        
        results = {}
        for symbol, df in raw_results.items():
            if df is not None and not df.empty:
                last_date = df.index[-1] if len(df) > 0 else None
                last_close = float(df["Close"].iloc[-1]) if len(df) > 0 else None
                
                version = DataVersion(
                    hash=_compute_hash(df),
                    timestamp=datetime.now(timezone.utc),
                    source="prices",
                    metadata={
                        "last_date": str(last_date.date()) if last_date else None,
                        "last_close": last_close,
                        "row_count": len(df),
                    },
                )
                results[symbol] = (df, version)
        
        return results
    
    async def search_tickers(
        self,
        query: str,
        max_results: int = 10,
        save_to_db: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Search for tickers and save ALL results to DB.
        
        All search results are persisted for future local-first lookups.
        """
        if len(query.strip()) < 2:
            return []
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            _executor,
            self._search_tickers_sync,
            query,
            max_results * 2,  # Fetch more, filter later
        )
        
        if save_to_db and results:
            # Save to symbol_search_results for future local search
            expires_at = datetime.now(timezone.utc) + timedelta(days=30)
            async with get_session() as session:
                for r in results:
                    try:
                        stmt = insert(SymbolSearchResult).values(
                            symbol=r["symbol"],
                            name=r.get("name"),
                            exchange=r.get("exchange"),
                            quote_type=r.get("quote_type"),
                            sector=r.get("sector"),
                            industry=r.get("industry"),
                            market_cap=r.get("market_cap"),
                            relevance_score=r.get("relevance_score"),
                            confidence_score=r.get("confidence_score") or r.get("relevance_score"),
                            expires_at=expires_at,
                            search_query=query,
                        )
                        # Use COALESCE for upsert - keep existing non-null values
                        stmt = stmt.on_conflict_do_update(
                            constraint="uq_symbol_search_result",
                            set_={
                                "name": stmt.excluded.name,
                                "exchange": stmt.excluded.exchange,
                                "quote_type": stmt.excluded.quote_type,
                                "sector": stmt.excluded.sector,
                                "industry": stmt.excluded.industry,
                                "market_cap": stmt.excluded.market_cap,
                                "relevance_score": stmt.excluded.relevance_score,
                                "confidence_score": stmt.excluded.confidence_score,
                                "expires_at": stmt.excluded.expires_at,
                            },
                        )
                        await session.execute(stmt)
                    except Exception as e:
                        logger.debug(f"Failed to cache search result {r.get('symbol')}: {e}")
                await session.commit()
        
        return results[:max_results]
    
    async def get_calendar(
        self,
        symbol: str,
        skip_cache: bool = False,
    ) -> tuple[Optional[dict[str, Any]], Optional[DataVersion]]:
        """
        Get earnings/dividend calendar with caching and version tracking.
        """
        symbol = symbol.upper()
        cache_key = f"calendar:{symbol}"
        
        if not skip_cache:
            cached = await self._calendar_cache.get(cache_key)
            if cached:
                version = DataVersion(
                    hash=_compute_hash(cached),
                    timestamp=datetime.now(timezone.utc),
                    source="calendar",
                )
                return cached, version
        
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(_executor, self._fetch_calendar_sync, symbol)
        
        if data:
            await self._calendar_cache.set(cache_key, data)
        
        version = DataVersion(
            hash=_compute_hash(data),
            timestamp=datetime.now(timezone.utc),
            source="calendar",
            metadata=data or {},
        ) if data else None
        
        return data, version
    
    async def validate_symbol(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Validate a symbol exists and return basic info.
        
        Uses get_ticker_info internally, returns simplified response.
        """
        info = await self.get_ticker_info(symbol)
        if not info:
            return None
        
        return {
            "symbol": info["symbol"],
            "name": info.get("name"),
            "valid": True,
            "sector": info.get("sector"),
            "summary": (info.get("summary") or "")[:500],
            "quote_type": info.get("quote_type"),
        }
    
    # =========================================================================
    # Version Tracking & Change Detection
    # =========================================================================
    
    async def get_data_version(
        self,
        symbol: str,
        source: Literal["prices", "fundamentals", "calendar"],
    ) -> Optional[DataVersion]:
        """Get current data version from database."""
        async with get_session() as session:
            result = await session.execute(
                select(
                    DataVersionORM.version_hash,
                    DataVersionORM.updated_at,
                    DataVersionORM.version_metadata,
                ).where(
                    DataVersionORM.symbol == symbol.upper(),
                    DataVersionORM.source == source,
                )
            )
            row = result.one_or_none()
        
        if not row:
            return None
        
        return DataVersion(
            hash=row.version_hash,
            timestamp=row.updated_at,
            source=source,
            metadata=row.version_metadata or {},
        )
    
    async def save_data_version(
        self,
        symbol: str,
        version: DataVersion,
    ) -> None:
        """Save data version for change tracking."""
        async with get_session() as session:
            stmt = insert(DataVersionORM).values(
                symbol=symbol.upper(),
                source=version.source,
                version_hash=version.hash,
                version_metadata=version.metadata,
                updated_at=datetime.now(timezone.utc),
            )
            stmt = stmt.on_conflict_do_update(
                constraint="uq_data_version_symbol_source",
                set_={
                    "version_hash": stmt.excluded.version_hash,
                    "version_metadata": stmt.excluded.version_metadata,
                    "updated_at": stmt.excluded.updated_at,
                },
            )
            await session.execute(stmt)
            await session.commit()
    
    async def has_data_changed(
        self,
        symbol: str,
        source: Literal["prices", "fundamentals", "calendar"],
        current_hash: str,
    ) -> bool:
        """Check if data has changed since last recorded version."""
        old_version = await self.get_data_version(symbol, source)
        if not old_version:
            return True  # No previous version = changed
        return old_version.hash != current_hash
    
    async def get_symbols_needing_refresh(
        self,
        source: Literal["prices", "fundamentals", "calendar"],
        max_age_hours: int = 24,
    ) -> list[str]:
        """Get symbols that need data refresh based on age."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        async with get_session() as session:
            result = await session.execute(
                select(Symbol.symbol)
                .outerjoin(
                    DataVersionORM,
                    (Symbol.symbol == DataVersionORM.symbol) & (DataVersionORM.source == source),
                )
                .where(
                    Symbol.is_active == True,
                    or_(
                        DataVersionORM.updated_at == None,
                        DataVersionORM.updated_at < cutoff,
                    ),
                )
                .order_by(DataVersionORM.updated_at.asc().nulls_first())
            )
            return [r[0] for r in result.all()]


# Singleton instance
_instance: Optional[YFinanceService] = None


def get_yfinance_service() -> YFinanceService:
    """Get singleton YFinanceService instance."""
    global _instance
    if _instance is None:
        _instance = YFinanceService()
    return _instance
