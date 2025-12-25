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

from app.core.logging import get_logger
from app.core.rate_limiter import get_yfinance_limiter
from app.cache.cache import Cache
from app.database.connection import fetch_one, fetch_all, execute

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


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data for version tracking."""
    if data is None:
        return ""
    if isinstance(data, pd.DataFrame):
        # For DataFrames, hash the shape and last few values
        content = f"{data.shape}:{data.index[-1] if len(data) > 0 else ''}:{data.iloc[-1].to_dict() if len(data) > 0 else {}}"
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
    
    def _fetch_ticker_info_sync(self, symbol: str) -> Optional[dict[str, Any]]:
        """Fetch complete ticker info from yfinance (blocking)."""
        if not self._limiter.acquire_sync():
            logger.warning(f"Rate limit timeout for {symbol}")
            return None
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            
            if not info or not info.get("symbol"):
                return None
            
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
                
                # Risk
                "beta": _safe_float(info.get("beta")),
                "short_ratio": _safe_float(info.get("shortRatio")),
                "short_percent_of_float": _safe_float(info.get("shortPercentOfFloat")),
                "held_percent_insiders": _safe_float(info.get("heldPercentInsiders")),
                "held_percent_institutions": _safe_float(info.get("heldPercentInstitutions")),
                
                # Metadata
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.warning(f"yfinance ticker info failed for {symbol}: {e}")
            return None
    
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
            # yfinance search is not directly available, use Ticker with fuzzy matching
            # For now, we rely on cached search results in DB
            # This is a placeholder - actual search uses yfinance's search endpoint
            results = []
            
            # Try direct ticker lookup first
            ticker = yf.Ticker(query.upper())
            info = ticker.info
            if info and info.get("symbol"):
                quote_type = (info.get("quoteType") or "EQUITY").upper()
                results.append({
                    "symbol": info["symbol"],
                    "name": info.get("shortName") or info.get("longName"),
                    "exchange": info.get("exchange"),
                    "quote_type": quote_type,
                    "sector": None if _is_etf_or_index(info["symbol"], quote_type) else info.get("sector"),
                    "market_cap": _safe_int(info.get("marketCap")),
                })
            
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
            db_row = await fetch_one(
                """
                SELECT data FROM yfinance_info_cache 
                WHERE symbol = $1 AND expires_at > NOW()
                """,
                symbol,
            )
            if db_row and db_row.get("data"):
                logger.debug(f"Cache hit L3 (db) for {symbol}")
                data = db_row["data"]
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
            await execute(
                """
                INSERT INTO yfinance_info_cache (symbol, data, expires_at, fetched_at)
                VALUES ($1, $2, NOW() + INTERVAL '24 hours', NOW())
                ON CONFLICT (symbol) DO UPDATE SET
                    data = EXCLUDED.data,
                    expires_at = EXCLUDED.expires_at,
                    fetched_at = EXCLUDED.fetched_at
                """,
                symbol,
                json.dumps(data, default=str),
            )
        
        return data
    
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
            for r in results:
                try:
                    await execute(
                        """
                        INSERT INTO symbol_search_results 
                            (symbol, name, exchange, quote_type, sector, market_cap, expires_at, search_query)
                        VALUES ($1, $2, $3, $4, $5, $6, NOW() + INTERVAL '30 days', $7)
                        ON CONFLICT (symbol) DO UPDATE SET
                            name = COALESCE(EXCLUDED.name, symbol_search_results.name),
                            exchange = COALESCE(EXCLUDED.exchange, symbol_search_results.exchange),
                            quote_type = COALESCE(EXCLUDED.quote_type, symbol_search_results.quote_type),
                            sector = COALESCE(EXCLUDED.sector, symbol_search_results.sector),
                            market_cap = COALESCE(EXCLUDED.market_cap, symbol_search_results.market_cap),
                            expires_at = EXCLUDED.expires_at
                        """,
                        r["symbol"],
                        r.get("name"),
                        r.get("exchange"),
                        r.get("quote_type"),
                        r.get("sector"),
                        r.get("market_cap"),
                        query,
                    )
                except Exception as e:
                    logger.debug(f"Failed to cache search result {r.get('symbol')}: {e}")
        
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
        row = await fetch_one(
            """
            SELECT version_hash, updated_at, version_metadata
            FROM data_versions
            WHERE symbol = $1 AND source = $2
            """,
            symbol.upper(),
            source,
        )
        
        if not row:
            return None
        
        return DataVersion(
            hash=row["version_hash"],
            timestamp=row["updated_at"],
            source=source,
            metadata=row.get("version_metadata") or {},
        )
    
    async def save_data_version(
        self,
        symbol: str,
        version: DataVersion,
    ) -> None:
        """Save data version for change tracking."""
        await execute(
            """
            INSERT INTO data_versions (symbol, source, version_hash, version_metadata, updated_at)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (symbol, source) DO UPDATE SET
                version_hash = EXCLUDED.version_hash,
                version_metadata = EXCLUDED.version_metadata,
                updated_at = NOW()
            """,
            symbol.upper(),
            version.source,
            version.hash,
            json.dumps(version.metadata, default=str),
        )
    
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
        rows = await fetch_all(
            """
            SELECT s.symbol
            FROM symbols s
            LEFT JOIN data_versions dv ON s.symbol = dv.symbol AND dv.source = $1
            WHERE s.is_active = TRUE
              AND (dv.updated_at IS NULL OR dv.updated_at < NOW() - INTERVAL '%s hours')
            ORDER BY dv.updated_at ASC NULLS FIRST
            """ % max_age_hours,
            source,
        )
        return [r["symbol"] for r in rows]


# Singleton instance
_instance: Optional[YFinanceService] = None


def get_yfinance_service() -> YFinanceService:
    """Get singleton YFinanceService instance."""
    global _instance
    if _instance is None:
        _instance = YFinanceService()
    return _instance
