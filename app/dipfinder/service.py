"""DipFinder service with price provider and caching.

Orchestrates the full signal computation pipeline with:
- Price data fetching/caching
- Rate-limited yfinance access
- Signal caching
- Database persistence
- Background job integration
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
import yfinance as yf

from app.cache.cache import Cache
from app.core.logging import get_logger
from app.database.connection import fetch_all, fetch_one, execute, execute_many

from .config import DipFinderConfig, get_dipfinder_config
from .dip import DipMetrics
from .fundamentals import compute_quality_score, fetch_stock_info, QualityMetrics
from .stability import compute_stability_score, StabilityMetrics
from .signal import compute_signal, DipSignal

logger = get_logger("dipfinder.service")

# Thread pool for yfinance calls
_executor = ThreadPoolExecutor(max_workers=4)

# Rate limiting for yfinance downloads
_last_download_time: float = 0.0
_download_lock = asyncio.Lock()


class PriceProvider(Protocol):
    """Protocol for price data providers."""

    async def get_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Get price data for a ticker."""
        ...

    async def get_prices_batch(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.DataFrame]:
        """Get price data for multiple tickers."""
        ...


class YFinancePriceProvider:
    """Price provider using yfinance with caching and rate limiting."""

    def __init__(self, config: Optional[DipFinderConfig] = None):
        """Initialize provider with optional config."""
        self.config = config or get_dipfinder_config()
        self._cache = Cache(prefix="prices", default_ttl=self.config.price_cache_ttl)

    def _normalize_yf_dataframe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Normalize yfinance DataFrame to handle MultiIndex columns."""
        if df.empty:
            return df

        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            # Get the ticker-specific columns
            ticker_upper = ticker.upper()
            try:
                # For single ticker downloads, columns are like ('Close', 'AAPL')
                if ticker_upper in df.columns.get_level_values(1):
                    df = df.xs(ticker_upper, axis=1, level=1)
                elif ticker_upper in df.columns.get_level_values(0):
                    df = df[ticker_upper]
            except Exception:
                # Fallback: droplevel if we can
                try:
                    df.columns = df.columns.droplevel(1)
                except Exception:
                    pass

        # Ensure we have proper column names
        expected_cols = {"Close", "Open", "High", "Low", "Volume"}
        if not any(col in df.columns for col in expected_cols):
            # Try to find columns with these names at any level
            for col in list(df.columns):
                if isinstance(col, tuple) and len(col) > 0:
                    df = df.rename(columns={col: col[0]})

        return df

    def _download_prices_sync(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Synchronously download prices (for thread pool)."""
        try:
            if len(tickers) == 1:
                df = yf.download(
                    tickers[0],
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    timeout=30,
                )
            else:
                df = yf.download(
                    " ".join(tickers),
                    start=start,
                    end=end,
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    timeout=30,
                )
            return df
        except Exception as e:
            logger.warning(f"yfinance download failed for {tickers}: {e}")
            return pd.DataFrame()

    async def _rate_limited_download(
        self,
        tickers: List[str],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Download with rate limiting."""
        global _last_download_time

        async with _download_lock:
            elapsed = time.time() - _last_download_time
            if elapsed < self.config.yf_batch_delay:
                await asyncio.sleep(self.config.yf_batch_delay - elapsed)
            _last_download_time = time.time()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._download_prices_sync,
            tickers,
            start,
            end,
        )

    async def get_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Get price data for a single ticker."""
        result = await self.get_prices_batch([ticker], start_date, end_date)
        return result.get(ticker)

    async def get_prices_batch(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.DataFrame]:
        """Get price data for multiple tickers with batching."""
        results: Dict[str, pd.DataFrame] = {}
        tickers_to_fetch: List[str] = []

        # Check cache first
        for ticker in tickers:
            cache_key = f"{ticker}:{start_date}:{end_date}"
            cached = await self._cache.get(cache_key)
            if cached is not None:
                # Reconstruct DataFrame from cached dict
                try:
                    df = pd.DataFrame(cached)
                    df.index = pd.to_datetime(df.index)
                    results[ticker] = df
                except Exception:
                    tickers_to_fetch.append(ticker)
            else:
                tickers_to_fetch.append(ticker)

        if not tickers_to_fetch:
            return results

        # Batch download
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        # Process in batches to respect rate limits
        batch_size = self.config.yf_batch_size
        for i in range(0, len(tickers_to_fetch), batch_size):
            batch = tickers_to_fetch[i : i + batch_size]
            df = await self._rate_limited_download(batch, start_str, end_str)

            if df.empty:
                continue

            # Extract individual ticker data
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_df = self._normalize_yf_dataframe(df, ticker)
                    elif isinstance(df.columns, pd.MultiIndex):
                        if ticker in df.columns.get_level_values(0):
                            ticker_df = df[ticker]
                        elif ticker.upper() in df.columns.get_level_values(1):
                            # New yfinance format: ('Price', 'TICKER')
                            ticker_df = df.xs(ticker.upper(), axis=1, level=1)
                        else:
                            continue
                    else:
                        ticker_df = df

                    if not ticker_df.empty:
                        results[ticker] = ticker_df

                        # Cache the result
                        cache_key = f"{ticker}:{start_date}:{end_date}"
                        cache_data = ticker_df.to_dict()
                        # Convert index to strings for JSON
                        cache_data["index"] = [str(idx) for idx in ticker_df.index]
                        await self._cache.set(
                            cache_key, cache_data, ttl=self.config.price_cache_ttl
                        )

                except Exception as e:
                    logger.warning(f"Failed to extract data for {ticker}: {e}")

        return results


class DatabasePriceProvider:
    """Price provider using cached database prices.

    Falls back to yfinance if not in database.
    """

    def __init__(self, config: Optional[DipFinderConfig] = None):
        """Initialize with optional config."""
        self.config = config or get_dipfinder_config()
        self._yf_provider = YFinancePriceProvider(config)

    async def get_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """Get price data from database or yfinance."""
        # Try database first
        rows = await fetch_all(
            """
            SELECT date, open, high, low, close, adj_close, volume
            FROM price_history
            WHERE symbol = $1 AND date >= $2 AND date <= $3
            ORDER BY date ASC
            """,
            ticker.upper(),
            start_date,
            end_date,
        )

        if rows and len(rows) > 0:
            df = pd.DataFrame([dict(r) for r in rows])
            df.set_index("date", inplace=True)
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "adj_close": "Adj Close",
                    "volume": "Volume",
                },
                inplace=True,
            )

            # Check if we have enough data
            expected_days = (
                end_date - start_date
            ).days * 0.7  # Approximate trading days
            if len(df) >= expected_days * 0.8:  # Allow 20% missing
                return df

        # Fallback to yfinance
        yf_df = await self._yf_provider.get_prices(ticker, start_date, end_date)

        # Cache in database
        if yf_df is not None and not yf_df.empty:
            await self._save_prices_to_db(ticker, yf_df)

        return yf_df

    async def get_prices_batch(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.DataFrame]:
        """Get prices for multiple tickers."""
        results: Dict[str, pd.DataFrame] = {}
        tickers_to_fetch: List[str] = []

        for ticker in tickers:
            df = await self.get_prices(ticker, start_date, end_date)
            if df is not None and not df.empty:
                results[ticker] = df
            else:
                tickers_to_fetch.append(ticker)

        return results

    async def _save_prices_to_db(self, ticker: str, df: pd.DataFrame) -> None:
        """Save price data to database."""
        try:
            rows = []
            for idx, row in df.iterrows():
                rows.append(
                    (
                        ticker.upper(),
                        idx.date() if hasattr(idx, "date") else idx,
                        float(row.get("Open", 0))
                        if pd.notna(row.get("Open"))
                        else None,
                        float(row.get("High", 0))
                        if pd.notna(row.get("High"))
                        else None,
                        float(row.get("Low", 0)) if pd.notna(row.get("Low")) else None,
                        float(row["Close"]) if pd.notna(row["Close"]) else None,
                        float(row.get("Adj Close", row["Close"]))
                        if pd.notna(row.get("Adj Close", row["Close"]))
                        else None,
                        int(row.get("Volume", 0))
                        if pd.notna(row.get("Volume"))
                        else None,
                    )
                )

            if rows:
                await execute_many(
                    """
                    INSERT INTO price_history (symbol, date, open, high, low, close, adj_close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume = EXCLUDED.volume
                    """,
                    rows,
                )
                logger.debug(f"Cached {len(rows)} price records for {ticker}")
        except Exception as e:
            logger.warning(f"Failed to cache prices for {ticker}: {e}")


class DipFinderService:
    """Main service for DipFinder signal computation."""

    def __init__(
        self,
        config: Optional[DipFinderConfig] = None,
        price_provider: Optional[PriceProvider] = None,
    ):
        """Initialize service with optional config and price provider."""
        self.config = config or get_dipfinder_config()
        self.price_provider = price_provider or DatabasePriceProvider(self.config)
        self._signal_cache = Cache(
            prefix="dipfinder", default_ttl=self.config.signal_cache_ttl
        )

    async def get_signal(
        self,
        ticker: str,
        benchmark: Optional[str] = None,
        window: Optional[int] = None,
        force_refresh: bool = False,
    ) -> Optional[DipSignal]:
        """
        Get dip signal for a single ticker.

        Args:
            ticker: Stock ticker symbol
            benchmark: Benchmark ticker (default from config)
            window: Window for dip calculation (default from config)
            force_refresh: If True, bypass cache

        Returns:
            DipSignal or None if computation fails
        """
        ticker = ticker.upper()
        benchmark = benchmark or self.config.default_benchmark
        window = window or self.config.windows[1]  # Default to 30-day
        today = date.today()

        # Check cache
        if not force_refresh:
            cache_key = f"{ticker}:{benchmark}:{window}:{today}"
            cached = await self._signal_cache.get(cache_key)
            if cached:
                return self._deserialize_signal(cached)

        # Compute signal
        signal = await self._compute_signal(ticker, benchmark, window, today)

        if signal:
            # Cache result
            cache_key = f"{ticker}:{benchmark}:{window}:{today}"
            await self._signal_cache.set(cache_key, signal.to_dict())

            # Save to database
            await self._save_signal_to_db(signal)

        return signal

    async def get_signals(
        self,
        tickers: List[str],
        benchmark: Optional[str] = None,
        window: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[DipSignal]:
        """
        Get dip signals for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            benchmark: Benchmark ticker
            window: Window for dip calculation
            force_refresh: If True, bypass cache

        Returns:
            List of DipSignals (excluding failures)
        """
        signals = []

        for ticker in tickers:
            try:
                signal = await self.get_signal(ticker, benchmark, window, force_refresh)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Failed to compute signal for {ticker}: {e}")

        return signals

    async def get_latest_signals(
        self,
        limit: int = 50,
        min_final_score: Optional[float] = None,
        only_alerts: bool = False,
    ) -> List[DipSignal]:
        """
        Get latest computed signals from database.

        Args:
            limit: Maximum number of signals to return
            min_final_score: Minimum final score filter
            only_alerts: If True, only return alerts

        Returns:
            List of DipSignals
        """
        conditions = ["1=1"]
        params = []
        param_idx = 1

        if min_final_score is not None:
            conditions.append(f"final_score >= ${param_idx}")
            params.append(min_final_score)
            param_idx += 1

        if only_alerts:
            conditions.append("should_alert = TRUE")

        where_clause = " AND ".join(conditions)

        rows = await fetch_all(
            f"""
            SELECT * FROM dipfinder_latest_signals
            WHERE {where_clause}
            ORDER BY final_score DESC
            LIMIT ${param_idx}
            """,
            *params,
            limit,
        )

        return [self._row_to_signal(dict(r)) for r in rows if r]

    async def get_dip_history(
        self,
        ticker: str,
        days: int = 90,
    ) -> List[Dict[str, Any]]:
        """
        Get dip history for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of days of history

        Returns:
            List of history records
        """
        since = datetime.utcnow() - timedelta(days=days)

        rows = await fetch_all(
            """
            SELECT * FROM dipfinder_history
            WHERE ticker = $1 AND recorded_at >= $2
            ORDER BY recorded_at DESC
            """,
            ticker.upper(),
            since,
        )

        return [dict(r) for r in rows]

    async def _compute_signal(
        self,
        ticker: str,
        benchmark: str,
        window: int,
        as_of_date: date,
    ) -> Optional[DipSignal]:
        """Compute signal for a ticker."""
        # Calculate date range
        history_days = self.config.history_years * 365
        start_date = as_of_date - timedelta(days=history_days)

        # Fetch prices
        prices = await self.price_provider.get_prices_batch(
            [ticker, benchmark],
            start_date,
            as_of_date,
        )

        stock_df = prices.get(ticker)
        benchmark_df = prices.get(benchmark)

        if stock_df is None or stock_df.empty:
            logger.warning(f"No price data for {ticker}")
            return None

        if benchmark_df is None or benchmark_df.empty:
            logger.warning(f"No price data for benchmark {benchmark}")
            return None

        # Extract close prices as numpy arrays
        stock_prices = stock_df["Close"].dropna().to_numpy()
        benchmark_prices = benchmark_df["Close"].dropna().to_numpy()

        if len(stock_prices) < window:
            logger.warning(
                f"Insufficient price data for {ticker}: {len(stock_prices)} < {window}"
            )
            return None

        # Fetch yfinance info for quality/stability
        info = await fetch_stock_info(ticker, self.config)

        # Compute quality metrics
        quality = await compute_quality_score(ticker, info, self.config)

        # Compute stability metrics
        stability = compute_stability_score(ticker, stock_prices, info, self.config)

        # Compute full signal
        signal = compute_signal(
            ticker=ticker,
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker=benchmark,
            window=window,
            quality_metrics=quality,
            stability_metrics=stability,
            as_of_date=as_of_date.isoformat(),
            config=self.config,
        )

        return signal

    async def _save_signal_to_db(self, signal: DipSignal) -> None:
        """Save signal to database."""
        try:
            expires_at = datetime.utcnow() + timedelta(days=7)
            
            # Convert as_of_date string to date object if needed
            as_of_date = signal.as_of_date
            if isinstance(as_of_date, str):
                as_of_date = date.fromisoformat(as_of_date)

            # Fetch ATH-based dip values from dip_state (source of truth)
            dip_state = await fetch_one(
                """
                SELECT ath_price, dip_percentage, dip_start_date
                FROM dip_state
                WHERE symbol = $1
                """,
                signal.ticker,
            )

            # Use ATH-based values if available, otherwise fall back to computed values
            if dip_state:
                peak_stock = float(dip_state["ath_price"])
                dip_stock = float(dip_state["dip_percentage"]) / 100.0  # Convert to fraction
                persist_days = (date.today() - dip_state["dip_start_date"]).days if dip_state["dip_start_date"] else 0
                # Recalculate dip_score based on ATH dip
                dip_score = min(100.0, dip_stock * 100 * 5)
                # Recalculate final_score
                final_score = (signal.quality_metrics.score + signal.stability_metrics.score + dip_score) / 3
            else:
                # Fallback to computed values
                peak_stock = signal.dip_metrics.peak_price
                dip_stock = signal.dip_metrics.dip_pct
                persist_days = signal.dip_metrics.persist_days
                dip_score = signal.dip_score
                final_score = signal.final_score

            await execute(
                """
                INSERT INTO dipfinder_signals (
                    ticker, benchmark, window_days, as_of_date,
                    dip_stock, peak_stock, dip_pctl, dip_vs_typical, persist_days,
                    dip_mkt, excess_dip, dip_class,
                    quality_score, stability_score, dip_score, final_score,
                    alert_level, should_alert, reason,
                    quality_factors, stability_factors,
                    expires_at
                ) VALUES (
                    $1, $2, $3, $4,
                    $5, $6, $7, $8, $9,
                    $10, $11, $12,
                    $13, $14, $15, $16,
                    $17, $18, $19,
                    $20, $21,
                    $22
                )
                ON CONFLICT (ticker, benchmark, window_days, as_of_date) DO UPDATE SET
                    dip_stock = EXCLUDED.dip_stock,
                    peak_stock = EXCLUDED.peak_stock,
                    dip_pctl = EXCLUDED.dip_pctl,
                    dip_vs_typical = EXCLUDED.dip_vs_typical,
                    persist_days = EXCLUDED.persist_days,
                    dip_mkt = EXCLUDED.dip_mkt,
                    excess_dip = EXCLUDED.excess_dip,
                    dip_class = EXCLUDED.dip_class,
                    quality_score = EXCLUDED.quality_score,
                    stability_score = EXCLUDED.stability_score,
                    dip_score = EXCLUDED.dip_score,
                    final_score = EXCLUDED.final_score,
                    alert_level = EXCLUDED.alert_level,
                    should_alert = EXCLUDED.should_alert,
                    reason = EXCLUDED.reason,
                    quality_factors = EXCLUDED.quality_factors,
                    stability_factors = EXCLUDED.stability_factors,
                    expires_at = EXCLUDED.expires_at
                """,
                signal.ticker,
                signal.benchmark,
                signal.window,
                as_of_date,
                dip_stock,
                peak_stock,
                signal.dip_metrics.dip_percentile,
                signal.dip_metrics.dip_vs_typical,
                persist_days,
                signal.market_context.dip_mkt,
                signal.market_context.excess_dip,
                signal.market_context.dip_class.value,
                signal.quality_metrics.score,
                signal.stability_metrics.score,
                dip_score,
                final_score,
                signal.alert_level.value,
                signal.should_alert,
                signal.reason,
                json.dumps(signal.quality_metrics.to_dict()),
                json.dumps(signal.stability_metrics.to_dict()),
                expires_at,
            )

            # Check if this is a state change that should be logged
            await self._check_and_log_history(signal)

        except Exception as e:
            logger.warning(f"Failed to save signal for {signal.ticker}: {e}")

    async def _check_and_log_history(self, signal: DipSignal) -> None:
        """Check for dip state changes and log to history."""
        try:
            # Get previous signal for this ticker/window
            prev = await fetch_one(
                """
                SELECT should_alert, dip_stock, final_score, dip_class
                FROM dipfinder_signals
                WHERE ticker = $1 AND window_days = $2 AND as_of_date < $3
                ORDER BY as_of_date DESC
                LIMIT 1
                """,
                signal.ticker,
                signal.window,
                signal.as_of_date,
            )

            event_type = None

            if prev is None and signal.dip_metrics.is_meaningful:
                event_type = "entered_dip"
            elif prev is not None:
                was_meaningful = prev["dip_stock"] >= self.config.min_dip_abs
                is_meaningful = signal.dip_metrics.is_meaningful

                if not was_meaningful and is_meaningful:
                    event_type = "entered_dip"
                elif was_meaningful and not is_meaningful:
                    event_type = "exited_dip"
                elif (
                    is_meaningful
                    and signal.dip_metrics.dip_pct > prev["dip_stock"] * 1.1
                ):
                    event_type = "deepened"
                elif (
                    is_meaningful
                    and signal.dip_metrics.dip_pct < prev["dip_stock"] * 0.9
                ):
                    event_type = "recovered"

            if signal.should_alert and (
                prev is None or not prev.get("should_alert", False)
            ):
                event_type = "alert_triggered"

            if event_type:
                await execute(
                    """
                    INSERT INTO dipfinder_history (
                        ticker, event_type, window_days, dip_pct, final_score, dip_class
                    ) VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    signal.ticker,
                    event_type,
                    signal.window,
                    signal.dip_metrics.dip_pct,
                    signal.final_score,
                    signal.market_context.dip_class.value,
                )

        except Exception as e:
            logger.debug(f"Could not log dip history: {e}")

    def _deserialize_signal(self, data: Dict[str, Any]) -> Optional[DipSignal]:
        """Deserialize signal from cache/db format."""
        try:
            from .signal import DipClass, AlertLevel, MarketContext

            # Reconstruct nested objects
            dip_metrics = DipMetrics(
                ticker=data["ticker"],
                window=data["window"],
                dip_pct=data["dip_stock"],
                peak_price=data["peak_stock"],
                current_price=data.get("current_price", 0),
                dip_percentile=data["dip_pctl"],
                dip_vs_typical=data["dip_vs_typical"],
                typical_dip=data.get("typical_dip", 0),
                persist_days=data["persist_days"],
                days_since_peak=data.get("days_since_peak", data["persist_days"]),  # fallback to persist_days
                is_meaningful=data.get("is_meaningful", False),
            )

            market_context = MarketContext(
                benchmark_ticker=data["benchmark"],
                dip_mkt=data["dip_mkt"],
                dip_stock=data["dip_stock"],
                excess_dip=data["excess_dip"],
                dip_class=DipClass(data["dip_class"]),
            )

            quality_factors = data.get("quality_factors", {})
            if isinstance(quality_factors, str):
                quality_factors = json.loads(quality_factors)

            quality_metrics = QualityMetrics(
                ticker=data["ticker"],
                score=data["quality_score"],
                **{
                    k: v
                    for k, v in quality_factors.items()
                    if k not in ("ticker", "score")
                },
            )

            stability_factors = data.get("stability_factors", {})
            if isinstance(stability_factors, str):
                stability_factors = json.loads(stability_factors)

            stability_metrics = StabilityMetrics(
                ticker=data["ticker"],
                score=data["stability_score"],
                **{
                    k: v
                    for k, v in stability_factors.items()
                    if k not in ("ticker", "score")
                },
            )

            return DipSignal(
                ticker=data["ticker"],
                window=data["window"],
                benchmark=data["benchmark"],
                as_of_date=data["as_of_date"],
                dip_metrics=dip_metrics,
                market_context=market_context,
                quality_metrics=quality_metrics,
                stability_metrics=stability_metrics,
                dip_score=data["dip_score"],
                final_score=data["final_score"],
                alert_level=AlertLevel(data["alert_level"]),
                should_alert=data["should_alert"],
                reason=data["reason"],
            )
        except Exception as e:
            logger.warning(f"Failed to deserialize signal: {e}")
            return None

    def _row_to_signal(self, row: Dict[str, Any]) -> Optional[DipSignal]:
        """Convert database row to DipSignal."""
        return self._deserialize_signal(
            {
                "ticker": row["ticker"],
                "window": row["window_days"],
                "benchmark": row["benchmark"],
                "as_of_date": str(row["as_of_date"]),
                "dip_stock": float(row["dip_stock"]) if row["dip_stock"] else 0,
                "peak_stock": float(row["peak_stock"]) if row["peak_stock"] else 0,
                "dip_pctl": float(row["dip_pctl"]) if row["dip_pctl"] else 0,
                "dip_vs_typical": float(row["dip_vs_typical"])
                if row["dip_vs_typical"]
                else 0,
                "persist_days": row["persist_days"] or 0,
                "dip_mkt": float(row["dip_mkt"]) if row["dip_mkt"] else 0,
                "excess_dip": float(row["excess_dip"]) if row["excess_dip"] else 0,
                "dip_class": row["dip_class"],
                "quality_score": float(row["quality_score"])
                if row["quality_score"]
                else 50,
                "stability_score": float(row["stability_score"])
                if row["stability_score"]
                else 50,
                "dip_score": float(row["dip_score"]) if row["dip_score"] else 0,
                "final_score": float(row["final_score"]) if row["final_score"] else 0,
                "alert_level": row["alert_level"],
                "should_alert": row["should_alert"],
                "reason": row["reason"],
                "quality_factors": row.get("quality_factors", {}),
                "stability_factors": row.get("stability_factors", {}),
            }
        )


# Singleton service instance
_service: Optional[DipFinderService] = None


def get_dipfinder_service() -> DipFinderService:
    """Get singleton DipFinder service."""
    global _service
    if _service is None:
        _service = DipFinderService()
    return _service
