"""DipFinder service with price provider and caching.

MIGRATED: Now uses unified YFinanceService for yfinance calls.
Orchestrates the full signal computation pipeline with:
- Price data fetching/caching
- Signal caching
- Database persistence
- Background job integration
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd

from app.cache.cache import Cache
from app.core.logging import get_logger
from app.repositories import price_history_orm as price_history_repo
from app.repositories import dipfinder_orm as dipfinder_repo
from app.services.data_providers import get_yfinance_service

from .config import DipFinderConfig, get_dipfinder_config
from .dip import DipMetrics
from .fundamentals import compute_quality_score, fetch_stock_info, QualityMetrics
from .stability import compute_stability_score, StabilityMetrics
from .signal import compute_signal, DipSignal

logger = get_logger("dipfinder.service")


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
    """Price provider using unified YFinanceService with caching."""

    def __init__(self, config: Optional[DipFinderConfig] = None):
        """Initialize provider with optional config."""
        self.config = config or get_dipfinder_config()
        self._cache = Cache(prefix="prices", default_ttl=self.config.price_cache_ttl)
        self._service = get_yfinance_service()

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

        # Check local cache first
        for ticker in tickers:
            cache_key = f"{ticker}:{start_date}:{end_date}"
            cached = await self._cache.get(cache_key)
            if cached is not None:
                # Reconstruct DataFrame from cached dict
                try:
                    index_dates = cached.pop("_index_dates", None)
                    df = pd.DataFrame(cached)
                    if index_dates is not None:
                        df.index = pd.to_datetime(index_dates)
                    else:
                        df.index = pd.to_datetime(df.index)
                    results[ticker] = df
                except Exception:
                    tickers_to_fetch.append(ticker)
            else:
                tickers_to_fetch.append(ticker)

        if not tickers_to_fetch:
            return results

        # Use unified service for batch download
        batch_results = await self._service.get_price_history_batch(
            tickers_to_fetch, start_date, end_date
        )

        # Process results and cache
        for ticker, (df, version) in batch_results.items():
            if df is not None and not df.empty:
                results[ticker] = df

                # Cache the result
                cache_key = f"{ticker}:{start_date}:{end_date}"
                df_for_cache = df.copy()
                df_for_cache.index = df_for_cache.index.strftime("%Y-%m-%d")
                cache_data = df_for_cache.to_dict()
                cache_data["_index_dates"] = list(df_for_cache.index)
                await self._cache.set(
                    cache_key, cache_data, ttl=self.config.price_cache_ttl
                )

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
        df = await price_history_repo.get_prices_as_dataframe(
            ticker.upper(),
            start_date,
            end_date,
        )

        if df is not None and not df.empty:
            expected_days = (end_date - start_date).days * 0.7
            coverage_ok = len(df) >= expected_days * 0.8

            last_idx = df.index.max()
            last_date = last_idx.date() if hasattr(last_idx, "date") else last_idx

            if coverage_ok and last_date and last_date >= (end_date - timedelta(days=1)):
                return df

            if last_date and last_date < end_date:
                # Use at least 5-day window to avoid holiday no-data errors
                fetch_start = last_date + timedelta(days=1)
                min_fetch_days = 5
                if (end_date - fetch_start).days < min_fetch_days:
                    fetch_start = end_date - timedelta(days=min_fetch_days)
                if fetch_start <= end_date:
                    yf_df = await self._yf_provider.get_prices(ticker, fetch_start, end_date)
                    if yf_df is not None and not yf_df.empty:
                        await self._save_prices_to_db(ticker, yf_df)
                        merged = pd.concat(
                            [df.copy(), yf_df.copy()],
                            axis=0,
                        )
                        merged.index = pd.to_datetime(merged.index).date
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                        return merged

        # Fallback to yfinance (full range)
        yf_df = await self._yf_provider.get_prices(ticker, start_date, end_date)

        if yf_df is not None and not yf_df.empty:
            await self._save_prices_to_db(ticker, yf_df)
            return yf_df

        return df

    async def get_prices_batch(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, pd.DataFrame]:
        """Get prices for multiple tickers."""
        results: Dict[str, pd.DataFrame] = {}

        for ticker in tickers:
            df = await self.get_prices(ticker, start_date, end_date)
            if df is not None and not df.empty:
                results[ticker] = df

        return results

    async def _save_prices_to_db(self, ticker: str, df: pd.DataFrame) -> None:
        """Save price data to database."""
        try:
            count = await price_history_repo.save_prices(ticker, df)
            logger.debug(f"Cached {count} price records for {ticker}")
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
        rows = await dipfinder_repo.get_latest_signals(
            limit=limit,
            min_final_score=min_final_score,
            only_alerts=only_alerts,
        )

        return [self._row_to_signal(self._orm_to_dict(r)) for r in rows if r]

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
        rows = await dipfinder_repo.get_dip_history(ticker, days)

        return [
            {
                "ticker": r.ticker,
                "event_type": r.event_type,
                "window_days": r.window_days,
                "dip_pct": float(r.dip_pct) if r.dip_pct else None,
                "final_score": float(r.final_score) if r.final_score else None,
                "dip_class": r.dip_class,
                "recorded_at": r.recorded_at,
            }
            for r in rows
        ]

    def _orm_to_dict(self, obj) -> Dict[str, Any]:
        """Convert ORM object to dict for signal deserialization."""
        return {
            "ticker": obj.ticker,
            "benchmark": obj.benchmark,
            "window": obj.window_days,
            "window_days": obj.window_days,
            "as_of_date": obj.as_of_date,
            "dip_stock": float(obj.dip_stock) if obj.dip_stock else 0,
            "peak_stock": float(obj.peak_stock) if obj.peak_stock else 0,
            "dip_pctl": float(obj.dip_pctl) if obj.dip_pctl else 0,
            "dip_vs_typical": float(obj.dip_vs_typical) if obj.dip_vs_typical else 0,
            "persist_days": obj.persist_days or 0,
            "dip_mkt": float(obj.dip_mkt) if obj.dip_mkt else 0,
            "excess_dip": float(obj.excess_dip) if obj.excess_dip else 0,
            "dip_class": obj.dip_class,
            "quality_score": float(obj.quality_score) if obj.quality_score else 0,
            "stability_score": float(obj.stability_score) if obj.stability_score else 0,
            "dip_score": float(obj.dip_score) if obj.dip_score else 0,
            "final_score": float(obj.final_score) if obj.final_score else 0,
            "alert_level": obj.alert_level,
            "should_alert": obj.should_alert,
            "reason": obj.reason,
            "quality_factors": obj.quality_factors,
            "stability_factors": obj.stability_factors,
        }

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
            expires_at = datetime.now(timezone.utc) + timedelta(days=7)
            
            # Convert as_of_date string to date object if needed
            as_of_date = signal.as_of_date
            if isinstance(as_of_date, str):
                as_of_date = date.fromisoformat(as_of_date)

            # Fetch ATH-based dip values from dip_state (source of truth)
            dip_state = await dipfinder_repo.get_dip_state(signal.ticker)

            # Use ATH-based values if available, otherwise fall back to computed values
            if dip_state:
                peak_stock = float(dip_state.ath_price)
                dip_stock = float(dip_state.dip_percentage) / 100.0  # Convert to fraction
                persist_days = (date.today() - dip_state.dip_start_date).days if dip_state.dip_start_date else 0
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

            await dipfinder_repo.save_signal(
                ticker=signal.ticker,
                benchmark=signal.benchmark,
                window_days=signal.window,
                as_of_date=as_of_date,
                dip_stock=float(dip_stock),
                peak_stock=float(peak_stock),
                dip_pctl=float(signal.dip_metrics.dip_percentile),
                dip_vs_typical=float(signal.dip_metrics.dip_vs_typical),
                persist_days=int(persist_days),
                dip_mkt=float(signal.market_context.dip_mkt),
                excess_dip=float(signal.market_context.excess_dip),
                dip_class=signal.market_context.dip_class.value,
                quality_score=float(signal.quality_metrics.score),
                stability_score=float(signal.stability_metrics.score),
                dip_score=float(dip_score),
                final_score=float(final_score),
                alert_level=signal.alert_level.value,
                should_alert=bool(signal.should_alert),
                reason=signal.reason,
                quality_factors=signal.quality_metrics.to_dict(),
                stability_factors=signal.stability_metrics.to_dict(),
                expires_at=expires_at,
            )

            # Check if this is a state change that should be logged
            await self._check_and_log_history(signal)

        except Exception as e:
            logger.warning(f"Failed to save signal for {signal.ticker}: {e}")

    async def _check_and_log_history(self, signal: DipSignal) -> None:
        """Check for dip state changes and log to history."""
        try:
            # Convert as_of_date string to date object if needed
            as_of_date = signal.as_of_date
            if isinstance(as_of_date, str):
                as_of_date = date.fromisoformat(as_of_date)
            
            # Get previous signal for this ticker/window
            prev = await dipfinder_repo.get_previous_signal(
                signal.ticker,
                signal.window,
                as_of_date,
            )

            event_type = None

            if prev is None and signal.dip_metrics.is_meaningful:
                event_type = "entered_dip"
            elif prev is not None:
                prev_dip_stock = float(prev.dip_stock) if prev.dip_stock else 0
                was_meaningful = prev_dip_stock >= self.config.min_dip_abs
                is_meaningful = signal.dip_metrics.is_meaningful

                if not was_meaningful and is_meaningful:
                    event_type = "entered_dip"
                elif was_meaningful and not is_meaningful:
                    event_type = "exited_dip"
                elif (
                    is_meaningful
                    and signal.dip_metrics.dip_pct > prev_dip_stock * 1.1
                ):
                    event_type = "deepened"
                elif (
                    is_meaningful
                    and signal.dip_metrics.dip_pct < prev_dip_stock * 0.9
                ):
                    event_type = "recovered"

            if signal.should_alert and (
                prev is None or not prev.should_alert
            ):
                event_type = "alert_triggered"

            if event_type:
                await dipfinder_repo.log_history_event(
                    ticker=signal.ticker,
                    event_type=event_type,
                    window_days=signal.window,
                    dip_pct=signal.dip_metrics.dip_pct,
                    final_score=signal.final_score,
                    dip_class=signal.market_context.dip_class.value,
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
