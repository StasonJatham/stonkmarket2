"""
Crash Testing & Accumulation Metrics.

This module evaluates strategy performance during historical crashes
and measures accumulation efficiency for bear market buying.

Key Crash Periods:
- 2008 Financial Crisis (Oct 2007 - Mar 2009)
- 2020 COVID Crash (Feb 2020 - Mar 2020)
- 2022 Tech Crash (Jan 2022 - Oct 2022)

For each crash, we measure:
1. How well did the strategy protect capital during the drawdown?
2. How effectively did it accumulate shares at low prices?
3. How quickly did it recover once the market turned?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrashPeriod(str, Enum):
    """Named crash periods for backtesting."""

    FINANCIAL_CRISIS_2008 = "FINANCIAL_CRISIS_2008"
    COVID_CRASH_2020 = "COVID_CRASH_2020"
    TECH_CRASH_2022 = "TECH_CRASH_2022"
    DOT_COM_BUST_2000 = "DOT_COM_BUST_2000"


@dataclass
class CrashDefinition:
    """Definition of a crash period for analysis."""

    period: CrashPeriod
    name: str
    start_date: date
    bottom_date: date
    recovery_date: date | None  # Date when previous high was recovered
    context: str

    # Pre-crash window for comparison
    lookback_months: int = 12


# Historical crash definitions
CRASH_PERIODS: dict[CrashPeriod, CrashDefinition] = {
    CrashPeriod.DOT_COM_BUST_2000: CrashDefinition(
        period=CrashPeriod.DOT_COM_BUST_2000,
        name="Dot-Com Bust",
        start_date=date(2000, 3, 10),  # NASDAQ peak
        bottom_date=date(2002, 10, 9),  # NASDAQ bottom
        recovery_date=date(2015, 3, 2),  # NASDAQ finally recovered
        context="Tech bubble burst after Y2K euphoria",
    ),
    CrashPeriod.FINANCIAL_CRISIS_2008: CrashDefinition(
        period=CrashPeriod.FINANCIAL_CRISIS_2008,
        name="Financial Crisis",
        start_date=date(2007, 10, 9),  # SPY all-time high before crash
        bottom_date=date(2009, 3, 9),  # SPY bottom
        recovery_date=date(2013, 3, 28),  # SPY recovered previous high
        context="Subprime mortgage crisis, Lehman collapse",
    ),
    CrashPeriod.COVID_CRASH_2020: CrashDefinition(
        period=CrashPeriod.COVID_CRASH_2020,
        name="COVID Crash",
        start_date=date(2020, 2, 19),  # SPY pre-COVID high
        bottom_date=date(2020, 3, 23),  # SPY COVID low
        recovery_date=date(2020, 8, 18),  # SPY recovered
        context="Global pandemic, fastest crash in history",
    ),
    CrashPeriod.TECH_CRASH_2022: CrashDefinition(
        period=CrashPeriod.TECH_CRASH_2022,
        name="2022 Tech Crash",
        start_date=date(2022, 1, 3),  # Start of 2022
        bottom_date=date(2022, 10, 12),  # Approximate bottom
        recovery_date=date(2024, 1, 19),  # SPY recovered in early 2024
        context="Inflation, rate hikes, tech valuation reset",
    ),
}


@dataclass
class AccumulationMetrics:
    """Metrics measuring accumulation effectiveness during a crash."""

    # Share accumulation
    shares_accumulated_strategy: float
    shares_accumulated_dca: float
    shares_accumulated_lump_bottom: float  # Perfect timing (theoretical max)

    # Accumulation score: strategy vs DCA (>1.0 means beat DCA)
    accumulation_score: float

    # Cost basis comparison
    avg_cost_strategy: float
    avg_cost_dca: float
    avg_cost_lump_bottom: float
    cost_improvement_vs_dca_pct: float

    # Capital efficiency
    cash_deployed_pct: float  # What % of available cash was deployed
    average_deployment_lag_days: int  # Average days between cash available and deployed


@dataclass
class RecoveryMetrics:
    """Metrics measuring recovery after the crash."""

    # Time to recover
    recovery_days_strategy: int | None
    recovery_days_bh: int | None

    # Return from bottom
    return_from_bottom_strategy: float
    return_from_bottom_bh: float

    # Recovery improvement
    recovery_improvement_days: int | None  # Positive = faster recovery


@dataclass
class DrawdownMetrics:
    """Metrics measuring capital preservation during the crash."""

    # Maximum drawdown
    max_drawdown_strategy: float
    max_drawdown_bh: float
    drawdown_reduction_pct: float

    # Volatility during crash
    daily_vol_strategy: float
    daily_vol_bh: float

    # Did strategy reduce exposure before bottom?
    reduced_exposure_before_bottom: bool
    exposure_at_bottom_pct: float


@dataclass
class CrashTestResult:
    """Complete result of crash testing."""

    crash: CrashDefinition
    symbol: str

    # Strategy performance
    strategy_return: float
    strategy_max_drawdown: float
    strategy_recovery_days: int | None

    # Benchmark performance
    bh_return: float
    bh_max_drawdown: float
    bh_recovery_days: int | None

    # Detailed metrics
    accumulation: AccumulationMetrics | None = None
    recovery: RecoveryMetrics | None = None
    drawdown: DrawdownMetrics | None = None

    # Did strategy outperform during crash?
    outperformed: bool = False
    alpha_vs_bh: float = 0.0

    # Verdict
    verdict: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "crash_name": self.crash.name,
            "crash_period": self.crash.period.value,
            "symbol": self.symbol,
            "strategy_return": self.strategy_return,
            "strategy_max_drawdown": self.strategy_max_drawdown,
            "strategy_recovery_days": self.strategy_recovery_days,
            "bh_return": self.bh_return,
            "bh_max_drawdown": self.bh_max_drawdown,
            "bh_recovery_days": self.bh_recovery_days,
            "outperformed": self.outperformed,
            "alpha_vs_bh": self.alpha_vs_bh,
            "verdict": self.verdict,
            "recommendation": self.recommendation,
            "accumulation_score": self.accumulation.accumulation_score if self.accumulation else None,
            "cost_improvement_pct": self.accumulation.cost_improvement_vs_dca_pct if self.accumulation else None,
            "recovery_improvement_days": self.recovery.recovery_improvement_days if self.recovery else None,
        }


class CrashTester:
    """
    Crash Testing Engine.

    Tests strategy performance during historical crashes with focus on:
    1. Capital preservation (drawdown reduction)
    2. Accumulation efficiency (buying the dip)
    3. Recovery speed (getting back to even)
    """

    def test_crash(
        self,
        prices: pd.Series,
        strategy_trades: list[dict[str, Any]],
        crash_period: CrashPeriod,
        initial_capital: float = 10_000,
        monthly_contribution: float = 1_000,
    ) -> CrashTestResult | None:
        """
        Test strategy performance during a specific crash.

        Args:
            prices: Full price series with DatetimeIndex
            strategy_trades: List of trades from strategy backtest
            crash_period: Which crash to analyze
            initial_capital: Starting capital
            monthly_contribution: Monthly DCA amount

        Returns:
            CrashTestResult or None if crash period not in data
        """
        crash = CRASH_PERIODS[crash_period]

        # Filter prices to crash period + recovery
        crash_start = pd.Timestamp(crash.start_date)
        crash_bottom = pd.Timestamp(crash.bottom_date)

        # Include recovery period if available
        if crash.recovery_date:
            crash_end = pd.Timestamp(crash.recovery_date) + pd.Timedelta(days=90)
        else:
            crash_end = crash_bottom + pd.Timedelta(days=365)

        # Check if we have data for this period
        if prices.index[0] > crash_start or prices.index[-1] < crash_bottom:
            logger.warning(f"Insufficient data for {crash.name}")
            return None

        # Filter to crash period
        mask = (prices.index >= crash_start) & (prices.index <= crash_end)
        crash_prices = prices[mask]

        if len(crash_prices) < 20:
            return None

        # Calculate B&H metrics
        bh_return = (crash_prices.iloc[-1] / crash_prices.iloc[0] - 1) * 100
        bh_max_drawdown = self._calculate_max_drawdown(crash_prices)
        bh_recovery_days = self._calculate_recovery_days(
            crash_prices, crash_prices.iloc[0]
        )

        # Calculate strategy metrics from trades
        strategy_metrics = self._calculate_strategy_metrics(
            crash_prices, strategy_trades, crash_start, crash_end
        )

        # Calculate accumulation metrics
        accumulation = self._calculate_accumulation(
            crash_prices,
            strategy_trades,
            initial_capital,
            monthly_contribution,
            crash_start,
            crash_bottom,
        )

        # Calculate recovery metrics
        recovery = self._calculate_recovery(
            crash_prices,
            strategy_trades,
            crash_bottom,
            bh_recovery_days,
        )

        # Calculate drawdown metrics
        drawdown = self._calculate_drawdown(
            crash_prices,
            strategy_trades,
            crash_bottom,
            bh_max_drawdown,
        )

        # Determine if strategy outperformed
        strategy_return = strategy_metrics.get("return", 0)
        alpha = strategy_return - bh_return
        outperformed = alpha > 0 or strategy_metrics.get("max_drawdown", 100) < bh_max_drawdown

        # Generate verdict
        verdict, recommendation = self._generate_verdict(
            crash,
            strategy_return,
            bh_return,
            strategy_metrics.get("max_drawdown", 0),
            bh_max_drawdown,
            accumulation,
        )

        return CrashTestResult(
            crash=crash,
            symbol=prices.name if hasattr(prices, "name") else "UNKNOWN",
            strategy_return=strategy_return,
            strategy_max_drawdown=strategy_metrics.get("max_drawdown", 0),
            strategy_recovery_days=strategy_metrics.get("recovery_days"),
            bh_return=bh_return,
            bh_max_drawdown=bh_max_drawdown,
            bh_recovery_days=bh_recovery_days,
            accumulation=accumulation,
            recovery=recovery,
            drawdown=drawdown,
            outperformed=outperformed,
            alpha_vs_bh=alpha,
            verdict=verdict,
            recommendation=recommendation,
        )

    def test_all_crashes(
        self,
        prices: pd.Series,
        strategy_trades: list[dict[str, Any]],
        initial_capital: float = 10_000,
        monthly_contribution: float = 1_000,
    ) -> list[CrashTestResult]:
        """Test strategy against all available crash periods."""
        results = []

        for crash_period in CrashPeriod:
            result = self.test_crash(
                prices,
                strategy_trades,
                crash_period,
                initial_capital,
                monthly_contribution,
            )
            if result:
                results.append(result)

        return results

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        rolling_max = prices.cummax()
        drawdown = (prices / rolling_max - 1) * 100
        return abs(float(drawdown.min()))

    def _calculate_recovery_days(
        self,
        prices: pd.Series,
        target_price: float,
    ) -> int | None:
        """Calculate days to recover to target price."""
        recovered = prices >= target_price

        if not recovered.any():
            return None

        first_recovery = recovered.idxmax()
        return (first_recovery - prices.index[0]).days

    def _calculate_strategy_metrics(
        self,
        crash_prices: pd.Series,
        trades: list[dict[str, Any]],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[str, Any]:
        """Calculate strategy metrics during crash period."""
        # Filter trades to crash period
        crash_trades = [
            t for t in trades
            if start <= pd.Timestamp(t.get("timestamp", t.get("date", ""))) <= end
        ]

        # Simulate equity curve from trades
        equity_values = []
        current_value = crash_prices.iloc[0] * 100  # Assume starting position

        for date, price in crash_prices.items():
            # Update with trades
            for t in crash_trades:
                t_date = pd.Timestamp(t.get("timestamp", t.get("date", "")))
                if t_date == date:
                    if t.get("type") == "buy":
                        current_value += t.get("value", 0)
                    elif t.get("type") == "sell":
                        current_value -= t.get("value", 0)

            equity_values.append(current_value)

        if not equity_values:
            return {"return": 0, "max_drawdown": 0}

        equity = pd.Series(equity_values, index=crash_prices.index)
        max_dd = self._calculate_max_drawdown(equity)
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

        return {
            "return": total_return,
            "max_drawdown": max_dd,
            "n_trades": len(crash_trades),
        }

    def _calculate_accumulation(
        self,
        crash_prices: pd.Series,
        trades: list[dict[str, Any]],
        initial_capital: float,
        monthly_contribution: float,
        start: pd.Timestamp,
        bottom: pd.Timestamp,
    ) -> AccumulationMetrics | None:
        """Calculate accumulation metrics during crash drawdown phase."""
        # Drawdown phase: from start to bottom
        mask = (crash_prices.index >= start) & (crash_prices.index <= bottom)
        drawdown_prices = crash_prices[mask]

        if len(drawdown_prices) < 5:
            return None

        # Filter trades during drawdown
        buy_trades = [
            t for t in trades
            if (
                t.get("type") == "buy"
                and start <= pd.Timestamp(t.get("timestamp", t.get("date", ""))) <= bottom
            )
        ]

        # Strategy accumulation
        strategy_shares = sum(t.get("shares", 0) for t in buy_trades)
        strategy_value = sum(t.get("value", 0) for t in buy_trades)
        avg_cost_strategy = strategy_value / strategy_shares if strategy_shares > 0 else 0

        # DCA accumulation (monthly buys)
        months_in_drawdown = (bottom - start).days // 30
        dca_capital = initial_capital + (monthly_contribution * months_in_drawdown)
        avg_price = float(drawdown_prices.mean())
        dca_shares = dca_capital / avg_price
        avg_cost_dca = avg_price

        # Perfect timing (lump sum at bottom)
        bottom_price = float(drawdown_prices.iloc[-1])
        lump_bottom_shares = dca_capital / bottom_price

        # Scores
        accumulation_score = strategy_shares / dca_shares if dca_shares > 0 else 0
        cost_improvement = (avg_cost_dca - avg_cost_strategy) / avg_cost_dca * 100 if avg_cost_dca > 0 else 0

        return AccumulationMetrics(
            shares_accumulated_strategy=strategy_shares,
            shares_accumulated_dca=dca_shares,
            shares_accumulated_lump_bottom=lump_bottom_shares,
            accumulation_score=accumulation_score,
            avg_cost_strategy=avg_cost_strategy,
            avg_cost_dca=avg_cost_dca,
            avg_cost_lump_bottom=bottom_price,
            cost_improvement_vs_dca_pct=cost_improvement,
            cash_deployed_pct=strategy_value / dca_capital * 100 if dca_capital > 0 else 0,
            average_deployment_lag_days=0,  # Would need more detailed tracking
        )

    def _calculate_recovery(
        self,
        crash_prices: pd.Series,
        trades: list[dict[str, Any]],
        bottom: pd.Timestamp,
        bh_recovery_days: int | None,
    ) -> RecoveryMetrics:
        """Calculate recovery metrics after the bottom."""
        # Recovery phase: from bottom onward
        mask = crash_prices.index >= bottom
        recovery_prices = crash_prices[mask]

        if len(recovery_prices) < 5:
            return RecoveryMetrics(
                recovery_days_strategy=None,
                recovery_days_bh=bh_recovery_days,
                return_from_bottom_strategy=0,
                return_from_bottom_bh=0,
                recovery_improvement_days=None,
            )

        # Return from bottom
        return_from_bottom = (recovery_prices.iloc[-1] / recovery_prices.iloc[0] - 1) * 100

        # Recovery days (simplified - assumes same as B&H for now)
        strategy_recovery = bh_recovery_days  # Would need equity curve tracking

        improvement = None
        if bh_recovery_days and strategy_recovery:
            improvement = bh_recovery_days - strategy_recovery

        return RecoveryMetrics(
            recovery_days_strategy=strategy_recovery,
            recovery_days_bh=bh_recovery_days,
            return_from_bottom_strategy=return_from_bottom,
            return_from_bottom_bh=return_from_bottom,
            recovery_improvement_days=improvement,
        )

    def _calculate_drawdown(
        self,
        crash_prices: pd.Series,
        trades: list[dict[str, Any]],
        bottom: pd.Timestamp,
        bh_max_drawdown: float,
    ) -> DrawdownMetrics:
        """Calculate drawdown and volatility metrics."""
        # Daily volatility
        daily_returns = crash_prices.pct_change().dropna()
        daily_vol = float(daily_returns.std() * 100)

        # Check if strategy reduced exposure (had sells before bottom)
        sells_before_bottom = [
            t for t in trades
            if t.get("type") == "sell" and pd.Timestamp(t.get("timestamp", t.get("date", ""))) < bottom
        ]
        reduced_exposure = len(sells_before_bottom) > 0

        return DrawdownMetrics(
            max_drawdown_strategy=bh_max_drawdown,  # Simplified
            max_drawdown_bh=bh_max_drawdown,
            drawdown_reduction_pct=0,  # Would need equity curve
            daily_vol_strategy=daily_vol,
            daily_vol_bh=daily_vol,
            reduced_exposure_before_bottom=reduced_exposure,
            exposure_at_bottom_pct=100,  # Simplified
        )

    def _generate_verdict(
        self,
        crash: CrashDefinition,
        strategy_return: float,
        bh_return: float,
        strategy_dd: float,
        bh_dd: float,
        accumulation: AccumulationMetrics | None,
    ) -> tuple[str, str]:
        """Generate human-readable verdict and recommendation."""
        verdicts = []
        recommendations = []

        alpha = strategy_return - bh_return

        # Return comparison
        if alpha > 5:
            verdicts.append(f"Outperformed B&H by {alpha:.1f}%")
        elif alpha > 0:
            verdicts.append(f"Slightly beat B&H (+{alpha:.1f}%)")
        elif alpha > -5:
            verdicts.append(f"Matched B&H ({alpha:+.1f}%)")
        else:
            verdicts.append(f"Underperformed B&H ({alpha:+.1f}%)")

        # Drawdown comparison
        dd_improvement = bh_dd - strategy_dd
        if dd_improvement > 10:
            verdicts.append(f"Protected capital ({dd_improvement:.0f}% less drawdown)")
        elif dd_improvement < -10:
            verdicts.append(f"More volatile ({-dd_improvement:.0f}% more drawdown)")

        # Accumulation
        if accumulation and accumulation.accumulation_score > 1.1:
            verdicts.append(f"Strong accumulation (score: {accumulation.accumulation_score:.2f})")
            recommendations.append("Strategy effectively bought the dip")
        elif accumulation and accumulation.accumulation_score < 0.9:
            recommendations.append("Consider more aggressive accumulation in bear markets")

        # Overall recommendation
        if alpha > 0 and (not accumulation or accumulation.accumulation_score >= 1.0):
            recommendations.append(f"Strategy performed well during {crash.name}")
        else:
            recommendations.append(f"Review strategy parameters for better crash performance")

        return "; ".join(verdicts), "; ".join(recommendations)


def get_crash_summary(results: list[CrashTestResult]) -> dict[str, Any]:
    """Summarize crash test results across all periods."""
    if not results:
        return {"n_crashes_tested": 0}

    avg_alpha = np.mean([r.alpha_vs_bh for r in results])
    avg_accumulation = np.mean([
        r.accumulation.accumulation_score
        for r in results
        if r.accumulation
    ]) if any(r.accumulation for r in results) else 0

    outperformed_count = sum(1 for r in results if r.outperformed)

    return {
        "n_crashes_tested": len(results),
        "avg_alpha_vs_bh": avg_alpha,
        "avg_accumulation_score": avg_accumulation,
        "crashes_outperformed": outperformed_count,
        "pct_crashes_outperformed": outperformed_count / len(results) * 100,
        "worst_crash": min(results, key=lambda r: r.alpha_vs_bh).crash.name,
        "best_crash": max(results, key=lambda r: r.alpha_vs_bh).crash.name,
    }


# =============================================================================
# Price Data Fetching (DB first, yfinance fallback)
# =============================================================================

async def get_prices_for_crash_testing(
    symbol: str,
    crash_period: CrashPeriod | None = None,
) -> pd.Series | None:
    """
    Fetch price data for crash testing from DB (with yfinance fallback).
    
    This uses the unified PriceService which:
    1. First checks the database for existing prices
    2. Falls back to yfinance only if data is missing
    3. Validates and saves fetched data to DB for future use
    
    Args:
        symbol: Stock ticker symbol
        crash_period: Optional specific crash period to fetch.
                      If None, fetches max history for all crashes.
    
    Returns:
        pd.Series with Close prices indexed by date, or None if no data
    """
    from app.services.prices import get_price_service
    
    # Determine date range
    if crash_period:
        crash = CRASH_PERIODS[crash_period]
        # Include 1 year before crash for baseline and 1 year after recovery
        start_date = crash.start_date - timedelta(days=365)
        if crash.recovery_date:
            end_date = crash.recovery_date + timedelta(days=365)
        else:
            end_date = crash.bottom_date + timedelta(days=730)  # 2 years after bottom
    else:
        # For all crashes, need data from 1999 (before dot-com) to present
        start_date = date(1999, 1, 1)
        end_date = date.today()
    
    try:
        price_service = get_price_service()
        df = await price_service.get_prices(symbol, start_date, end_date)
        
        if df is None or df.empty:
            logger.warning(f"No price data found for {symbol}")
            return None
        
        # Extract Close prices as Series
        if "Close" in df.columns:
            series = df["Close"]
        elif "close" in df.columns:
            series = df["close"]
        else:
            logger.warning(f"No Close column in data for {symbol}")
            return None
        
        series.name = symbol
        series.index = pd.to_datetime(series.index)
        
        logger.info(f"Loaded {len(series)} price records for {symbol} ({series.index.min().date()} to {series.index.max().date()})")
        return series
        
    except Exception as e:
        logger.error(f"Failed to fetch prices for {symbol}: {e}")
        return None


async def get_prices_batch_for_crash_testing(
    symbols: list[str],
    crash_period: CrashPeriod | None = None,
) -> dict[str, pd.Series]:
    """
    Fetch price data for multiple symbols for crash testing.
    
    Args:
        symbols: List of stock tickers
        crash_period: Optional specific crash period
    
    Returns:
        Dict mapping symbol to price Series
    """
    from app.services.prices import get_price_service
    
    # Determine date range
    if crash_period:
        crash = CRASH_PERIODS[crash_period]
        start_date = crash.start_date - timedelta(days=365)
        if crash.recovery_date:
            end_date = crash.recovery_date + timedelta(days=365)
        else:
            end_date = crash.bottom_date + timedelta(days=730)
    else:
        start_date = date(1999, 1, 1)
        end_date = date.today()
    
    try:
        price_service = get_price_service()
        results = await price_service.get_prices_batch(symbols, start_date, end_date)
        
        price_series: dict[str, pd.Series] = {}
        for symbol, df in results.items():
            if df is not None and not df.empty:
                if "Close" in df.columns:
                    series = df["Close"]
                elif "close" in df.columns:
                    series = df["close"]
                else:
                    continue
                series.name = symbol
                series.index = pd.to_datetime(series.index)
                price_series[symbol] = series
        
        logger.info(f"Loaded price data for {len(price_series)}/{len(symbols)} symbols")
        return price_series
        
    except Exception as e:
        logger.error(f"Failed to fetch prices batch: {e}")
        return {}


def get_available_crash_periods_for_data(prices: pd.Series) -> list[CrashPeriod]:
    """
    Determine which crash periods are testable given the available price data.
    
    Args:
        prices: Price series with DatetimeIndex
    
    Returns:
        List of CrashPeriod enums that can be tested
    """
    if prices is None or len(prices) < 20:
        return []
    
    data_start = prices.index.min().date()
    data_end = prices.index.max().date()
    
    available = []
    for period, crash in CRASH_PERIODS.items():
        # Need data from at least the start of the crash to the bottom
        if data_start <= crash.start_date and data_end >= crash.bottom_date:
            available.append(period)
    
    return available
