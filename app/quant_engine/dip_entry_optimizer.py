"""
Dip Entry Optimizer - Find optimal buy-more levels for stocks you already hold.

This module answers the question:
"I already hold NVDA/BAC/etc. When should I buy MORE?"

The goal is NOT to beat buy-and-hold, but to OPTIMIZE buy-and-hold by:
1. Finding the optimal dip threshold (5%, 10%, 15%, 20%?)
2. Calculating expected recovery rates and times
3. Factoring in volatility regime and earnings calendar
4. Computing risk-adjusted entry scores

Key insight: Strong-trend stocks WILL recover. The question is:
- How deep should the dip be before adding?
- What's the expected gain from buying at each level?
- How long until recovery?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DipEvent:
    """A single dip event in price history."""
    
    dip_date: datetime
    dip_price: float
    peak_price: float
    drawdown_pct: float  # Negative, e.g., -15%
    
    # Recovery info
    recovered: bool = False
    recovery_date: datetime | None = None
    recovery_days: int | None = None
    
    # Returns if bought at dip
    return_30d: float | None = None  # Return after 30 days
    return_60d: float | None = None
    return_90d: float | None = None
    max_return: float | None = None  # Max return within holding period
    
    # Context
    volatility_percentile: float = 50.0  # Where in vol distribution
    near_earnings: bool = False  # Within 2 weeks of earnings
    near_dividend: bool = False  # Within 2 weeks of ex-dividend date (opportunity)


@dataclass
class DipThresholdStats:
    """Statistics for a specific dip threshold (e.g., -10%)."""
    
    threshold_pct: float  # e.g., -10.0
    
    # Frequency
    n_occurrences: int = 0
    avg_per_year: float = 0.0
    
    # Recovery
    recovery_rate: float = 0.0  # % that recovered to peak
    avg_recovery_days: float = 0.0
    median_recovery_days: float = 0.0
    
    # Returns from buying at dip
    avg_return_30d: float = 0.0
    avg_return_60d: float = 0.0
    avg_return_90d: float = 0.0
    win_rate_30d: float = 0.0  # % positive after 30 days
    win_rate_60d: float = 0.0
    win_rate_90d: float = 0.0
    
    # Risk metrics
    max_further_drawdown: float = 0.0  # How much more it dropped after entry
    avg_further_drawdown: float = 0.0
    
    # Score (higher = better entry)
    entry_score: float = 0.0


@dataclass
class OptimalDipEntry:
    """Result of dip entry optimization for a symbol."""
    
    symbol: str
    analysis_date: datetime = field(default_factory=datetime.now)
    
    # Current state
    current_price: float = 0.0
    recent_high: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Optimal levels
    optimal_dip_threshold: float = 0.0  # Best dip % to buy at
    optimal_entry_price: float = 0.0  # Price to set buy order
    
    # Current recommendation
    is_buy_now: bool = False
    buy_signal_strength: float = 0.0  # 0-100
    signal_reason: str = ""
    
    # Threshold analysis
    threshold_stats: list[DipThresholdStats] = field(default_factory=list)
    
    # Historical dip events
    recent_dips: list[DipEvent] = field(default_factory=list)
    
    # Stock characteristics
    avg_annual_dips_10pct: float = 0.0  # How many 10%+ dips per year
    avg_annual_dips_15pct: float = 0.0
    avg_annual_dips_20pct: float = 0.0
    typical_recovery_days: float = 0.0
    volatility_regime: Literal["low", "normal", "high"] = "normal"
    
    # Fundamental context
    fundamentals_healthy: bool = True
    fundamental_notes: list[str] = field(default_factory=list)


# =============================================================================
# Dip Entry Optimizer
# =============================================================================


class DipEntryOptimizer:
    """
    Optimizes dip entry points for stocks you already hold.
    
    Instead of trying to beat buy-and-hold with timing, this helps you
    OPTIMIZE your buy-and-hold strategy by finding the best levels to add.
    
    Usage:
        optimizer = DipEntryOptimizer()
        result = optimizer.analyze(df, "NVDA", fundamentals)
        
        print(f"Optimal buy level: {result.optimal_dip_threshold}% dip")
        print(f"Set limit order at: ${result.optimal_entry_price}")
    """
    
    # Dynamic thresholds: test every percentage from 1% to 50%
    # This gives much finer granularity than preset buckets
    DIP_THRESHOLDS = list(range(-1, -51, -1))  # [-1, -2, -3, ..., -50]
    
    def __init__(
        self,
        lookback_years: int = 5,
        recovery_max_days: int = 180,
        min_dips_for_stats: int = 2,  # Lower threshold since we test more granular levels
    ):
        """
        Args:
            lookback_years: Years of history to analyze
            recovery_max_days: Max days to consider for recovery
            min_dips_for_stats: Minimum dip occurrences for valid stats
        """
        self.lookback_years = lookback_years
        self.recovery_max_days = recovery_max_days
        self.min_dips_for_stats = min_dips_for_stats
    
    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        fundamentals: dict | None = None,
        earnings_dates: list[datetime] | None = None,
        dividend_dates: list[datetime] | None = None,
    ) -> OptimalDipEntry:
        """
        Analyze a stock to find optimal dip entry points.
        
        Args:
            df: OHLCV DataFrame with 'close' column
            symbol: Stock ticker
            fundamentals: Dict with PE, FCF, debt ratios, etc.
            earnings_dates: List of earnings announcement dates
            dividend_dates: List of ex-dividend dates (opportunity windows)
        
        Returns:
            OptimalDipEntry with recommendations and analysis
        """
        logger.info(f"Analyzing dip entry points for {symbol}")
        
        # Prepare data
        df = self._prepare_data(df)
        
        # Identify all dip events
        dip_events = self._find_all_dips(df, earnings_dates, dividend_dates)
        
        # Calculate stats for each threshold
        threshold_stats = []
        for threshold in self.DIP_THRESHOLDS:
            stats = self._calculate_threshold_stats(dip_events, threshold, df)
            if stats.n_occurrences >= self.min_dips_for_stats:
                threshold_stats.append(stats)
        
        # Find optimal threshold
        optimal = self._find_optimal_threshold(threshold_stats)
        
        # Current state analysis
        current_price = float(df["close"].iloc[-1])
        recent_high = float(df["close"].rolling(252).max().iloc[-1])  # 1-year high
        current_drawdown = (current_price / recent_high - 1) * 100
        
        # Calculate entry price
        optimal_entry_price = recent_high * (1 + optimal / 100) if optimal else current_price * 0.9
        
        # Determine if now is a buy
        is_buy_now, signal_strength, reason = self._evaluate_current_opportunity(
            current_drawdown, optimal, threshold_stats, fundamentals
        )
        
        # Stock characteristics
        years = len(df) / 252
        dips_10 = sum(1 for d in dip_events if d.drawdown_pct <= -10) / max(years, 1)
        dips_15 = sum(1 for d in dip_events if d.drawdown_pct <= -15) / max(years, 1)
        dips_20 = sum(1 for d in dip_events if d.drawdown_pct <= -20) / max(years, 1)
        
        # Volatility regime
        recent_vol = df["close"].pct_change().tail(60).std() * np.sqrt(252)
        historical_vol = df["close"].pct_change().std() * np.sqrt(252)
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        vol_regime: Literal["low", "normal", "high"] = (
            "low" if vol_ratio < 0.8 else "high" if vol_ratio > 1.3 else "normal"
        )
        
        # Typical recovery
        recovered_dips = [d for d in dip_events if d.recovered and d.recovery_days]
        typical_recovery = np.median([d.recovery_days for d in recovered_dips]) if recovered_dips else 60.0
        
        # Check fundamentals
        fund_healthy, fund_notes = self._check_fundamentals(fundamentals)
        
        return OptimalDipEntry(
            symbol=symbol,
            analysis_date=datetime.now(),
            current_price=current_price,
            recent_high=recent_high,
            current_drawdown_pct=current_drawdown,
            optimal_dip_threshold=optimal,
            optimal_entry_price=optimal_entry_price,
            is_buy_now=is_buy_now,
            buy_signal_strength=signal_strength,
            signal_reason=reason,
            threshold_stats=threshold_stats,
            recent_dips=dip_events[-10:],  # Last 10 dips
            avg_annual_dips_10pct=dips_10,
            avg_annual_dips_15pct=dips_15,
            avg_annual_dips_20pct=dips_20,
            typical_recovery_days=typical_recovery,
            volatility_regime=vol_regime,
            fundamentals_healthy=fund_healthy,
            fundamental_notes=fund_notes,
        )
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with required columns."""
        df = df.copy()
        
        # Ensure lowercase columns
        df.columns = [str(c).lower() for c in df.columns]
        
        # Calculate rolling max (for drawdown)
        df["rolling_max"] = df["close"].expanding().max()
        df["drawdown"] = (df["close"] / df["rolling_max"] - 1) * 100
        
        # Daily returns
        df["return"] = df["close"].pct_change()
        
        # Volatility (20-day rolling)
        df["volatility"] = df["return"].rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _find_all_dips(
        self,
        df: pd.DataFrame,
        earnings_dates: list[datetime] | None = None,
        dividend_dates: list[datetime] | None = None,
    ) -> list[DipEvent]:
        """Find all significant dip events in the data."""
        dip_events = []
        
        # Get local minima in drawdown (significant dips)
        drawdown = df["drawdown"]
        close = df["close"]
        
        # Find dip troughs (local minima)
        for i in range(5, len(df) - 1):
            current_dd = drawdown.iloc[i]
            
            # Is this a local minimum? (dip trough)
            if current_dd <= -5:  # At least 5% dip
                # Check if it's a local minimum
                window_before = drawdown.iloc[max(0, i-5):i]
                window_after = drawdown.iloc[i+1:min(len(df), i+6)]
                
                if len(window_after) > 0:
                    is_local_min = (
                        current_dd <= window_before.min() and
                        current_dd < window_after.min()
                    )
                    
                    if is_local_min:
                        dip_date = df.index[i] if hasattr(df.index[i], 'date') else datetime.now()
                        dip_price = float(close.iloc[i])
                        peak_price = float(df["rolling_max"].iloc[i])
                        
                        # Calculate recovery
                        future_close = close.iloc[i+1:]
                        recovered = any(future_close >= peak_price)
                        recovery_idx = None
                        if recovered:
                            recovery_mask = future_close >= peak_price
                            if recovery_mask.any():
                                recovery_idx = recovery_mask.idxmax()
                                recovery_days = (recovery_idx - df.index[i]).days if hasattr(recovery_idx, 'days') else int(recovery_mask.argmax())
                            else:
                                recovery_days = None
                        else:
                            recovery_days = None
                        
                        # Returns after N days
                        return_30d = self._calc_future_return(close, i, 30)
                        return_60d = self._calc_future_return(close, i, 60)
                        return_90d = self._calc_future_return(close, i, 90)
                        max_return = self._calc_max_return(close, i, 180)
                        
                        # Volatility percentile
                        vol_percentile = self._calc_vol_percentile(df, i)
                        
                        # Near earnings? (risk - volatility spike around earnings)
                        near_earnings = self._is_near_event(dip_date, earnings_dates)
                        
                        # Near dividend? (opportunity - can capture dividend)
                        near_dividend = self._is_near_event(dip_date, dividend_dates)
                        
                        event = DipEvent(
                            dip_date=dip_date if isinstance(dip_date, datetime) else datetime.now(),
                            dip_price=dip_price,
                            peak_price=peak_price,
                            drawdown_pct=current_dd,
                            recovered=recovered,
                            recovery_date=recovery_idx if isinstance(recovery_idx, datetime) else None,
                            recovery_days=recovery_days if isinstance(recovery_days, int) else None,
                            return_30d=return_30d,
                            return_60d=return_60d,
                            return_90d=return_90d,
                            max_return=max_return,
                            volatility_percentile=vol_percentile,
                            near_earnings=near_earnings,
                            near_dividend=near_dividend,
                        )
                        dip_events.append(event)
        
        return dip_events
    
    def _calc_future_return(self, close: pd.Series, idx: int, days: int) -> float | None:
        """Calculate return N days after index."""
        if idx + days >= len(close):
            return None
        return float((close.iloc[idx + days] / close.iloc[idx] - 1) * 100)
    
    def _calc_max_return(self, close: pd.Series, idx: int, days: int) -> float | None:
        """Calculate max return within N days after index."""
        end_idx = min(idx + days, len(close))
        if idx >= end_idx:
            return None
        future = close.iloc[idx:end_idx]
        return float((future.max() / close.iloc[idx] - 1) * 100)
    
    def _calc_vol_percentile(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate volatility percentile at given index."""
        if "volatility" not in df.columns or idx < 60:
            return 50.0
        current_vol = df["volatility"].iloc[idx]
        historical_vol = df["volatility"].iloc[:idx]
        if historical_vol.isna().all():
            return 50.0
        return float((historical_vol < current_vol).mean() * 100)
    
    def _is_near_event(
        self,
        date: datetime,
        event_dates: list[datetime] | None,
        days_window: int = 14,
    ) -> bool:
        """Check if date is within N days of any event date."""
        if not event_dates:
            return False
        for ed in event_dates:
            if abs((date - ed).days) <= days_window:
                return True
        return False
    
    def _calculate_threshold_stats(
        self,
        dip_events: list[DipEvent],
        threshold: float,
        df: pd.DataFrame,
    ) -> DipThresholdStats:
        """Calculate statistics for a specific dip threshold."""
        # Filter dips that reached this threshold
        relevant_dips = [d for d in dip_events if d.drawdown_pct <= threshold]
        
        n = len(relevant_dips)
        years = len(df) / 252
        
        if n == 0:
            return DipThresholdStats(
                threshold_pct=threshold,
                n_occurrences=0,
            )
        
        # Recovery stats
        recovered = [d for d in relevant_dips if d.recovered]
        recovery_rate = len(recovered) / n * 100
        recovery_days = [d.recovery_days for d in recovered if d.recovery_days]
        
        # Return stats
        returns_30d = [d.return_30d for d in relevant_dips if d.return_30d is not None]
        returns_60d = [d.return_60d for d in relevant_dips if d.return_60d is not None]
        returns_90d = [d.return_90d for d in relevant_dips if d.return_90d is not None]
        
        # Further drawdown (how much more it dropped after hitting threshold)
        # This would require more tracking in DipEvent, simplified here
        
        # Calculate entry score
        # Higher score = better entry opportunity
        # Factors: win rate, avg return, recovery rate, frequency (not too rare)
        win_rate_60 = (sum(1 for r in returns_60d if r > 0) / len(returns_60d) * 100) if returns_60d else 0
        avg_ret_60 = np.mean(returns_60d) if returns_60d else 0
        
        # Score formula: balance return, win rate, and practical frequency
        frequency_score = min(n / years, 3) / 3 * 100  # Cap at 3 per year = 100
        return_score = min(max(avg_ret_60, 0), 30) / 30 * 100  # Cap at 30% = 100
        win_score = win_rate_60
        recovery_score = recovery_rate
        
        entry_score = (
            return_score * 0.35 +
            win_score * 0.30 +
            recovery_score * 0.20 +
            frequency_score * 0.15
        )
        
        return DipThresholdStats(
            threshold_pct=threshold,
            n_occurrences=n,
            avg_per_year=n / years if years > 0 else 0,
            recovery_rate=recovery_rate,
            avg_recovery_days=np.mean(recovery_days) if recovery_days else 0,
            median_recovery_days=np.median(recovery_days) if recovery_days else 0,
            avg_return_30d=np.mean(returns_30d) if returns_30d else 0,
            avg_return_60d=np.mean(returns_60d) if returns_60d else 0,
            avg_return_90d=np.mean(returns_90d) if returns_90d else 0,
            win_rate_30d=(sum(1 for r in returns_30d if r > 0) / len(returns_30d) * 100) if returns_30d else 0,
            win_rate_60d=win_rate_60,
            win_rate_90d=(sum(1 for r in returns_90d if r > 0) / len(returns_90d) * 100) if returns_90d else 0,
            entry_score=entry_score,
        )
    
    def _find_optimal_threshold(
        self,
        stats: list[DipThresholdStats],
    ) -> float:
        """Find the optimal dip threshold based on entry scores."""
        if not stats:
            return -10.0  # Default to 10% dip
        
        # Sort by entry score (descending)
        sorted_stats = sorted(stats, key=lambda s: s.entry_score, reverse=True)
        
        # Return the best threshold
        return sorted_stats[0].threshold_pct
    
    def _evaluate_current_opportunity(
        self,
        current_drawdown: float,
        optimal_threshold: float,
        stats: list[DipThresholdStats],
        fundamentals: dict | None,
    ) -> tuple[bool, float, str]:
        """Evaluate if current price is a buy opportunity."""
        # Check fundamentals FIRST - they gate the buy decision
        fund_healthy, fund_notes = self._check_fundamentals(fundamentals)
        
        # Find stats for current drawdown level
        applicable_stats = [s for s in stats if current_drawdown <= s.threshold_pct]
        
        if not applicable_stats:
            # Not at any significant dip level
            if current_drawdown > -3:
                return False, 0.0, f"Near highs (only {current_drawdown:.1f}% from peak)"
            else:
                return False, 20.0, f"Minor pullback ({current_drawdown:.1f}%), wait for deeper dip"
        
        # Use the deepest applicable threshold
        best_stats = max(applicable_stats, key=lambda s: abs(s.threshold_pct))
        
        # Calculate signal strength
        # Based on: how deep the dip, win rate, and fundamentals
        depth_score = min(abs(current_drawdown) / 20, 1.0) * 40  # Max 40 points for 20%+ dip
        win_score = best_stats.win_rate_60d * 0.4  # Max 40 points for 100% win rate
        return_score = min(best_stats.avg_return_60d / 15, 1.0) * 20  # Max 20 points for 15%+ avg return
        
        # Fundamentals adjustment: reduce strength if unhealthy
        fundamentals_score = 0.0
        if fund_healthy:
            fundamentals_score = 10.0  # Bonus for healthy fundamentals
        else:
            fundamentals_score = -20.0  # Penalty for unhealthy fundamentals
        
        signal_strength = depth_score + win_score + return_score + fundamentals_score
        
        # Determine if buy - MUST have healthy fundamentals
        is_buy = (
            current_drawdown <= optimal_threshold 
            and signal_strength >= 50 
            and fund_healthy  # Gate: fundamentals must be healthy
        )
        
        # Generate reason
        if is_buy:
            reason = (
                f"At {current_drawdown:.1f}% dip (optimal: {optimal_threshold:.0f}%). "
                f"Historically {best_stats.win_rate_60d:.0f}% win rate, "
                f"avg {best_stats.avg_return_60d:.1f}% return in 60 days. "
                f"Recovery in ~{best_stats.avg_recovery_days:.0f} days. "
                f"Fundamentals: {'✓ healthy' if fund_healthy else '⚠ concerns'}."
            )
        elif not fund_healthy:
            reason = (
                f"At {current_drawdown:.1f}% dip but fundamentals unhealthy: {'; '.join(fund_notes)}. "
                f"May be falling knife - wait for clarity."
            )
        else:
            reason = (
                f"At {current_drawdown:.1f}% dip. "
                f"Wait for {optimal_threshold:.0f}% dip for optimal entry. "
                f"Entry price target: ${best_stats.threshold_pct}% from recent high."
            )
        
        return is_buy, signal_strength, reason
    
    def _check_fundamentals(
        self,
        fundamentals: dict | None,
    ) -> tuple[bool, list[str]]:
        """Check if fundamentals support buying the dip."""
        if not fundamentals:
            return True, ["No fundamental data - proceed with caution"]
        
        notes = []
        issues = 0
        
        # Check key metrics
        pe = fundamentals.get("pe_ratio")
        if pe and pe > 50:
            notes.append(f"High P/E ratio ({pe:.1f})")
            issues += 1
        
        debt_to_equity = fundamentals.get("debt_to_equity")
        if debt_to_equity and debt_to_equity > 2:
            notes.append(f"High debt/equity ({debt_to_equity:.1f})")
            issues += 1
        
        revenue_growth = fundamentals.get("revenue_growth")
        if revenue_growth and float(str(revenue_growth).replace("%", "")) < 0:
            notes.append("Negative revenue growth")
            issues += 1
        
        fcf = fundamentals.get("free_cash_flow")
        if fcf and fcf < 0:
            notes.append("Negative free cash flow")
            issues += 1
        
        if not notes:
            notes.append("Fundamentals look healthy")
        
        return issues < 2, notes


# =============================================================================
# Convenience Functions
# =============================================================================


def analyze_dip_opportunity(
    df: pd.DataFrame,
    symbol: str,
    fundamentals: dict | None = None,
) -> OptimalDipEntry:
    """
    Quick analysis of dip entry opportunity.
    
    Usage:
        result = analyze_dip_opportunity(df, "NVDA")
        print(f"Optimal dip to buy: {result.optimal_dip_threshold}%")
        print(f"Set limit order at: ${result.optimal_entry_price:.2f}")
        print(f"Current: {result.current_drawdown_pct:.1f}% from high")
        if result.is_buy_now:
            print(f"BUY NOW: {result.signal_reason}")
    """
    optimizer = DipEntryOptimizer()
    return optimizer.analyze(df, symbol, fundamentals)


def get_dip_summary(result: OptimalDipEntry) -> dict:
    """Convert OptimalDipEntry to a summary dict for API/frontend."""
    return {
        "symbol": result.symbol,
        "current_price": result.current_price,
        "recent_high": result.recent_high,
        "current_drawdown_pct": result.current_drawdown_pct,
        "optimal_dip_threshold": result.optimal_dip_threshold,
        "optimal_entry_price": result.optimal_entry_price,
        "is_buy_now": result.is_buy_now,
        "buy_signal_strength": result.buy_signal_strength,
        "signal_reason": result.signal_reason,
        "volatility_regime": result.volatility_regime,
        "typical_recovery_days": result.typical_recovery_days,
        "avg_dips_per_year": {
            "10_pct": result.avg_annual_dips_10pct,
            "15_pct": result.avg_annual_dips_15pct,
            "20_pct": result.avg_annual_dips_20pct,
        },
        "fundamentals_healthy": result.fundamentals_healthy,
        "fundamental_notes": result.fundamental_notes,
        "threshold_analysis": [
            {
                "threshold": s.threshold_pct,
                "occurrences": s.n_occurrences,
                "per_year": s.avg_per_year,
                "win_rate_60d": s.win_rate_60d,
                "avg_return_60d": s.avg_return_60d,
                "recovery_rate": s.recovery_rate,
                "avg_recovery_days": s.avg_recovery_days,
                "entry_score": s.entry_score,
            }
            for s in result.threshold_stats
        ],
    }
