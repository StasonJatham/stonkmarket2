"""
Dip Entry Optimizer - Find optimal buy-more levels for stocks you already hold.

This module answers the question:
"I already hold NVDA/BAC/etc. When should I buy MORE?"

The goal is NOT to beat buy-and-hold, but to OPTIMIZE buy-and-hold by:
1. Finding the optimal dip threshold (5%, 10%, 15%, 20%?)
2. Calculating expected recovery rates and times
3. Factoring in volatility regime and earnings calendar
4. Computing risk-adjusted entry scores with MAE (Maximum Adverse Excursion)

Key insight: Strong-trend stocks WILL recover. The question is:
- How deep should the dip be before adding?
- What's the expected gain from buying at each level?
- How long until recovery?
- What's the risk of further drops after entry?

V2 CHANGES (Risk-Adjusted):
- Threshold-crossing detection instead of trough-based
- Maximum Adverse Excursion (MAE) tracking after entry
- Sharpe/Sortino ratios for risk-adjusted returns
- CVaR (Conditional Value at Risk) for tail risk
- Outlier filtering for black swan events
- Continuation probability (P(further drop >= X%))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.quant_engine.core.config import QUANT_LIMITS

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DipOptimizerConfig:
    """
    Configuration for the dip entry optimizer.
    
    All holding period limits come from QUANT_LIMITS (central config).
    Other parameters control scoring and filtering behavior.
    
    Holding periods are tested from 1 to max_holding_days (every single day).
    The optimal period is DISCOVERED per symbol via Sharpe optimization.
    """
    
    # Risk metrics
    cvar_percentile: float = 0.95  # 95% CVaR (worst 5% of returns)
    
    # MAE (Maximum Adverse Excursion) gates
    mae_gate_multiplier: float = 1.5  # Reject if avg_MAE > threshold * this
    mae_disqualify_multiplier: float = 2.0  # Hard reject if MAE > threshold * this
    
    # Continuation probability
    further_drop_threshold: float = 10.0  # Track P(drops another X%)
    max_continuation_prob: float = 0.50  # Reject if P(further drop) > this
    
    # Outlier filtering
    min_frequency_per_year: float = 0.3  # Filter dips rarer than this
    outlier_zscore: float = 2.5  # Flag dips beyond this z-score as outliers
    
    # Scoring weights (must sum to 1.0)
    weight_return: float = 0.30
    weight_risk_adjusted: float = 0.25  # Sharpe/Sortino
    weight_win_rate: float = 0.20
    weight_frequency: float = 0.15
    weight_recovery: float = 0.10
    
    # Recovery threshold
    recovery_win_threshold: float = 80.0  # 80% = recovered 80% of the drop counts as "win"
    
    # Optimal threshold selection
    min_optimal_threshold: float = -5.0  # Don't consider dips shallower than this as "optimal"
    min_occurrences_for_optimal: int = 3  # Need at least this many occurrences
    
    # =========================================================================
    # All limits come from central config - NO hardcoded values here
    # =========================================================================
    
    @property
    def max_holding_days(self) -> int:
        """Maximum holding period from central config."""
        return QUANT_LIMITS.max_holding_days
    
    @property
    def holding_periods(self) -> tuple[int, ...]:
        """Full range of holding periods to test: 1, 2, 3, ..., max.
        
        Tests EVERY day from 1 to max_holding_days.
        The optimal period is DISCOVERED per symbol by Sharpe optimization.
        """
        return tuple(QUANT_LIMITS.holding_days_range())
    
    @property
    def lookback_years(self) -> int:
        """Lookback period from central config."""
        return QUANT_LIMITS.lookback_years
    
    @property
    def min_dips_for_stats(self) -> int:
        """Minimum samples from central config."""
        return QUANT_LIMITS.min_samples_for_stats
    
    @property
    def min_years_high_confidence(self) -> int:
        """Minimum years for confidence from central config."""
        return QUANT_LIMITS.min_years_for_confidence
    
    @property
    def max_recovery_days(self) -> int:
        """Maximum days to track recovery from central config."""
        return QUANT_LIMITS.max_recovery_days
    
    @property
    def primary_holding_period(self) -> int:
        """Primary holding period for scoring from central config."""
        return QUANT_LIMITS.primary_holding_period


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class DipEvent:
    """A single dip event in price history (threshold-crossing based)."""
    
    # Entry point (when threshold was first crossed)
    entry_date: datetime
    entry_price: float
    threshold_crossed: float  # e.g., -10.0 (the threshold that triggered this event)
    peak_price: float  # Recent high before the dip
    
    # Maximum Adverse Excursion (how much worse it got after entry)
    min_price_after_entry: float = 0.0  # Lowest price in holding period
    max_adverse_excursion: float = 0.0  # % drop from entry to lowest (negative)
    further_drop_pct: float = 0.0  # How much more it dropped after entry
    
    # Recovery info
    recovered: bool = False
    recovery_date: datetime | None = None
    recovery_days: int | None = None  # Days to FULL (100%) recovery
    recovery_pct: float = 0.0  # What % of the dip was recovered within max_recovery_days
    
    # Dynamic recovery tracking - how fast did it recover?
    days_to_threshold_recovery: int | None = None  # Days to hit recovery_win_threshold %
    recovery_velocity: float = 0.0  # recovery_pct / days (higher = faster bounce)
    
    # Return at recovery (sell when price returns to entry price)
    return_at_recovery: float = 0.0  # Return % when selling at break-even
    days_held_to_recovery: int = 0  # Days held until break-even or max period
    
    # Returns if bought at entry (keyed by holding period days)
    returns: dict[int, float | None] = field(default_factory=dict)  # {30: 5.2, 60: 8.1, 90: 12.3}
    max_return: float | None = None  # Max return within holding period
    
    # Context
    volatility_percentile: float = 50.0
    near_earnings: bool = False
    near_dividend: bool = False
    
    # Outlier flag
    is_outlier: bool = False


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
    
    # Returns from buying at dip (keyed by holding period)
    avg_returns: dict[int, float] = field(default_factory=dict)  # {30: 5.2, 60: 8.1, 90: 12.3}
    total_profit: dict[int, float] = field(default_factory=dict)  # Simple sum: N × avg_return
    total_profit_compounded: dict[int, float] = field(default_factory=dict)  # Compounded: (1+r1)*(1+r2)*...
    win_rates: dict[int, float] = field(default_factory=dict)  # {30: 60.0, 60: 70.0, 90: 75.0}
    
    # Recovery-based metrics (sell at recovery, not fixed hold)
    avg_return_at_recovery: float = 0.0  # Avg return when selling at break-even
    total_profit_at_recovery: float = 0.0  # Compounded return selling at recovery
    avg_days_at_recovery: float = 0.0  # Avg days held until recovery
    
    # Recovery-based metrics (more meaningful than raw win rate)
    # recovery_threshold_rate uses config.recovery_win_threshold (default 80%)
    recovery_threshold_rate: float = 0.0  # % that recovered at least X% of the drop
    full_recovery_rate: float = 0.0  # % that fully recovered to peak (100%)
    avg_recovery_pct: float = 0.0  # Average recovery percentage
    
    # Dynamic recovery metrics - how FAST does it recover?
    avg_days_to_threshold: float = 0.0  # Avg days to hit recovery threshold
    avg_recovery_velocity: float = 0.0  # Avg recovery_pct / days (higher = faster)
    
    # Risk metrics - Maximum Adverse Excursion
    max_further_drawdown: float = 0.0  # Worst MAE across all events
    avg_further_drawdown: float = 0.0  # Average MAE
    
    # Risk-adjusted metrics (keyed by holding period)
    sharpe_ratios: dict[int, float] = field(default_factory=dict)
    sortino_ratios: dict[int, float] = field(default_factory=dict)
    cvar: dict[int, float] = field(default_factory=dict)
    
    # Continuation probability
    prob_further_drop: float = 0.0  # P(drops another X% after entry)
    continuation_risk: Literal["low", "medium", "high"] = "medium"
    
    # Outlier info
    n_outliers: int = 0
    is_outlier_dominated: bool = False  # True if most events are outliers
    
    # Scores
    entry_score: float = 0.0  # Risk-adjusted score
    
    # Confidence
    confidence: Literal["low", "medium", "high"] = "medium"
    
    # =========================================================================
    # OPTIMAL HOLDING PERIOD - Dynamically computed
    # =========================================================================
    
    @property
    def optimal_holding_days(self) -> int:
        """Find the optimal holding period based on Sharpe ratio.
        
        The optimal period is DISCOVERED by testing a full range (5, 10, 15, ... max)
        and selecting the period with the best risk-adjusted returns.
        
        This balances:
        - Higher returns (longer holds)
        - Risk-adjusted performance (Sharpe ratio)
        - Capital efficiency (shorter holds = more trades possible)
        """
        if not self.sharpe_ratios:
            return 60  # Default fallback
        
        # Find period with best Sharpe ratio
        best_period = 90
        best_sharpe = self.sharpe_ratios.get(90, 0)
        
        for period, sharpe in self.sharpe_ratios.items():
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_period = period
        
        return best_period
    
    @property
    def optimal_avg_return(self) -> float:
        """Average return at the optimal holding period."""
        return self.avg_returns.get(self.optimal_holding_days, 0.0)
    
    @property
    def optimal_win_rate(self) -> float:
        """Win rate at the optimal holding period."""
        return self.win_rates.get(self.optimal_holding_days, 0.0)
    
    @property
    def optimal_total_profit(self) -> float:
        """Total compounded profit at the optimal holding period."""
        return self.total_profit_compounded.get(self.optimal_holding_days, 0.0)
    
    @property
    def optimal_sharpe(self) -> float:
        """Sharpe ratio at the optimal holding period."""
        return self.sharpe_ratios.get(self.optimal_holding_days, 0.0)


@dataclass
class OptimalDipEntry:
    """Result of dip entry optimization for a symbol."""
    
    symbol: str
    analysis_date: datetime = field(default_factory=datetime.now)
    
    # Current state
    current_price: float = 0.0
    recent_high: float = 0.0
    current_drawdown_pct: float = 0.0
    
    # Risk-adjusted optimal (less pain, better timing)
    # This is the PRIMARY recommendation - minimizes MAE while maximizing quality
    optimal_dip_threshold: float = 0.0  # Best risk-adjusted dip %
    optimal_entry_price: float = 0.0  # Price to set buy order
    
    # Max profit optimal (more opportunities, higher total return)
    # Secondary recommendation - maximizes total expected profit
    max_profit_threshold: float = 0.0  # Best total profit dip %
    max_profit_entry_price: float = 0.0  # Price for max profit strategy
    max_profit_total_return: float = 0.0  # Expected total return from this threshold
    
    # Current recommendation
    is_buy_now: bool = False
    buy_signal_strength: float = 0.0  # 0-100
    signal_reason: str = ""
    continuation_risk: Literal["low", "medium", "high"] = "medium"
    
    # Threshold analysis
    threshold_stats: list[DipThresholdStats] = field(default_factory=list)
    
    # Historical dip events
    recent_dips: list[DipEvent] = field(default_factory=list)  # Last 10 for display
    all_dip_events: list[DipEvent] = field(default_factory=list)  # All events for trigger generation
    
    # Outlier events (black swans) - stored separately for UI
    outlier_events: list[DipEvent] = field(default_factory=list)
    
    # Stock characteristics
    avg_annual_dips_10pct: float = 0.0  # How many 10%+ dips per year
    avg_annual_dips_15pct: float = 0.0
    avg_annual_dips_20pct: float = 0.0
    typical_recovery_days: float = 0.0
    volatility_regime: Literal["low", "normal", "high"] = "normal"
    
    # Fundamental context
    fundamentals_healthy: bool = True
    fundamental_notes: list[str] = field(default_factory=list)
    
    # Confidence level
    data_years: float = 0.0
    confidence: Literal["low", "medium", "high"] = "medium"


# =============================================================================
# Dip Entry Optimizer
# =============================================================================


class DipEntryOptimizer:
    """
    Optimizes dip entry points for stocks you already hold.
    
    V2: Uses threshold-crossing detection with MAE tracking and risk-adjusted scoring.
    
    Instead of trying to beat buy-and-hold with timing, this helps you
    OPTIMIZE your buy-and-hold strategy by finding the best levels to add.
    
    Key improvements over V1:
    - Tracks what happens AFTER you enter at a threshold (not just trough stats)
    - Computes Maximum Adverse Excursion (how much worse it gets after entry)
    - Risk-adjusted scoring with Sharpe/Sortino ratios
    - Outlier filtering for black swan events
    
    Usage:
        config = DipOptimizerConfig(max_holding_days=60)
        optimizer = DipEntryOptimizer(config=config)
        result = optimizer.analyze(df, "NVDA", fundamentals)
        
        print(f"Optimal buy level: {result.optimal_dip_threshold}% dip")
        print(f"Set limit order at: ${result.optimal_entry_price}")
    """
    
    # Dynamic thresholds from central config: test every % from 1 to max
    DIP_THRESHOLDS = list(QUANT_LIMITS.dip_thresholds_range())  # [-1, -2, -3, ..., -50]
    
    def __init__(self, config: DipOptimizerConfig | None = None):
        """
        Initialize the dip entry optimizer.
        
        Args:
            config: Configuration object (uses defaults from QUANT_LIMITS if None)
        """
        self.config = config or DipOptimizerConfig()
    
    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str,
        fundamentals: dict | None = None,
        earnings_dates: list[datetime] | None = None,
        dividend_dates: list[datetime] | None = None,
        min_dip_threshold: float | None = None,
    ) -> OptimalDipEntry:
        """
        Analyze a stock to find optimal dip entry points.
        
        Args:
            df: OHLCV DataFrame with 'close' column
            symbol: Stock ticker
            fundamentals: Dict with PE, FCF, debt ratios, etc.
            earnings_dates: List of earnings announcement dates
            dividend_dates: List of ex-dividend dates (opportunity windows)
            min_dip_threshold: Minimum dip % from symbol config in DB (e.g., -10.0)
                              If None, uses config.min_optimal_threshold
        
        Returns:
            OptimalDipEntry with recommendations and analysis
        """
        logger.info(f"Analyzing dip entry points for {symbol}")
        
        # Use symbol-specific min threshold if provided, otherwise config default
        effective_min_threshold = min_dip_threshold if min_dip_threshold is not None else self.config.min_optimal_threshold
        
        # Prepare data
        df = self._prepare_data(df)
        years = len(df) / 252
        
        # Identify all threshold-crossing events (not trough-based)
        all_dip_events = self._find_threshold_crossings(df, earnings_dates, dividend_dates)
        
        # Separate outliers from regular events (for flagging, not filtering)
        regular_events, outlier_events = self._filter_outliers(all_dip_events, df)
        
        # Calculate stats for each threshold using ALL events (not just regular)
        # Outliers are flagged but included - they provide valuable return data
        # especially for volatile stocks where deep dips are rare but informative
        threshold_stats = []
        for threshold in self.DIP_THRESHOLDS:
            stats = self._calculate_threshold_stats(all_dip_events, threshold, df)
            if stats.n_occurrences >= self.config.min_dips_for_stats:
                threshold_stats.append(stats)
        
        # Find RISK-ADJUSTED optimal threshold (less pain, better timing)
        optimal = self._find_optimal_threshold(threshold_stats)
        
        # Find MAX PROFIT threshold (respects symbol's min_dip_threshold from DB)
        max_profit_threshold, max_profit_total = self._find_max_profit_threshold(
            threshold_stats, effective_min_threshold
        )
        
        # Current state analysis
        current_price = float(df["close"].iloc[-1])
        recent_high = float(df["close"].rolling(252).max().iloc[-1])  # 1-year high
        current_drawdown = (current_price / recent_high - 1) * 100
        
        # Calculate entry prices for both strategies
        optimal_entry_price = recent_high * (1 + optimal / 100) if optimal else current_price * 0.9
        max_profit_entry_price = recent_high * (1 + max_profit_threshold / 100)
        
        # Determine if now is a buy (using new risk-adjusted logic)
        is_buy_now, signal_strength, reason, cont_risk = self._evaluate_current_opportunity(
            current_drawdown, optimal, threshold_stats, fundamentals
        )
        
        # Stock characteristics
        dips_10 = sum(1 for d in regular_events if d.threshold_crossed <= -10) / max(years, 1)
        dips_15 = sum(1 for d in regular_events if d.threshold_crossed <= -15) / max(years, 1)
        dips_20 = sum(1 for d in regular_events if d.threshold_crossed <= -20) / max(years, 1)
        
        # Volatility regime
        recent_vol = df["close"].pct_change().tail(60).std() * np.sqrt(252)
        historical_vol = df["close"].pct_change().std() * np.sqrt(252)
        vol_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        vol_regime: Literal["low", "normal", "high"] = (
            "low" if vol_ratio < 0.8 else "high" if vol_ratio > 1.3 else "normal"
        )
        
        # Typical recovery
        recovered_dips = [d for d in regular_events if d.recovered and d.recovery_days]
        typical_recovery = np.median([d.recovery_days for d in recovered_dips]) if recovered_dips else 60.0
        
        # Check fundamentals
        fund_healthy, fund_notes = self._check_fundamentals(fundamentals)
        
        # Confidence based on data years
        confidence: Literal["low", "medium", "high"] = (
            "high" if years >= self.config.min_years_high_confidence 
            else "medium" if years >= 2 
            else "low"
        )
        
        return OptimalDipEntry(
            symbol=symbol,
            analysis_date=datetime.now(),
            current_price=current_price,
            recent_high=recent_high,
            current_drawdown_pct=current_drawdown,
            # Risk-adjusted optimal (primary recommendation)
            optimal_dip_threshold=optimal,
            optimal_entry_price=optimal_entry_price,
            # Max profit optimal (secondary recommendation)
            max_profit_threshold=max_profit_threshold,
            max_profit_entry_price=max_profit_entry_price,
            max_profit_total_return=max_profit_total,
            # Current opportunity
            is_buy_now=is_buy_now,
            buy_signal_strength=signal_strength,
            signal_reason=reason,
            continuation_risk=cont_risk,
            threshold_stats=threshold_stats,
            recent_dips=regular_events[-10:],  # Last 10 dips for display
            all_dip_events=regular_events,  # All events for trigger generation
            outlier_events=outlier_events,
            avg_annual_dips_10pct=dips_10,
            avg_annual_dips_15pct=dips_15,
            avg_annual_dips_20pct=dips_20,
            typical_recovery_days=typical_recovery,
            volatility_regime=vol_regime,
            fundamentals_healthy=fund_healthy,
            fundamental_notes=fund_notes,
            data_years=years,
            confidence=confidence,
        )
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame with required columns."""
        df = df.copy()
        
        # Ensure lowercase columns
        df.columns = [str(c).lower() for c in df.columns]
        
        # Calculate rolling max for drawdown
        # Use 252-day (1-year) rolling max instead of expanding max
        # This allows stocks that crash and recover to have "new" dip opportunities
        # instead of being stuck in one perpetual drawdown from ATH
        df["rolling_max"] = df["close"].rolling(window=252, min_periods=1).max()
        df["drawdown"] = (df["close"] / df["rolling_max"] - 1) * 100
        
        # Also track ATH for context
        df["ath"] = df["close"].expanding().max()
        df["drawdown_from_ath"] = (df["close"] / df["ath"] - 1) * 100
        
        # Daily returns
        df["return"] = df["close"].pct_change()
        
        # Volatility (20-day rolling)
        df["volatility"] = df["return"].rolling(20).std() * np.sqrt(252)
        
        return df
    
    def _find_threshold_crossings(
        self,
        df: pd.DataFrame,
        earnings_dates: list[datetime] | None = None,
        dividend_dates: list[datetime] | None = None,
    ) -> list[DipEvent]:
        """
        Find all threshold-crossing events (V2 approach).
        
        Unlike V1 trough-based detection, this finds the FIRST time price crosses
        each threshold level, then tracks what happens afterward (MAE, recovery, returns).
        
        This means a dip that starts at -6% and continues to -18% will:
        - Create an event at -6% with high MAE (further drop of -12%)
        - Create an event at -10% with moderate MAE
        - Create an event at -15% with low MAE
        - Create an event at -18% (the trough) with zero MAE
        
        This properly captures the risk of buying at shallow dips.
        """
        dip_events = []
        drawdown = df["drawdown"]
        close = df["close"]
        
        # Track which thresholds have been crossed in the current drawdown cycle
        # Reset when price recovers to a new high
        active_crossings: dict[int, int] = {}  # threshold -> index where crossed
        
        max_holding = max(self.config.holding_periods)
        
        for i in range(1, len(df)):
            current_dd = drawdown.iloc[i]
            prev_dd = drawdown.iloc[i - 1]
            
            # Check if we've recovered to a new high (reset cycle)
            if current_dd >= -1:  # Within 1% of high
                # Process all active crossings from this cycle
                for threshold, cross_idx in active_crossings.items():
                    event = self._create_dip_event_from_crossing(
                        df, cross_idx, threshold, earnings_dates, dividend_dates
                    )
                    if event:
                        dip_events.append(event)
                active_crossings = {}
                continue
            
            # Check for new threshold crossings
            for threshold in self.DIP_THRESHOLDS:
                # Skip if already crossed in this cycle
                if threshold in active_crossings:
                    continue
                
                # Did we just cross this threshold?
                if current_dd <= threshold < prev_dd:
                    active_crossings[threshold] = i
        
        # Process any remaining active crossings at end of data
        for threshold, cross_idx in active_crossings.items():
            # Only process if we have enough forward data
            if cross_idx + 30 < len(df):  # Need at least 30 days forward
                event = self._create_dip_event_from_crossing(
                    df, cross_idx, threshold, earnings_dates, dividend_dates
                )
                if event:
                    dip_events.append(event)
        
        return dip_events
    
    def _create_dip_event_from_crossing(
        self,
        df: pd.DataFrame,
        cross_idx: int,
        threshold: float,
        earnings_dates: list[datetime] | None,
        dividend_dates: list[datetime] | None,
    ) -> DipEvent | None:
        """Create a DipEvent from a threshold crossing point."""
        close = df["close"]
        max_holding = max(self.config.holding_periods)
        
        # Need enough forward data
        if cross_idx + 30 >= len(df):
            return None
        
        entry_price = float(close.iloc[cross_idx])
        peak_price = float(df["rolling_max"].iloc[cross_idx])
        entry_date = df.index[cross_idx] if hasattr(df.index[cross_idx], 'date') else datetime.now()
        
        # Calculate returns for each holding period
        returns: dict[int, float | None] = {}
        for period in self.config.holding_periods:
            returns[period] = self._calc_future_return(close, cross_idx, period)
        
        # Max return within holding period
        max_return = self._calc_max_return(close, cross_idx, max_holding)
        
        # Calculate Maximum Adverse Excursion (MAE)
        # How much lower did it go after entry?
        end_idx = min(cross_idx + max_holding, len(close))
        future_prices = close.iloc[cross_idx:end_idx]
        min_price_after = float(future_prices.min())
        max_adverse_excursion = ((min_price_after / entry_price) - 1) * 100  # Negative %
        further_drop_pct = abs(max_adverse_excursion)  # Positive number for easier comparison
        
        # Recovery calculation - both full recovery and recovery percentage
        # Use configurable max_recovery_days instead of full holding period
        recovery_end_idx = min(cross_idx + self.config.max_recovery_days, len(close))
        future_close = close.iloc[cross_idx + 1:recovery_end_idx]
        recovered = any(future_close >= peak_price) if len(future_close) > 0 else False
        recovery_days = None
        recovery_date = None
        
        # Calculate recovery percentage: how much of the dip was recovered?
        # If price dropped from 100 to 80 (peak to entry), and recovered to 95,
        # that's recovering 15 of the 20 point drop = 75% recovery
        dip_size = peak_price - entry_price  # The size of the drop
        recovery_prices = close.iloc[cross_idx:recovery_end_idx]
        max_price_after = float(recovery_prices.max()) if len(recovery_prices) > 0 else entry_price
        
        # Track day-by-day recovery to find when threshold was hit
        days_to_threshold_recovery = None
        recovery_threshold = self.config.recovery_win_threshold
        
        if dip_size > 0:
            recovery_amount = max_price_after - entry_price
            recovery_pct = min(100.0, (recovery_amount / dip_size) * 100)
            
            # Find first day that hit recovery threshold
            for day_offset in range(1, len(recovery_prices)):
                price_at_day = recovery_prices.iloc[day_offset]
                day_recovery_pct = ((price_at_day - entry_price) / dip_size) * 100
                if day_recovery_pct >= recovery_threshold:
                    days_to_threshold_recovery = day_offset
                    break
        else:
            recovery_pct = 100.0 if max_price_after >= peak_price else 0.0
        
        # Recovery velocity = how fast it recovered (% per day)
        # Higher is better - recovering 60% in 10 days (6%/day) beats 80% in 80 days (1%/day)
        if days_to_threshold_recovery and days_to_threshold_recovery > 0:
            # Velocity based on reaching threshold
            recovery_velocity = recovery_threshold / days_to_threshold_recovery
        elif recovery_pct > 0:
            # Fallback: use final recovery_pct over max_recovery_days
            recovery_velocity = recovery_pct / self.config.max_recovery_days
        else:
            recovery_velocity = 0.0
        
        if recovered:
            recovery_mask = future_close >= peak_price
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                if hasattr(recovery_idx, 'date'):
                    recovery_date = recovery_idx
                    recovery_days = (recovery_idx - df.index[cross_idx]).days
                else:
                    recovery_days = int(recovery_mask.argmax())
        
        # Volatility percentile
        vol_percentile = self._calc_vol_percentile(df, cross_idx)
        
        # Near earnings/dividend
        if isinstance(entry_date, datetime):
            near_earnings = self._is_near_event(entry_date, earnings_dates)
            near_dividend = self._is_near_event(entry_date, dividend_dates)
        else:
            near_earnings = False
            near_dividend = False
        
        # Calculate return at recovery (sell when price returns to entry price)
        # This is the realistic "sell at break-even" strategy
        return_at_recovery = 0.0
        days_held_to_recovery = 0
        max_wait_days = max(self.config.holding_periods)  # Use max holding as timeout
        
        for day_offset in range(1, min(max_wait_days, len(close) - cross_idx)):
            price_at_day = float(close.iloc[cross_idx + day_offset])
            if price_at_day >= entry_price:
                # Price recovered to entry - calculate actual return (may be slightly positive)
                return_at_recovery = ((price_at_day / entry_price) - 1) * 100
                days_held_to_recovery = day_offset
                break
        else:
            # Did not recover within max_wait_days - calculate return at end of period
            end_idx = min(cross_idx + max_wait_days, len(close) - 1)
            final_price = float(close.iloc[end_idx])
            return_at_recovery = ((final_price / entry_price) - 1) * 100
            days_held_to_recovery = max_wait_days
        
        return DipEvent(
            entry_date=entry_date if isinstance(entry_date, datetime) else datetime.now(),
            entry_price=entry_price,
            threshold_crossed=threshold,
            peak_price=peak_price,
            min_price_after_entry=min_price_after,
            max_adverse_excursion=max_adverse_excursion,
            further_drop_pct=further_drop_pct,
            recovered=recovered,
            recovery_pct=recovery_pct,
            recovery_date=recovery_date,
            recovery_days=recovery_days,
            days_to_threshold_recovery=days_to_threshold_recovery,
            recovery_velocity=recovery_velocity,
            return_at_recovery=return_at_recovery,
            days_held_to_recovery=days_held_to_recovery,
            returns=returns,
            max_return=max_return,
            volatility_percentile=vol_percentile,
            near_earnings=near_earnings,
            near_dividend=near_dividend,
            is_outlier=False,  # Will be set by _filter_outliers
        )
    
    def _filter_outliers(
        self,
        dip_events: list[DipEvent],
        df: pd.DataFrame,
    ) -> tuple[list[DipEvent], list[DipEvent]]:
        """
        Separate regular dip events from BLACK SWAN outliers.
        
        Outliers are ONLY extreme crash events that we want to exclude from
        optimal threshold calculation - things like:
        - HOOD's -90% post-IPO crash
        - COVID crash events
        - 2008 financial crisis levels
        
        We do NOT filter out -20%, -25%, -30% dips just because they're rare.
        Those are exactly the buying opportunities we're looking for!
        
        Outlier criteria (must meet BOTH):
        1. Extreme depth: threshold <= -40% (absolute floor for most stocks)
        2. Extreme MAE: dropped another 30%+ after entry (continuation disaster)
        
        OR:
        3. Statistical outlier: z-score > 3.0 (very extreme, not 2.5)
        
        Returns:
            (regular_events, outlier_events)
        """
        if not dip_events:
            return [], []
        
        # Get all unique thresholds crossed
        thresholds = [e.threshold_crossed for e in dip_events]
        
        # Calculate z-scores for threshold depths
        mean_threshold = np.mean(thresholds)
        std_threshold = np.std(thresholds) if len(thresholds) > 1 else 1.0
        
        regular_events = []
        outlier_events = []
        
        # Absolute threshold for black swan detection
        # -40% is already a crash for most stocks
        BLACK_SWAN_THRESHOLD = -40.0
        CATASTROPHIC_MAE = -30.0  # If it dropped another 30% after entry
        EXTREME_ZSCORE = 3.0  # More permissive than before (was 2.5)
        
        for event in dip_events:
            mae = event.max_adverse_excursion or 0
            
            # Check for black swan: extremely deep AND continued to crash
            is_black_swan = (
                event.threshold_crossed <= BLACK_SWAN_THRESHOLD and
                mae <= CATASTROPHIC_MAE
            )
            
            # Check z-score: statistical outlier (very extreme)
            z_score = (event.threshold_crossed - mean_threshold) / std_threshold if std_threshold > 0 else 0
            is_statistical_outlier = abs(z_score) > EXTREME_ZSCORE
            
            if is_black_swan or is_statistical_outlier:
                event.is_outlier = True
                outlier_events.append(event)
            else:
                regular_events.append(event)
        
        return regular_events, outlier_events
    
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
        """
        Calculate statistics for a specific dip threshold (V2: risk-adjusted).
        
        With threshold-crossing detection, we filter events where this exact
        threshold was crossed. Each event tracks MAE and forward returns.
        
        V2 additions:
        - Sharpe/Sortino ratios for risk-adjusted returns
        - CVaR (Conditional Value at Risk) for tail risk
        - MAE statistics for continuation risk
        - Continuation probability (P(drops another X%))
        """
        # Filter events where this exact threshold was crossed
        relevant_dips = [d for d in dip_events if int(d.threshold_crossed) == int(threshold)]
        
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
        
        # Recovery percentage stats - more meaningful than just "recovered" yes/no
        # recovery_pct: how much of the dip was recovered (100% = full recovery)
        recovery_pcts = [d.recovery_pct for d in relevant_dips]
        avg_recovery_pct = float(np.mean(recovery_pcts)) if recovery_pcts else 0.0
        # Use configurable threshold for "good enough" recovery
        recovery_threshold = self.config.recovery_win_threshold
        recovery_threshold_rate = sum(1 for r in recovery_pcts if r >= recovery_threshold) / n * 100 if n > 0 else 0.0
        full_recovery_rate = sum(1 for r in recovery_pcts if r >= 100) / n * 100 if n > 0 else 0.0
        
        # Dynamic recovery metrics - how FAST did it recover?
        days_to_threshold = [d.days_to_threshold_recovery for d in relevant_dips if d.days_to_threshold_recovery is not None]
        avg_days_to_threshold = float(np.mean(days_to_threshold)) if days_to_threshold else self.config.max_recovery_days
        
        velocities = [d.recovery_velocity for d in relevant_dips if d.recovery_velocity > 0]
        avg_recovery_velocity = float(np.mean(velocities)) if velocities else 0.0
        
        # Return stats for each holding period
        avg_returns: dict[int, float] = {}
        total_profit: dict[int, float] = {}
        total_profit_compounded: dict[int, float] = {}
        win_rates: dict[int, float] = {}
        sharpe_ratios: dict[int, float] = {}
        sortino_ratios: dict[int, float] = {}
        cvar: dict[int, float] = {}
        
        for period in self.config.holding_periods:
            returns = [d.returns.get(period) for d in relevant_dips if d.returns.get(period) is not None]
            if returns:
                avg_ret = float(np.mean(returns))
                avg_returns[period] = avg_ret
                total_profit[period] = n * avg_ret  # Simple sum: N × avg_return
                
                # Compounded return: (1+r1) × (1+r2) × ... × (1+rn) - 1
                # Convert percentages to decimals for compounding
                compounded = 1.0
                for r in returns:
                    compounded *= (1 + r / 100)
                total_profit_compounded[period] = (compounded - 1) * 100
                
                win_rates[period] = sum(1 for r in returns if r > 0) / len(returns) * 100
                
                # Sharpe ratio: mean / std (annualized for holding period)
                std_returns = np.std(returns) if len(returns) > 1 else 1.0
                sharpe_ratios[period] = float(np.mean(returns) / std_returns) if std_returns > 0 else 0.0
                
                # Sortino ratio: mean / downside_std (only negative returns)
                negative_returns = [r for r in returns if r < 0]
                downside_std = np.std(negative_returns) if len(negative_returns) > 1 else 1.0
                sortino_ratios[period] = float(np.mean(returns) / downside_std) if downside_std > 0 else 0.0
                
                # CVaR (Conditional Value at Risk) - mean of worst X% of returns
                sorted_returns = sorted(returns)
                n_tail = max(1, int(len(sorted_returns) * (1 - self.config.cvar_percentile)))
                cvar[period] = float(np.mean(sorted_returns[:n_tail]))
            else:
                avg_returns[period] = 0.0
                total_profit[period] = 0.0
                total_profit_compounded[period] = 0.0
                win_rates[period] = 0.0
                sharpe_ratios[period] = 0.0
                sortino_ratios[period] = 0.0
                cvar[period] = 0.0
        
        # Recovery-based metrics (sell at break-even, not fixed hold period)
        # This is the realistic "I'll sell when I break even" strategy
        recovery_returns = [d.return_at_recovery for d in relevant_dips]
        recovery_days_held = [d.days_held_to_recovery for d in relevant_dips]
        
        avg_return_at_recovery = float(np.mean(recovery_returns)) if recovery_returns else 0.0
        avg_days_at_recovery = float(np.mean(recovery_days_held)) if recovery_days_held else 0.0
        
        # Compounded total profit at recovery
        compounded_recovery = 1.0
        for r in recovery_returns:
            compounded_recovery *= (1 + r / 100)
        total_profit_at_recovery = (compounded_recovery - 1) * 100
        
        # MAE (Maximum Adverse Excursion) statistics
        mae_values = [d.max_adverse_excursion for d in relevant_dips]
        max_further_drawdown = float(min(mae_values)) if mae_values else 0.0  # Most negative
        avg_further_drawdown = float(np.mean(mae_values)) if mae_values else 0.0
        
        # Continuation probability: P(drops another X% after entry)
        further_drop_threshold = self.config.further_drop_threshold
        continuation_events = [d for d in relevant_dips if d.further_drop_pct >= further_drop_threshold]
        prob_further_drop = len(continuation_events) / n * 100 if n > 0 else 0.0
        
        # Continuation risk level
        if prob_further_drop < 25:
            continuation_risk: Literal["low", "medium", "high"] = "low"
        elif prob_further_drop < 50:
            continuation_risk = "medium"
        else:
            continuation_risk = "high"
        
        # Outlier stats
        n_outliers = sum(1 for d in relevant_dips if d.is_outlier)
        is_outlier_dominated = n_outliers > n / 2
        
        # =========================================================================
        # SCORING V2: Risk-Adjusted Entry Score
        # =========================================================================
        
        primary_period = self.config.primary_holding_period
        
        # Get primary metrics
        primary_return = avg_returns.get(primary_period, 0.0)
        primary_win_rate = win_rates.get(primary_period, 0.0)
        primary_sharpe = sharpe_ratios.get(primary_period, 0.0)
        primary_sortino = sortino_ratios.get(primary_period, 0.0)
        
        occurrences_per_year = n / years if years > 0 else 0
        
        # --- Component scores (0-100 scale) ---
        
        # Return score: Reward returns up to 25%
        return_score = min(primary_return / 25, 1.0) * 100 if primary_return > 0 else 0
        
        # Risk-adjusted score: Use Sharpe/Sortino (whichever is better)
        # Sharpe > 1.0 is excellent, > 0.5 is good
        best_ratio = max(primary_sharpe, primary_sortino)
        risk_adjusted_score = min(best_ratio / 1.5, 1.0) * 100
        
        # Win rate score - USE RECOVERY-BASED WIN RATE
        # The old win_rate just checks "is price higher at day 90?" which is flawed
        # The real question: "did the stock recover back to (or near) highs?"
        win_score = recovery_threshold_rate  # Uses config.recovery_win_threshold
        
        # Frequency score: Sweet spot is 1-3 per year
        if occurrences_per_year < 0.5:
            frequency_score = 20
        elif occurrences_per_year < 1.0:
            frequency_score = 50
        elif occurrences_per_year < 1.5:
            frequency_score = 70
        elif occurrences_per_year <= 3.0:
            frequency_score = 100
        elif occurrences_per_year <= 5.0:
            frequency_score = 85
        else:
            frequency_score = 70
        
        # Recovery score - use recovery_threshold_rate not just "full recovery"
        recovery_score = recovery_threshold_rate
        
        # --- Gates and penalties ---
        
        # MAE gate: Penalize if avg MAE exceeds threshold * gate_multiplier
        abs_threshold = abs(threshold)
        max_acceptable_mae = abs_threshold * self.config.mae_gate_multiplier
        avg_mae = abs(avg_further_drawdown)
        
        if avg_mae > abs_threshold * self.config.mae_disqualify_multiplier:
            mae_penalty = 0.1  # Disqualify - too risky
        elif avg_mae > max_acceptable_mae:
            # Linear penalty between gate and disqualify
            excess = avg_mae - max_acceptable_mae
            max_excess = abs_threshold * (self.config.mae_disqualify_multiplier - self.config.mae_gate_multiplier)
            mae_penalty = 0.5 - (excess / max_excess * 0.4) if max_excess > 0 else 0.5
        else:
            mae_penalty = 1.0  # No penalty
        
        # Continuation probability gate
        if prob_further_drop > self.config.max_continuation_prob * 100:
            continuation_penalty = 0.3  # High continuation risk
        elif prob_further_drop > self.config.max_continuation_prob * 100 * 0.7:
            continuation_penalty = 0.6  # Moderate continuation risk
        else:
            continuation_penalty = 1.0  # Low continuation risk
        
        # Extreme threshold penalty (same as V1)
        if abs_threshold <= 25:
            threshold_penalty = 1.0
        elif abs_threshold <= 30:
            threshold_penalty = 0.95
        elif abs_threshold <= 35:
            threshold_penalty = 0.80
        elif abs_threshold <= 40:
            threshold_penalty = 0.60
        elif abs_threshold <= 45:
            threshold_penalty = 0.40
        else:
            threshold_penalty = 0.25
        
        # Sample size confidence
        if n < self.config.min_dips_for_stats:
            sample_confidence = 0.0
        elif n < 4:
            sample_confidence = 0.6
        elif n < 8:
            sample_confidence = 0.8
        else:
            sample_confidence = 1.0
        
        # Recovery gate (same as V1)
        if recovery_rate < 50:
            recovery_gate = 0.1
        elif recovery_rate < 60:
            recovery_gate = 0.4
        elif recovery_rate < 70:
            recovery_gate = 0.6
        elif recovery_rate < 80:
            recovery_gate = 0.8
        else:
            recovery_gate = 1.0
        
        # --- Calculate scores ---
        
        # NEW V2 score: Uses configurable weights and includes risk-adjusted metrics
        config = self.config
        raw_v2_score = (
            return_score * config.weight_return +
            risk_adjusted_score * config.weight_risk_adjusted +
            win_score * config.weight_win_rate +
            frequency_score * config.weight_frequency +
            recovery_score * config.weight_recovery
        )
        
        # Apply V2 gates (includes MAE and continuation)
        entry_score = (
            raw_v2_score * 
            sample_confidence * 
            threshold_penalty * 
            recovery_gate * 
            mae_penalty * 
            continuation_penalty
        )
        
        # Confidence level
        if n >= 8 and years >= self.config.min_years_high_confidence:
            confidence: Literal["low", "medium", "high"] = "high"
        elif n >= 4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return DipThresholdStats(
            threshold_pct=threshold,
            n_occurrences=n,
            avg_per_year=occurrences_per_year,
            recovery_rate=recovery_rate,
            avg_recovery_days=float(np.mean(recovery_days)) if recovery_days else 0.0,
            median_recovery_days=float(np.median(recovery_days)) if recovery_days else 0.0,
            recovery_threshold_rate=recovery_threshold_rate,
            full_recovery_rate=full_recovery_rate,
            avg_recovery_pct=avg_recovery_pct,
            avg_days_to_threshold=avg_days_to_threshold,
            avg_recovery_velocity=avg_recovery_velocity,
            avg_returns=avg_returns,
            total_profit=total_profit,
            total_profit_compounded=total_profit_compounded,
            win_rates=win_rates,
            # Recovery-based metrics (sell at break-even strategy)
            avg_return_at_recovery=avg_return_at_recovery,
            total_profit_at_recovery=total_profit_at_recovery,
            avg_days_at_recovery=avg_days_at_recovery,
            max_further_drawdown=max_further_drawdown,
            avg_further_drawdown=avg_further_drawdown,
            sharpe_ratios=sharpe_ratios,
            sortino_ratios=sortino_ratios,
            cvar=cvar,
            prob_further_drop=prob_further_drop,
            continuation_risk=continuation_risk,
            n_outliers=n_outliers,
            is_outlier_dominated=is_outlier_dominated,
            entry_score=entry_score,
            confidence=confidence,
        )
    
    def _find_optimal_threshold(
        self,
        stats: list[DipThresholdStats],
    ) -> float:
        """
        Find the optimal dip threshold - WHERE THE STOCK ACTUALLY BOTTOMS OUT.
        
        The goal is NOT to find the "safest" shallow dip.
        The goal IS to find where the stock typically stops dropping and recovers.
        
        Key insight: If -10% has MAE of -8%, that means it typically drops to -18%
        before recovering. So -10% is NOT optimal - you're buying 8% too early!
        
        The optimal entry has LOW MAE RATIO - meaning you're buying close to 
        where it actually bottoms out. A dip with 0% MAE = perfect timing.
        
        Scoring priorities:
        1. LOW MAE RATIO (primary) - buying near the actual bottom
        2. Good recovery rate - it does bounce back from there
        3. Reasonable Sharpe - risk-adjusted returns are decent
        4. Enough occurrences - statistically meaningful
        """
        if not stats:
            return -10.0  # Default to 10% dip
        
        # Filter to meaningful dip thresholds only
        min_threshold = self.config.min_optimal_threshold
        min_occurrences = self.config.min_occurrences_for_optimal
        
        valid_stats = [
            s for s in stats
            if s.threshold_pct <= min_threshold  # At least -5% (deeper is more negative)
            and s.n_occurrences >= min_occurrences  # Enough data
        ]
        
        if not valid_stats:
            # Fallback: find deepest threshold with any occurrences
            with_occurrences = [s for s in stats if s.n_occurrences >= 2]
            if with_occurrences:
                # Pick deepest one as safer default for volatile stocks
                return min(s.threshold_pct for s in with_occurrences)
            return -10.0  # Default fallback
        
        # =========================================================================
        # V4: Find where the stock ACTUALLY bottoms out (LOW MAE = near bottom)
        # =========================================================================
        
        best_threshold = valid_stats[0].threshold_pct
        best_composite_score = float('-inf')
        
        for s in valid_stats:
            threshold = s.threshold_pct
            abs_threshold = abs(threshold)
            
            # Get metrics
            primary_return = s.avg_returns.get(self.config.primary_holding_period, 0)
            recovery_win_rate = s.recovery_threshold_rate
            sharpe = s.sharpe_ratios.get(self.config.primary_holding_period, 0)
            
            # MAE ratio: how much further does it drop after entry?
            # Lower = better (you're buying closer to the actual bottom)
            avg_mae = abs(s.avg_further_drawdown) if s.avg_further_drawdown else 0
            mae_ratio = avg_mae / abs_threshold if abs_threshold > 0 else 1.0
            
            # =====================================================
            # SCORING: Prioritize buying near the actual bottom
            # =====================================================
            
            # 1. MAE EFFICIENCY (0-50 points) - PRIMARY FACTOR
            # mae_ratio < 0.2 = excellent (buying very close to bottom)
            # mae_ratio = 0.5 = okay (drops 50% more after entry)
            # mae_ratio > 1.0 = BAD (drops more than the threshold itself - way too early!)
            if mae_ratio <= 0.15:
                mae_score = 50  # Near-perfect timing
            elif mae_ratio <= 0.25:
                mae_score = 45
            elif mae_ratio <= 0.35:
                mae_score = 40
            elif mae_ratio <= 0.50:
                mae_score = 35
            elif mae_ratio <= 0.60:
                mae_score = 28
            elif mae_ratio <= 0.70:
                mae_score = 22
            elif mae_ratio <= 0.80:
                mae_score = 16
            elif mae_ratio <= 0.90:
                mae_score = 10
            elif mae_ratio <= 1.0:
                mae_score = 5  # Barely acceptable - drops as much as threshold
            else:
                # PENALTY: Drops MORE than the threshold = buying way too early
                mae_score = max(-20, 5 - (mae_ratio - 1.0) * 25)
            
            # 2. Recovery rate (0-25 points)
            # Does it actually bounce back from this level?
            recovery_score = min(recovery_win_rate / 4, 25)  # 100% = 25 pts
            
            # 3. Sharpe ratio (0-15 points)
            # Risk-adjusted returns matter
            sharpe_score = min(max(sharpe * 10, 0), 15)
            
            # 4. Return potential (0-10 points)
            # Higher returns from deeper dips
            if primary_return > 0:
                return_score = min(primary_return / 3, 10)  # 30% = 10 pts
            else:
                return_score = max(-10, primary_return / 2)  # Penalty for losses
            
            # 5. Depth bonus (0-10 points)
            # Deeper dips = bigger opportunities, slight bonus
            depth_bonus = min(abs_threshold / 5, 10)  # -50% = 10 pts
            
            # 6. Sample size confidence (0.6-1.0 multiplier)
            n = s.n_occurrences
            if n < 3:
                sample_mult = 0.5
            elif n < 5:
                sample_mult = 0.7
            elif n < 8:
                sample_mult = 0.85
            else:
                sample_mult = 1.0
            
            # Composite score
            raw_score = mae_score + recovery_score + sharpe_score + return_score + depth_bonus
            composite = raw_score * sample_mult
            
            if composite > best_composite_score:
                best_composite_score = composite
                best_threshold = threshold
        
        return best_threshold
    
    def _find_max_profit_threshold(
        self,
        stats: list[DipThresholdStats],
        min_threshold: float,
    ) -> tuple[float, float]:
        """
        Find the threshold that maximizes TOTAL expected profit.
        
        This is different from risk-adjusted optimal:
        - Risk-adjusted: minimizes pain (low MAE) with good returns
        - Max profit: maximizes N × avg_return regardless of MAE
        
        Args:
            stats: List of threshold statistics
            min_threshold: Minimum dip threshold from symbol config (e.g., -10%)
                          We won't consider shallower dips than this.
        
        Returns:
            tuple of (optimal_threshold, total_profit)
        """
        if not stats:
            return -10.0, 0.0
        
        # Filter to dips at or deeper than the symbol's minimum threshold
        # min_threshold is negative (e.g., -10), so we want threshold <= min_threshold
        # Also require positive profit - negative profit is not "profit optimized"
        valid_stats = [
            s for s in stats
            if s.threshold_pct <= min_threshold  # Respect minimum dip threshold
            and s.n_occurrences >= 2  # Need at least 2 occurrences for any confidence
            and s.total_profit_at_recovery > 0  # Must have positive profit
        ]
        
        if not valid_stats:
            # Fallback: try to find any threshold with positive profit (ignore min_threshold)
            fallback_stats = [
                s for s in stats
                if s.n_occurrences >= 2
                and s.total_profit_at_recovery > 0
            ]
            if fallback_stats:
                # Pick the one with best profit among fallbacks
                best = max(fallback_stats, key=lambda s: s.total_profit_at_recovery)
                return best.threshold_pct, best.total_profit_at_recovery
            # No positive profit thresholds - return minimum with 0 profit
            return min_threshold, 0.0
        
        # Find threshold with highest total profit at recovery (compounded)
        best_threshold = valid_stats[0].threshold_pct
        best_total_profit = 0.0
        
        for s in valid_stats:
            # Use recovery-based profit (sell at break-even), not fixed 90d hold
            total_profit = s.total_profit_at_recovery
            
            if total_profit > best_total_profit:
                best_total_profit = total_profit
                best_threshold = s.threshold_pct
        
        return best_threshold, best_total_profit
    
    def _evaluate_current_opportunity(
        self,
        current_drawdown: float,
        optimal_threshold: float,
        stats: list[DipThresholdStats],
        fundamentals: dict | None,
    ) -> tuple[bool, float, str, Literal["low", "medium", "high"]]:
        """Evaluate if current price is a buy opportunity.
        
        V2: Now includes continuation risk in decision-making.
        
        Returns:
            tuple of (is_buy, signal_strength, reason, continuation_risk)
        """
        # Check fundamentals FIRST - they gate the buy decision
        fund_healthy, fund_notes = self._check_fundamentals(fundamentals)
        
        # Find stats for current drawdown level
        applicable_stats = [s for s in stats if current_drawdown <= s.threshold_pct]
        
        if not applicable_stats:
            # Not at any significant dip level
            if current_drawdown > -3:
                return False, 0.0, f"Near highs (only {current_drawdown:.1f}% from peak)", "low"
            else:
                return False, 20.0, f"Minor pullback ({current_drawdown:.1f}%), wait for deeper dip", "low"
        
        # Use the deepest applicable threshold
        best_stats = max(applicable_stats, key=lambda s: abs(s.threshold_pct))
        
        # Get metrics for primary holding period
        primary_period = self.config.primary_holding_period
        
        primary_return = best_stats.avg_returns.get(primary_period, 0.0)
        primary_win_rate = best_stats.win_rates.get(primary_period, 0.0)
        primary_sharpe = best_stats.sharpe_ratios.get(primary_period, 0.0)
        primary_sortino = best_stats.sortino_ratios.get(primary_period, 0.0)
        continuation_risk = best_stats.continuation_risk
        
        # Calculate signal strength (V2: includes risk-adjusted metrics)
        
        # Depth score: How deep is the dip?
        depth_score = min(abs(current_drawdown) / 20, 1.0) * 30  # Max 30 points for 20%+ dip
        
        # Win rate score
        win_score = primary_win_rate * 0.25  # Max 25 points for 100% win rate
        
        # Return score
        return_score = min(primary_return / 15, 1.0) * 15  # Max 15 points for 15%+ avg return
        
        # Risk-adjusted score (NEW): Sharpe/Sortino contribution
        best_ratio = max(primary_sharpe, primary_sortino)
        risk_adjusted_score = min(best_ratio / 1.0, 1.0) * 15  # Max 15 points for Sharpe >= 1.0
        
        # Continuation risk penalty (NEW)
        if continuation_risk == "high":
            continuation_penalty = -20.0  # Heavy penalty for high continuation risk
        elif continuation_risk == "medium":
            continuation_penalty = -10.0  # Moderate penalty
        else:
            continuation_penalty = 0.0  # No penalty for low risk
        
        # MAE penalty (NEW): Penalize if avg further drawdown is excessive
        abs_threshold = abs(best_stats.threshold_pct)
        avg_mae = abs(best_stats.avg_further_drawdown)
        max_acceptable_mae = abs_threshold * self.config.mae_gate_multiplier
        
        if avg_mae > abs_threshold * self.config.mae_disqualify_multiplier:
            mae_penalty = -30.0  # Disqualify - way too risky
        elif avg_mae > max_acceptable_mae:
            mae_penalty = -15.0  # Significant penalty
        else:
            mae_penalty = 0.0  # Acceptable
        
        # Fundamentals adjustment
        fundamentals_score = 0.0
        if fund_healthy:
            fundamentals_score = 10.0  # Bonus for healthy fundamentals
        else:
            fundamentals_score = -20.0  # Penalty for unhealthy fundamentals
        
        signal_strength = (
            depth_score + 
            win_score + 
            return_score + 
            risk_adjusted_score + 
            continuation_penalty + 
            mae_penalty + 
            fundamentals_score
        )
        
        # Clamp to 0-100
        signal_strength = max(0.0, min(100.0, signal_strength))
        
        # Determine if buy - MUST have healthy fundamentals AND acceptable continuation risk
        is_buy = (
            current_drawdown <= optimal_threshold 
            and signal_strength >= 50 
            and fund_healthy  # Gate: fundamentals must be healthy
            and continuation_risk != "high"  # Gate: don't buy into falling knife
        )
        
        # Generate reason
        if is_buy:
            reason = (
                f"At {current_drawdown:.1f}% dip (optimal: {optimal_threshold:.0f}%). "
                f"Historically {primary_win_rate:.0f}% win rate, "
                f"avg {primary_return:.1f}% return in {primary_period} days. "
                f"Sharpe: {primary_sharpe:.2f}. "
                f"Recovery in ~{best_stats.avg_recovery_days:.0f} days. "
                f"Continuation risk: {continuation_risk}. "
                f"Fundamentals: {'✓ healthy' if fund_healthy else '⚠ concerns'}."
            )
        elif continuation_risk == "high":
            reason = (
                f"At {current_drawdown:.1f}% dip but HIGH continuation risk "
                f"({best_stats.prob_further_drop:.0f}% chance of dropping {self.config.further_drop_threshold:.0f}%+ more). "
                f"Avg MAE: {best_stats.avg_further_drawdown:.1f}%. "
                f"Wait for more confirmation or deeper dip."
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
                f"Entry score: {best_stats.entry_score:.1f}."
            )
        
        return is_buy, signal_strength, reason, continuation_risk
    
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
        # Risk-adjusted optimal (less pain, better timing)
        "optimal_dip_threshold": result.optimal_dip_threshold,
        "optimal_entry_price": result.optimal_entry_price,
        # Max profit optimal (more opportunities, higher total return)
        "max_profit_threshold": result.max_profit_threshold,
        "max_profit_entry_price": result.max_profit_entry_price,
        "max_profit_total_return": result.max_profit_total_return,
        # Current opportunity
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
        "continuation_risk": result.continuation_risk,
        "data_years": result.data_years,
        "confidence": result.confidence,
        "outlier_events": result.outlier_events,
        "threshold_analysis": [
            {
                "threshold": s.threshold_pct,
                "occurrences": s.n_occurrences,
                "per_year": s.avg_per_year,
                # OPTIMAL HOLDING PERIOD - dynamically computed per threshold
                "optimal_holding_days": s.optimal_holding_days,
                "optimal_avg_return": s.optimal_avg_return,
                "optimal_win_rate": s.optimal_win_rate,
                "optimal_total_profit": s.optimal_total_profit,
                "optimal_sharpe": s.optimal_sharpe,
                # V2: Use dict-based metrics with primary period fallback
                "win_rate": s.win_rates.get(90, s.win_rates.get(60, 0.0)),
                "avg_return": s.avg_returns.get(90, s.avg_returns.get(60, 0.0)),
                "total_profit": s.total_profit.get(90, s.total_profit.get(60, 0.0)),
                "total_profit_compounded": s.total_profit_compounded.get(90, s.total_profit_compounded.get(60, 0.0)),
                "sharpe_ratio": s.sharpe_ratios.get(90, s.sharpe_ratios.get(60, 0.0)),
                "sortino_ratio": s.sortino_ratios.get(90, s.sortino_ratios.get(60, 0.0)),
                "cvar": s.cvar.get(90, s.cvar.get(60, 0.0)),
                "recovery_rate": s.recovery_rate,
                "recovery_threshold_rate": s.recovery_threshold_rate,
                "avg_recovery_days": s.avg_recovery_days,
                "avg_days_to_threshold": s.avg_days_to_threshold,
                "avg_recovery_velocity": s.avg_recovery_velocity,
                # Recovery-based metrics (sell at break-even strategy)
                "avg_return_at_recovery": s.avg_return_at_recovery,
                "total_profit_at_recovery": s.total_profit_at_recovery,
                "avg_days_at_recovery": s.avg_days_at_recovery,
                "max_further_drawdown": s.max_further_drawdown,
                "avg_further_drawdown": s.avg_further_drawdown,
                "prob_further_drop": s.prob_further_drop,
                "continuation_risk": s.continuation_risk,
                "entry_score": s.entry_score,
                "confidence": s.confidence,
                # Legacy aliases for backward compatibility
                "win_rate_60d": s.win_rate_60d,
                "avg_return_60d": s.avg_return_60d,
            }
            for s in result.threshold_stats
        ],
    }


def get_dip_signal_triggers(result: OptimalDipEntry) -> dict[str, Any]:
    """Convert OptimalDipEntry to signal triggers for chart overlay.
    
    Returns a dict matching the format expected by the API:
    {
        "threshold_pct": -0.15,
        "n_trades": 8,
        "win_rate": 0.75,
        "total_return_pct": 0.45,
        "triggers": [
            {"date": "2024-01-15", "signal_type": "entry", "price": 145.50, ...},
            {"date": "2024-02-20", "signal_type": "exit", "price": 158.30, ...},
        ]
    }
    """
    triggers = []
    
    # Use max profit threshold for determining which dips to show
    # (this is the threshold that the backtest is based on)
    target_threshold = result.max_profit_threshold
    if not target_threshold:
        target_threshold = result.optimal_dip_threshold
    if not target_threshold:
        return {"threshold_pct": 0.0, "n_trades": 0, "triggers": []}
    
    # Get events that match the target threshold from ALL dip events
    # Use all_dip_events which contains the full history, not just recent 10
    all_events = result.all_dip_events if result.all_dip_events else result.recent_dips
    
    matching_events = [
        e for e in all_events
        if abs(e.threshold_crossed - target_threshold) < 2.0  # Within 2% of target
    ]
    
    # Also include events close to target threshold
    for event in all_events:
        if event not in matching_events:
            # Include if close to target threshold
            if abs(event.threshold_crossed - target_threshold) < 5.0:
                matching_events.append(event)
    
    # Sort by entry date
    matching_events.sort(key=lambda e: e.entry_date)
    
    # Compute metrics for these specific events
    n_trades = len(matching_events)
    wins = 0
    total_return = 0.0
    
    for event in matching_events:
        # Get optimal holding period return
        optimal_holding = getattr(result, "typical_recovery_days", 60) or 60
        optimal_holding = min(60, max(1, int(optimal_holding)))
        
        # Try to get return at optimal holding, fallback to best available
        trade_return = event.returns.get(optimal_holding)
        if trade_return is None:
            # Try nearby periods
            for period in [60, 40, 30, 20]:
                trade_return = event.returns.get(period)
                if trade_return is not None:
                    optimal_holding = period
                    break
        
        if trade_return is None:
            trade_return = event.max_return or 0.0
        
        is_win = trade_return > 0
        if is_win:
            wins += 1
        total_return += trade_return
        
        # Add entry trigger
        entry_date_str = event.entry_date.strftime("%Y-%m-%d") if hasattr(event.entry_date, "strftime") else str(event.entry_date)[:10]
        triggers.append({
            "date": entry_date_str,
            "signal_type": "entry",
            "price": event.entry_price,
            "threshold_pct": event.threshold_crossed / 100.0,  # Convert to decimal
            "return_pct": 0.0,
            "holding_days": 0,
        })
        
        # Add exit trigger (entry_date + optimal holding days)
        exit_date = event.entry_date + timedelta(days=optimal_holding) if hasattr(event.entry_date, "__add__") else None
        if exit_date:
            exit_date_str = exit_date.strftime("%Y-%m-%d") if hasattr(exit_date, "strftime") else str(exit_date)[:10]
            # Estimate exit price from return
            exit_price = event.entry_price * (1 + trade_return / 100.0) if trade_return else event.entry_price
            triggers.append({
                "date": exit_date_str,
                "signal_type": "exit",
                "price": round(exit_price, 2),
                "threshold_pct": event.threshold_crossed / 100.0,
                "return_pct": trade_return / 100.0 if trade_return else 0.0,  # Convert to decimal
                "holding_days": optimal_holding,
            })
    
    win_rate = wins / n_trades if n_trades > 0 else 0.0
    avg_return = total_return / n_trades if n_trades > 0 else 0.0
    
    return {
        "threshold_pct": target_threshold / 100.0,  # Convert to decimal
        "n_trades": n_trades,
        "win_rate": win_rate,
        "total_return_pct": total_return / 100.0,  # Convert to decimal
        "triggers": triggers,
    }
