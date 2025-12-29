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
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration - Hyperparameters for future optimization
# =============================================================================


@dataclass
class DipOptimizerConfig:
    """
    Configurable parameters for the dip entry optimizer.
    
    All values have sensible defaults but can be tuned via hyperparameter
    optimization in the future.
    """
    # Holding periods for return calculations
    holding_periods: tuple[int, ...] = (30, 60, 90)
    primary_holding_period: int = 90  # Used for Sharpe/Sortino
    secondary_holding_period: int = 60  # For backward compatibility
    
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
    
    # Lookback and recovery
    lookback_years: int = 5
    recovery_max_days: int = 180
    min_dips_for_stats: int = 2
    
    # Confidence thresholds
    min_years_high_confidence: int = 3  # Flag as "low confidence" if less
    
    # Optimal threshold selection
    min_optimal_threshold: float = -5.0  # Don't consider dips shallower than this as "optimal"
    min_occurrences_for_optimal: int = 3  # Need at least this many occurrences


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
    recovery_days: int | None = None
    
    # Returns if bought at entry (keyed by holding period days)
    returns: dict[int, float | None] = field(default_factory=dict)  # {30: 5.2, 60: 8.1, 90: 12.3}
    max_return: float | None = None  # Max return within holding period
    
    # Context
    volatility_percentile: float = 50.0
    near_earnings: bool = False
    near_dividend: bool = False
    
    # Outlier flag
    is_outlier: bool = False
    
    # Legacy compatibility properties
    @property
    def dip_date(self) -> datetime:
        return self.entry_date
    
    @property
    def dip_price(self) -> float:
        return self.entry_price
    
    @property
    def drawdown_pct(self) -> float:
        return self.threshold_crossed
    
    @property
    def return_30d(self) -> float | None:
        return self.returns.get(30)
    
    @property
    def return_60d(self) -> float | None:
        return self.returns.get(60)
    
    @property
    def return_90d(self) -> float | None:
        return self.returns.get(90)


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
    win_rates: dict[int, float] = field(default_factory=dict)  # {30: 60.0, 60: 70.0, 90: 75.0}
    
    # Legacy compatibility
    @property
    def avg_return_30d(self) -> float:
        return self.avg_returns.get(30, 0.0)
    
    @property
    def avg_return_60d(self) -> float:
        return self.avg_returns.get(60, 0.0)
    
    @property
    def avg_return_90d(self) -> float:
        return self.avg_returns.get(90, 0.0)
    
    @property
    def win_rate_30d(self) -> float:
        return self.win_rates.get(30, 0.0)
    
    @property
    def win_rate_60d(self) -> float:
        return self.win_rates.get(60, 0.0)
    
    @property
    def win_rate_90d(self) -> float:
        return self.win_rates.get(90, 0.0)
    
    # Risk metrics - Maximum Adverse Excursion
    max_further_drawdown: float = 0.0  # Worst MAE across all events
    avg_further_drawdown: float = 0.0  # Average MAE
    
    # Risk-adjusted metrics (keyed by holding period)
    sharpe_ratios: dict[int, float] = field(default_factory=dict)  # {60: 0.8, 90: 1.2}
    sortino_ratios: dict[int, float] = field(default_factory=dict)  # {60: 1.0, 90: 1.5}
    cvar: dict[int, float] = field(default_factory=dict)  # {60: -5.2, 90: -3.1}
    
    # Legacy compatibility
    @property
    def sharpe_ratio(self) -> float:
        return self.sharpe_ratios.get(90, 0.0)
    
    @property
    def sortino_ratio(self) -> float:
        return self.sortino_ratios.get(90, 0.0)
    
    @property
    def cvar_95(self) -> float:
        return self.cvar.get(90, 0.0)
    
    # Continuation probability
    prob_further_drop: float = 0.0  # P(drops another X% after entry)
    continuation_risk: Literal["low", "medium", "high"] = "medium"
    
    # Outlier info
    n_outliers: int = 0
    is_outlier_dominated: bool = False  # True if most events are outliers
    
    # Scores
    entry_score: float = 0.0  # New risk-adjusted score
    legacy_entry_score: float = 0.0  # Old score for comparison
    
    # Confidence
    confidence: Literal["low", "medium", "high"] = "medium"


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
    continuation_risk: Literal["low", "medium", "high"] = "medium"
    
    # Threshold analysis
    threshold_stats: list[DipThresholdStats] = field(default_factory=list)
    
    # Historical dip events
    recent_dips: list[DipEvent] = field(default_factory=list)
    
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
        config = DipOptimizerConfig(primary_holding_period=90)
        optimizer = DipEntryOptimizer(config=config)
        result = optimizer.analyze(df, "NVDA", fundamentals)
        
        print(f"Optimal buy level: {result.optimal_dip_threshold}% dip")
        print(f"Set limit order at: ${result.optimal_entry_price}")
    """
    
    # Dynamic thresholds: test every percentage from 1% to 50%
    DIP_THRESHOLDS = list(range(-1, -51, -1))  # [-1, -2, -3, ..., -50]
    
    def __init__(
        self,
        config: DipOptimizerConfig | None = None,
        # Legacy parameters for backward compatibility
        lookback_years: int | None = None,
        recovery_max_days: int | None = None,
        min_dips_for_stats: int | None = None,
    ):
        """
        Args:
            config: Configuration object with all parameters
            lookback_years: (Legacy) Years of history to analyze
            recovery_max_days: (Legacy) Max days to consider for recovery
            min_dips_for_stats: (Legacy) Minimum dip occurrences for valid stats
        """
        self.config = config or DipOptimizerConfig()
        
        # Apply legacy overrides if provided
        if lookback_years is not None:
            self.config.lookback_years = lookback_years
        if recovery_max_days is not None:
            self.config.recovery_max_days = recovery_max_days
        if min_dips_for_stats is not None:
            self.config.min_dips_for_stats = min_dips_for_stats
        
        # Shortcuts for frequently used config values
        self.lookback_years = self.config.lookback_years
        self.recovery_max_days = self.config.recovery_max_days
        self.min_dips_for_stats = self.config.min_dips_for_stats
    
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
            if stats.n_occurrences >= self.min_dips_for_stats:
                threshold_stats.append(stats)
        
        # Find optimal threshold (using new risk-adjusted scoring)
        optimal = self._find_optimal_threshold(threshold_stats)
        
        # Current state analysis
        current_price = float(df["close"].iloc[-1])
        recent_high = float(df["close"].rolling(252).max().iloc[-1])  # 1-year high
        current_drawdown = (current_price / recent_high - 1) * 100
        
        # Calculate entry price
        optimal_entry_price = recent_high * (1 + optimal / 100) if optimal else current_price * 0.9
        
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
            optimal_dip_threshold=optimal,
            optimal_entry_price=optimal_entry_price,
            is_buy_now=is_buy_now,
            buy_signal_strength=signal_strength,
            signal_reason=reason,
            continuation_risk=cont_risk,
            threshold_stats=threshold_stats,
            recent_dips=regular_events[-10:],  # Last 10 dips
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
        
        # Recovery calculation
        future_close = close.iloc[cross_idx + 1:]
        recovered = any(future_close >= peak_price)
        recovery_days = None
        recovery_date = None
        
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
        
        return DipEvent(
            entry_date=entry_date if isinstance(entry_date, datetime) else datetime.now(),
            entry_price=entry_price,
            threshold_crossed=threshold,
            peak_price=peak_price,
            min_price_after_entry=min_price_after,
            max_adverse_excursion=max_adverse_excursion,
            further_drop_pct=further_drop_pct,
            recovered=recovered,
            recovery_date=recovery_date,
            recovery_days=recovery_days,
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
    
    def _find_all_dips(
        self,
        df: pd.DataFrame,
        earnings_dates: list[datetime] | None = None,
        dividend_dates: list[datetime] | None = None,
    ) -> list[DipEvent]:
        """
        LEGACY: Find all significant dip events using trough-based detection.
        Kept for backward compatibility. Use _find_threshold_crossings() instead.
        """
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
        
        # Return stats for each holding period
        avg_returns: dict[int, float] = {}
        win_rates: dict[int, float] = {}
        sharpe_ratios: dict[int, float] = {}
        sortino_ratios: dict[int, float] = {}
        cvar: dict[int, float] = {}
        
        for period in self.config.holding_periods:
            returns = [d.returns.get(period) for d in relevant_dips if d.returns.get(period) is not None]
            if returns:
                avg_returns[period] = float(np.mean(returns))
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
                win_rates[period] = 0.0
                sharpe_ratios[period] = 0.0
                sortino_ratios[period] = 0.0
                cvar[period] = 0.0
        
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
        
        # Win rate score
        win_score = primary_win_rate
        
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
        
        # Recovery score
        recovery_score = recovery_rate
        
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
        
        # LEGACY V1 score (for comparison)
        raw_v1_score = (
            return_score * 0.40 +
            win_score * 0.30 +
            frequency_score * 0.20 +
            recovery_score * 0.10
        )
        
        # V1 win rate gate
        if primary_win_rate < 40:
            win_rate_gate = 0.4
        elif primary_win_rate < 50:
            win_rate_gate = 0.6
        elif primary_win_rate < 60:
            win_rate_gate = 0.75
        elif primary_win_rate < 70:
            win_rate_gate = 0.85
        elif primary_win_rate < 80:
            win_rate_gate = 0.95
        else:
            win_rate_gate = 1.0
        
        legacy_entry_score = raw_v1_score * sample_confidence * threshold_penalty * recovery_gate * win_rate_gate
        
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
            avg_returns=avg_returns,
            win_rates=win_rates,
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
            legacy_entry_score=legacy_entry_score,
            confidence=confidence,
        )
    
    def _find_optimal_threshold(
        self,
        stats: list[DipThresholdStats],
    ) -> float:
        """
        Find the optimal dip threshold based on entry scores.
        
        V2: Filters out:
        - Shallow thresholds (< min_optimal_threshold) - daily noise, not real dips
        - Low occurrence thresholds - not statistically significant
        - Negative score thresholds - too risky
        
        V3 MAE-ADJUSTED SELECTION:
        The key insight is: if -15% typically drops another 8% (MAE=-8%), then
        buying at -15% means you'll likely see prices at -23% before recovery.
        
        The "effective entry" = threshold + avg_MAE
        
        The optimal threshold minimizes the gap between where you buy and where
        the price actually bottoms out. This is measured by:
        1. "Proximity score" = how close to the real bottom (low MAE = good)
        2. "Efficiency score" = MAE relative to threshold (MAE/threshold ratio)
        
        A -20% dip with -3% MAE is better than -10% with -8% MAE because:
        - First: you're buying at -20%, bottom is ~-23%, only 3% slippage
        - Second: you're buying at -10%, bottom is ~-18%, 8% slippage
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
        # V3: MAE-Adjusted Optimal Threshold Selection
        # =========================================================================
        
        # Calculate composite score for each threshold:
        # - High win rate (primary goal: make money)
        # - Low MAE ratio (buy close to bottom)
        # - Good Sharpe ratio (risk-adjusted returns)
        # - Reasonable frequency (not too rare)
        
        best_threshold = valid_stats[0].threshold_pct
        best_composite_score = float('-inf')
        
        for s in valid_stats:
            threshold = s.threshold_pct
            abs_threshold = abs(threshold)
            
            # Skip if no return data (too recent)
            primary_return = s.avg_returns.get(self.config.primary_holding_period, 0)
            primary_win_rate = s.win_rates.get(self.config.primary_holding_period, 0)
            
            # If no valid returns data, use a penalty
            if primary_win_rate == 0 and s.avg_returns.get(self.config.primary_holding_period) is None:
                # Recent events with no 90-day data yet - lower priority
                data_penalty = 0.5
            else:
                data_penalty = 1.0
            
            # 1. Win rate component (0-40 points)
            # 80%+ win rate = 40 points, 50% = 20 points
            win_component = min(primary_win_rate / 2, 40)
            
            # 2. MAE efficiency component (0-30 points)
            # Measures how close to the bottom you're buying
            # Lower MAE relative to threshold = better
            avg_mae = abs(s.avg_further_drawdown) if s.avg_further_drawdown else 0
            mae_ratio = avg_mae / abs_threshold if abs_threshold > 0 else 1.0
            
            # mae_ratio < 0.3 = excellent (30 pts), 0.5 = good (20 pts), > 1.0 = bad (0 pts)
            if mae_ratio <= 0.3:
                mae_component = 30
            elif mae_ratio <= 0.5:
                mae_component = 25
            elif mae_ratio <= 0.7:
                mae_component = 18
            elif mae_ratio <= 1.0:
                mae_component = 10
            else:
                mae_component = max(0, 10 - (mae_ratio - 1.0) * 10)
            
            # 3. Return component (0-20 points)
            # Reward positive returns, penalize losses
            if primary_return > 0:
                return_component = min(primary_return / 1.5, 20)  # 30% return = 20 pts
            else:
                return_component = max(-10, primary_return / 3)  # Losses penalized
            
            # 4. Sharpe component (0-10 points)
            sharpe = s.sharpe_ratios.get(self.config.primary_holding_period, 0)
            sharpe_component = min(max(sharpe * 10, 0), 10)
            
            # 5. Continuation risk penalty (-20 to 0)
            prob_drop = s.prob_further_drop
            if prob_drop > 60:
                continuation_penalty = -20
            elif prob_drop > 50:
                continuation_penalty = -15
            elif prob_drop > 40:
                continuation_penalty = -10
            elif prob_drop > 30:
                continuation_penalty = -5
            else:
                continuation_penalty = 0
            
            # Composite score
            composite = (
                win_component +
                mae_component +
                return_component +
                sharpe_component +
                continuation_penalty
            ) * data_penalty
            
            if composite > best_composite_score:
                best_composite_score = composite
                best_threshold = threshold
        
        return best_threshold
    
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
        "continuation_risk": result.continuation_risk,
        "data_years": result.data_years,
        "confidence": result.confidence,
        "outlier_events": result.outlier_events,
        "threshold_analysis": [
            {
                "threshold": s.threshold_pct,
                "occurrences": s.n_occurrences,
                "per_year": s.avg_per_year,
                # V2: Use dict-based metrics with primary period fallback
                "win_rate": s.win_rates.get(90, s.win_rates.get(60, 0.0)),
                "avg_return": s.avg_returns.get(90, s.avg_returns.get(60, 0.0)),
                "sharpe_ratio": s.sharpe_ratios.get(90, s.sharpe_ratios.get(60, 0.0)),
                "sortino_ratio": s.sortino_ratios.get(90, s.sortino_ratios.get(60, 0.0)),
                "cvar": s.cvar.get(90, s.cvar.get(60, 0.0)),
                "recovery_rate": s.recovery_rate,
                "avg_recovery_days": s.avg_recovery_days,
                "max_further_drawdown": s.max_further_drawdown,
                "avg_further_drawdown": s.avg_further_drawdown,
                "prob_further_drop": s.prob_further_drop,
                "continuation_risk": s.continuation_risk,
                "entry_score": s.entry_score,
                "legacy_entry_score": s.legacy_entry_score,
                "confidence": s.confidence,
                # Legacy aliases for backward compatibility
                "win_rate_60d": s.win_rate_60d,
                "avg_return_60d": s.avg_return_60d,
            }
            for s in result.threshold_stats
        ],
    }
