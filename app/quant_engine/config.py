"""
Quant Engine Central Configuration.

ALL optimization limits are defined HERE and ONLY HERE.
No hardcoded values anywhere else in the quant engine.

Philosophy:
- Parameters are DISCOVERED via hyperparameter optimization, not preset
- Only LIMITS are set (max values, edge case filters)
- All discovered parameters must be statistically validated (vs buy-and-hold)
- Overfitting prevention via walk-forward validation
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantLimits:
    """
    Central limits for hyperparameter optimization.
    
    These are CEILINGS and FILTERS, not preset values.
    The optimizer tests ALL values from 1 to max and finds the best.
    
    Example: max_holding_days=60 means test holding periods 1,2,3,...,60
    and pick the one with best risk-adjusted returns.
    """
    
    # =========================================================================
    # HOLDING PERIOD LIMITS
    # =========================================================================
    
    # Maximum holding period in days
    # Optimizer tests EVERY day from 1 to this value
    # Analysis shows beyond 60 days, alpha vs buy-and-hold goes negative
    max_holding_days: int = 60
    
    # Minimum holding period (filter out noise)
    min_holding_days: int = 1
    
    # =========================================================================
    # DIP THRESHOLD LIMITS
    # =========================================================================
    
    # Deepest dip to consider (e.g., -50 means test -1%, -2%, ..., -50%)
    max_dip_threshold_pct: int = 50
    
    # Shallowest dip to consider as meaningful
    min_dip_threshold_pct: int = 1
    
    # =========================================================================
    # DATA QUALITY FILTERS (Edge Case Removal)
    # =========================================================================
    
    # Max single-day gain to consider valid (filter data errors)
    max_daily_gain_pct: float = 100.0
    
    # Max single-day loss to consider valid (filter data errors)
    max_daily_loss_pct: float = 50.0
    
    # Minimum data points for statistical significance
    min_samples_for_stats: int = 3
    
    # Minimum years of data for high confidence
    min_years_for_confidence: int = 3
    
    # =========================================================================
    # OPTIMIZATION CONSTRAINTS
    # =========================================================================
    
    # Lookback years for backtesting
    lookback_years: int = 5
    
    # Walk-forward validation folds
    walkforward_folds: int = 4
    
    # Train/test split ratio for each fold
    train_ratio: float = 0.7
    
    # =========================================================================
    # STATISTICAL VALIDATION THRESHOLDS
    # =========================================================================
    
    # Minimum Sharpe ratio to consider a strategy valid
    min_sharpe_ratio: float = 0.0
    
    # Minimum alpha vs buy-and-hold to accept strategy
    min_alpha_vs_buyhold: float = 0.0
    
    # Maximum p-value for statistical significance
    max_p_value: float = 0.05
    
    # =========================================================================
    # STRATEGY PARAMETER OPTIMIZATION RANGES
    # =========================================================================
    # These define the SEARCH SPACE for hyperparameter optimization
    # NO hardcoded defaults - the optimizer finds the best combination
    
    # Moving average periods to test
    ma_periods_fast: tuple[int, ...] = (5, 8, 10, 13, 15, 20, 25, 30)
    ma_periods_slow: tuple[int, ...] = (20, 30, 40, 50, 60, 80, 100, 150, 200)
    
    # RSI thresholds to test
    rsi_oversold_range: tuple[int, int] = (15, 40)  # Test 15, 16, 17, ..., 40
    rsi_overbought_range: tuple[int, int] = (60, 85)  # Test 60, 61, ..., 85
    
    # Drawdown thresholds for buy-the-dip (as positive values, e.g., 10 = -10%)
    drawdown_threshold_range: tuple[int, int] = (5, 40)  # Test -5%, -6%, ..., -40%
    
    # Stochastic oversold thresholds
    stochastic_oversold_range: tuple[int, int] = (10, 35)
    
    # Bollinger band thresholds
    bb_lower_threshold_range: tuple[float, float] = (0.05, 0.35)
    
    # Z-score thresholds
    zscore_threshold_range: tuple[float, float] = (-3.0, -0.5)
    
    # ATR lookback periods
    atr_lookback_range: tuple[int, int] = (5, 50)
    
    # Momentum/breakout windows
    breakout_window_range: tuple[int, int] = (5, 60)
    momentum_days_range: tuple[int, int] = (5, 40)
    
    # Volatility thresholds (for crash avoidance)
    volatility_threshold_range: tuple[float, float] = (1.0, 4.0)
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def holding_days_range(self) -> range:
        """Full range of holding days to test: 1, 2, 3, ..., max."""
        return range(self.min_holding_days, self.max_holding_days + 1)
    
    def dip_thresholds_range(self) -> range:
        """Full range of dip thresholds to test: -1, -2, -3, ..., -max."""
        return range(-self.min_dip_threshold_pct, -self.max_dip_threshold_pct - 1, -1)
    
    def ma_crossover_combinations(self) -> list[tuple[int, int]]:
        """All valid fast/slow MA combinations where fast < slow."""
        return [
            (fast, slow)
            for fast in self.ma_periods_fast
            for slow in self.ma_periods_slow
            if fast < slow
        ]
    
    def rsi_oversold_values(self) -> range:
        """Range of RSI oversold thresholds to test."""
        return range(self.rsi_oversold_range[0], self.rsi_oversold_range[1] + 1)
    
    def drawdown_thresholds(self) -> list[float]:
        """Range of drawdown thresholds as decimals (e.g., -0.05 to -0.40)."""
        return [
            -pct / 100.0
            for pct in range(self.drawdown_threshold_range[0], self.drawdown_threshold_range[1] + 1)
        ]


# Singleton instance - import this everywhere
QUANT_LIMITS = QuantLimits()
