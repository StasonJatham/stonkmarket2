"""
Advanced Strategy Optimizer with Recency Weighting and Fundamental Filters.

This module extends the backtest engine with:
1. RECENCY WEIGHTING: Recent performance (last 6-12 months) matters 3x more
2. FUNDAMENTAL FILTERS: Only enter when financials are healthy
3. INDICATOR OPTIMIZATION: Find which indicators work NOW, not just which params
4. OUT-OF-SAMPLE FOCUS: 2025 must be profitable, not just overall

Key Philosophy:
- A strategy that made 1000% 5 years ago but lost 20% this year is USELESS
- We want strategies that work in CURRENT market conditions
- Fundamentals filter out value traps (cheap for good reason)
- Buy & hold beats most strategies, so we wait for REAL opportunities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd

from .backtest_engine import (
    BacktestEngine,
    TradingConfig,
    StrategyResult,
    ValidationReport,
    Trade,
    compute_indicators,
    compute_buy_and_hold_return,
    compare_to_benchmark,
    STRATEGIES,
)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Recency Weighting Configuration
# =============================================================================

@dataclass
class RecencyConfig:
    """Configuration for recency-weighted performance evaluation."""
    
    # Half-life for exponential decay (months)
    # Trades 6 months ago worth 50% as much as recent trades
    half_life_months: float = 6.0
    
    # Minimum weight for very old trades (don't completely ignore)
    min_weight: float = 0.1
    
    # Recent period that gets full weight (days)
    recent_full_weight_days: int = 90
    
    # Cutoff for "current year" validation (must be profitable here)
    current_year_start: str = "2025-01-01"
    
    # Minimum win rate in current year to consider strategy valid
    min_current_year_win_rate: float = 0.45
    
    # Minimum trades in current year for validation
    min_current_year_trades: int = 3


@dataclass
class FundamentalFilter:
    """Fundamental quality requirements for entry signals."""
    
    # Profitability requirements
    min_profit_margin: float | None = 0.0  # Must be positive
    min_fcf_yield: float | None = 0.02     # 2% FCF yield minimum
    
    # Valuation requirements  
    max_pe_ratio: float | None = 50.0      # Don't buy insanely expensive
    min_margin_of_safety: float | None = -0.20  # Max 20% overvalued
    
    # Financial health
    max_debt_to_equity: float | None = 3.0  # Not overleveraged
    min_current_ratio: float | None = 1.0   # Can pay short-term debts
    
    # Analyst sentiment (1=Strong Buy, 5=Strong Sell)
    max_analyst_rating: float | None = 3.0  # At least neutral
    
    # Growth
    min_revenue_growth: float | None = -0.10  # Not shrinking too fast
    
    @classmethod
    def relaxed(cls) -> "FundamentalFilter":
        """Relaxed filters for high volatility / oversold stocks."""
        return cls(
            min_profit_margin=-0.10,      # Can be slightly unprofitable
            min_fcf_yield=-0.05,          # Allow some cash burn
            max_pe_ratio=100.0,           # Higher PE ok if growing
            min_margin_of_safety=-0.50,   # Allow 50% overvalued
            max_debt_to_equity=5.0,
            min_current_ratio=0.5,
            max_analyst_rating=4.0,
            min_revenue_growth=-0.20,
        )
    
    @classmethod
    def strict(cls) -> "FundamentalFilter":
        """Strict value investor filters."""
        return cls(
            min_profit_margin=0.05,
            min_fcf_yield=0.04,
            max_pe_ratio=25.0,
            min_margin_of_safety=0.10,    # 10% undervalued
            max_debt_to_equity=1.5,
            min_current_ratio=1.5,
            max_analyst_rating=2.5,
            min_revenue_growth=0.0,
        )


@dataclass  
class OptimizationResult:
    """Result of full strategy optimization."""
    
    symbol: str
    best_strategy_name: str
    best_params: dict[str, Any]
    
    # Performance metrics
    total_return_pct: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown_pct: float
    n_trades: int
    
    # Recency-weighted metrics
    recency_weighted_return: float
    current_year_return_pct: float
    current_year_win_rate: float
    current_year_trades: int
    
    # Comparison to benchmarks
    vs_buy_hold: float  # Excess return
    vs_spy: float | None
    beats_buy_hold: bool
    beats_spy: bool  # Strategy beats SPY benchmark
    
    # Current signal status
    has_active_signal: bool
    signal_type: str  # "BUY", "SELL", "HOLD", "WAIT"
    signal_reason: str
    
    # Fundamental check
    fundamentals_healthy: bool
    fundamental_concerns: list[str]
    
    # Validation
    is_statistically_valid: bool
    validation_report: ValidationReport | None
    
    # Recent trades
    recent_trades: list[dict]
    
    # Dip entry analysis
    typical_recovery_days: int | None = None  # From dip entry optimizer
    
    # Optimization metadata
    optimization_timestamp: datetime = field(default_factory=datetime.now)
    indicators_used: list[str] = field(default_factory=list)


# =============================================================================
# Recency-Weighted Performance Calculator
# =============================================================================

class RecencyWeightedEvaluator:
    """Evaluates strategy performance with recency weighting."""
    
    def __init__(self, config: RecencyConfig | None = None):
        self.config = config or RecencyConfig()
    
    def compute_trade_weights(
        self, 
        trades: list[Trade], 
        reference_date: pd.Timestamp | None = None,
    ) -> np.ndarray:
        """
        Compute weights for each trade based on recency.
        
        Recent trades get weight ~1.0, old trades decay exponentially.
        """
        if not trades:
            return np.array([])
        
        if reference_date is None:
            reference_date = pd.Timestamp.now()
        
        weights = []
        half_life_days = self.config.half_life_months * 30
        
        for trade in trades:
            if trade.exit_date is None:
                # Open trade - use entry date
                trade_date = trade.entry_date
            else:
                trade_date = trade.exit_date
            
            days_ago = (reference_date - trade_date).days
            
            if days_ago <= self.config.recent_full_weight_days:
                # Recent trades get full weight
                weight = 1.0
            else:
                # Exponential decay with half-life
                adjusted_days = days_ago - self.config.recent_full_weight_days
                weight = 0.5 ** (adjusted_days / half_life_days)
                weight = max(weight, self.config.min_weight)
            
            weights.append(weight)
        
        return np.array(weights)
    
    def recency_weighted_return(
        self,
        trades: list[Trade],
        reference_date: pd.Timestamp | None = None,
    ) -> float:
        """Calculate recency-weighted average return per trade."""
        if not trades:
            return 0.0
        
        weights = self.compute_trade_weights(trades, reference_date)
        returns = np.array([t.pnl_pct for t in trades])
        
        if weights.sum() == 0:
            return 0.0
        
        return float(np.average(returns, weights=weights))
    
    def recency_weighted_sharpe(
        self,
        trades: list[Trade],
        risk_free_rate: float = 0.04,  # 4% annual
    ) -> float:
        """Calculate recency-weighted Sharpe ratio from trades."""
        if len(trades) < 5:
            return 0.0
        
        weights = self.compute_trade_weights(trades)
        returns = np.array([t.pnl_pct / 100 for t in trades])
        
        if weights.sum() == 0:
            return 0.0
        
        weighted_mean = np.average(returns, weights=weights)
        
        # Weighted standard deviation
        weighted_var = np.average((returns - weighted_mean) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        
        if weighted_std == 0:
            return 0.0
        
        # Annualize (assume average holding period)
        avg_holding = np.mean([t.holding_days for t in trades])
        trades_per_year = 252 / max(avg_holding, 1)
        
        annual_return = weighted_mean * trades_per_year
        annual_std = weighted_std * np.sqrt(trades_per_year)
        
        return (annual_return - risk_free_rate) / annual_std
    
    def current_year_performance(
        self,
        trades: list[Trade],
    ) -> dict:
        """Extract performance for current year only."""
        cutoff = pd.Timestamp(self.config.current_year_start)
        
        current_year_trades = [
            t for t in trades 
            if t.entry_date >= cutoff
        ]
        
        if not current_year_trades:
            return {
                "trades": 0,
                "return_pct": 0.0,
                "win_rate": 0.0,
                "is_valid": False,
            }
        
        # Only count closed trades
        closed = [t for t in current_year_trades if not t.is_open]
        
        if not closed:
            return {
                "trades": 0,
                "return_pct": 0.0,
                "win_rate": 0.0,
                "is_valid": False,
            }
        
        total_return = sum(t.pnl_pct for t in closed)
        wins = sum(1 for t in closed if t.pnl_pct > 0)
        win_rate = wins / len(closed) if closed else 0
        
        is_valid = (
            len(closed) >= self.config.min_current_year_trades and
            win_rate >= self.config.min_current_year_win_rate
        )
        
        return {
            "trades": len(closed),
            "return_pct": total_return,
            "win_rate": win_rate * 100,  # As percentage
            "is_valid": is_valid,
        }


# =============================================================================
# Fundamental Quality Checker
# =============================================================================

class FundamentalChecker:
    """Checks if fundamentals pass quality filters."""
    
    def __init__(self, filters: FundamentalFilter | None = None):
        self.filters = filters or FundamentalFilter()
    
    def check_fundamentals(
        self,
        fundamentals: dict | None,
    ) -> tuple[bool, list[str]]:
        """
        Check if fundamentals pass all filters.
        
        Returns:
            Tuple of (passes: bool, concerns: list[str])
        """
        if fundamentals is None:
            return False, ["No fundamentals data available"]
        
        concerns = []
        f = self.filters
        
        # Profitability
        profit_margin = fundamentals.get("profit_margin")
        if profit_margin is not None and f.min_profit_margin is not None:
            if profit_margin < f.min_profit_margin:
                concerns.append(f"Low profit margin: {profit_margin:.1%}")
        
        # FCF Yield
        fcf = fundamentals.get("free_cash_flow")
        market_cap = fundamentals.get("market_cap")
        if fcf is not None and market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap
            if f.min_fcf_yield is not None and fcf_yield < f.min_fcf_yield:
                concerns.append(f"Low FCF yield: {fcf_yield:.1%}")
        
        # Valuation
        pe_ratio = fundamentals.get("pe_ratio")
        if pe_ratio is not None and f.max_pe_ratio is not None:
            if pe_ratio > f.max_pe_ratio:
                concerns.append(f"High P/E ratio: {pe_ratio:.1f}")
        
        # Margin of safety (from ValuationAgent)
        margin_of_safety = fundamentals.get("margin_of_safety")
        if margin_of_safety is not None and f.min_margin_of_safety is not None:
            if margin_of_safety < f.min_margin_of_safety:
                overvalued = abs(margin_of_safety) * 100
                concerns.append(f"Overvalued by {overvalued:.0f}%")
        
        # Debt
        debt_to_equity = fundamentals.get("debt_to_equity")
        if debt_to_equity is not None and f.max_debt_to_equity is not None:
            if debt_to_equity > f.max_debt_to_equity:
                concerns.append(f"High debt: D/E={debt_to_equity:.1f}")
        
        # Liquidity
        current_ratio = fundamentals.get("current_ratio")
        if current_ratio is not None and f.min_current_ratio is not None:
            if current_ratio < f.min_current_ratio:
                concerns.append(f"Low liquidity: Current ratio={current_ratio:.2f}")
        
        # Analyst rating
        analyst_rating = fundamentals.get("recommendation_mean")
        if analyst_rating is not None and f.max_analyst_rating is not None:
            if analyst_rating > f.max_analyst_rating:
                concerns.append(f"Bearish analyst rating: {analyst_rating:.1f}/5")
        
        # Revenue growth
        revenue_growth = fundamentals.get("revenue_growth")
        if revenue_growth is not None and f.min_revenue_growth is not None:
            if revenue_growth < f.min_revenue_growth:
                concerns.append(f"Revenue declining: {revenue_growth:.1%}")
        
        passes = len(concerns) == 0
        return passes, concerns
    
    def get_intrinsic_value_signal(
        self,
        fundamentals: dict | None,
        current_price: float,
    ) -> tuple[str, str]:
        """
        Get signal based on intrinsic value vs price.
        
        Returns:
            Tuple of (signal: "BUY"/"SELL"/"HOLD", reason: str)
        """
        if fundamentals is None or current_price <= 0:
            return "HOLD", "Insufficient data for valuation"
        
        # Try to get intrinsic value
        intrinsic_value = fundamentals.get("intrinsic_value")
        dcf_value = fundamentals.get("dcf_value")
        
        # Use either intrinsic or DCF value
        fair_value = intrinsic_value or dcf_value
        
        if fair_value is None or fair_value <= 0:
            # Fall back to analyst target
            target_price = fundamentals.get("target_mean_price")
            if target_price and target_price > 0:
                discount = (target_price - current_price) / current_price
                if discount > 0.20:
                    return "BUY", f"20%+ below analyst target ${target_price:.0f}"
                elif discount < -0.15:
                    return "SELL", f"15%+ above analyst target ${target_price:.0f}"
                else:
                    return "HOLD", f"Near analyst target ${target_price:.0f}"
            return "HOLD", "No intrinsic value estimate available"
        
        margin_of_safety = (fair_value - current_price) / current_price
        
        if margin_of_safety > 0.25:
            return "BUY", f"25%+ margin of safety (fair value ${fair_value:.0f})"
        elif margin_of_safety > 0.10:
            return "BUY", f"Undervalued by {margin_of_safety:.0%}"
        elif margin_of_safety < -0.20:
            return "SELL", f"Overvalued by {abs(margin_of_safety):.0%}"
        else:
            return "HOLD", f"Fairly valued (±10% of ${fair_value:.0f})"


# =============================================================================
# Indicator Combination Optimizer
# =============================================================================

# Define which indicators to combine
INDICATOR_SETS = {
    "momentum": ["rsi", "stochastic", "williams_r", "roc"],
    "trend": ["sma_50", "sma_200", "ema_20", "adx"],
    "volatility": ["bb_upper", "bb_lower", "atr", "keltner_upper"],
    "volume": ["obv", "vwap", "volume_sma"],
    "mean_reversion": ["bb_pct", "zscore", "rsi_divergence"],
}

# Strategy indicator requirements
STRATEGY_INDICATORS = {
    "mean_reversion_rsi": ["rsi"],
    "mean_reversion_zscore": ["zscore"],
    "momentum_breakout": ["atr", "sma_50", "adx"],
    "trend_following_sma": ["sma_50", "sma_200"],
    "macd_crossover": ["macd", "macd_signal"],
    "bollinger_squeeze": ["bb_upper", "bb_lower", "bb_pct"],
    "stochastic_oversold": ["stoch_k", "stoch_d"],
    "rsi_divergence": ["rsi"],
    "combined_oversold": ["rsi", "stoch_k", "bb_pct"],
    "volatility_contraction": ["atr", "bb_upper", "bb_lower"],
}


class IndicatorOptimizer:
    """Optimizes which indicators to use, not just params."""
    
    def __init__(
        self,
        config: TradingConfig | None = None,
        recency_config: RecencyConfig | None = None,
    ):
        self.config = config or TradingConfig()
        self.recency_config = recency_config or RecencyConfig()
        self.engine = BacktestEngine(self.config)
        self.recency_eval = RecencyWeightedEvaluator(self.recency_config)
    
    def optimize_indicator_combination(
        self,
        df: pd.DataFrame,
        n_trials: int = 100,
        fundamental_filter: FundamentalFilter | None = None,
    ) -> tuple[str, dict, StrategyResult]:
        """
        Use Optuna to find best indicator combination AND params.
        
        Unlike standard optimization that just tunes params, this also
        selects which strategy/indicators perform best in RECENT data.
        """
        if not OPTUNA_AVAILABLE:
            raise RuntimeError("Optuna required for optimization")
        
        # Pre-compute indicators
        df_ind = compute_indicators(df.copy())
        
        best_result = None
        best_strategy = ""
        best_params = {}
        best_score = float("-inf")
        
        def objective(trial: optuna.Trial) -> float:
            # Select strategy
            strategy_name = trial.suggest_categorical(
                "strategy", list(STRATEGIES.keys())
            )
            
            # Get strategy's param space
            strat_func, default_params, param_space = STRATEGIES[strategy_name]
            
            # Suggest params within space
            params = {}
            for param_name, space in param_space.items():
                if isinstance(space, list):
                    # Categorical choices
                    params[param_name] = trial.suggest_categorical(param_name, space)
                elif isinstance(space, tuple) and len(space) == 2:
                    low, high = space
                    if isinstance(low, int) and isinstance(high, int):
                        params[param_name] = trial.suggest_int(param_name, low, high)
                    else:
                        params[param_name] = trial.suggest_float(param_name, float(low), float(high))
                else:
                    # Use default if space not understood
                    params[param_name] = default_params.get(param_name)
            
            try:
                result = self.engine.backtest_strategy(df_ind, strategy_name, params)
            except Exception:
                return float("-inf")
            
            if result.n_trades < 5:
                return float("-inf")
            
            # Score with heavy recency weighting
            current_year = self.recency_eval.current_year_performance(result.trades)
            recency_return = self.recency_eval.recency_weighted_return(result.trades)
            recency_sharpe = self.recency_eval.recency_weighted_sharpe(result.trades)
            
            # Composite score emphasizing recent performance
            score = (
                recency_sharpe * 0.40 +           # Recency-weighted Sharpe
                current_year.get("return_pct", 0) / 100 * 0.30 +  # Current year return
                result.sharpe_ratio * 0.20 +       # Overall Sharpe
                (result.win_rate / 100 - 0.5) * 0.10  # Win rate bonus
            )
            
            # Penalty if current year is losing
            if current_year.get("return_pct", 0) < 0:
                score *= 0.5
            
            # Penalty for high drawdown
            if result.max_drawdown_pct < -30:
                score *= 0.8
            
            nonlocal best_result, best_strategy, best_params, best_score
            if score > best_score and current_year.get("is_valid", False):
                best_score = score
                best_result = result
                best_strategy = strategy_name
                best_params = params
            
            return score
        
        # Run optimization
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        if best_result is None:
            # Fall back to best overall strategy
            for strat_name in STRATEGIES:
                try:
                    _, default_params, _ = STRATEGIES[strat_name]
                    result = self.engine.backtest_strategy(
                        df_ind, strat_name, default_params
                    )
                    if result.n_trades >= 5 and result.sharpe_ratio > best_score:
                        best_score = result.sharpe_ratio
                        best_result = result
                        best_strategy = strat_name
                        best_params = default_params
                except Exception:
                    continue
        
        return best_strategy, best_params, best_result


# =============================================================================
# Main Strategy Optimizer
# =============================================================================

class StrategyOptimizer:
    """
    Main optimizer that combines all techniques for production signals.
    
    This is what runs nightly to find the best strategy for each stock.
    """
    
    def __init__(
        self,
        config: TradingConfig | None = None,
        recency_config: RecencyConfig | None = None,
        fundamental_filter: FundamentalFilter | None = None,
    ):
        self.config = config or TradingConfig()
        self.recency_config = recency_config or RecencyConfig()
        self.fundamental_filter = fundamental_filter or FundamentalFilter()
        
        self.engine = BacktestEngine(self.config)
        self.recency_eval = RecencyWeightedEvaluator(self.recency_config)
        self.fundamental_checker = FundamentalChecker(self.fundamental_filter)
        self.indicator_optimizer = IndicatorOptimizer(
            self.config, self.recency_config
        )
    
    def optimize_for_symbol(
        self,
        df: pd.DataFrame,
        symbol: str,
        fundamentals: dict | None = None,
        spy_prices: pd.Series | None = None,
        n_trials: int = 100,
    ) -> OptimizationResult:
        """
        Run full optimization for a single symbol.
        
        This is the main entry point for nightly batch processing.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock ticker
            fundamentals: Dict of fundamental data (PE, FCF, etc)
            spy_prices: SPY close prices for benchmark
            n_trials: Optuna trials for optimization
        
        Returns:
            OptimizationResult with strategy, signals, and metrics
        """
        logger.info(f"Optimizing strategy for {symbol}")
        
        # Ensure columns are lowercase
        df = df.copy()
        if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        df.attrs["symbol"] = symbol
        
        # Compute all indicators
        df_ind = compute_indicators(df)
        
        # Run optimization
        best_strategy, best_params, result = self.indicator_optimizer.optimize_indicator_combination(
            df, n_trials=n_trials
        )
        
        # Handle case where no valid strategy found
        if result is None:
            return self._build_no_signal_result(symbol, "No valid strategy found")
        
        # Compute recency metrics
        current_year = self.recency_eval.current_year_performance(result.trades)
        recency_return = self.recency_eval.recency_weighted_return(result.trades)
        
        # Benchmark comparison
        buy_hold = compute_buy_and_hold_return(df_ind["close"])
        comparison = compare_to_benchmark(result, df_ind["close"], spy_prices)
        
        # Check fundamentals
        fundamentals_pass, concerns = self.fundamental_checker.check_fundamentals(
            fundamentals
        )
        
        # Get current signal
        current_price = float(df_ind["close"].iloc[-1])
        signal_type, signal_reason = self._get_current_signal(
            df_ind, best_strategy, best_params, result,
            fundamentals, current_price, fundamentals_pass
        )
        
        # Check for active signal
        strat_func, _, _ = STRATEGIES[best_strategy]
        entries = strat_func(df_ind, best_params)
        has_active_signal = bool(entries.iloc[-1] == 1) if len(entries) > 0 else False
        
        # Build validation report
        try:
            _, _, validation = self.engine.find_best_strategy(
                df_ind,
                strategies=[best_strategy],
                n_trials_per_strategy=20,
            )
        except Exception:
            validation = None
        
        # Recent trades (last 5)
        recent_trades = [
            {
                "entry_date": str(t.entry_date.date()),
                "exit_date": str(t.exit_date.date()) if t.exit_date else None,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "pnl_pct": round(t.pnl_pct, 1),
                "exit_reason": t.exit_reason,
                "holding_days": t.holding_days,
            }
            for t in (result.trades[-5:] if result.trades else [])
        ]
        
        return OptimizationResult(
            symbol=symbol,
            best_strategy_name=best_strategy,
            best_params=best_params,
            
            # Performance
            total_return_pct=result.total_return_pct,
            sharpe_ratio=result.sharpe_ratio,
            win_rate=result.win_rate,
            max_drawdown_pct=result.max_drawdown_pct,
            n_trades=result.n_trades,
            
            # Recency
            recency_weighted_return=recency_return,
            current_year_return_pct=current_year.get("return_pct", 0),
            current_year_win_rate=current_year.get("win_rate", 0),
            current_year_trades=current_year.get("trades", 0),
            
            # Benchmarks
            vs_buy_hold=comparison.get("excess_vs_stock", 0),
            vs_spy=comparison.get("excess_vs_spy"),
            beats_buy_hold=comparison.get("beats_stock", False),
            beats_spy=comparison.get("beats_spy", False),
            
            # Current signal
            has_active_signal=has_active_signal,
            signal_type=signal_type,
            signal_reason=signal_reason,
            
            # Fundamentals
            fundamentals_healthy=fundamentals_pass,
            fundamental_concerns=concerns,
            
            # Validation
            is_statistically_valid=validation.is_valid if validation else False,
            validation_report=validation,
            
            # Trades
            recent_trades=recent_trades,
            
            # Metadata
            indicators_used=STRATEGY_INDICATORS.get(best_strategy, []),
        )
    
    def _get_current_signal(
        self,
        df_ind: pd.DataFrame,
        strategy_name: str,
        params: dict,
        result: StrategyResult,
        fundamentals: dict | None,
        current_price: float,
        fundamentals_pass: bool,
    ) -> tuple[str, str]:
        """Determine the current signal to display."""
        
        # Get technical signal
        strat_func, _, _ = STRATEGIES[strategy_name]
        entries = strat_func(df_ind, params)
        has_tech_signal = bool(entries.iloc[-1] == 1) if len(entries) > 0 else False
        
        # Check if in position
        in_position = False
        if result.trades:
            last_trade = result.trades[-1]
            if last_trade.is_open:
                in_position = True
        
        # Get fundamental signal
        fund_signal, fund_reason = self.fundamental_checker.get_intrinsic_value_signal(
            fundamentals, current_price
        )
        
        # Combine signals
        if in_position:
            # Already in position - check for exit signals
            if fund_signal == "SELL":
                return "SELL", f"Exit signal: {fund_reason}"
            return "HOLD", f"In position from '{strategy_name}'"
        
        if has_tech_signal:
            if not fundamentals_pass:
                return "WAIT", f"Tech signal active but fundamentals weak"
            if fund_signal == "SELL":
                return "WAIT", f"Tech signal but {fund_reason}"
            return "BUY", f"'{strategy_name}' signals entry, fundamentals healthy"
        
        # No active signal
        if fund_signal == "BUY" and fundamentals_pass:
            return "WATCH", f"Fundamentally attractive, waiting for tech signal"
        
        return "WAIT", f"Waiting for '{strategy_name}' entry signal"
    
    def _build_no_signal_result(
        self,
        symbol: str,
        reason: str,
    ) -> OptimizationResult:
        """Build result when no valid strategy found."""
        return OptimizationResult(
            symbol=symbol,
            best_strategy_name="none",
            best_params={},
            total_return_pct=0,
            sharpe_ratio=0,
            win_rate=0,
            max_drawdown_pct=0,
            n_trades=0,
            recency_weighted_return=0,
            current_year_return_pct=0,
            current_year_win_rate=0,
            current_year_trades=0,
            vs_buy_hold=0,
            vs_spy=None,
            beats_buy_hold=False,
            beats_spy=False,
            has_active_signal=False,
            signal_type="HOLD",
            signal_reason=reason,
            fundamentals_healthy=False,
            fundamental_concerns=[reason],
            is_statistically_valid=False,
            validation_report=None,
            recent_trades=[],
        )


# =============================================================================
# Batch Processing Functions
# =============================================================================

async def optimize_all_symbols(
    symbols: list[str],
    get_prices_func: Callable,
    get_fundamentals_func: Callable | None = None,
    spy_prices: pd.Series | None = None,
    config: TradingConfig | None = None,
    n_trials: int = 100,
) -> dict[str, OptimizationResult]:
    """
    Run optimization for all symbols in parallel.
    
    This is called by the nightly job.
    
    Args:
        symbols: List of stock symbols
        get_prices_func: Async function(symbol) -> DataFrame
        get_fundamentals_func: Async function(symbol) -> dict
        spy_prices: SPY prices for benchmark
        config: Trading configuration
        n_trials: Optuna trials per symbol
    
    Returns:
        Dict mapping symbol -> OptimizationResult
    """
    import asyncio
    
    optimizer = StrategyOptimizer(config=config)
    results = {}
    
    async def process_symbol(symbol: str) -> tuple[str, OptimizationResult | None]:
        try:
            # Get price data
            df = await get_prices_func(symbol)
            if df is None or len(df) < 100:
                logger.warning(f"Insufficient data for {symbol}")
                return symbol, None
            
            # Get fundamentals
            fundamentals = None
            if get_fundamentals_func:
                try:
                    fundamentals = await get_fundamentals_func(symbol)
                except Exception as e:
                    logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
            
            # Run optimization
            result = optimizer.optimize_for_symbol(
                df=df,
                symbol=symbol,
                fundamentals=fundamentals,
                spy_prices=spy_prices,
                n_trials=n_trials,
            )
            
            return symbol, result
            
        except Exception as e:
            logger.exception(f"Failed to optimize {symbol}: {e}")
            return symbol, None
    
    # Process in batches to avoid overwhelming the system
    batch_size = 5
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_results = await asyncio.gather(
            *[process_symbol(s) for s in batch],
            return_exceptions=True,
        )
        
        for item in batch_results:
            if isinstance(item, tuple):
                sym, res = item
                if res is not None:
                    results[sym] = res
    
    return results


def result_to_dict(result: OptimizationResult) -> dict:
    """Convert OptimizationResult to JSON-serializable dict."""
    return {
        "symbol": result.symbol,
        "best_strategy_name": result.best_strategy_name,
        "best_params": result.best_params,
        
        "metrics": {
            "total_return_pct": round(result.total_return_pct, 1),
            "sharpe_ratio": round(result.sharpe_ratio, 2),
            "win_rate": round(result.win_rate, 1),
            "max_drawdown_pct": round(result.max_drawdown_pct, 1),
            "n_trades": result.n_trades,
        },
        
        "recency": {
            "weighted_return": round(result.recency_weighted_return, 2),
            "current_year_return_pct": round(result.current_year_return_pct, 1),
            "current_year_win_rate": round(result.current_year_win_rate, 1),
            "current_year_trades": result.current_year_trades,
        },
        
        "benchmarks": {
            "vs_buy_hold": round(result.vs_buy_hold, 1),
            "vs_spy": round(result.vs_spy, 1) if result.vs_spy else None,
            "beats_buy_hold": result.beats_buy_hold,
        },
        
        "signal": {
            "has_active": result.has_active_signal,
            "type": result.signal_type,
            "reason": result.signal_reason,
        },
        
        "fundamentals": {
            "healthy": result.fundamentals_healthy,
            "concerns": result.fundamental_concerns,
        },
        
        "validation": {
            "is_valid": result.is_statistically_valid,
        },
        
        "recent_trades": result.recent_trades,
        "indicators_used": result.indicators_used,
        "optimized_at": result.optimization_timestamp.isoformat(),
    }
