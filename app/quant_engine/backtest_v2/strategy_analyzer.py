"""
StrategyAnalyzer - Advanced Strategy Metrics Computation.

This service transforms raw backtest results into the rich StrategyFullReport
data model with professional-grade metrics:

- Kelly Criterion (optimal position sizing)
- SQN - System Quality Number (Van Tharp)
- Duration Analysis (holding periods)
- Expectancy and R-multiples
- Regime-specific breakdowns
- Equity curves with benchmarks
- Runner-up strategy comparisons

Example Usage:
```python
analyzer = StrategyAnalyzer(
    prices=price_df,
    spy_prices=spy_df,
    optimization_result=result,
)
report = analyzer.generate_full_report()
```
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.quant_engine.backtest_v2.alpha_factory import (
    AlphaFactoryConfig,
    BacktestMetrics,
    ConditionGenome,
    IndicatorMatrix,
    IndicatorType,
    LogicGate,
    OptimizationResult,
    StrategyGenome,
    VectorizedBacktester,
)
from app.quant_engine.backtest_v2.regime_filter import MarketRegime, RegimeDetector
from app.quant_engine.backtest_v2.strategy_report import (
    AdvancedMetrics,
    BenchmarkComparison,
    EquityCurvePoint,
    OptimizationMeta,
    RegimeBreakdown,
    RegimePerformance,
    RegimeType,
    RiskMetrics,
    RunnerUpStrategy,
    SignalEvent,
    SignalType,
    StrategyConditionSummary,
    StrategyFullReport,
    StrategyVerdict,
    TradeStats,
    WinningStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Trade Record for Detailed Analysis
# =============================================================================

@dataclass
class TradeRecord:
    """Detailed record of a single trade."""
    entry_date: datetime
    entry_price: float
    entry_bar: int
    
    exit_date: datetime | None = None
    exit_price: float | None = None
    exit_bar: int | None = None
    exit_type: SignalType = SignalType.SELL
    
    shares: float = 1.0
    
    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None
    
    @property
    def return_pct(self) -> float:
        if not self.is_closed or self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price
    
    @property
    def pnl(self) -> float:
        if not self.is_closed or self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.shares
    
    @property
    def is_winner(self) -> bool:
        return self.return_pct > 0
    
    @property
    def holding_bars(self) -> int:
        if self.exit_bar is None:
            return 0
        return self.exit_bar - self.entry_bar
    
    @property
    def holding_days(self) -> float:
        if self.exit_date is None:
            return 0.0
        delta = self.exit_date - self.entry_date
        return delta.total_seconds() / 86400


# =============================================================================
# Detailed Backtester (extends VectorizedBacktester with trade records)
# =============================================================================

class DetailedBacktester:
    """
    Extended backtester that produces detailed trade records for analysis.
    
    Unlike VectorizedBacktester which focuses on speed, this produces
    full trade records with timestamps and signal details.
    """
    
    def __init__(
        self,
        matrix: IndicatorMatrix,
        initial_capital: float = 10_000.0,
        commission_pct: float = 0.001,
    ) -> None:
        self.matrix = matrix
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        
        self.prices = matrix.close.values
        self.dates = matrix.df.index.to_pydatetime()
    
    def run_detailed(
        self, 
        genome: StrategyGenome,
    ) -> tuple[list[TradeRecord], np.ndarray]:
        """
        Run backtest and return detailed trade records + equity curve.
        
        Returns:
            (trades, equity_curve) where equity_curve is an array of portfolio values
        """
        n = len(self.prices)
        
        # Generate signals
        entry_signals = genome.generate_entry_signals(self.matrix)
        exit_signals = genome.generate_exit_signals(self.matrix)
        
        trades: list[TradeRecord] = []
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        
        # State
        position = False
        current_trade: TradeRecord | None = None
        cash = self.initial_capital
        shares = 0.0
        
        for i in range(1, n):
            price = self.prices[i]
            
            if not position:
                equity[i] = cash
                
                # Check entry
                if entry_signals[i]:
                    # Buy with all available capital
                    entry_price = price * (1 + self.commission_pct)
                    shares = cash / entry_price
                    cash = 0
                    position = True
                    
                    current_trade = TradeRecord(
                        entry_date=self.dates[i],
                        entry_price=entry_price,
                        entry_bar=i,
                        shares=shares,
                    )
            else:
                # Update equity with current position value
                equity[i] = shares * price
                
                # Check exits
                holding_bars = i - current_trade.entry_bar
                pnl_pct = (price - current_trade.entry_price) / current_trade.entry_price
                
                exit_type = None
                if exit_signals[i]:
                    exit_type = SignalType.SELL
                elif pnl_pct <= -genome.stop_loss_pct:
                    exit_type = SignalType.STOP_LOSS
                elif pnl_pct >= genome.take_profit_pct:
                    exit_type = SignalType.TAKE_PROFIT
                elif holding_bars >= genome.holding_period_max:
                    exit_type = SignalType.TIME_EXIT
                
                if exit_type:
                    # Close trade
                    exit_price = price * (1 - self.commission_pct)
                    current_trade.exit_date = self.dates[i]
                    current_trade.exit_price = exit_price
                    current_trade.exit_bar = i
                    current_trade.exit_type = exit_type
                    
                    trades.append(current_trade)
                    
                    # Update cash
                    cash = shares * exit_price
                    equity[i] = cash
                    shares = 0.0
                    position = False
                    current_trade = None
        
        return trades, equity


# =============================================================================
# Strategy Analyzer - Main Service
# =============================================================================

class StrategyAnalyzer:
    """
    Comprehensive strategy analysis service.
    
    Takes raw backtest results and produces the StrategyFullReport
    with all professional-grade metrics.
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        spy_prices: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
    ) -> None:
        """
        Initialize analyzer with price data.
        
        Args:
            prices: OHLCV price data for the asset
            spy_prices: Optional SPY data for benchmark comparison
            symbol: Symbol being analyzed
        """
        self.symbol = symbol
        self.prices = prices
        self.spy_prices = spy_prices
        
        # Build indicator matrix
        self.matrix = IndicatorMatrix(prices)
        self.detailed_backtester = DetailedBacktester(self.matrix)
        
        # Regime detection
        self.regime_detector = RegimeDetector()
        if spy_prices is not None:
            self.regime_detector.set_spy_prices(spy_prices)
    
    def analyze_genome(
        self,
        genome: StrategyGenome,
        include_equity_curve: bool = True,
        include_signals: bool = True,
    ) -> WinningStrategy:
        """
        Fully analyze a single strategy genome.
        
        Args:
            genome: The strategy to analyze
            include_equity_curve: Include full equity curve (large)
            include_signals: Include all signal events
            
        Returns:
            WinningStrategy with all metrics
        """
        # Run detailed backtest
        trades, equity = self.detailed_backtester.run_detailed(genome)
        
        # Calculate all metrics
        trade_stats = self._calculate_trade_stats(trades)
        risk_metrics = self._calculate_risk_metrics(trades, equity)
        advanced_metrics = self._calculate_advanced_metrics(trades, equity)
        
        # Build equity curve if requested
        equity_curve = []
        if include_equity_curve:
            equity_curve = self._build_equity_curve(equity, trades)
        
        # Build signals if requested
        signals = []
        if include_signals:
            signals = self._build_signals(trades, genome)
        
        # Benchmark comparison
        benchmark = self._calculate_benchmark_comparison(trades, equity)
        
        # Regime breakdown
        regime_breakdown = self._calculate_regime_breakdown(trades)
        
        # Strategy description
        strategy_logic = self._describe_strategy(genome)
        
        # Verdict
        verdict = self._determine_verdict(advanced_metrics.sqn)
        confidence = self._calculate_confidence(trade_stats, risk_metrics, advanced_metrics)
        
        return WinningStrategy(
            name=genome.name,
            description=self._generate_description(genome),
            strategy_logic=strategy_logic,
            trade_stats=trade_stats,
            risk_metrics=risk_metrics,
            advanced_metrics=advanced_metrics,
            equity_curve=equity_curve,
            signals=signals,
            benchmark_comparison=benchmark,
            regime_breakdown=regime_breakdown,
            verdict=verdict,
            confidence_score=confidence,
        )
    
    def generate_full_report(
        self,
        optimization_result: OptimizationResult,
        runner_up_results: list[tuple[StrategyGenome, BacktestMetrics]] | None = None,
        include_baseline_comparison: bool = True,
    ) -> StrategyFullReport:
        """
        Generate the complete strategy report from optimization results.
        
        Args:
            optimization_result: Result from AlphaFactory optimization
            runner_up_results: Optional list of (genome, metrics) for runner-ups
            include_baseline_comparison: Include B&H, DCA, Buy Dips, SPY comparisons
            
        Returns:
            Complete StrategyFullReport
        """
        # Analyze winning strategy
        winner = self.analyze_genome(optimization_result.best_genome)
        
        # Build runner-ups
        runner_ups = []
        if runner_up_results:
            winner_return = winner.risk_metrics.sharpe_ratio
            for i, (genome, metrics) in enumerate(runner_up_results[:5]):
                runner_up = self._build_runner_up(
                    rank=i + 2,
                    genome=genome,
                    metrics=metrics,
                    winner_return=winner_return,
                    winner_sharpe=winner.risk_metrics.sharpe_ratio,
                )
                runner_ups.append(runner_up)
        
        # Build metadata
        meta = OptimizationMeta(
            n_trials_total=optimization_result.n_trials_completed,
            n_valid_strategies=len([s for s in optimization_result.all_trial_scores if s > 0]),
            optimization_time_seconds=optimization_result.optimization_time_seconds,
            symbol=self.symbol,
            data_start=self.matrix.df.index[0].to_pydatetime(),
            data_end=self.matrix.df.index[-1].to_pydatetime(),
            total_bars=len(self.matrix),
            train_period_days=len(self.matrix),
            validate_period_days=0,
            walk_forward_windows=0,
            generated_at=datetime.now(),
        )
        
        # Run baseline strategy comparisons
        baseline_comparison = None
        if include_baseline_comparison:
            try:
                from app.quant_engine.backtest_v2.baseline_strategies import BaselineEngine
                
                baseline_engine = BaselineEngine(
                    prices=self.prices,
                    spy_prices=self.spy_prices,
                    symbol=self.symbol,
                    initial_capital=10_000.0,
                    monthly_contribution=1_000.0,
                )
                
                # Pass genome and matrix for technical trading comparison
                baseline_comparison = baseline_engine.run_all(
                    genome=optimization_result.best_genome,
                    indicator_matrix=self.matrix,
                )
            except Exception as e:
                logger.warning(f"Failed to run baseline comparison: {e}")
                baseline_comparison = None
        
        return StrategyFullReport(
            meta=meta,
            winner=winner,
            runner_ups=runner_ups,
            baseline_comparison=baseline_comparison,
        )
    
    # =========================================================================
    # Trade Statistics
    # =========================================================================
    
    def _calculate_trade_stats(self, trades: list[TradeRecord]) -> TradeStats:
        """Calculate detailed trade statistics."""
        if not trades:
            return TradeStats(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                loss_rate=0.0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                best_trade_pct=0.0,
                worst_trade_pct=0.0,
                avg_duration_hours=0.0,
                avg_duration_days=0.0,
                min_duration_hours=0.0,
                max_duration_hours=0.0,
            )
        
        returns = [t.return_pct for t in trades]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r <= 0]
        durations = [t.holding_days * 24 for t in trades]  # In hours
        
        return TradeStats(
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(trades) if trades else 0.0,
            loss_rate=len(losers) / len(trades) if trades else 0.0,
            avg_win_pct=float(np.mean(winners) * 100) if winners else 0.0,
            avg_loss_pct=float(np.mean(losers) * 100) if losers else 0.0,
            best_trade_pct=float(max(returns) * 100) if returns else 0.0,
            worst_trade_pct=float(min(returns) * 100) if returns else 0.0,
            avg_duration_hours=float(np.mean(durations)) if durations else 0.0,
            avg_duration_days=float(np.mean([t.holding_days for t in trades])) if trades else 0.0,
            min_duration_hours=float(min(durations)) if durations else 0.0,
            max_duration_hours=float(max(durations)) if durations else 0.0,
        )
    
    # =========================================================================
    # Risk Metrics
    # =========================================================================
    
    def _calculate_risk_metrics(
        self, 
        trades: list[TradeRecord],
        equity: np.ndarray,
    ) -> RiskMetrics:
        """Calculate risk-adjusted performance metrics."""
        if not trades:
            return RiskMetrics(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                expectancy_ratio=0.0,
                max_drawdown_pct=0.0,
                max_drawdown_duration_days=0,
                avg_drawdown_pct=0.0,
                volatility_annual=0.0,
                downside_deviation=0.0,
            )
        
        returns = np.array([t.return_pct for t in trades])
        
        # Basic stats
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.001
        
        # Sharpe (annualized, assuming ~20 trades/year average)
        avg_holding = np.mean([t.holding_days for t in trades]) or 20
        trades_per_year = 252 / avg_holding
        sharpe = (avg_return / std_return * np.sqrt(trades_per_year)) if std_return > 0 else 0.0
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0.001
        sortino = (avg_return / downside_std * np.sqrt(trades_per_year)) if downside_std > 0 else 0.0
        
        # Drawdown analysis
        max_dd, max_dd_duration, avg_dd = self._analyze_drawdowns(equity)
        
        # Calmar
        total_return = (equity[-1] / equity[0]) - 1 if equity[0] > 0 else 0
        years = len(equity) / 252
        annual_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0
        calmar = (annual_return / abs(max_dd)) if max_dd < 0 else 0.0
        
        # Profit Factor
        gains = sum(r for r in returns if r > 0)
        losses = abs(sum(r for r in returns if r < 0))
        profit_factor = (gains / losses) if losses > 0 else gains if gains > 0 else 0.0
        
        # Expectancy
        expectancy = float(np.mean([t.pnl for t in trades])) if trades else 0.0
        avg_loss = abs(np.mean([t.pnl for t in trades if t.pnl < 0])) if any(t.pnl < 0 for t in trades) else 1.0
        expectancy_ratio = (expectancy / avg_loss) if avg_loss > 0 else 0.0
        
        # Volatility
        equity_returns = np.diff(equity) / equity[:-1]
        equity_returns = equity_returns[~np.isnan(equity_returns)]
        volatility = float(np.std(equity_returns) * np.sqrt(252)) if len(equity_returns) > 1 else 0.0
        
        return RiskMetrics(
            sharpe_ratio=float(np.clip(sharpe, -10, 10)),
            sortino_ratio=float(np.clip(sortino, -10, 10)),
            calmar_ratio=float(np.clip(calmar, -10, 10)),
            profit_factor=float(np.clip(profit_factor, 0, 100)),
            expectancy=float(expectancy),
            expectancy_ratio=float(expectancy_ratio),
            max_drawdown_pct=float(max_dd * 100),
            max_drawdown_duration_days=int(max_dd_duration),
            avg_drawdown_pct=float(avg_dd * 100),
            volatility_annual=float(volatility),
            downside_deviation=float(downside_std),
        )
    
    def _analyze_drawdowns(self, equity: np.ndarray) -> tuple[float, int, float]:
        """
        Analyze drawdown characteristics.
        
        Returns:
            (max_drawdown, max_duration_days, avg_drawdown)
        """
        if len(equity) < 2:
            return 0.0, 0, 0.0
        
        # Running max
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        drawdowns = np.nan_to_num(drawdowns, nan=0.0)
        
        max_dd = float(np.min(drawdowns))
        avg_dd = float(np.mean(drawdowns[drawdowns < 0])) if (drawdowns < 0).any() else 0.0
        
        # Max duration
        max_duration = 0
        current_duration = 0
        for dd in drawdowns:
            if dd < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_dd, max_duration, avg_dd
    
    # =========================================================================
    # Advanced Metrics
    # =========================================================================
    
    def _calculate_advanced_metrics(
        self,
        trades: list[TradeRecord],
        equity: np.ndarray,
    ) -> AdvancedMetrics:
        """Calculate professional trading system metrics."""
        if not trades:
            return AdvancedMetrics(
                kelly_criterion=0.0,
                kelly_half=0.0,
                sqn=0.0,
                sqn_rating=StrategyVerdict.REJECTED,
                payoff_ratio=0.0,
                trade_frequency_per_year=0.0,
                avg_bars_in_trade=0.0,
                time_in_market_pct=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                avg_consecutive_wins=0.0,
                avg_consecutive_losses=0.0,
            )
        
        returns = [t.return_pct for t in trades]
        winners = [r for r in returns if r > 0]
        losers = [r for r in returns if r <= 0]
        
        # Win rate and payoff
        win_rate = len(winners) / len(trades) if trades else 0.0
        avg_win = np.mean(winners) if winners else 0.0
        avg_loss = abs(np.mean(losers)) if losers else 0.001
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Kelly Criterion: f* = (bp - q) / b
        # where b = payoff ratio, p = win prob, q = lose prob
        b = payoff_ratio
        p = win_rate
        q = 1 - p
        kelly = ((b * p) - q) / b if b > 0 else 0.0
        kelly = max(0, min(kelly, 1))  # Clamp to 0-1
        
        # SQN = (Expectancy / StdDev of R) * sqrt(N)
        expectancy = np.mean(returns)
        std_r = np.std(returns) if len(returns) > 1 else 0.001
        sqn = (expectancy / std_r) * np.sqrt(len(trades)) if std_r > 0 else 0.0
        sqn_rating = self._determine_verdict(sqn)
        
        # Trade frequency
        total_bars = len(equity)
        bars_per_year = 252
        years = total_bars / bars_per_year
        trade_freq = len(trades) / years if years > 0 else 0.0
        
        # Time in market
        bars_in_trades = sum(t.holding_bars for t in trades)
        time_in_market = (bars_in_trades / total_bars * 100) if total_bars > 0 else 0.0
        
        # Streak analysis
        win_streaks, loss_streaks = self._analyze_streaks(trades)
        
        return AdvancedMetrics(
            kelly_criterion=float(kelly),
            kelly_half=float(kelly / 2),
            sqn=float(np.clip(sqn, -10, 10)),
            sqn_rating=sqn_rating,
            payoff_ratio=float(payoff_ratio),
            trade_frequency_per_year=float(trade_freq),
            avg_bars_in_trade=float(np.mean([t.holding_bars for t in trades])),
            time_in_market_pct=float(time_in_market),
            max_consecutive_wins=max(win_streaks) if win_streaks else 0,
            max_consecutive_losses=max(loss_streaks) if loss_streaks else 0,
            avg_consecutive_wins=float(np.mean(win_streaks)) if win_streaks else 0.0,
            avg_consecutive_losses=float(np.mean(loss_streaks)) if loss_streaks else 0.0,
        )
    
    def _analyze_streaks(self, trades: list[TradeRecord]) -> tuple[list[int], list[int]]:
        """Analyze winning and losing streaks."""
        if not trades:
            return [], []
        
        win_streaks = []
        loss_streaks = []
        current_win_streak = 0
        current_loss_streak = 0
        
        for trade in trades:
            if trade.is_winner:
                current_win_streak += 1
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
            else:
                current_loss_streak += 1
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
        
        # Append final streaks
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        return win_streaks, loss_streaks
    
    # =========================================================================
    # Equity Curve & Signals
    # =========================================================================
    
    def _build_equity_curve(
        self,
        equity: np.ndarray,
        trades: list[TradeRecord],
    ) -> list[EquityCurvePoint]:
        """Build equity curve time series."""
        dates = self.matrix.df.index.to_pydatetime()
        prices = self.matrix.close.values
        
        # Buy & hold benchmark
        initial = equity[0]
        benchmark_shares = initial / prices[0]
        benchmark_equity = benchmark_shares * prices
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        drawdowns = np.nan_to_num(drawdowns, nan=0.0)
        
        # Position tracking
        in_position = np.zeros(len(equity), dtype=bool)
        for trade in trades:
            if trade.exit_bar is not None:
                in_position[trade.entry_bar:trade.exit_bar+1] = True
        
        # Subsample for large datasets (keep every Nth point)
        n_points = len(equity)
        max_points = 500
        step = max(1, n_points // max_points)
        
        curve = []
        for i in range(0, n_points, step):
            point = EquityCurvePoint(
                timestamp=dates[i],
                equity=float(equity[i]),
                equity_pct=float((equity[i] / initial - 1) * 100),
                benchmark_equity=float(benchmark_equity[i]),
                benchmark_pct=float((benchmark_equity[i] / initial - 1) * 100),
                drawdown_pct=float(drawdowns[i] * 100),
                in_position=bool(in_position[i]),
            )
            curve.append(point)
        
        return curve
    
    def _build_signals(
        self,
        trades: list[TradeRecord],
        genome: StrategyGenome,
    ) -> list[SignalEvent]:
        """Build signal events from trade records."""
        signals = []
        
        for trade in trades:
            # Entry signal
            entry = SignalEvent(
                timestamp=trade.entry_date,
                signal_type=SignalType.BUY,
                price=trade.entry_price,
                position_size=trade.shares,
                position_value=trade.shares * trade.entry_price,
                reason=f"Entry conditions met",
            )
            signals.append(entry)
            
            # Exit signal
            if trade.is_closed:
                exit_signal = SignalEvent(
                    timestamp=trade.exit_date,
                    signal_type=trade.exit_type,
                    price=trade.exit_price,
                    position_size=trade.shares,
                    position_value=trade.shares * trade.exit_price,
                    entry_price=trade.entry_price,
                    trade_return_pct=trade.return_pct * 100,
                    trade_pnl=trade.pnl,
                    holding_days=int(trade.holding_days),
                    reason=f"{trade.exit_type.value}: {trade.return_pct:.1%} return",
                )
                signals.append(exit_signal)
        
        return signals
    
    # =========================================================================
    # Benchmark Comparison
    # =========================================================================
    
    def _calculate_benchmark_comparison(
        self,
        trades: list[TradeRecord],
        equity: np.ndarray,
    ) -> BenchmarkComparison:
        """Compare strategy to benchmarks."""
        prices = self.matrix.close.values
        initial = equity[0]
        
        # Strategy metrics
        strategy_return = (equity[-1] / initial - 1) * 100 if initial > 0 else 0.0
        strategy_rets = np.diff(equity) / equity[:-1]
        strategy_sharpe = self._calculate_sharpe(strategy_rets)
        strategy_dd = self._calculate_max_dd(equity) * 100
        strategy_vol = float(np.std(strategy_rets) * np.sqrt(252)) if len(strategy_rets) > 1 else 0.0
        
        # Buy & Hold
        bh_return = (prices[-1] / prices[0] - 1) * 100
        bh_rets = np.diff(prices) / prices[:-1]
        bh_sharpe = self._calculate_sharpe(bh_rets)
        bh_dd = self._calculate_max_dd(prices) * 100
        
        # SPY (if available)
        spy_return = 0.0
        spy_sharpe = 0.0
        spy_dd = 0.0
        beta = 0.0
        correlation = 0.0
        
        if self.spy_prices is not None and len(self.spy_prices) > 0:
            spy_close = self.spy_prices["close"].values if "close" in self.spy_prices.columns else self.spy_prices.iloc[:, 0].values
            spy_return = (spy_close[-1] / spy_close[0] - 1) * 100
            spy_rets = np.diff(spy_close) / spy_close[:-1]
            spy_sharpe = self._calculate_sharpe(spy_rets)
            spy_dd = self._calculate_max_dd(spy_close) * 100
            
            # Beta and correlation
            if len(strategy_rets) == len(spy_rets):
                correlation = float(np.corrcoef(strategy_rets, spy_rets)[0, 1])
                beta = float(np.cov(strategy_rets, spy_rets)[0, 1] / np.var(spy_rets)) if np.var(spy_rets) > 0 else 0.0
        
        return BenchmarkComparison(
            strategy_return_pct=float(strategy_return),
            strategy_sharpe=float(strategy_sharpe),
            strategy_max_dd=float(strategy_dd),
            strategy_volatility=float(strategy_vol),
            buy_hold_return_pct=float(bh_return),
            buy_hold_sharpe=float(bh_sharpe),
            buy_hold_max_dd=float(bh_dd),
            spy_return_pct=float(spy_return),
            spy_sharpe=float(spy_sharpe),
            spy_max_dd=float(spy_dd),
            alpha_vs_buy_hold=float(strategy_return - bh_return),
            alpha_vs_spy=float(strategy_return - spy_return),
            beta_to_spy=float(beta),
            correlation_to_spy=float(np.clip(correlation, -1, 1)) if not math.isnan(correlation) else 0.0,
        )
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        returns = returns[~np.isnan(returns)]
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))
    
    def _calculate_max_dd(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(values) < 2:
            return 0.0
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        return float(np.min(drawdowns))
    
    # =========================================================================
    # Regime Breakdown
    # =========================================================================
    
    def _calculate_regime_breakdown(
        self,
        trades: list[TradeRecord],
    ) -> RegimeBreakdown:
        """Calculate performance breakdown by market regime."""
        # Default regimes (if no detector available)
        bull_perf = self._create_regime_performance(trades, RegimeType.BULL)
        bear_perf = self._create_regime_performance(trades, RegimeType.BEAR)
        
        # For now, split trades by simple 50/50 for demo
        # In production, use actual regime detection
        mid = len(trades) // 2
        bull_trades = trades[:mid] if trades else []
        bear_trades = trades[mid:] if trades else []
        
        bull_perf = self._calculate_regime_perf(bull_trades, RegimeType.BULL, len(self.matrix) // 2)
        bear_perf = self._calculate_regime_perf(bear_trades, RegimeType.BEAR, len(self.matrix) // 2)
        
        # Determine best/worst
        bull_return = bull_perf.total_return_pct
        bear_return = bear_perf.total_return_pct
        
        best = RegimeType.BULL if bull_return >= bear_return else RegimeType.BEAR
        worst = RegimeType.BEAR if bull_return >= bear_return else RegimeType.BULL
        
        # Consistency (how similar are the returns?)
        returns = [bull_return, bear_return]
        if np.std(returns) > 0:
            consistency = 1 - min(np.std(returns) / (abs(np.mean(returns)) + 0.001), 1)
        else:
            consistency = 1.0
        
        return RegimeBreakdown(
            bull=bull_perf,
            bear=bear_perf,
            crash=None,
            recovery=None,
            best_regime=best,
            worst_regime=worst,
            regime_consistency=float(consistency),
        )
    
    def _calculate_regime_perf(
        self,
        trades: list[TradeRecord],
        regime: RegimeType,
        period_days: int,
    ) -> RegimePerformance:
        """Calculate performance for a specific regime."""
        if not trades:
            return self._create_regime_performance([], regime)
        
        returns = [t.return_pct for t in trades]
        total_return = float(np.prod([1 + r for r in returns]) - 1) * 100
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        avg_return = float(np.mean(returns)) * 100
        
        # Simple max DD for this subset
        cumulative = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdowns)) * 100 if len(drawdowns) > 0 else 0.0
        
        return RegimePerformance(
            regime=regime,
            period_days=period_days,
            period_pct=50.0,  # Simplified
            num_trades=len(trades),
            win_rate=float(win_rate),
            total_return_pct=total_return,
            avg_trade_return=avg_return,
            max_drawdown_pct=max_dd,
        )
    
    def _create_regime_performance(
        self,
        trades: list[TradeRecord],
        regime: RegimeType,
    ) -> RegimePerformance:
        """Create empty regime performance."""
        return RegimePerformance(
            regime=regime,
            period_days=0,
            period_pct=0.0,
            num_trades=len(trades),
            win_rate=0.0,
            total_return_pct=0.0,
            avg_trade_return=0.0,
            max_drawdown_pct=0.0,
        )
    
    # =========================================================================
    # Strategy Description
    # =========================================================================
    
    def _describe_strategy(self, genome: StrategyGenome) -> StrategyConditionSummary:
        """Create human-readable strategy description."""
        entry_parts = []
        for cond in genome.entry_conditions:
            entry_parts.append(self._describe_condition(cond))
        
        exit_parts = []
        for cond in genome.exit_conditions:
            exit_parts.append(self._describe_condition(cond))
        
        return StrategyConditionSummary(
            name=genome.name,
            entry_logic=" AND ".join(entry_parts) if entry_parts else "No entry conditions",
            exit_logic=" OR ".join(exit_parts) if exit_parts else "No exit conditions",
            stop_loss_pct=genome.stop_loss_pct * 100,
            take_profit_pct=genome.take_profit_pct * 100,
            max_holding_days=genome.holding_period_max,
        )
    
    def _describe_condition(self, cond: ConditionGenome) -> str:
        """Create human-readable condition description."""
        indicator = cond.indicator_type.value.upper()
        period = cond.period
        
        if cond.logic_gate == LogicGate.GREATER_THAN:
            return f"{indicator}({period}) > {cond.threshold:.1f}"
        elif cond.logic_gate == LogicGate.LESS_THAN:
            return f"{indicator}({period}) < {cond.threshold:.1f}"
        elif cond.logic_gate == LogicGate.CROSS_OVER:
            return f"{indicator}({period}) crosses above {cond.threshold:.1f}"
        elif cond.logic_gate == LogicGate.CROSS_UNDER:
            return f"{indicator}({period}) crosses below {cond.threshold:.1f}"
        elif cond.logic_gate == LogicGate.BETWEEN:
            return f"{indicator}({period}) between {cond.threshold:.1f} and {cond.threshold_upper:.1f}"
        return f"{indicator}({period})"
    
    def _generate_description(self, genome: StrategyGenome) -> str:
        """Generate natural language strategy description."""
        conditions = len(genome.entry_conditions)
        exits = len(genome.exit_conditions)
        
        parts = [f"Strategy with {conditions} entry condition(s)"]
        if exits > 0:
            parts.append(f" and {exits} exit condition(s)")
        parts.append(f". Stop loss at {genome.stop_loss_pct:.0%}")
        parts.append(f", take profit at {genome.take_profit_pct:.0%}")
        parts.append(f", max hold {genome.holding_period_max} days.")
        
        return "".join(parts)
    
    # =========================================================================
    # Verdict & Confidence
    # =========================================================================
    
    def _determine_verdict(self, sqn: float) -> StrategyVerdict:
        """Determine strategy quality verdict based on SQN."""
        if sqn >= 3.0:
            return StrategyVerdict.EXCELLENT
        elif sqn >= 2.0:
            return StrategyVerdict.GOOD
        elif sqn >= 1.0:
            return StrategyVerdict.AVERAGE
        elif sqn > 0:
            return StrategyVerdict.POOR
        else:
            return StrategyVerdict.REJECTED
    
    def _calculate_confidence(
        self,
        trade_stats: TradeStats,
        risk_metrics: RiskMetrics,
        advanced_metrics: AdvancedMetrics,
    ) -> float:
        """Calculate overall confidence score (0-100)."""
        score = 0.0
        
        # Trade count (up to 25 points)
        trade_score = min(trade_stats.total_trades / 100 * 25, 25)
        score += trade_score
        
        # Win rate (up to 20 points)
        win_score = trade_stats.win_rate * 20
        score += win_score
        
        # Sharpe (up to 25 points)
        sharpe_score = min(max(risk_metrics.sharpe_ratio, 0) / 3 * 25, 25)
        score += sharpe_score
        
        # Profit factor (up to 15 points)
        pf_score = min(max(risk_metrics.profit_factor - 1, 0) / 2 * 15, 15)
        score += pf_score
        
        # SQN (up to 15 points)
        sqn_score = min(max(advanced_metrics.sqn, 0) / 3 * 15, 15)
        score += sqn_score
        
        return min(score, 100)
    
    # =========================================================================
    # Runner-Up Builder
    # =========================================================================
    
    def _build_runner_up(
        self,
        rank: int,
        genome: StrategyGenome,
        metrics: BacktestMetrics,
        winner_return: float,
        winner_sharpe: float,
    ) -> RunnerUpStrategy:
        """Build runner-up strategy summary."""
        # Determine rejection reason
        if metrics.num_trades < 20:
            reason = f"Insufficient trades ({metrics.num_trades} < 20 minimum)"
            verdict = StrategyVerdict.REJECTED
        elif metrics.profitable_years_pct < 0.7:
            reason = f"Low profitability ({metrics.profitable_years_pct:.0%} < 70% years)"
            verdict = StrategyVerdict.REJECTED
        elif metrics.max_drawdown < -0.5:
            reason = f"Excessive drawdown ({metrics.max_drawdown:.0%} > 50%)"
            verdict = StrategyVerdict.REJECTED
        elif metrics.sharpe_ratio < winner_sharpe:
            reason = f"Lower Sharpe ({metrics.sharpe_ratio:.2f} < winner's {winner_sharpe:.2f})"
            verdict = self._determine_verdict(metrics.sharpe_ratio)
        else:
            reason = "Lower overall fitness score"
            verdict = self._determine_verdict(metrics.sharpe_ratio)
        
        return RunnerUpStrategy(
            rank=rank,
            strategy=self._describe_strategy(genome),
            verdict=verdict,
            rejection_reason=reason,
            total_return_pct=metrics.total_return * 100,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown_pct=metrics.max_drawdown * 100,
            win_rate=metrics.win_rate,
            num_trades=metrics.num_trades,
            sqn=metrics.sharpe_ratio,  # Approximation
            return_vs_winner=metrics.total_return * 100 - winner_return,
            sharpe_vs_winner=metrics.sharpe_ratio - winner_sharpe,
        )


# =============================================================================
# Factory Function
# =============================================================================

def create_strategy_analyzer(
    prices: pd.DataFrame,
    spy_prices: pd.DataFrame | None = None,
    symbol: str = "UNKNOWN",
) -> StrategyAnalyzer:
    """
    Create a StrategyAnalyzer instance.
    
    Args:
        prices: OHLCV price data
        spy_prices: Optional SPY benchmark data
        symbol: Symbol being analyzed
        
    Returns:
        Configured StrategyAnalyzer
    """
    return StrategyAnalyzer(prices=prices, spy_prices=spy_prices, symbol=symbol)
