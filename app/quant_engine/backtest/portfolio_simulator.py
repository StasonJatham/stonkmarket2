"""
Portfolio Simulator with DCA and Regime-Adaptive Position Sizing.

This module simulates realistic portfolio performance including:
- Initial capital deployment
- Monthly DCA contributions
- Regime-aware position sizing (scale-in during bear markets)
- Transaction costs and slippage
- Multiple scenarios for comparison
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.quant_engine.core import (
    MarketRegime,
    RegimeService,
    RegimeState,
    StrategyMode,
)
from app.quant_engine.backtest.fundamental_guardrail import (
    FundamentalGuardrail,
    FundamentalData,
    GuardrailResult,
)

logger = logging.getLogger(__name__)


def _normalize_tz(series: pd.Series | pd.DataFrame | None) -> pd.Series | pd.DataFrame | None:
    """Normalize a Series/DataFrame to timezone-naive for safe comparisons."""
    if series is None:
        return None
    if hasattr(series.index, 'tz') and series.index.tz is not None:
        return series.tz_localize(None) if hasattr(series, 'tz_localize') else series.copy()
    return series


class SimulationScenario(str, Enum):
    """Scenarios to simulate."""

    BUY_AND_HOLD = "BUY_AND_HOLD"  # Lump sum, hold forever
    BUY_AND_HOLD_DCA = "BUY_AND_HOLD_DCA"  # Monthly contributions
    DIP_STRATEGY = "DIP_STRATEGY"  # Timing entries only (lump sum)
    DIP_STRATEGY_DCA = "DIP_STRATEGY_DCA"  # Monthly + dip timing (timed DCA)
    TECHNICAL_STRATEGY = "TECHNICAL_STRATEGY"  # Best technical strategy


@dataclass
class PortfolioConfig:
    """Portfolio simulation configuration."""

    initial_capital: float = 10_000.0
    monthly_contribution: float = 1_000.0

    # Trading costs
    cost_per_trade: float = 1.0  # Flat fee
    slippage_bps: float = 5.0  # 5 basis points

    # Cash management
    max_months_cash_drag: int = 3  # Force deploy if cash sits too long

    # Position sizing (overridden by regime in bear mode)
    default_position_pct: float = 100.0


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""

    date: pd.Timestamp
    cash: float
    shares: float
    share_price: float
    portfolio_value: float
    cumulative_invested: float
    cumulative_return_pct: float
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class Trade:
    """A single trade with full details."""

    date: pd.Timestamp
    type: Literal["buy", "sell"]
    shares: float
    price: float
    value: float
    cost: float  # Transaction cost
    reason: str
    regime: MarketRegime | None = None
    pnl_pct: float | None = None  # For sells


@dataclass
class ScenarioResult:
    """Result of a single scenario simulation."""

    scenario: SimulationScenario

    # Final values
    final_value: float
    total_invested: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float

    # Trade statistics
    n_trades: int
    win_rate: float
    avg_trade_return: float

    # Trade markers for charting
    trades: list[Trade] = field(default_factory=list)

    # Equity curve
    equity_curve: list[PortfolioSnapshot] = field(default_factory=list)

    def to_markers(self) -> list[dict[str, Any]]:
        """Convert trades to frontend marker format."""
        return [
            {
                "timestamp": str(t.date),
                "price": t.price,
                "type": t.type,
                "shares": t.shares,
                "value": t.value,
                "reason": t.reason,
                "regime": t.regime.value if t.regime else None,
                "pnl_pct": t.pnl_pct,
            }
            for t in self.trades
        ]


@dataclass
class SimulationResult:
    """Complete simulation across all scenarios."""

    symbol: str
    period_years: float
    period_start: pd.Timestamp
    period_end: pd.Timestamp

    # Results by scenario
    scenarios: dict[SimulationScenario, ScenarioResult]

    # Winner
    best_scenario: SimulationScenario
    best_return_pct: float

    # Comparison metrics
    strategy_vs_bh: float
    dca_vs_lump_sum: float
    dip_vs_regular_dca: float


class PortfolioSimulator:
    """
    Realistic portfolio simulation with regime-adaptive behavior.

    Key Features:
    - Simulates actual capital allocation (not just percentages)
    - Regime-aware: different behavior in bull vs bear markets
    - DCA and scale-in logic for accumulation
    - Transaction costs and slippage
    """

    def __init__(
        self,
        config: PortfolioConfig | None = None,
        regime_service: RegimeService | None = None,
        fundamental_guardrail: FundamentalGuardrail | None = None,
    ):
        self.config = config or PortfolioConfig()
        self.regime_service = regime_service
        self.fundamental_guardrail = fundamental_guardrail

    def simulate_all_scenarios(
        self,
        prices: pd.Series,
        symbol: str,
        dip_signals: pd.Series | None = None,
        technical_signals: pd.Series | None = None,
        fundamentals_history: pd.DataFrame | None = None,
        regime_series: pd.Series | None = None,
    ) -> SimulationResult:
        """
        Run all simulation scenarios.

        Args:
            prices: Price series with DatetimeIndex
            symbol: Stock symbol
            dip_signals: Series with 1 for dip buy signals
            technical_signals: Series with 1 for technical buy, -1 for sell
            fundamentals_history: Historical fundamentals for bear mode checks
            regime_series: Pre-computed regime for each date (from RegimeService)

        Returns:
            SimulationResult with all scenarios and comparison metrics
        """
        # Normalize all timestamps to timezone-naive for safe comparisons
        # (yfinance returns tz-aware America/New_York timestamps)
        if hasattr(prices.index, 'tz') and prices.index.tz is not None:
            prices = prices.copy()
            prices.index = prices.index.tz_localize(None)
        if dip_signals is not None and hasattr(dip_signals.index, 'tz') and dip_signals.index.tz is not None:
            dip_signals = dip_signals.copy()
            dip_signals.index = dip_signals.index.tz_localize(None)
        if technical_signals is not None and hasattr(technical_signals.index, 'tz') and technical_signals.index.tz is not None:
            technical_signals = technical_signals.copy()
            technical_signals.index = technical_signals.index.tz_localize(None)
        if regime_series is not None and hasattr(regime_series.index, 'tz') and regime_series.index.tz is not None:
            regime_series = regime_series.copy()
            regime_series.index = regime_series.index.tz_localize(None)
        if fundamentals_history is not None and hasattr(fundamentals_history.index, 'tz') and fundamentals_history.index.tz is not None:
            fundamentals_history = fundamentals_history.copy()
            fundamentals_history.index = fundamentals_history.index.tz_localize(None)
        
        results: dict[SimulationScenario, ScenarioResult] = {}

        # Scenario 1: Pure Buy & Hold (lump sum)
        results[SimulationScenario.BUY_AND_HOLD] = self._simulate_buy_hold(
            prices, lump_sum=True
        )

        # Scenario 2: Buy & Hold with DCA
        results[SimulationScenario.BUY_AND_HOLD_DCA] = self._simulate_buy_hold(
            prices, lump_sum=False
        )

        # Scenario 3: Dip Strategy (timing entries, lump sum)
        if dip_signals is not None:
            results[SimulationScenario.DIP_STRATEGY] = self._simulate_dip_strategy(
                prices, dip_signals, with_dca=False, 
                fundamentals_history=fundamentals_history, regime_series=regime_series
            )

            # Scenario 4: Dip Strategy + DCA (timed DCA)
            results[SimulationScenario.DIP_STRATEGY_DCA] = self._simulate_dip_strategy(
                prices, dip_signals, with_dca=True, 
                fundamentals_history=fundamentals_history, regime_series=regime_series
            )

        # Scenario 5: Technical Strategy
        if technical_signals is not None:
            results[SimulationScenario.TECHNICAL_STRATEGY] = self._simulate_technical(
                prices, technical_signals
            )

        # Find best scenario
        best = max(results.items(), key=lambda x: x[1].total_return_pct)

        # Compute comparisons
        bh_return = results[SimulationScenario.BUY_AND_HOLD].total_return_pct
        bh_dca_return = results[SimulationScenario.BUY_AND_HOLD_DCA].total_return_pct

        strategy_vs_bh = 0.0
        if SimulationScenario.TECHNICAL_STRATEGY in results:
            strategy_vs_bh = results[SimulationScenario.TECHNICAL_STRATEGY].total_return_pct - bh_return
        elif SimulationScenario.DIP_STRATEGY_DCA in results:
            strategy_vs_bh = results[SimulationScenario.DIP_STRATEGY_DCA].total_return_pct - bh_return

        dip_vs_dca = 0.0
        if SimulationScenario.DIP_STRATEGY_DCA in results:
            dip_vs_dca = results[SimulationScenario.DIP_STRATEGY_DCA].total_return_pct - bh_dca_return

        return SimulationResult(
            symbol=symbol,
            period_years=len(prices) / 252,
            period_start=prices.index[0],
            period_end=prices.index[-1],
            scenarios=results,
            best_scenario=best[0],
            best_return_pct=best[1].total_return_pct,
            strategy_vs_bh=strategy_vs_bh,
            dca_vs_lump_sum=bh_dca_return - bh_return,
            dip_vs_regular_dca=dip_vs_dca,
        )

    def _simulate_buy_hold(
        self,
        prices: pd.Series,
        lump_sum: bool,
    ) -> ScenarioResult:
        """Simulate buy and hold (with or without DCA)."""
        equity_curve: list[PortfolioSnapshot] = []
        trades: list[Trade] = []

        cash = self.config.initial_capital
        shares = 0.0
        cumulative_invested = self.config.initial_capital
        last_contribution_month: tuple[int, int] | None = None
        realized_pnl = 0.0

        for date, price in prices.items():
            date = pd.Timestamp(date)
            month = (date.year, date.month)

            # Monthly contribution (if DCA)
            if not lump_sum and month != last_contribution_month:
                cash += self.config.monthly_contribution
                cumulative_invested += self.config.monthly_contribution
                last_contribution_month = month

            # Buy logic
            if lump_sum and shares == 0 and cash > price:
                # Lump sum: buy everything on day 1
                cost = self.config.cost_per_trade
                slippage = price * (self.config.slippage_bps / 10000)
                effective_price = price + slippage
                shares_to_buy = (cash - cost) / effective_price
                
                trades.append(Trade(
                    date=date,
                    type="buy",
                    shares=shares_to_buy,
                    price=price,
                    value=shares_to_buy * price,
                    cost=cost,
                    reason="Initial lump sum",
                ))
                
                shares = shares_to_buy
                cash = 0.0

            elif not lump_sum and cash >= self.config.monthly_contribution * 0.9:
                # DCA: buy monthly contribution worth
                cost = self.config.cost_per_trade
                slippage = price * (self.config.slippage_bps / 10000)
                effective_price = price + slippage
                amount_to_invest = min(cash, self.config.monthly_contribution)
                shares_to_buy = (amount_to_invest - cost) / effective_price
                
                if shares_to_buy > 0:
                    trades.append(Trade(
                        date=date,
                        type="buy",
                        shares=shares_to_buy,
                        price=price,
                        value=shares_to_buy * price,
                        cost=cost,
                        reason="Monthly DCA",
                    ))
                    
                    shares += shares_to_buy
                    cash -= amount_to_invest

            # Record snapshot
            portfolio_value = cash + shares * price
            unrealized_pnl = shares * price - (cumulative_invested - cash)
            
            equity_curve.append(PortfolioSnapshot(
                date=date,
                cash=cash,
                shares=shares,
                share_price=price,
                portfolio_value=portfolio_value,
                cumulative_invested=cumulative_invested,
                cumulative_return_pct=(portfolio_value / cumulative_invested - 1) * 100 if cumulative_invested > 0 else 0,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
            ))

        return self._build_scenario_result(
            SimulationScenario.BUY_AND_HOLD if lump_sum else SimulationScenario.BUY_AND_HOLD_DCA,
            equity_curve,
            trades,
        )

    def _simulate_dip_strategy(
        self,
        prices: pd.Series,
        dip_signals: pd.Series,
        with_dca: bool,
        fundamentals_history: pd.DataFrame | None = None,
        regime_series: pd.Series | None = None,
    ) -> ScenarioResult:
        """
        Simulate dip buying strategy with regime awareness.

        Key behavior:
        - In BULL mode: Use dip signals with standard position sizing
        - In BEAR mode: Require fundamental check + scale-in
        - In CRASH mode: Maximum accumulation (if quality passes)
        """
        equity_curve: list[PortfolioSnapshot] = []
        trades: list[Trade] = []

        cash = self.config.initial_capital
        shares = 0.0
        cumulative_invested = self.config.initial_capital
        pending_contribution = 0.0
        last_contribution_month: tuple[int, int] | None = None
        realized_pnl = 0.0
        months_without_deployment = 0
        avg_cost_basis = 0.0

        for date, price in prices.items():
            date = pd.Timestamp(date)
            month = (date.year, date.month)

            # Monthly contribution
            if with_dca and month != last_contribution_month:
                pending_contribution += self.config.monthly_contribution
                cumulative_invested += self.config.monthly_contribution
                last_contribution_month = month
                months_without_deployment += 1

            # Get regime at this date from pre-computed series
            regime_value = None
            if regime_series is not None and date in regime_series.index:
                regime_value = regime_series.loc[date]

            # Check for dip signal
            is_dip = dip_signals.get(date, 0) == 1
            available_cash = cash + pending_contribution

            if is_dip and available_cash > price:
                # Determine if we should buy based on regime
                should_buy = True
                buy_reason = "Dip signal"
                position_pct = 100.0

                if regime_value is not None:
                    # Get strategy config for this regime
                    try:
                        regime = MarketRegime(regime_value)
                        from app.quant_engine.core import REGIME_STRATEGY_CONFIGS
                        config = REGIME_STRATEGY_CONFIGS.get(regime)
                        
                        if config:
                            # In bear/crash mode, could require fundamental check
                            if config.use_fundamentals and self.fundamental_guardrail is not None:
                                should_buy = True  # Placeholder - integrate with PIT fundamentals
                            buy_reason = f"Dip signal ({regime.value} mode)"
                            position_pct = config.position_size_pct
                    except ValueError:
                        pass  # Unknown regime value, use defaults

                if should_buy:
                    cost = self.config.cost_per_trade
                    slippage = price * (self.config.slippage_bps / 10000)
                    effective_price = price + slippage

                    # Calculate amount to invest based on position sizing
                    target_investment = available_cash * (position_pct / 100)
                    shares_to_buy = (target_investment - cost) / effective_price

                    if shares_to_buy > 0:
                        # Update cost basis
                        old_value = shares * avg_cost_basis
                        new_value = shares_to_buy * price
                        avg_cost_basis = (old_value + new_value) / (shares + shares_to_buy) if (shares + shares_to_buy) > 0 else price

                        # Determine regime for trade record
                        trade_regime = None
                        if regime_value:
                            try:
                                trade_regime = MarketRegime(regime_value)
                            except ValueError:
                                pass

                        trades.append(Trade(
                            date=date,
                            type="buy",
                            shares=shares_to_buy,
                            price=price,
                            value=shares_to_buy * price,
                            cost=cost,
                            reason=buy_reason,
                            regime=trade_regime,
                        ))

                        shares += shares_to_buy
                        spent = target_investment
                        if pending_contribution >= spent:
                            pending_contribution -= spent
                        else:
                            cash -= (spent - pending_contribution)
                            pending_contribution = 0

                        months_without_deployment = 0

            # Cash drag limit: force deploy if cash sits too long
            if with_dca and months_without_deployment >= self.config.max_months_cash_drag and pending_contribution > price:
                cost = self.config.cost_per_trade
                slippage = price * (self.config.slippage_bps / 10000)
                effective_price = price + slippage
                shares_to_buy = (pending_contribution - cost) / effective_price

                if shares_to_buy > 0:
                    old_value = shares * avg_cost_basis
                    new_value = shares_to_buy * price
                    avg_cost_basis = (old_value + new_value) / (shares + shares_to_buy) if (shares + shares_to_buy) > 0 else price

                    trades.append(Trade(
                        date=date,
                        type="buy",
                        shares=shares_to_buy,
                        price=price,
                        value=shares_to_buy * price,
                        cost=cost,
                        reason=f"Cash drag limit ({self.config.max_months_cash_drag} months)",
                        regime=trade_regime,
                    ))

                    shares += shares_to_buy
                    pending_contribution = 0
                    months_without_deployment = 0

            # Record snapshot
            portfolio_value = cash + shares * price + pending_contribution
            unrealized_pnl = shares * (price - avg_cost_basis) if shares > 0 else 0

            equity_curve.append(PortfolioSnapshot(
                date=date,
                cash=cash + pending_contribution,
                shares=shares,
                share_price=price,
                portfolio_value=portfolio_value,
                cumulative_invested=cumulative_invested,
                cumulative_return_pct=(portfolio_value / cumulative_invested - 1) * 100 if cumulative_invested > 0 else 0,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
            ))

        return self._build_scenario_result(
            SimulationScenario.DIP_STRATEGY_DCA if with_dca else SimulationScenario.DIP_STRATEGY,
            equity_curve,
            trades,
        )

    def _simulate_technical(
        self,
        prices: pd.Series,
        signals: pd.Series,
    ) -> ScenarioResult:
        """
        Simulate technical trading strategy.

        Signals: 1 = buy, -1 = sell, 0 = hold
        """
        equity_curve: list[PortfolioSnapshot] = []
        trades: list[Trade] = []

        cash = self.config.initial_capital
        shares = 0.0
        cumulative_invested = self.config.initial_capital
        realized_pnl = 0.0
        entry_price = 0.0

        for date, price in prices.items():
            date = pd.Timestamp(date)
            signal = signals.get(date, 0)

            # Buy signal
            if signal == 1 and shares == 0 and cash > price:
                cost = self.config.cost_per_trade
                slippage = price * (self.config.slippage_bps / 10000)
                effective_price = price + slippage
                shares_to_buy = (cash - cost) / effective_price

                trades.append(Trade(
                    date=date,
                    type="buy",
                    shares=shares_to_buy,
                    price=price,
                    value=shares_to_buy * price,
                    cost=cost,
                    reason="Technical buy signal",
                    regime=None,
                ))

                shares = shares_to_buy
                cash = 0
                entry_price = price

            # Sell signal
            elif signal == -1 and shares > 0:
                cost = self.config.cost_per_trade
                slippage = price * (self.config.slippage_bps / 10000)
                effective_price = price - slippage
                sale_value = shares * effective_price - cost

                pnl_pct = (price / entry_price - 1) * 100 if entry_price > 0 else 0
                realized_pnl += sale_value - (shares * entry_price)

                trades.append(Trade(
                    date=date,
                    type="sell",
                    shares=shares,
                    price=price,
                    value=shares * price,
                    cost=cost,
                    reason="Technical sell signal",
                    regime=None,
                    pnl_pct=pnl_pct,
                ))

                cash = sale_value
                shares = 0

            # Record snapshot
            portfolio_value = cash + shares * price
            unrealized_pnl = shares * (price - entry_price) if shares > 0 and entry_price > 0 else 0

            equity_curve.append(PortfolioSnapshot(
                date=date,
                cash=cash,
                shares=shares,
                share_price=price,
                portfolio_value=portfolio_value,
                cumulative_invested=cumulative_invested,
                cumulative_return_pct=(portfolio_value / cumulative_invested - 1) * 100 if cumulative_invested > 0 else 0,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
            ))

        return self._build_scenario_result(
            SimulationScenario.TECHNICAL_STRATEGY,
            equity_curve,
            trades,
        )

    def _build_scenario_result(
        self,
        scenario: SimulationScenario,
        equity_curve: list[PortfolioSnapshot],
        trades: list[Trade],
    ) -> ScenarioResult:
        """Build ScenarioResult from equity curve and trades."""
        if not equity_curve:
            return ScenarioResult(
                scenario=scenario,
                final_value=0,
                total_invested=0,
                total_return_pct=0,
                annualized_return_pct=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                calmar_ratio=0,
                n_trades=0,
                win_rate=0,
                avg_trade_return=0,
                trades=[],
                equity_curve=[],
            )

        # Extract metrics
        final = equity_curve[-1]
        first = equity_curve[0]

        # Calculate returns
        total_return_pct = final.cumulative_return_pct
        years = len(equity_curve) / 252
        annualized_return_pct = ((1 + total_return_pct / 100) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Calculate max drawdown
        portfolio_values = pd.Series([e.portfolio_value for e in equity_curve])
        rolling_max = portfolio_values.cummax()
        drawdowns = (portfolio_values / rolling_max - 1) * 100
        max_drawdown_pct = abs(float(drawdowns.min()))

        # Calculate Sharpe ratio
        daily_returns = portfolio_values.pct_change().dropna()
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0

        # Calculate Calmar ratio
        calmar_ratio = annualized_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0.0

        # Trade statistics
        n_trades = len(trades)
        sell_trades = [t for t in trades if t.type == "sell" and t.pnl_pct is not None]
        win_rate = len([t for t in sell_trades if t.pnl_pct > 0]) / len(sell_trades) * 100 if sell_trades else 0
        avg_trade_return = np.mean([t.pnl_pct for t in sell_trades]) if sell_trades else 0

        return ScenarioResult(
            scenario=scenario,
            final_value=final.portfolio_value,
            total_invested=final.cumulative_invested,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            n_trades=n_trades,
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            trades=trades,
            equity_curve=equity_curve,
        )


def calculate_accumulation_metrics(
    strategy_result: ScenarioResult,
    dca_result: ScenarioResult,
) -> dict[str, Any]:
    """
    Calculate accumulation efficiency metrics.

    Compares how many shares were acquired and at what cost.
    """
    if not strategy_result.equity_curve or not dca_result.equity_curve:
        return {}

    strategy_shares = strategy_result.equity_curve[-1].shares
    dca_shares = dca_result.equity_curve[-1].shares

    strategy_invested = strategy_result.total_invested
    dca_invested = dca_result.total_invested

    # Accumulation score: > 1.0 means strategy acquired more shares
    accumulation_score = strategy_shares / dca_shares if dca_shares > 0 else 0

    # Average cost basis
    avg_cost_strategy = strategy_invested / strategy_shares if strategy_shares > 0 else 0
    avg_cost_dca = dca_invested / dca_shares if dca_shares > 0 else 0

    cost_improvement_pct = (avg_cost_dca - avg_cost_strategy) / avg_cost_dca * 100 if avg_cost_dca > 0 else 0

    return {
        "shares_acquired_strategy": strategy_shares,
        "shares_acquired_dca": dca_shares,
        "accumulation_score": accumulation_score,
        "avg_cost_strategy": avg_cost_strategy,
        "avg_cost_dca": avg_cost_dca,
        "cost_improvement_pct": cost_improvement_pct,
    }
