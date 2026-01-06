"""
Baseline Strategies - Benchmark comparison strategies.

These strategies represent common investment approaches to compare against
optimized trading strategies:

1. Buy & Hold (B&H): Buy once, hold forever
2. DCA (Dollar Cost Average): Buy fixed $ amount monthly
3. Buy Dips (Perfect Dip): Buy only on detected dips, hold
4. Dip Trading: Buy dips, sell on recovery, repeat
5. SPY Baseline: Just buy SPY instead

These provide context for evaluating if a complex strategy is worth the effort.
If SPY DCA beats everything, the recommendation should be "just buy SPY monthly".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from app.quant_engine.core import (
    RegimeService,
    MarketRegime,
    StrategyMode,
)


logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class BaselineStrategyType(str, Enum):
    """Types of baseline strategies."""
    BUY_HOLD = "BUY_HOLD"
    LUMP_SUM = "LUMP_SUM"  # Buy all capital on day 1 (informational only)
    DCA_MONTHLY = "DCA_MONTHLY"
    BUY_DIPS_HOLD = "BUY_DIPS_HOLD"
    DIP_TRADING = "DIP_TRADING"  # Perfect dip trading with compounding
    TECHNICAL_TRADING = "TECHNICAL_TRADING"  # AlphaFactory optimized strategy
    REGIME_AWARE_TECHNICAL = "REGIME_AWARE_TECHNICAL"  # Regime-aware technical trading
    SPY_BUY_HOLD = "SPY_BUY_HOLD"
    SPY_DCA = "SPY_DCA"


# =============================================================================
# Trade Detail Model
# =============================================================================

class TradeDetail(BaseModel):
    """Detailed record of a single trade for API responses."""
    
    trade_num: int = Field(description="Trade number (1-indexed)")
    
    # Entry
    entry_date: str = Field(description="Entry date (YYYY-MM-DD)")
    entry_price: float = Field(description="Entry price")
    entry_reason: str = Field(default="", description="Why we entered")
    entry_regime: str = Field(default="UNKNOWN", description="Market regime at entry")
    
    # Exit (None if still open)
    exit_date: str | None = Field(None, description="Exit date (YYYY-MM-DD)")
    exit_price: float | None = Field(None, description="Exit price")
    exit_reason: str = Field(default="", description="Why we exited")
    
    # Results
    shares: float = Field(default=1.0, description="Shares traded")
    return_pct: float = Field(default=0.0, description="Return % on this trade")
    pnl: float = Field(default=0.0, description="Profit/loss in $")
    holding_days: int = Field(default=0, description="Days held")
    is_winner: bool = Field(default=False, description="Was this a winning trade?")


class RecommendationType(str, Enum):
    """Investment recommendation type."""
    OPTIMIZED_STRATEGY = "OPTIMIZED_STRATEGY"  # Use the found strategy
    BUY_AND_HOLD = "BUY_AND_HOLD"              # Just buy and hold the stock
    DCA = "DCA"                                 # Dollar cost average
    BUY_DIPS = "BUY_DIPS"                      # Buy on dips only
    SWITCH_TO_SPY = "SWITCH_TO_SPY"            # Forget this stock, buy SPY
    SPY_DCA = "SPY_DCA"                        # DCA into SPY


# =============================================================================
# Result Models
# =============================================================================

class BaselineResult(BaseModel):
    """Results from running a baseline strategy."""
    
    strategy_type: BaselineStrategyType = Field(description="Strategy type")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="Strategy description")
    
    # Performance
    initial_capital: float = Field(description="Starting capital")
    final_value: float = Field(description="Final portfolio value")
    total_return_pct: float = Field(description="Total return % (ROI on invested capital)")
    annualized_return_pct: float = Field(description="Annualized return %")
    profit: float = Field(default=0.0, description="Absolute $ profit (final - invested)")
    
    # Risk
    max_drawdown_pct: float = Field(le=0, description="Maximum drawdown %")
    sharpe_ratio: float = Field(description="Sharpe ratio")
    volatility_pct: float = Field(ge=0, description="Annualized volatility %")
    
    # Activity
    total_buys: int = Field(ge=0, description="Number of buy transactions")
    total_invested: float = Field(description="Total $ invested over time")
    avg_cost_basis: float = Field(description="Average purchase price")
    
    # Time
    years: float = Field(description="Years of data")
    
    # For DIP strategies
    dips_detected: int = Field(default=0, description="Number of dips detected")
    avg_dip_depth_pct: float = Field(default=0.0, description="Avg dip depth %")
    
    # Trade details (for API responses)
    trade_details: list[TradeDetail] = Field(
        default_factory=list, 
        description="Detailed trade history (only for active trading strategies)"
    )
    win_rate_pct: float = Field(default=0.0, description="Win rate percentage")
    avg_win_pct: float = Field(default=0.0, description="Average winning trade %")
    avg_loss_pct: float = Field(default=0.0, description="Average losing trade %")
    profit_factor: float = Field(default=0.0, description="Total wins / total losses")
    
    @field_validator("*", mode="before")
    @classmethod
    def handle_nan(cls, v: Any) -> Any:
        """Convert NaN/Inf to 0.0."""
        import math
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return 0.0
        return v


class InvestmentRecommendation(BaseModel):
    """The final investment recommendation based on all comparisons."""
    
    recommendation: RecommendationType = Field(description="Recommended approach")
    headline: str = Field(description="Short recommendation headline")
    reasoning: str = Field(description="Why this is recommended")
    
    # Comparison summary
    best_strategy_name: str = Field(description="Name of best performing strategy")
    best_strategy_return_pct: float = Field(description="Best strategy return")
    spy_return_pct: float = Field(description="SPY return over same period")
    
    # Key stats
    alpha_vs_spy: float = Field(description="Best strategy alpha vs SPY")
    risk_adjusted_winner: str = Field(description="Best risk-adjusted strategy name")
    
    # Actionable advice
    action_items: list[str] = Field(
        default_factory=list,
        description="Specific actionable steps"
    )
    
    # Caveats
    warnings: list[str] = Field(
        default_factory=list,
        description="Important warnings/caveats"
    )


class BaselineComparison(BaseModel):
    """Complete comparison of all baseline strategies."""
    
    symbol: str = Field(description="Symbol analyzed")
    
    # Main strategies (fair comparison - all use $10k + $1k/month)
    dca: BaselineResult = Field(description="DCA - buy every month")
    buy_dips: BaselineResult = Field(description="Buy on Dips & Hold")
    dip_trading: BaselineResult | None = Field(None, description="Perfect Dip Trading (compound)")
    technical_trading: BaselineResult | None = Field(None, description="AlphaFactory optimized strategy")
    regime_aware_technical: BaselineResult | None = Field(None, description="Regime-aware technical trading")
    
    # Reference strategies (different capital, for context)
    buy_hold: BaselineResult = Field(description="Buy & Hold (initial capital only)")
    lump_sum: BaselineResult = Field(description="Lump Sum (informational - all capital day 1)")
    
    # SPY comparison
    spy_buy_hold: BaselineResult = Field(description="SPY Buy & Hold")
    spy_dca: BaselineResult = Field(description="SPY DCA")
    
    # Rankings
    ranked_by_return: list[str] = Field(
        description="Strategy names ranked by total return"
    )
    ranked_by_sharpe: list[str] = Field(
        description="Strategy names ranked by Sharpe ratio"
    )
    
    # The recommendation
    recommendation: InvestmentRecommendation = Field(
        description="Investment recommendation"
    )


# =============================================================================
# Baseline Strategy Implementations
# =============================================================================

class BaselineEngine:
    """
    Engine to run all baseline strategy simulations.
    
    Uses the DipEntryOptimizer to calculate statistically optimal dip thresholds
    instead of hardcoded values.
    
    Usage:
        engine = BaselineEngine(prices, spy_prices, initial_capital=10000)
        comparison = engine.run_all()
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        spy_prices: pd.DataFrame | None = None,
        symbol: str = "UNKNOWN",
        initial_capital: float = 10_000.0,
        monthly_contribution: float = 1_000.0,
        recovery_target_pct: float = 80.0,
    ) -> None:
        """
        Initialize the baseline engine.
        
        Args:
            prices: OHLCV DataFrame for the asset
            spy_prices: OHLCV DataFrame for SPY (benchmark)
            symbol: Symbol name
            initial_capital: Starting capital
            monthly_contribution: Monthly DCA amount
            recovery_target_pct: % of dip to recover before selling (dip trading)
        """
        self.symbol = symbol
        self.prices = prices
        self.spy_prices = spy_prices
        
        self.initial_capital = initial_capital
        self.monthly_contribution = monthly_contribution
        self.recovery_target_pct = recovery_target_pct
        
        # Extract close prices
        self.close = self._get_close(prices)
        self.spy_close = self._get_close(spy_prices) if spy_prices is not None else None
        
        # Years of data
        self.years = len(self.close) / 252
        
        # Calculate optimal dip threshold using DipEntryOptimizer
        self.dip_threshold_pct, self.dip_analysis = self._calculate_optimal_dip_threshold()
    
    def _calculate_optimal_dip_threshold(self) -> tuple[float, Any]:
        """
        Use DipEntryOptimizer to find the statistically optimal dip threshold.
        
        Returns:
            (optimal_threshold_pct, full_analysis)
        """
        try:
            from app.quant_engine.dip_entry_optimizer import DipEntryOptimizer
            
            optimizer = DipEntryOptimizer()
            analysis = optimizer.analyze(self.prices, self.symbol)
            
            # Use the optimal dip threshold from analysis
            # This is the statistically calculated best threshold
            optimal_threshold = analysis.optimal_dip_threshold
            
            # If no optimal found, fall back to max profit threshold
            if optimal_threshold == 0 or optimal_threshold is None:
                optimal_threshold = analysis.max_profit_threshold
            
            # If still nothing, use a reasonable default based on stock volatility
            if optimal_threshold == 0 or optimal_threshold is None:
                # Use typical dip frequency as guide
                if analysis.avg_annual_dips_10pct >= 2:
                    optimal_threshold = -10.0
                elif analysis.avg_annual_dips_15pct >= 1:
                    optimal_threshold = -15.0
                else:
                    optimal_threshold = -20.0
            
            logger.info(
                f"{self.symbol}: Optimal dip threshold = {optimal_threshold:.1f}% "
                f"(from DipEntryOptimizer, {analysis.data_years:.1f} years of data)"
            )
            
            return optimal_threshold, analysis
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal dip: {e}. Using default -15%")
            return -15.0, None
    
    def _get_close(self, df: pd.DataFrame) -> pd.Series:
        """Extract close prices from DataFrame."""
        if "Close" in df.columns:
            return df["Close"]
        elif "close" in df.columns:
            return df["close"]
        elif "Adj Close" in df.columns:
            return df["Adj Close"]
        else:
            raise ValueError("No close price column found")
    
    def run_all(
        self,
        optimized_return_pct: float = 0.0,
        genome: Any = None,
        indicator_matrix: Any = None,
    ) -> BaselineComparison:
        """
        Run all baseline strategies and generate comparison.
        
        Args:
            optimized_return_pct: Return from optimized strategy (for comparison) - deprecated
            genome: Optional StrategyGenome from AlphaFactory for technical trading
            indicator_matrix: Optional IndicatorMatrix for technical trading
            
        Returns:
            Complete BaselineComparison with recommendation
        """
        # Run all strategies for the target symbol
        buy_hold = self.run_buy_hold(self.close, self.symbol)
        dca = self.run_dca(self.close, self.symbol)
        buy_dips = self.run_buy_dips_hold(self.close, self.symbol)
        dip_trading = self.run_dip_trading(self.close, self.symbol)
        
        # Run technical trading if genome provided
        technical_trading = None
        if genome is not None:
            technical_trading = self.run_technical_trading(
                self.close, self.symbol, genome, indicator_matrix
            )
        
        # Run regime-aware technical trading (always, uses simple SMA strategy)
        regime_aware_technical = self.run_regime_aware_technical(
            self.close, self.symbol
        )
        
        # Run lump sum with same capital as DCA (informational only)
        lump_sum = self.run_lump_sum(self.close, self.symbol, dca.total_invested)
        
        # Run SPY strategies
        if self.spy_close is not None:
            spy_buy_hold = self.run_buy_hold(self.spy_close, "SPY", is_spy=True)
            spy_dca = self.run_dca(self.spy_close, "SPY", is_spy=True)
        else:
            # Fallback - no SPY data
            spy_buy_hold = buy_hold.model_copy(update={
                "strategy_type": BaselineStrategyType.SPY_BUY_HOLD,
                "name": "SPY Buy & Hold (no data)"
            })
            spy_dca = dca.model_copy(update={
                "strategy_type": BaselineStrategyType.SPY_DCA,
                "name": "SPY DCA (no data)"
            })
        
        # Collect all results for ranking (only include comparable strategies)
        # Fair comparison: all use $10k + $1k/month
        all_results = {
            "DCA Monthly": dca,
            "Buy Dips & Hold": buy_dips,
            "SPY DCA": spy_dca,
        }
        if dip_trading:
            all_results["Perfect Dip Trading"] = dip_trading
        if technical_trading:
            all_results["Technical Trading"] = technical_trading
        if regime_aware_technical:
            all_results["Regime-Aware Technical"] = regime_aware_technical
        
        # Add reference strategies (different capital, for context only)
        reference_results = {
            "Buy & Hold": buy_hold,
            "Lump Sum": lump_sum,
            "SPY Buy & Hold": spy_buy_hold,
        }
        
        # Combine for ranking
        all_for_ranking = {**all_results, **reference_results}
        
        # Rank by return
        ranked_by_return = sorted(
            all_for_ranking.keys(),
            key=lambda k: all_for_ranking[k].total_return_pct,
            reverse=True
        )
        
        # Rank by Sharpe
        ranked_by_sharpe = sorted(
            all_for_ranking.keys(),
            key=lambda k: all_for_ranking[k].sharpe_ratio,
            reverse=True
        )
        
        # Use technical trading return for recommendation if available
        optimized_pct = 0.0
        if technical_trading:
            optimized_pct = technical_trading.total_return_pct
        elif optimized_return_pct != 0.0:
            optimized_pct = optimized_return_pct
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            all_for_ranking,
            ranked_by_return,
            ranked_by_sharpe,
            optimized_pct,
        )
        
        return BaselineComparison(
            symbol=self.symbol,
            dca=dca,
            buy_dips=buy_dips,
            dip_trading=dip_trading,
            technical_trading=technical_trading,
            regime_aware_technical=regime_aware_technical,
            buy_hold=buy_hold,
            lump_sum=lump_sum,
            spy_buy_hold=spy_buy_hold,
            spy_dca=spy_dca,
            ranked_by_return=ranked_by_return,
            ranked_by_sharpe=ranked_by_sharpe,
            recommendation=recommendation,
        )
    
    def run_buy_hold(
        self,
        close: pd.Series,
        symbol: str,
        is_spy: bool = False,
    ) -> BaselineResult:
        """
        Buy & Hold strategy: Buy on day 1, hold forever.
        """
        start_price = close.iloc[0]
        end_price = close.iloc[-1]
        
        shares = self.initial_capital / start_price
        final_value = shares * end_price
        
        total_return_pct = (final_value / self.initial_capital - 1) * 100
        years = len(close) / 252
        annualized = ((final_value / self.initial_capital) ** (1 / max(years, 0.5)) - 1) * 100
        
        # Calculate risk metrics
        returns = close.pct_change().dropna()
        sharpe = self._calculate_sharpe(returns)
        volatility = returns.std() * np.sqrt(252) * 100
        max_dd = self._calculate_max_drawdown(close)
        
        strategy_type = BaselineStrategyType.SPY_BUY_HOLD if is_spy else BaselineStrategyType.BUY_HOLD
        
        return BaselineResult(
            strategy_type=strategy_type,
            name=f"{symbol} Buy & Hold",
            description=f"Buy ${self.initial_capital:.0f} of {symbol} on day 1 and hold",
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            profit=final_value - self.initial_capital,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            volatility_pct=volatility,
            total_buys=1,
            total_invested=self.initial_capital,
            avg_cost_basis=start_price,
            years=years,
        )
    
    def run_lump_sum(
        self,
        close: pd.Series,
        symbol: str,
        total_dca_capital: float,
    ) -> BaselineResult:
        """
        Lump Sum strategy: Buy all DCA-equivalent capital on day 1.
        
        This provides a fair comparison to DCA by using the same total capital.
        If DCA invests $10k + $1k/month for 5 years = $70k total,
        this invests $70k on day 1.
        """
        start_price = close.iloc[0]
        end_price = close.iloc[-1]
        
        shares = total_dca_capital / start_price
        final_value = shares * end_price
        
        total_return_pct = (final_value / total_dca_capital - 1) * 100
        years = len(close) / 252
        annualized = ((final_value / total_dca_capital) ** (1 / max(years, 0.5)) - 1) * 100
        
        # Calculate risk metrics
        returns = close.pct_change().dropna()
        sharpe = self._calculate_sharpe(returns)
        volatility = returns.std() * np.sqrt(252) * 100
        max_dd = self._calculate_max_drawdown(close)
        
        return BaselineResult(
            strategy_type=BaselineStrategyType.LUMP_SUM,
            name=f"{symbol} Lump Sum (${total_dca_capital:,.0f})",
            description=f"Invest ${total_dca_capital:,.0f} on day 1 (same capital as DCA)",
            initial_capital=total_dca_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            profit=final_value - total_dca_capital,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            volatility_pct=volatility,
            total_buys=1,
            total_invested=total_dca_capital,
            avg_cost_basis=start_price,
            years=years,
        )
    
    def run_dca(
        self,
        close: pd.Series,
        symbol: str,
        is_spy: bool = False,
    ) -> BaselineResult:
        """
        Dollar Cost Average: Invest fixed amount monthly.
        
        - Start with initial_capital on day 1
        - Add monthly_contribution each month
        """
        shares = 0.0
        total_invested = 0.0
        total_buys = 0
        
        # Initial buy
        shares = self.initial_capital / close.iloc[0]
        total_invested = self.initial_capital
        total_buys = 1
        
        # Monthly contributions
        last_month = None
        for i, (date, price) in enumerate(close.items()):
            if hasattr(date, 'month'):
                current_month = (date.year, date.month)
                if last_month is not None and current_month != last_month:
                    # New month - invest
                    shares += self.monthly_contribution / price
                    total_invested += self.monthly_contribution
                    total_buys += 1
                last_month = current_month
        
        final_value = shares * close.iloc[-1]
        total_return_pct = (final_value / total_invested - 1) * 100
        
        years = len(close) / 252
        annualized = ((final_value / total_invested) ** (1 / max(years, 0.5)) - 1) * 100
        
        avg_cost = total_invested / shares if shares > 0 else close.iloc[0]
        
        # Risk metrics on the portfolio value over time
        # Simulate portfolio value at each point
        portfolio_values = self._simulate_dca_equity(close)
        port_returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe = self._calculate_sharpe(port_returns)
        volatility = port_returns.std() * np.sqrt(252) * 100 if len(port_returns) > 0 else 0.0
        max_dd = self._calculate_max_drawdown(pd.Series(portfolio_values))
        
        strategy_type = BaselineStrategyType.SPY_DCA if is_spy else BaselineStrategyType.DCA_MONTHLY
        
        return BaselineResult(
            strategy_type=strategy_type,
            name=f"{symbol} DCA (${self.monthly_contribution:.0f}/month)",
            description=f"Start with ${self.initial_capital:.0f}, add ${self.monthly_contribution:.0f}/month = ${total_invested:.0f} total invested",
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            profit=final_value - total_invested,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            volatility_pct=volatility,
            total_buys=total_buys,
            total_invested=total_invested,
            avg_cost_basis=avg_cost,
            years=years,
        )
    
    def _simulate_dca_equity(self, close: pd.Series) -> list[float]:
        """Simulate DCA portfolio value over time."""
        shares = self.initial_capital / close.iloc[0]
        equity = [self.initial_capital]
        
        last_month = None
        for i, (date, price) in enumerate(close.items()):
            if i == 0:
                continue
            if hasattr(date, 'month'):
                current_month = (date.year, date.month)
                if last_month is not None and current_month != last_month:
                    shares += self.monthly_contribution / price
                last_month = current_month
            equity.append(shares * price)
        
        return equity
    
    def run_buy_dips_hold(
        self,
        close: pd.Series,
        symbol: str,
    ) -> BaselineResult:
        """
        Buy Dips & Hold: Only buy when price dips, then hold.
        
        - Initial capital waits for first dip
        - Monthly contribution only invested on dip months
        - Never sell
        """
        dips = self._detect_dips(close)
        
        shares = 0.0
        total_invested = 0.0
        total_buys = 0
        cash_waiting = self.initial_capital
        
        last_month = None
        for i, (date, price) in enumerate(close.items()):
            is_dip = i in dips["indices"]
            
            if hasattr(date, 'month'):
                current_month = (date.year, date.month)
                if last_month is not None and current_month != last_month:
                    # New month - add to waiting cash
                    cash_waiting += self.monthly_contribution
                last_month = current_month
            
            if is_dip and cash_waiting > 0:
                # Buy with all waiting cash
                shares += cash_waiting / price
                total_invested += cash_waiting
                total_buys += 1
                cash_waiting = 0
        
        # If we never got a dip, invest remaining at end
        if shares == 0 and cash_waiting > 0:
            shares = cash_waiting / close.iloc[-1]
            total_invested = cash_waiting
            total_buys = 1
            cash_waiting = 0
        
        final_value = shares * close.iloc[-1] + cash_waiting
        total_return_pct = (final_value / max(total_invested, 1) - 1) * 100
        
        years = len(close) / 252
        annualized = ((final_value / max(total_invested, 1)) ** (1 / max(years, 0.5)) - 1) * 100
        
        avg_cost = total_invested / shares if shares > 0 else close.iloc[0]
        
        # Simulate equity curve
        portfolio = self._simulate_buy_dips_equity(close, dips)
        port_returns = pd.Series(portfolio).pct_change().dropna()
        sharpe = self._calculate_sharpe(port_returns)
        volatility = port_returns.std() * np.sqrt(252) * 100 if len(port_returns) > 0 else 0.0
        max_dd = self._calculate_max_drawdown(pd.Series(portfolio))
        
        avg_dip_depth = dips["avg_depth"] if dips["count"] > 0 else 0.0
        optimal_thresh = dips.get("optimal_threshold", self.dip_threshold_pct)
        
        # Get extra info from dip analysis if available
        typical_recovery = 60.0
        if self.dip_analysis is not None:
            typical_recovery = getattr(self.dip_analysis, 'typical_recovery_days', 60.0)
        
        # Calculate maximum possible invested (same as DCA)
        months = int(years * 12)
        max_possible_invested = self.initial_capital + self.monthly_contribution * months
        
        return BaselineResult(
            strategy_type=BaselineStrategyType.BUY_DIPS_HOLD,
            name=f"{symbol} Buy Dips & Hold (optimal: {optimal_thresh:.0f}%)",
            description=f"Save ${self.monthly_contribution:.0f}/month, deploy on {abs(optimal_thresh):.0f}% dips. Invested ${total_invested:.0f} of ${max_possible_invested:.0f} available (recovery: {typical_recovery:.0f}d)",
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            profit=final_value - total_invested,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            volatility_pct=volatility,
            total_buys=total_buys,
            total_invested=total_invested,
            avg_cost_basis=avg_cost,
            years=years,
            dips_detected=dips["count"],
            avg_dip_depth_pct=avg_dip_depth,
        )
    
    def _simulate_buy_dips_equity(
        self,
        close: pd.Series,
        dips: dict,
    ) -> list[float]:
        """Simulate Buy Dips portfolio value over time."""
        shares = 0.0
        cash = self.initial_capital
        equity = []
        
        last_month = None
        dip_indices = set(dips["indices"])
        
        for i, (date, price) in enumerate(close.items()):
            if hasattr(date, 'month'):
                current_month = (date.year, date.month)
                if last_month is not None and current_month != last_month:
                    cash += self.monthly_contribution
                last_month = current_month
            
            if i in dip_indices and cash > 0:
                shares += cash / price
                cash = 0
            
            equity.append(shares * price + cash)
        
        return equity
    
    def run_dip_trading(
        self,
        close: pd.Series,
        symbol: str,
    ) -> BaselineResult | None:
        """
        Perfect Dip Trading with Compounding.
        
        - Start with initial_capital
        - Add monthly_contribution each month (saved as cash if not in position)
        - Buy when price drops to optimal threshold
        - Sell when price recovers by recovery_target_pct
        - Reinvest ALL gains + accumulated cash on next dip (compounding)
        - Repeat
        
        This uses the FULL portfolio value (initial + monthly + gains) for each trade.
        """
        dips = self._detect_dips(close)
        
        if dips["count"] == 0:
            return None
        
        cash = self.initial_capital
        shares = 0.0
        position = False
        entry_price = 0.0
        entry_dip_depth = 0.0
        
        total_trades = 0
        winning_trades = 0
        total_contributions = self.initial_capital  # Track total $ put in
        
        equity = [cash]
        
        dip_indices = set(dips["indices"])
        dip_depths = {idx: depth for idx, depth in zip(dips["indices"], dips["depths"])}
        
        # Track months for monthly contributions
        last_month = None
        dates = close.index if hasattr(close.index, '__iter__') else range(len(close))
        
        for i, (date, price) in enumerate(zip(dates, close.values)):
            # Add monthly contribution (whether in position or not)
            if hasattr(date, 'month'):
                current_month = (date.year, date.month)
                if last_month is not None and current_month != last_month:
                    # New month - add contribution
                    if position:
                        # If in position, add to shares at current price
                        shares += self.monthly_contribution / price
                    else:
                        # If not in position, add to cash pile
                        cash += self.monthly_contribution
                    total_contributions += self.monthly_contribution
                last_month = current_month
            
            if i == 0:
                equity.append(cash + shares * price)
                continue
            
            if not position:
                # Look for dip entry - use ALL available cash
                if i in dip_indices and cash > 0:
                    shares = cash / price
                    entry_price = price
                    entry_dip_depth = dip_depths[i]
                    cash = 0
                    position = True
                    total_trades += 1
            else:
                # Look for recovery exit
                drop_amount = abs(entry_dip_depth) / 100 * entry_price
                target_price = entry_price + drop_amount * (self.recovery_target_pct / 100)
                
                if price >= target_price:
                    # Sell ALL shares - capture gains
                    cash = shares * price
                    if price > entry_price:
                        winning_trades += 1
                    shares = 0
                    position = False
            
            equity.append(cash + shares * price)
        
        # Close any open position at end
        if position:
            cash = shares * close.iloc[-1]
            shares = 0
        
        final_value = cash
        profit = final_value - total_contributions
        total_return_pct = (final_value / total_contributions - 1) * 100
        
        years = len(close) / 252
        annualized = ((final_value / total_contributions) ** (1 / max(years, 0.5)) - 1) * 100
        
        port_returns = pd.Series(equity).pct_change().dropna()
        sharpe = self._calculate_sharpe(port_returns)
        volatility = port_returns.std() * np.sqrt(252) * 100 if len(port_returns) > 0 else 0.0
        max_dd = self._calculate_max_drawdown(pd.Series(equity))
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Get typical recovery from dip analysis
        typical_recovery = 60.0
        if self.dip_analysis is not None:
            typical_recovery = getattr(self.dip_analysis, 'typical_recovery_days', 60.0)
        
        return BaselineResult(
            strategy_type=BaselineStrategyType.DIP_TRADING,
            name=f"{symbol} Perfect Dip Trading",
            description=f"Buy {abs(self.dip_threshold_pct):.0f}% dips, sell +{self.recovery_target_pct:.0f}% recovery. {total_trades} trades, {win_rate:.0f}% win rate. Compounding gains.",
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            profit=profit,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            volatility_pct=volatility,
            total_buys=total_trades,
            total_invested=total_contributions,
            avg_cost_basis=total_contributions / max(total_trades, 1),
            years=years,
            dips_detected=dips["count"],
            avg_dip_depth_pct=dips["avg_depth"],
        )
    
    def run_technical_trading(
        self,
        close: pd.Series,
        symbol: str,
        genome: Any,  # StrategyGenome from AlphaFactory
        indicator_matrix: Any = None,  # IndicatorMatrix from AlphaFactory
    ) -> BaselineResult | None:
        """
        Technical Trading with Compounding using AlphaFactory optimized strategy.
        
        - Start with initial_capital
        - Add monthly_contribution each month
        - Trade using the optimized entry/exit signals
        - Reinvest ALL gains + accumulated cash (compounding)
        - Use stop-loss and take-profit from genome
        
        Args:
            close: Price series
            symbol: Stock symbol
            genome: StrategyGenome from AlphaFactory optimization
            indicator_matrix: Pre-computed IndicatorMatrix (optional, will build if not provided)
            
        Returns:
            BaselineResult with performance metrics
        """
        if genome is None:
            return None
        
        # Build indicator matrix if not provided
        if indicator_matrix is None:
            try:
                from app.quant_engine.backtest_v2.alpha_factory import IndicatorMatrix
                indicator_matrix = IndicatorMatrix(self.prices)
            except Exception as e:
                logger.warning(f"Could not build indicator matrix: {e}")
                return None
        
        # Generate entry/exit signals
        try:
            entry_signals = genome.generate_entry_signals(indicator_matrix)
            exit_signals = genome.generate_exit_signals(indicator_matrix)
        except Exception as e:
            logger.warning(f"Could not generate signals: {e}")
            return None
        
        cash = self.initial_capital
        shares = 0.0
        position = False
        entry_price = 0.0
        entry_bar = 0
        
        total_trades = 0
        winning_trades = 0
        total_contributions = self.initial_capital
        
        equity = [cash]
        
        # Track months for monthly contributions
        last_month = None
        dates = close.index if hasattr(close.index, '__iter__') else range(len(close))
        prices = close.values
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Add monthly contribution
            if hasattr(date, 'month'):
                current_month = (date.year, date.month)
                if last_month is not None and current_month != last_month:
                    if position:
                        shares += self.monthly_contribution / price
                    else:
                        cash += self.monthly_contribution
                    total_contributions += self.monthly_contribution
                last_month = current_month
            
            if i == 0:
                equity.append(cash + shares * price)
                continue
            
            if not position:
                # Check entry signal
                if entry_signals[i] and cash > 0:
                    shares = cash / price
                    entry_price = price
                    entry_bar = i
                    cash = 0
                    position = True
                    total_trades += 1
            else:
                # Check exits: signal, stop-loss, take-profit, max holding period
                holding_days = i - entry_bar
                pnl_pct = (price - entry_price) / entry_price
                
                should_exit = (
                    exit_signals[i] or
                    pnl_pct <= -genome.stop_loss_pct or
                    pnl_pct >= genome.take_profit_pct or
                    holding_days >= genome.holding_period_max
                )
                
                if should_exit:
                    cash = shares * price
                    if price > entry_price:
                        winning_trades += 1
                    shares = 0
                    position = False
            
            equity.append(cash + shares * price)
        
        # Close any open position
        if position:
            cash = shares * close.iloc[-1]
            shares = 0
        
        final_value = cash
        profit = final_value - total_contributions
        total_return_pct = (final_value / total_contributions - 1) * 100
        
        years = len(close) / 252
        annualized = ((final_value / total_contributions) ** (1 / max(years, 0.5)) - 1) * 100
        
        port_returns = pd.Series(equity).pct_change().dropna()
        sharpe = self._calculate_sharpe(port_returns)
        volatility = port_returns.std() * np.sqrt(252) * 100 if len(port_returns) > 0 else 0.0
        max_dd = self._calculate_max_drawdown(pd.Series(equity))
        
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        return BaselineResult(
            strategy_type=BaselineStrategyType.TECHNICAL_TRADING,
            name=f"{symbol} Technical Trading ({genome.name})",
            description=f"AlphaFactory optimized: {len(genome.entry_conditions)} entry, {len(genome.exit_conditions)} exit conditions. {total_trades} trades, {win_rate:.0f}% win rate. SL {genome.stop_loss_pct:.0%}, TP {genome.take_profit_pct:.0%}",
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            profit=profit,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            volatility_pct=volatility,
            total_buys=total_trades,
            total_invested=total_contributions,
            avg_cost_basis=total_contributions / max(total_trades, 1),
            years=years,
        )
    
    def run_regime_aware_technical(
        self,
        close: pd.Series,
        symbol: str,
        genome: Any = None,
        indicator_matrix: Any = None,
    ) -> BaselineResult | None:
        """
        Regime-Aware Technical Trading with Compounding.
        
        Adapts trading behavior based on market regime:
        
        BULL: Standard technical trading (SMA50 crossover), 10% stop loss
        BEAR: More patient entries, wider 20% stops, hold through volatility
        CRASH: Aggressive accumulation, no stop loss, wait for major recovery
        RECOVERY: Standard entries with trailing stop to protect gains
        
        Uses SPY prices for regime detection.
        
        Args:
            close: Price series for the stock
            symbol: Stock symbol
            genome: Optional - AlphaFactory genome for entry signals
            indicator_matrix: Optional - pre-computed indicators
            
        Returns:
            BaselineResult with performance metrics and trade details
        """
        if self.spy_close is None or len(self.spy_close) < 200:
            logger.warning(f"Insufficient SPY data for regime detection")
            return None
        
        # Initialize regime detector with SPY prices
        regime_detector = RegimeDetector(self.spy_close)
        
        # Calculate technical indicators for the stock
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        rolling_high = close.rolling(252).max()
        
        # Use genome signals if available, otherwise use SMA crossover
        use_genome = genome is not None and indicator_matrix is not None
        if use_genome:
            try:
                entry_signals = genome.generate_entry_signals(indicator_matrix)
                exit_signals = genome.generate_exit_signals(indicator_matrix)
            except Exception as e:
                logger.warning(f"Could not generate genome signals: {e}")
                use_genome = False
        
        cash = self.initial_capital
        shares = 0.0
        position = False
        entry_price = 0.0
        entry_bar = 0
        entry_regime = None
        highest_price_in_trade = 0.0  # For trailing stop
        
        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        total_wins = 0.0
        total_losses = 0.0
        total_contributions = self.initial_capital
        
        trade_details: list[TradeDetail] = []
        current_trade_entry_date = None
        current_trade_entry_reason = ""
        
        equity = [cash]
        
        # Track months for monthly contributions
        last_month = None
        dates = close.index if hasattr(close.index, '__iter__') else range(len(close))
        prices = close.values
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Add monthly contribution
            if hasattr(date, 'month'):
                current_month = (date.year, date.month)
                if last_month is not None and current_month != last_month:
                    if position:
                        shares += self.monthly_contribution / price
                    else:
                        cash += self.monthly_contribution
                    total_contributions += self.monthly_contribution
                last_month = current_month
            
            if i < 200:  # Need enough data for regime detection
                equity.append(cash + shares * price)
                continue
            
            # Detect current market regime
            try:
                regime_state = regime_detector.detect_at_date(date)
                regime = regime_state.regime
            except Exception:
                regime = MarketRegime.BULL  # Default to BULL if detection fails
            
            # Calculate P&L if in position
            pnl_pct = 0
            if position and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price
                highest_price_in_trade = max(highest_price_in_trade, price)
            
            if not position:
                should_enter = False
                entry_reason = ""
                
                # Entry logic - adapt based on regime
                if regime == MarketRegime.BULL:
                    # Standard: Use genome signals or SMA50 crossover
                    if use_genome and entry_signals[i]:
                        should_enter = True
                        entry_reason = "Genome entry signal (BULL)"
                    elif not use_genome and price > sma50.iloc[i]:
                        should_enter = True
                        entry_reason = "Price > SMA50 (BULL momentum)"
                
                elif regime == MarketRegime.BEAR:
                    # Bear: Wait for reversal signals - price crossing above SMA50
                    if i > 0 and price > sma50.iloc[i] and prices[i-1] < sma50.iloc[i-1]:
                        should_enter = True
                        entry_reason = "SMA50 crossover (BEAR reversal)"
                
                elif regime == MarketRegime.CRASH:
                    # Crash: Buy on significant dips (15%+ from high)
                    dd_from_high = (price - rolling_high.iloc[i]) / rolling_high.iloc[i]
                    if dd_from_high <= -0.15:
                        should_enter = True
                        entry_reason = f"Crash accumulation ({dd_from_high:.1%} from high)"
                
                elif regime == MarketRegime.RECOVERY:
                    # Recovery: Standard momentum entry
                    if use_genome and entry_signals[i]:
                        should_enter = True
                        entry_reason = "Genome entry signal (RECOVERY)"
                    elif not use_genome and price > sma50.iloc[i]:
                        should_enter = True
                        entry_reason = "Price > SMA50 (RECOVERY momentum)"
                
                if should_enter and cash > 0:
                    shares = cash / price  # Always full position
                    entry_price = price
                    entry_bar = i
                    entry_regime = regime
                    highest_price_in_trade = price
                    current_trade_entry_date = date
                    current_trade_entry_reason = entry_reason
                    cash = 0
                    position = True
                    total_trades += 1
            
            else:
                # In position - exit logic varies by regime
                should_exit = False
                exit_reason = ""
                holding_bars = i - entry_bar
                
                if regime == MarketRegime.BULL:
                    # Bull: Standard 10% stop, exit on SMA50 breakdown
                    if pnl_pct <= -0.10:
                        should_exit = True
                        exit_reason = "Stop loss 10% (BULL)"
                    elif price < sma50.iloc[i] * 0.98:
                        should_exit = True
                        exit_reason = "SMA50 breakdown (BULL)"
                    elif use_genome and exit_signals[i]:
                        should_exit = True
                        exit_reason = "Genome exit signal (BULL)"
                
                elif regime == MarketRegime.BEAR:
                    # Bear: Wider 20% stop, hold through volatility
                    if pnl_pct <= -0.20:
                        should_exit = True
                        exit_reason = "Stop loss 20% (BEAR)"
                    # Take profit on bounce to SMA200 with profit
                    elif price > sma200.iloc[i] and pnl_pct > 0.15:
                        should_exit = True
                        exit_reason = "Profit target at SMA200 (BEAR recovery)"
                
                elif regime == MarketRegime.CRASH:
                    # Crash: NO stop loss - hold for major recovery
                    if price > sma200.iloc[i] and pnl_pct > 0.20:
                        should_exit = True
                        exit_reason = "Major recovery +20% at SMA200 (CRASH)"
                
                elif regime == MarketRegime.RECOVERY:
                    # Recovery: Trailing stop to protect gains
                    trailing_stop_price = highest_price_in_trade * 0.92  # 8% trailing
                    if price < trailing_stop_price:
                        should_exit = True
                        exit_reason = "Trailing stop 8% (RECOVERY)"
                    elif pnl_pct <= -0.12:
                        should_exit = True
                        exit_reason = "Stop loss 12% (RECOVERY)"
                
                if should_exit:
                    cash = shares * price
                    trade_pnl = (price - entry_price) * shares
                    is_winner = price > entry_price
                    
                    if is_winner:
                        winning_trades += 1
                        total_wins += trade_pnl
                    else:
                        losing_trades += 1
                        total_losses += abs(trade_pnl)
                    
                    # Record trade detail
                    trade_details.append(TradeDetail(
                        trade_num=total_trades,
                        entry_date=str(current_trade_entry_date)[:10] if current_trade_entry_date else "",
                        entry_price=round(entry_price, 2),
                        entry_reason=current_trade_entry_reason,
                        entry_regime=entry_regime.value if entry_regime else "UNKNOWN",
                        exit_date=str(date)[:10] if hasattr(date, 'strftime') else str(date),
                        exit_price=round(price, 2),
                        exit_reason=exit_reason,
                        shares=round(shares, 4),
                        return_pct=round(pnl_pct * 100, 2),
                        pnl=round(trade_pnl, 2),
                        holding_days=holding_bars,
                        is_winner=is_winner,
                    ))
                    
                    shares = 0
                    position = False
                    highest_price_in_trade = 0
                    entry_regime = None
            
            equity.append(cash + shares * price)
        
        # Close any open position at end
        if position:
            final_price = close.iloc[-1]
            cash = shares * final_price
            trade_pnl = (final_price - entry_price) * shares
            is_winner = final_price > entry_price
            
            if is_winner:
                winning_trades += 1
                total_wins += trade_pnl
            else:
                losing_trades += 1
                total_losses += abs(trade_pnl)
            
            trade_details.append(TradeDetail(
                trade_num=total_trades,
                entry_date=str(current_trade_entry_date)[:10] if current_trade_entry_date else "",
                entry_price=round(entry_price, 2),
                entry_reason=current_trade_entry_reason,
                entry_regime=entry_regime.value if entry_regime else "UNKNOWN",
                exit_date=str(close.index[-1])[:10],
                exit_price=round(final_price, 2),
                exit_reason="End of period (position closed)",
                shares=round(shares, 4),
                return_pct=round((final_price - entry_price) / entry_price * 100, 2),
                pnl=round(trade_pnl, 2),
                holding_days=len(close) - entry_bar,
                is_winner=is_winner,
            ))
            shares = 0
        
        final_value = cash
        profit = final_value - total_contributions
        total_return_pct = (final_value / total_contributions - 1) * 100
        
        years = len(close) / 252
        annualized = ((final_value / total_contributions) ** (1 / max(years, 0.5)) - 1) * 100
        
        port_returns = pd.Series(equity).pct_change().dropna()
        sharpe = self._calculate_sharpe(port_returns)
        volatility = port_returns.std() * np.sqrt(252) * 100 if len(port_returns) > 0 else 0.0
        max_dd = self._calculate_max_drawdown(pd.Series(equity))
        
        # Calculate trade stats
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        avg_win = total_wins / winning_trades if winning_trades > 0 else 0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 999.99
        
        # Calculate avg % returns
        winning_returns = [t.return_pct for t in trade_details if t.is_winner]
        losing_returns = [t.return_pct for t in trade_details if not t.is_winner]
        avg_win_pct = sum(winning_returns) / len(winning_returns) if winning_returns else 0
        avg_loss_pct = sum(losing_returns) / len(losing_returns) if losing_returns else 0
        
        return BaselineResult(
            strategy_type=BaselineStrategyType.REGIME_AWARE_TECHNICAL,
            name=f"{symbol} Regime-Aware Technical",
            description=f"Adapts to BULL/BEAR/CRASH/RECOVERY regimes. {total_trades} trades, {win_rate:.0f}% win rate. Profit factor: {profit_factor:.2f}",
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized,
            profit=profit,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            volatility_pct=volatility,
            total_buys=total_trades,
            total_invested=total_contributions,
            avg_cost_basis=total_contributions / max(total_trades, 1),
            years=years,
            trade_details=trade_details,
            win_rate_pct=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            profit_factor=min(profit_factor, 999.99),
        )

    def _detect_dips(self, close: pd.Series) -> dict:
        """
        Detect dip entry points using the optimal threshold from DipEntryOptimizer.
        
        If DipEntryOptimizer analysis is available, uses the actual dip events
        from the analysis for more accurate detection.
        
        Returns:
            dict with 'indices', 'depths', 'count', 'avg_depth', 'optimal_threshold'
        """
        # Try to use dip events from DipEntryOptimizer analysis
        if self.dip_analysis is not None and hasattr(self.dip_analysis, 'all_dip_events'):
            dip_events = self.dip_analysis.all_dip_events
            
            # Get threshold stats for the optimal threshold
            optimal_threshold = self.dip_threshold_pct
            
            # Filter events at or below the optimal threshold
            relevant_events = [
                e for e in dip_events 
                if e.threshold_crossed <= optimal_threshold
            ]
            
            if relevant_events:
                # Map events to indices in the close series
                indices = []
                depths = []
                
                for event in relevant_events:
                    # Find the index in close series by date
                    try:
                        if hasattr(event, 'date') and event.date is not None:
                            # Try to find matching date
                            matching = close.index.get_indexer([event.date], method='nearest')
                            if len(matching) > 0 and matching[0] >= 0:
                                indices.append(matching[0])
                                depths.append(event.threshold_crossed)
                    except Exception:
                        pass
                
                if indices:
                    return {
                        "indices": indices,
                        "depths": depths,
                        "count": len(indices),
                        "avg_depth": float(np.mean(depths)),
                        "optimal_threshold": optimal_threshold,
                    }
        
        # Fallback: Simple threshold crossing detection
        rolling_high = close.rolling(window=252, min_periods=1).max()
        drawdown = (close - rolling_high) / rolling_high * 100
        
        # Find crossing points
        crossed = (drawdown <= self.dip_threshold_pct) & (drawdown.shift(1) > self.dip_threshold_pct)
        
        indices = []
        depths = []
        
        for i, is_cross in enumerate(crossed.values):
            if is_cross:
                indices.append(i)
                depths.append(drawdown.iloc[i])
        
        return {
            "indices": indices,
            "depths": depths,
            "count": len(indices),
            "avg_depth": float(np.mean(depths)) if depths else 0.0,
            "optimal_threshold": self.dip_threshold_pct,
        }
    
    def _calculate_sharpe(self, returns: pd.Series, rf_rate: float = 0.04) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 10:
            return 0.0
        
        excess_returns = returns - rf_rate / 252
        if returns.std() == 0:
            return 0.0
        
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
        
        if np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        
        return float(sharpe)
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak * 100
        return float(drawdown.min())
    
    def _generate_recommendation(
        self,
        results: dict[str, BaselineResult],
        ranked_by_return: list[str],
        ranked_by_sharpe: list[str],
        optimized_return_pct: float = 0.0,
    ) -> InvestmentRecommendation:
        """Generate the investment recommendation based on all results."""
        
        best_return_name = ranked_by_return[0]
        best_sharpe_name = ranked_by_sharpe[0]
        
        best_result = results[best_return_name]
        spy_bh = results.get("SPY Buy & Hold")
        spy_dca = results.get("SPY DCA")
        
        spy_return = spy_bh.total_return_pct if spy_bh else 0.0
        alpha_vs_spy = best_result.total_return_pct - spy_return
        
        # Determine recommendation
        if best_return_name.startswith("SPY"):
            # SPY wins - suggest SPY
            if "DCA" in best_return_name:
                rec_type = RecommendationType.SPY_DCA
                headline = f"Just DCA into SPY - it beats {self.symbol}"
                reasoning = (
                    f"SPY DCA returned {spy_dca.total_return_pct:+.1f}% vs "
                    f"{results['Buy & Hold'].total_return_pct:+.1f}% for {self.symbol} B&H. "
                    f"No complex strategy needed - just buy SPY monthly."
                )
            else:
                rec_type = RecommendationType.SWITCH_TO_SPY
                headline = f"Buy SPY instead of {self.symbol}"
                reasoning = (
                    f"SPY returned {spy_return:+.1f}% vs "
                    f"{results['Buy & Hold'].total_return_pct:+.1f}% for {self.symbol}. "
                    f"The market index outperformed this stock."
                )
            action_items = [
                "Consider switching allocation from this stock to SPY",
                f"Set up monthly ${self.monthly_contribution:.0f} auto-investment into SPY",
                "Review again in 6 months",
            ]
            warnings = [
                "Past performance doesn't guarantee future results",
                f"{self.symbol} may still be valuable for diversification",
            ]
            
        elif best_return_name == "Optimized Strategy":
            # Optimized strategy wins
            rec_type = RecommendationType.OPTIMIZED_STRATEGY
            headline = f"Use the optimized trading strategy"
            reasoning = (
                f"The AlphaFactory strategy returned {optimized_return_pct:+.1f}% vs "
                f"{results['Buy & Hold'].total_return_pct:+.1f}% for simple B&H. "
                f"Alpha vs SPY: {alpha_vs_spy:+.1f}%"
            )
            action_items = [
                "Review the strategy entry/exit conditions",
                "Paper trade for 1-3 months before committing capital",
                "Set position size per Kelly criterion (half-Kelly recommended)",
            ]
            warnings = [
                "Backtest results may not reflect live trading",
                "Ensure you can execute signals consistently",
            ]
            
        elif best_return_name == "DCA Monthly":
            rec_type = RecommendationType.DCA
            headline = f"DCA into {self.symbol} monthly"
            dca = results["DCA Monthly"]
            reasoning = (
                f"DCA returned {dca.total_return_pct:+.1f}% with "
                f"Sharpe {dca.sharpe_ratio:.2f} and max drawdown {dca.max_drawdown_pct:.1f}%. "
                f"Simple and effective."
            )
            action_items = [
                f"Set up automatic ${self.monthly_contribution:.0f}/month investment",
                "Don't try to time the market",
                "Review allocation annually",
            ]
            warnings = [
                "Stock still has individual company risk",
                "Consider diversifying across multiple stocks or ETFs",
            ]
            
        elif "Dip" in best_return_name:
            rec_type = RecommendationType.BUY_DIPS
            dips = results.get("Buy Dips & Hold", results.get("Dip Trading"))
            headline = f"Buy {self.symbol} on dips"
            reasoning = (
                f"Buying on dips returned {dips.total_return_pct:+.1f}% vs "
                f"{results['Buy & Hold'].total_return_pct:+.1f}% for B&H. "
                f"Detected {dips.dips_detected} dips (avg depth: {dips.avg_dip_depth_pct:.1f}%)."
            )
            action_items = [
                f"Set limit orders at {abs(self.dip_threshold_pct):.0f}% below recent highs",
                "Wait for dips - don't chase rallies",
                "Keep cash ready for opportunities",
            ]
            warnings = [
                "Dips may continue lower (falling knife risk)",
                "May miss gains while waiting for dips",
            ]
            
        else:
            # Default to Buy & Hold
            rec_type = RecommendationType.BUY_AND_HOLD
            bh = results["Buy & Hold"]
            headline = f"Just buy & hold {self.symbol}"
            reasoning = (
                f"Buy & Hold returned {bh.total_return_pct:+.1f}% with "
                f"less complexity than active strategies. "
                f"Simplicity wins."
            )
            action_items = [
                f"Buy {self.symbol} and hold for the long term",
                "Reinvest dividends if applicable",
                "Only add on significant dips (>15%)",
            ]
            warnings = [
                "Stock has individual company risk",
                "Consider position sizing relative to portfolio",
            ]
        
        return InvestmentRecommendation(
            recommendation=rec_type,
            headline=headline,
            reasoning=reasoning,
            best_strategy_name=best_return_name,
            best_strategy_return_pct=best_result.total_return_pct,
            spy_return_pct=spy_return,
            alpha_vs_spy=alpha_vs_spy,
            risk_adjusted_winner=best_sharpe_name,
            action_items=action_items,
            warnings=warnings,
        )
