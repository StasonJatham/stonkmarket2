"""
BacktestV2Service - Main Orchestrator for Regime-Adaptive Backtesting.

This service ties together all V2 components:
1. RegimeDetector - Identifies market regime (Bull/Bear/Crash/Recovery)
2. BearMarketStrategyFilter - META Rule fundamental checks for bear buys
3. PortfolioSimulator - Realistic portfolio simulation with DCA
4. AlphaGauntlet - Strategy validation (vs B&H, SPY, risk-adjusted)
5. CrashTester - Performance analysis during 2008, 2020, 2022 crashes
6. WalkForwardOptimizer - Out-of-sample validation

Key Philosophy:
- Bear markets are BUYING opportunities (not blocks)
- Fundamental checks prevent value traps during accumulation
- Technical signals are regime-dependent
- ALL history is used (not just 5 years) for robustness
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.quant_engine.core import (
    MarketRegime,
    RegimeService,
    RegimeState,
    StrategyMode,
    StrategyConfig,
)
from app.quant_engine.backtest_v2.fundamental_service import (
    FundamentalService,
    BearMarketStrategyFilter,
    MetaRuleResult,
    MetaRuleDecision,
    QuarterlyFundamentals,
)
from app.quant_engine.backtest_v2.portfolio_simulator import (
    PortfolioConfig,
    PortfolioSimulator,
    SimulationResult,
    SimulationScenario,
    ScenarioResult,
    Trade,
)
from app.quant_engine.backtest_v2.alpha_gauntlet import (
    AlphaGauntlet,
    GauntletConfig,
    GauntletResult,
    GauntletVerdict,
    quick_gauntlet,
)
from app.quant_engine.backtest_v2.crash_testing import (
    CrashTester,
    CrashTestResult,
    CRASH_PERIODS,
    get_crash_summary,
    identify_crash_periods,
)
from app.quant_engine.backtest_v2.walk_forward import (
    WalkForwardOptimizer,
    WFOConfig,
    WFOResult,
    WFOSummary,
)

# Note: Schemas are NOT imported here - API routes define their own response models
# from app.quant_engine.backtest_v2.schemas import ...

logger = logging.getLogger(__name__)


# Maximum history to fetch (30 years = ~7500 trading days)
MAX_LOOKBACK_DAYS = 7500


@dataclass
class BacktestV2Config:
    """Configuration for V2 backtest."""
    
    # Data range
    lookback_years: int = 20  # Use all available history
    
    # Portfolio
    initial_capital: float = 10_000.0
    monthly_contribution: float = 1_000.0
    
    # Scenarios to run
    run_dip_strategy: bool = True
    run_technical_strategy: bool = True
    
    # Validation
    run_alpha_gauntlet: bool = True
    run_crash_testing: bool = True
    run_walk_forward: bool = False  # Expensive, optional
    
    # Fundamental checks
    use_meta_rule: bool = True  # Apply fundamental guardrails in bear mode


@dataclass
class BacktestV2Result:
    """Complete V2 backtest result."""
    
    symbol: str
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    period_years: float
    
    # Current market context
    current_regime: RegimeState
    
    # Portfolio simulation results
    simulation: SimulationResult | None = None
    best_scenario: SimulationScenario | None = None
    best_return_pct: float = 0.0
    
    # Scenario comparisons
    strategy_vs_bh: float = 0.0
    dca_vs_lump_sum: float = 0.0
    dip_vs_regular_dca: float = 0.0
    
    # Validation results
    gauntlet: GauntletResult | None = None
    crash_tests: list[CrashTestResult] = field(default_factory=list)
    wfo_summary: WFOSummary | None = None
    
    # Regime breakdown
    regime_days: dict[MarketRegime, int] = field(default_factory=dict)
    regime_returns: dict[MarketRegime, float] = field(default_factory=dict)
    
    # Trade markers for frontend
    trade_markers: list[dict[str, Any]] = field(default_factory=list)
    
    # META Rule stats (bear mode)
    meta_rule_stats: dict[str, Any] = field(default_factory=dict)


class BacktestV2Service:
    """
    Main orchestrator for V2 backtesting.
    
    Ties together all components with proper flow:
    1. Fetch ALL historical data (prices + fundamentals)
    2. Detect regimes across history
    3. Generate signals (dip signals, technical signals)
    4. Apply META Rule in bear markets
    5. Simulate portfolio scenarios
    6. Validate with Alpha Gauntlet
    7. Stress test against historical crashes
    """
    
    def __init__(
        self,
        config: BacktestV2Config | None = None,
    ):
        self.config = config or BacktestV2Config()
        
        # Initialize components - use singleton RegimeService
        self.regime_service = RegimeService.get_instance()
        self.fundamental_service = FundamentalService()
        self.bear_filter = BearMarketStrategyFilter(self.fundamental_service)
        self.portfolio_simulator = PortfolioSimulator(
            config=PortfolioConfig(
                initial_capital=self.config.initial_capital,
                monthly_contribution=self.config.monthly_contribution,
            ),
            regime_service=self.regime_service,
        )
        self.gauntlet = AlphaGauntlet()
        self.crash_tester = CrashTester()
    
    async def run_full_backtest(
        self,
        symbol: str,
        prices: pd.DataFrame,
        spy_prices: pd.DataFrame | None = None,
        dip_signals: pd.Series | None = None,
        technical_signals: pd.Series | None = None,
    ) -> BacktestV2Result:
        """
        Run complete V2 backtest with all validations.
        
        Args:
            symbol: Stock ticker
            prices: DataFrame with OHLCV data (max history)
            spy_prices: SPY prices for benchmark (optional)
            dip_signals: Series with 1 for dip buy signals
            technical_signals: Series with 1 for buy, -1 for sell
            
        Returns:
            BacktestV2Result with complete analysis
        """
        logger.info(f"Starting V2 backtest for {symbol} with {len(prices)} days of data")
        
        # Extract close prices
        if "Close" in prices.columns:
            close_prices = prices["Close"]
        elif "close" in prices.columns:
            close_prices = prices["close"]
        else:
            close_prices = prices.iloc[:, 0]  # First column
        
        close_prices.index = pd.to_datetime(close_prices.index)
        
        # 1. Detect current regime - wrap close prices in a DataFrame for RegimeService
        price_df = pd.DataFrame({"close": close_prices})
        current_regime = self.regime_service.get_current_regime(price_df)
        logger.info(f"Current regime for {symbol}: {current_regime.regime.value}")
        
        # 2. Get historical regime states as a Series for backtesting
        regime_history = self.regime_service.get_regime_series(price_df)
        
        # 3. Apply META Rule to dip signals if in bear mode
        filtered_dip_signals = dip_signals
        meta_rule_stats: dict[str, Any] = {}
        
        if self.config.use_meta_rule and dip_signals is not None:
            filtered_signals, meta_stats = self._apply_meta_rule(
                symbol, close_prices, dip_signals, regime_history
            )
            filtered_dip_signals = filtered_signals
            meta_rule_stats = meta_stats
        
        # 4. Get fundamentals history for bear mode
        fundamentals_history = None
        if self.config.use_meta_rule:
            fundamentals_history = self.fundamental_service.get_aligned_fundamentals(
                symbol, close_prices.index
            )
        
        # 5. Run portfolio simulation
        simulation = self.portfolio_simulator.simulate_all_scenarios(
            prices=close_prices,
            symbol=symbol,
            dip_signals=filtered_dip_signals,
            technical_signals=technical_signals,
            fundamentals_history=fundamentals_history,
            regime_series=regime_history,
        )
        
        # 6. Run Alpha Gauntlet
        gauntlet_result = None
        if self.config.run_alpha_gauntlet:
            # Get SPY prices if not provided
            spy_close = None
            if spy_prices is not None:
                if "Close" in spy_prices.columns:
                    spy_close = spy_prices["Close"]
                elif "close" in spy_prices.columns:
                    spy_close = spy_prices["close"]
            
            if simulation.scenarios.get(SimulationScenario.DIP_STRATEGY_DCA):
                strategy_result = simulation.scenarios[SimulationScenario.DIP_STRATEGY_DCA]
            elif simulation.scenarios.get(SimulationScenario.TECHNICAL_STRATEGY):
                strategy_result = simulation.scenarios[SimulationScenario.TECHNICAL_STRATEGY]
            else:
                strategy_result = None
            
            if strategy_result:
                gauntlet_result = self.gauntlet.evaluate(
                    strategy_result=strategy_result,
                    buy_hold_result=simulation.scenarios[SimulationScenario.BUY_AND_HOLD],
                    spy_prices=spy_close,
                    regime_history=regime_history,
                )
        
        # 7. Run crash testing
        crash_tests = []
        if self.config.run_crash_testing and simulation:
            # Get trades from best strategy for crash testing
            best_result = simulation.scenarios.get(simulation.best_scenario)
            strategy_trades = []
            if best_result:
                # Convert Trade objects to dicts for crash tester
                strategy_trades = [
                    {
                        "date": t.date,
                        "type": t.type,
                        "price": t.price,
                        "shares": t.shares,
                        "value": t.value,
                    }
                    for t in best_result.trades
                ]
            
            for crash_name, crash_period in CRASH_PERIODS.items():
                # Check if we have data for this crash
                crash_start = pd.Timestamp(crash_period.start_date)
                if crash_start >= close_prices.index.min():
                    try:
                        crash_result = self.crash_tester.test_crash(
                            prices=close_prices,
                            strategy_trades=strategy_trades,
                            crash_period=crash_name,  # Pass the name/key
                        )
                        if crash_result:
                            crash_tests.append(crash_result)
                    except Exception as e:
                        logger.warning(f"Crash test failed for {crash_name}: {e}")
        
        # 8. Calculate regime breakdown
        regime_days: dict[MarketRegime, int] = {}
        regime_returns: dict[MarketRegime, float] = {}
        
        if not regime_history.empty:
            for regime in MarketRegime:
                regime_mask = regime_history == regime.value
                regime_days[regime] = int(regime_mask.sum())
                
                if regime_mask.any():
                    # Calculate return during this regime
                    regime_prices = close_prices[regime_mask]
                    if len(regime_prices) > 1:
                        regime_returns[regime] = (
                            (regime_prices.iloc[-1] / regime_prices.iloc[0] - 1) * 100
                        )
        
        # 9. Collect trade markers for frontend
        trade_markers = []
        if simulation:
            best_result = simulation.scenarios.get(simulation.best_scenario)
            if best_result:
                trade_markers = best_result.to_markers()
        
        return BacktestV2Result(
            symbol=symbol,
            period_start=close_prices.index[0],
            period_end=close_prices.index[-1],
            period_years=len(close_prices) / 252,
            current_regime=current_regime,
            simulation=simulation,
            best_scenario=simulation.best_scenario if simulation else None,
            best_return_pct=simulation.best_return_pct if simulation else 0.0,
            strategy_vs_bh=simulation.strategy_vs_bh if simulation else 0.0,
            dca_vs_lump_sum=simulation.dca_vs_lump_sum if simulation else 0.0,
            dip_vs_regular_dca=simulation.dip_vs_regular_dca if simulation else 0.0,
            gauntlet=gauntlet_result,
            crash_tests=crash_tests,
            regime_days=regime_days,
            regime_returns=regime_returns,
            trade_markers=trade_markers,
            meta_rule_stats=meta_rule_stats,
        )
    
    def _apply_meta_rule(
        self,
        symbol: str,
        prices: pd.Series,
        dip_signals: pd.Series,
        regime_history: pd.Series,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        Apply META Rule to filter dip signals in bear markets.
        
        Returns:
            - Filtered dip signals
            - Statistics about approvals/rejections
        """
        # Create filtered signals (copy)
        filtered = dip_signals.copy()
        
        stats = {
            "total_signals": 0,
            "bull_signals": 0,
            "bear_signals": 0,
            "bear_approved": 0,
            "bear_deep_value": 0,
            "bear_blocked": 0,
            "blocked_reasons": {},
        }
        
        # Find all signal dates
        signal_dates = dip_signals[dip_signals == 1].index
        stats["total_signals"] = len(signal_dates)
        
        for signal_date in signal_dates:
            # Get regime at this date
            if signal_date in regime_history.index:
                regime = regime_history.loc[signal_date]
            else:
                # Find nearest
                idx = regime_history.index.searchsorted(signal_date)
                if idx > 0:
                    regime = regime_history.iloc[idx - 1]
                else:
                    regime = MarketRegime.BULL.value
            
            # Only apply META Rule in bear/crash markets
            if regime in [MarketRegime.BEAR.value, MarketRegime.CRASH.value]:
                stats["bear_signals"] += 1
                
                # Check META Rule
                decision = self.bear_filter.check_buy_signal(
                    symbol=symbol,
                    signal_date=signal_date,
                )
                
                if decision.approved:
                    stats["bear_approved"] += 1
                    if decision.is_deep_value:
                        stats["bear_deep_value"] += 1
                else:
                    # Block this signal
                    filtered.loc[signal_date] = 0
                    stats["bear_blocked"] += 1
                    
                    reason = decision.result.value
                    stats["blocked_reasons"][reason] = (
                        stats["blocked_reasons"].get(reason, 0) + 1
                    )
            else:
                stats["bull_signals"] += 1
        
        return filtered, stats
    
    async def run_quick_backtest(
        self,
        symbol: str,
        prices: pd.DataFrame,
        dip_signals: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Run a quick backtest without full validation (for API responses).
        
        Returns simplified dict for frontend.
        """
        result = await self.run_full_backtest(
            symbol=symbol,
            prices=prices,
            dip_signals=dip_signals,
        )
        
        return {
            "symbol": symbol,
            "period_years": result.period_years,
            "current_regime": result.current_regime.regime.value,
            "strategy_mode": result.current_regime.strategy_mode.value,
            "best_scenario": result.best_scenario.value if result.best_scenario else None,
            "best_return_pct": result.best_return_pct,
            "strategy_vs_bh": result.strategy_vs_bh,
            "trade_count": len(result.trade_markers),
            "meta_rule_stats": result.meta_rule_stats,
        }


# Singleton pattern for service access
_backtest_v2_service: BacktestV2Service | None = None


def get_backtest_v2_service() -> BacktestV2Service:
    """Get or create the BacktestV2Service singleton."""
    global _backtest_v2_service
    if _backtest_v2_service is None:
        _backtest_v2_service = BacktestV2Service()
    return _backtest_v2_service


async def fetch_all_history(
    symbol: str,
    max_years: int = 30,
) -> pd.DataFrame:
    """
    Fetch ALL available price history for a symbol.
    
    Uses yfinance with period="max" to get maximum history.
    
    Args:
        symbol: Stock ticker
        max_years: Maximum years to fetch (default 30)
        
    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf
    
    logger.info(f"Fetching ALL history for {symbol} (up to {max_years} years)")
    
    try:
        # Use period="max" to get all available data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max")
        
        if df.empty:
            # Fallback to date-based fetch
            end_date = date.today()
            start_date = end_date - timedelta(days=max_years * 365)
            df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            logger.warning(f"No price history found for {symbol}")
            return pd.DataFrame()
        
        logger.info(f"Fetched {len(df)} days of history for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {e}")
        return pd.DataFrame()


async def fetch_all_fundamentals(
    symbol: str,
) -> pd.DataFrame:
    """
    Fetch ALL available quarterly fundamentals for a symbol.
    
    Unlike the FundamentalService (which is designed for point-in-time
    lookups during backtesting), this fetches raw quarterly data.
    
    Args:
        symbol: Stock ticker
        
    Returns:
        DataFrame with quarterly fundamentals
    """
    import yfinance as yf
    
    logger.info(f"Fetching ALL fundamentals for {symbol}")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Fetch quarterly statements (yfinance returns last 4-8 quarters)
        income_stmt = ticker.quarterly_income_stmt
        balance_sheet = ticker.quarterly_balance_sheet
        cash_flow = ticker.quarterly_cashflow
        
        # Also try to get annual for longer history
        annual_income = ticker.income_stmt
        annual_balance = ticker.balance_sheet
        annual_cashflow = ticker.cashflow
        
        # Count quarters
        quarterly_count = 0
        if income_stmt is not None:
            quarterly_count = len(income_stmt.columns)
        
        annual_count = 0
        if annual_income is not None:
            annual_count = len(annual_income.columns)
        
        logger.info(
            f"Fetched {quarterly_count} quarters and {annual_count} years "
            f"of fundamentals for {symbol}"
        )
        
        # Return quarterly (more granular for backtesting)
        # The FundamentalService will process this
        return income_stmt if income_stmt is not None else pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return pd.DataFrame()
