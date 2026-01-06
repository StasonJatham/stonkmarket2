"""
Backtest Engine V2 - Regime-Adaptive Strategy System.

This module implements a sophisticated backtesting system with:
- Regime detection (Bull/Bear/Crash/Recovery)
- Adaptive strategy selection based on market conditions
- Fundamental guardrails for bear market accumulation
- Walk-forward optimization with strict validation
- Portfolio simulation with DCA scenarios
- Alpha Gauntlet (hierarchy of truth)
- Crash Testing (2008, 2020, 2022)

Key Philosophy:
- Bear markets are BUYING opportunities for quality assets
- Fundamental checks prevent value traps
- Technical signals are regime-dependent
"""

# Core services (unified V3)
from app.quant_engine.core import (
    MarketRegime,
    StrategyMode,
    StrategyConfig,
    RegimeState,
    RegimeService,
    REGIME_STRATEGY_CONFIGS,
)

from app.quant_engine.backtest.crash_testing import (
    CrashPeriod,
    identify_crash_periods,
)
from app.quant_engine.backtest.fundamental_guardrail import (
    FundamentalGuardrail,
    GuardrailConfig,
    GuardrailResult,
    FundamentalData,
    PointInTimeFundamentals,
)
from app.quant_engine.backtest.schemas import (
    TradeMarkerSchema,
    ScenarioResultSchema,
    BacktestV2Response,
    AccumulationMetricsSchema,
    CrashTestResultSchema,
    FundamentalCheckSchema,
    GuardrailResultSchema,
    RegimeStateSchema,
    WFOResultSchema,
)
from app.quant_engine.backtest.portfolio_simulator import (
    PortfolioConfig,
    PortfolioSimulator,
    PortfolioSnapshot,
    ScenarioResult,
    SimulationResult,
    SimulationScenario,
    Trade,
    calculate_accumulation_metrics,
)
from app.quant_engine.backtest.walk_forward import (
    FoldResult,
    KillSwitchResult,
    WalkForwardOptimizer,
    WFOConfig,
    WFOResult,
    WFOSummary,
)
from app.quant_engine.backtest.alpha_gauntlet import (
    AlphaGauntlet,
    GauntletConfig,
    GauntletResult,
    GauntletTrial,
    GauntletVerdict,
    RegimePerformance,
    TrialResult,
    quick_gauntlet,
)
from app.quant_engine.backtest.crash_testing import (
    AccumulationMetrics,
    CrashDefinition,
    CrashPeriod,
    CrashTester,
    CrashTestResult,
    DrawdownMetrics,
    RecoveryMetrics,
    CRASH_PERIODS,
    get_crash_summary,
)
from app.quant_engine.backtest.fundamental_service import (
    FundamentalService,
    QuarterlyFundamentals,
    MetaRuleResult,
    MetaRuleDecision,
    BearMarketStrategyFilter,
    backtest_with_meta_rule,
)
from app.quant_engine.backtest.service import (
    BacktestV2Service,
    BacktestV2Config,
    BacktestV2Result,
    get_backtest_service,
    fetch_all_history,
    fetch_all_fundamentals,
    MAX_LOOKBACK_DAYS,
)
from app.quant_engine.backtest.alpha_factory import (
    AlphaFactory,
    AlphaFactoryConfig,
    BacktestMetrics,
    ConditionGenome,
    IndicatorMatrix,
    IndicatorType,
    LogicGate,
    OptimizationResult,
    StrategyGenome,
    VectorizedBacktester,
    create_alpha_factory,
    quick_optimize,
)
from app.quant_engine.backtest.strategy_report import (
    AdvancedMetrics,
    BenchmarkComparison,
    EquityCurvePoint,
    OptimizationMeta,
    RegimeBreakdown,
    RegimePerformance as ReportRegimePerformance,
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
from app.quant_engine.backtest.strategy_analyzer import (
    DetailedBacktester,
    StrategyAnalyzer,
    TradeRecord,
    create_strategy_analyzer,
)

__all__ = [
    # Regime - Use RegimeService from core for runtime access
    "MarketRegime",
    "StrategyMode",
    "StrategyConfig",
    "RegimeState",
    "identify_crash_periods",
    # Fundamentals (Guardrail)
    "FundamentalGuardrail",
    "GuardrailConfig",
    "GuardrailResult",
    "FundamentalData",
    "PointInTimeFundamentals",
    # Fundamentals (YFinance Service)
    "FundamentalService",
    "QuarterlyFundamentals",
    "MetaRuleResult",
    "MetaRuleDecision",
    "BearMarketStrategyFilter",
    "backtest_with_meta_rule",
    # Schemas
    "TradeMarkerSchema",
    "ScenarioResultSchema",
    "BacktestV2Response",
    "AccumulationMetricsSchema",
    "CrashTestResultSchema",
    "FundamentalCheckSchema",
    "GuardrailResultSchema",
    "RegimeStateSchema",
    "WFOResultSchema",
    # Portfolio Simulator
    "PortfolioConfig",
    "PortfolioSimulator",
    "PortfolioSnapshot",
    "ScenarioResult",
    "SimulationResult",
    "SimulationScenario",
    "Trade",
    "calculate_accumulation_metrics",
    # Walk-Forward Optimization
    "FoldResult",
    "KillSwitchResult",
    "WalkForwardOptimizer",
    "WFOConfig",
    "WFOResult",
    "WFOSummary",
    # Alpha Gauntlet
    "AlphaGauntlet",
    "GauntletConfig",
    "GauntletResult",
    "GauntletTrial",
    "GauntletVerdict",
    "RegimePerformance",
    "TrialResult",
    "quick_gauntlet",
    # Crash Testing
    "AccumulationMetrics",
    "CrashDefinition",
    "CrashPeriod",
    "CrashTester",
    "CrashTestResult",
    "DrawdownMetrics",
    "RecoveryMetrics",
    "CRASH_PERIODS",
    "get_crash_summary",
    # V2 Service
    "BacktestV2Service",
    "BacktestV2Config",
    "BacktestV2Result",
    "get_backtest_service",
    "fetch_all_history",
    "fetch_all_fundamentals",
    "MAX_LOOKBACK_DAYS",
    # Alpha Factory
    "AlphaFactory",
    "AlphaFactoryConfig",
    "BacktestMetrics",
    "ConditionGenome",
    "IndicatorMatrix",
    "IndicatorType",
    "LogicGate",
    "OptimizationResult",
    "StrategyGenome",
    "VectorizedBacktester",
    "create_alpha_factory",
    "quick_optimize",
    # Strategy Report (Deep Dive)
    "AdvancedMetrics",
    "BenchmarkComparison",
    "EquityCurvePoint",
    "OptimizationMeta",
    "RegimeBreakdown",
    "ReportRegimePerformance",
    "RegimeType",
    "RiskMetrics",
    "RunnerUpStrategy",
    "SignalEvent",
    "SignalType",
    "StrategyConditionSummary",
    "StrategyFullReport",
    "StrategyVerdict",
    "TradeStats",
    "WinningStrategy",
    # Strategy Analyzer
    "DetailedBacktester",
    "StrategyAnalyzer",
    "TradeRecord",
    "create_strategy_analyzer",
]
