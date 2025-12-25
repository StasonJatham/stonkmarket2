"""
Hedge Fund Analysis Module.

A comprehensive multi-agent investment analysis system using:
- Calculation-based agents (fundamentals, technicals, valuation, sentiment, risk)
- LLM-powered investor persona agents (Buffett, Lynch, Wood, etc.)
- Portfolio aggregation and decision making
- yfinance data integration
- OpenAI batch and realtime APIs

Usage:
    from app.hedge_fund import run_analysis
    
    # Analyze multiple stocks
    result = await run_analysis(["AAPL", "MSFT", "GOOGL"])
    
    for report in result.reports:
        print(f"{report.symbol}: {report.consensus_signal.value}")
    
    # Quick signal (no LLM, faster)
    from app.hedge_fund import get_quick_signal
    signal, confidence = await get_quick_signal("AAPL")
"""

from app.hedge_fund.orchestrator import (
    Orchestrator,
    get_all_agents,
    get_calculation_agents,
    get_persona_agents,
    get_quick_signal,
    run_analysis,
    run_single_analysis,
)
from app.hedge_fund.schemas import (
    AgentSignal,
    AgentType,
    AnalysisBundle,
    AnalysisRequest,
    Fundamentals,
    InvestorPersona,
    LLMMode,
    MarketData,
    PerTickerReport,
    PortfolioDecision,
    PricePoint,
    PriceSeries,
    RiskMetrics,
    SentimentMetrics,
    Signal,
    TechnicalIndicators,
    TickerInput,
    ValuationMetrics,
)

__all__ = [
    # Main entry points
    "run_analysis",
    "run_single_analysis",
    "get_quick_signal",
    "Orchestrator",
    # Agent access
    "get_all_agents",
    "get_calculation_agents",
    "get_persona_agents",
    # Schemas
    "AgentSignal",
    "AgentType",
    "AnalysisBundle",
    "AnalysisRequest",
    "Fundamentals",
    "InvestorPersona",
    "LLMMode",
    "MarketData",
    "PerTickerReport",
    "PortfolioDecision",
    "PricePoint",
    "PriceSeries",
    "RiskMetrics",
    "SentimentMetrics",
    "Signal",
    "TechnicalIndicators",
    "TickerInput",
    "ValuationMetrics",
]
