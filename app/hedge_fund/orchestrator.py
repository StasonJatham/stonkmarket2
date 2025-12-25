"""
Hedge Fund Orchestrator.

Main entry point for running multi-agent investment analysis.
Coordinates data fetching, agent execution, and result aggregation.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional

from app.hedge_fund.agents.base import AgentBase
from app.hedge_fund.agents.fundamentals import get_fundamentals_agent
from app.hedge_fund.agents.investor_persona import (
    get_all_persona_agents,
    get_persona_agent,
)
from app.hedge_fund.agents.portfolio_manager import get_portfolio_manager
from app.hedge_fund.agents.risk import get_risk_agent
from app.hedge_fund.agents.sentiment import get_sentiment_agent
from app.hedge_fund.agents.technicals import get_technicals_agent
from app.hedge_fund.agents.valuation import get_valuation_agent
from app.hedge_fund.data.yfinance_service import get_market_data, get_market_data_batch
from app.hedge_fund.llm.gateway import OpenAIGateway, get_gateway
from app.hedge_fund.schemas import (
    AgentSignal,
    AnalysisBundle,
    AnalysisRequest,
    LLMMode,
    MarketData,
    PerTickerReport,
    PortfolioDecision,
    Signal,
    TickerInput,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Default Agent Registry
# =============================================================================


def get_calculation_agents() -> list[AgentBase]:
    """Get all calculation-based agents (no LLM required)."""
    return [
        get_fundamentals_agent(),
        get_technicals_agent(),
        get_valuation_agent(),
        get_sentiment_agent(),
        get_risk_agent(),
    ]


def get_persona_agents(
    personas: Optional[list[str]] = None,
    gateway: Optional[OpenAIGateway] = None,
) -> list[AgentBase]:
    """Get investor persona agents."""
    if personas:
        agents = []
        for persona_id in personas:
            agent = get_persona_agent(persona_id, gateway)
            if agent:
                agents.append(agent)
        return agents
    return get_all_persona_agents(gateway)


def get_all_agents(
    include_personas: bool = True,
    personas: Optional[list[str]] = None,
    gateway: Optional[OpenAIGateway] = None,
) -> list[AgentBase]:
    """Get all available agents."""
    agents = get_calculation_agents()
    if include_personas:
        agents.extend(get_persona_agents(personas, gateway))
    return agents


# =============================================================================
# Orchestrator Class
# =============================================================================


class Orchestrator:
    """
    Coordinates multi-agent investment analysis.
    
    Features:
    - Parallel data fetching for multiple tickers
    - Parallel agent execution (calculation agents)
    - Batch or realtime LLM calls for persona agents
    - Aggregation via portfolio manager
    - Error handling and partial results
    """

    def __init__(
        self,
        gateway: Optional[OpenAIGateway] = None,
        include_personas: bool = True,
        include_external: bool = False,
        max_concurrent_agents: int = 10,
    ):
        self.gateway = gateway or get_gateway()
        self.include_personas = include_personas
        self.include_external = include_external
        self.max_concurrent_agents = max_concurrent_agents
        self.portfolio_manager = get_portfolio_manager()

    async def run_analysis(
        self,
        request: AnalysisRequest,
    ) -> AnalysisBundle:
        """
        Run complete analysis for all tickers in the request.
        
        This is the main entry point.
        """
        start_time = time.monotonic()
        run_id = request.run_id or uuid.uuid4().hex[:12]
        
        symbols = [t.symbol for t in request.tickers]
        logger.info(f"Starting analysis {run_id} for {len(symbols)} tickers: {symbols}")
        
        errors: list[str] = []
        total_agents = 0
        successful_agents = 0
        failed_agents = 0
        
        # 1. Fetch market data for all tickers
        logger.info(f"Fetching market data for {len(symbols)} symbols...")
        market_data = await get_market_data_batch(symbols)
        
        # 2. Get agents
        agents = get_calculation_agents()
        if self.include_personas:
            agents.extend(get_persona_agents(request.personas, self.gateway))
        
        if self.include_external:
            from app.hedge_fund.agents.external import get_all_external_adapters
            agents.extend(get_all_external_adapters())
        
        total_agents = len(agents) * len(symbols)
        
        # 3. Run analysis for each ticker
        reports: list[PerTickerReport] = []
        portfolio_decisions: list[PortfolioDecision] = []
        
        for symbol in symbols:
            data = market_data.get(symbol)
            if not data:
                errors.append(f"No market data for {symbol}")
                continue
            
            # Run all agents for this ticker
            signals, agent_errors = await self._run_agents_for_ticker(
                symbol=symbol,
                data=data,
                agents=agents,
                mode=request.mode,
                run_id=run_id,
            )
            
            successful_agents += len(signals)
            failed_agents += len(agent_errors)
            errors.extend(agent_errors)
            
            # Aggregate signals
            report = self.portfolio_manager.aggregate_signals(signals)
            reports.append(report)
            
            # Get risk score for portfolio decision
            risk_signal = next(
                (s for s in signals if s.agent_id == "risk"),
                None
            )
            risk_score = 0.5
            if risk_signal and risk_signal.metrics:
                risk_score = risk_signal.metrics.get("overall_risk_score", 0.5)
            
            # Create portfolio decision
            decision = self.portfolio_manager.create_portfolio_decision(
                report=report,
                risk_score=risk_score,
            )
            portfolio_decisions.append(decision)
        
        execution_time = time.monotonic() - start_time
        
        logger.info(
            f"Analysis {run_id} complete: {successful_agents}/{total_agents} agents succeeded "
            f"in {execution_time:.2f}s"
        )
        
        return AnalysisBundle(
            run_id=run_id,
            mode=request.mode,
            tickers=symbols,
            reports=reports,
            portfolio_decisions=portfolio_decisions,
            total_agents_run=total_agents,
            successful_agents=successful_agents,
            failed_agents=failed_agents,
            execution_time_seconds=execution_time,
            errors=errors,
        )

    async def _run_agents_for_ticker(
        self,
        symbol: str,
        data: MarketData,
        agents: list[AgentBase],
        mode: LLMMode,
        run_id: str,
    ) -> tuple[list[AgentSignal], list[str]]:
        """Run all agents for a single ticker."""
        signals: list[AgentSignal] = []
        errors: list[str] = []
        
        # Separate calculation and LLM agents
        calc_agents = [a for a in agents if not a.requires_llm]
        llm_agents = [a for a in agents if a.requires_llm]
        
        # Run calculation agents concurrently
        if calc_agents:
            calc_tasks = [
                self._run_agent_safe(agent, symbol, data, mode, run_id)
                for agent in calc_agents
            ]
            calc_results = await asyncio.gather(*calc_tasks)
            
            for result in calc_results:
                if isinstance(result, AgentSignal):
                    signals.append(result)
                else:
                    errors.append(result)
        
        # Run LLM agents
        if llm_agents:
            if mode == LLMMode.REALTIME:
                # Run LLM agents concurrently with semaphore
                sem = asyncio.Semaphore(self.max_concurrent_agents)
                
                async def run_with_sem(agent):
                    async with sem:
                        return await self._run_agent_safe(agent, symbol, data, mode, run_id)
                
                llm_tasks = [run_with_sem(agent) for agent in llm_agents]
                llm_results = await asyncio.gather(*llm_tasks)
                
                for result in llm_results:
                    if isinstance(result, AgentSignal):
                        signals.append(result)
                    else:
                        errors.append(result)
            else:
                # Batch mode - run one by one (batch handles parallelism)
                for agent in llm_agents:
                    result = await self._run_agent_safe(agent, symbol, data, mode, run_id)
                    if isinstance(result, AgentSignal):
                        signals.append(result)
                    else:
                        errors.append(result)
        
        return signals, errors

    async def _run_agent_safe(
        self,
        agent: AgentBase,
        symbol: str,
        data: MarketData,
        mode: LLMMode,
        run_id: str,
    ) -> AgentSignal | str:
        """Run an agent with error handling."""
        try:
            signal = await agent.run(symbol, data, mode=mode, run_id=run_id)
            return signal
        except Exception as e:
            error_msg = f"Agent {agent.agent_id} failed for {symbol}: {str(e)}"
            logger.error(error_msg)
            return error_msg


# =============================================================================
# Convenience Functions
# =============================================================================


async def run_analysis(
    symbols: list[str],
    *,
    mode: LLMMode = LLMMode.REALTIME,
    personas: Optional[list[str]] = None,
    include_external: bool = False,
    run_id: Optional[str] = None,
) -> AnalysisBundle:
    """
    Run investment analysis for a list of symbols.
    
    This is the primary entry point for the hedge fund module.
    
    Args:
        symbols: List of ticker symbols to analyze
        mode: LLM execution mode (realtime or batch)
        personas: Optional list of persona IDs to use (None = all)
        include_external: Whether to include external/legacy adapters
        run_id: Optional run ID for tracking
    
    Returns:
        AnalysisBundle with all results
    
    Example:
        >>> result = await run_analysis(["AAPL", "MSFT", "GOOGL"])
        >>> for report in result.reports:
        ...     print(f"{report.symbol}: {report.consensus_signal.value}")
    """
    request = AnalysisRequest(
        tickers=[TickerInput(symbol=s) for s in symbols],
        run_id=run_id,
        mode=mode,
        personas=personas,
    )
    
    orchestrator = Orchestrator(include_external=include_external)
    return await orchestrator.run_analysis(request)


async def run_single_analysis(
    symbol: str,
    *,
    mode: LLMMode = LLMMode.REALTIME,
    personas: Optional[list[str]] = None,
) -> PerTickerReport:
    """
    Convenience function to analyze a single symbol.
    
    Returns the PerTickerReport directly.
    """
    result = await run_analysis([symbol], mode=mode, personas=personas)
    if result.reports:
        return result.reports[0]
    
    # Return empty report on failure
    return PerTickerReport(
        symbol=symbol,
        signals=[],
        consensus_signal=Signal.HOLD,
        consensus_confidence=0.0,
        summary="Analysis failed",
    )


async def get_quick_signal(symbol: str) -> tuple[Signal, float]:
    """
    Get a quick signal for a symbol using only calculation agents.
    
    Faster than full analysis - no LLM calls.
    Returns (signal, confidence) tuple.
    """
    # Fetch data
    data = await get_market_data(symbol)
    
    # Run only calculation agents
    agents = get_calculation_agents()
    signals: list[AgentSignal] = []
    
    for agent in agents:
        try:
            signal = await agent.run(symbol, data)
            signals.append(signal)
        except Exception as e:
            logger.warning(f"Agent {agent.agent_id} failed: {e}")
    
    if not signals:
        return Signal.HOLD, 0.0
    
    # Aggregate
    pm = get_portfolio_manager()
    report = pm.aggregate_signals(signals)
    
    return report.consensus_signal, report.consensus_confidence


# =============================================================================
# Module Initialization
# =============================================================================


# Create init file for the module
__all__ = [
    "run_analysis",
    "run_single_analysis",
    "get_quick_signal",
    "Orchestrator",
    "get_all_agents",
    "get_calculation_agents",
    "get_persona_agents",
]
