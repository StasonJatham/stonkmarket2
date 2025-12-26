"""
External Agent Adapters.

Adapters to integrate existing internal agents into the hedge fund module
without modifying their public API.
"""

import logging
from typing import Optional

from app.hedge_fund.agents.base import AgentBase, AgentSignal
from app.hedge_fund.schemas import AgentType, LLMMode, MarketData, Signal


logger = logging.getLogger(__name__)


# =============================================================================
# Base Adapter
# =============================================================================


class ExternalAgentAdapter(AgentBase):
    """
    Base adapter for wrapping external agents.
    
    Converts between the hedge fund module's interface and external agent APIs.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        external_agent_id: str,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=AgentType.EXTERNAL,
            requires_llm=True,  # Most external agents use LLM
        )
        self.external_agent_id = external_agent_id


# =============================================================================
# AI Agents Service Adapter
# =============================================================================


class AIAgentsServiceAdapter(ExternalAgentAdapter):
    """
    Adapter for the existing ai_agents.py service.
    
    Maps the existing AgentVerdict to our AgentSignal format.
    """

    # Map existing signal types to our Signal enum
    SIGNAL_MAP = {
        "bullish": Signal.BUY,
        "bearish": Signal.SELL,
        "neutral": Signal.HOLD,
        "strong_buy": Signal.STRONG_BUY,
        "strong_sell": Signal.STRONG_SELL,
    }

    def __init__(self, external_agent_id: str, agent_name: str):
        super().__init__(
            agent_id=f"external_{external_agent_id}",
            agent_name=f"{agent_name} (Legacy)",
            external_agent_id=external_agent_id,
        )

    async def run(
        self,
        symbol: str,
        data: MarketData,
        *,
        mode: LLMMode = LLMMode.REALTIME,
        run_id: str | None = None,
    ) -> AgentSignal:
        """Run the external agent and convert result to AgentSignal."""
        from app.services.ai_agents import (
            AGENTS,
            _format_metrics_for_prompt,
            _run_single_agent,
        )

        # Get agent config
        agent_config = AGENTS.get(self.external_agent_id)
        if not agent_config:
            logger.error(f"Unknown external agent: {self.external_agent_id}")
            return self._build_signal(
                symbol=symbol,
                signal=Signal.HOLD.value,
                confidence=0.1,
                reasoning=f"External agent {self.external_agent_id} not found",
                key_factors=["Agent not found"],
            )

        # Get fundamentals in expected format
        fundamentals = self._convert_fundamentals(data.fundamentals)
        stock_data = self._convert_stock_data(data)

        # Format metrics text for the prompt
        metrics_text = _format_metrics_for_prompt(fundamentals, stock_data)

        try:
            # Call the existing agent with correct signature
            verdict = await _run_single_agent(
                agent_id=self.external_agent_id,
                symbol=symbol,
                metrics_text=metrics_text,
            )

            if verdict:
                # Convert verdict to AgentSignal
                signal = self.SIGNAL_MAP.get(verdict.signal, Signal.HOLD)
                confidence = verdict.confidence / 100.0  # Convert 0-100 to 0-1

                return self._build_signal(
                    symbol=symbol,
                    signal=signal.value,
                    confidence=confidence,
                    reasoning=verdict.reasoning,
                    key_factors=verdict.key_factors,
                    metrics={"source": "legacy_ai_agents"},
                )
            else:
                return self._build_signal(
                    symbol=symbol,
                    signal=Signal.HOLD.value,
                    confidence=0.3,
                    reasoning="External agent returned no verdict",
                    key_factors=["No verdict"],
                )

        except Exception as e:
            logger.error(f"External agent {self.external_agent_id} failed: {e}")
            return self._build_signal(
                symbol=symbol,
                signal=Signal.HOLD.value,
                confidence=0.1,
                reasoning=f"External agent failed: {e!s}",
                key_factors=["Error"],
            )

    def _convert_fundamentals(self, f) -> dict:
        """Convert our Fundamentals schema to the dict format expected by legacy agents."""
        return {
            "pe_ratio": f.pe_ratio,
            "forward_pe": f.forward_pe,
            "peg_ratio": f.peg_ratio,
            "price_to_book": f.price_to_book,
            "ev_to_ebitda": f.ev_to_ebitda,
            "profit_margin": f.profit_margin,
            "operating_margin": f.operating_margin,
            "return_on_equity": f.roe,
            "return_on_assets": f.roa,
            "debt_to_equity": f.debt_to_equity,
            "current_ratio": f.current_ratio,
            "free_cash_flow": f.free_cash_flow,
            "revenue_growth": f.revenue_growth,
            "earnings_growth": f.earnings_growth,
            "dividend_yield": f.dividend_yield,
        }

    def _convert_stock_data(self, data: MarketData) -> dict:
        """Convert MarketData to stock_data dict format."""
        f = data.fundamentals
        prices = data.prices

        current_price = prices.latest.close if prices.latest else None

        return {
            "name": f.name,
            "sector": f.sector,
            "industry": f.industry,
            "current_price": current_price,
            "market_cap": f.market_cap,
            "beta": f.beta,
        }


# =============================================================================
# Fundamentals Service Adapter (dipfinder)
# =============================================================================


class FundamentalsServiceAdapter(ExternalAgentAdapter):
    """
    Adapter for the dipfinder fundamentals.py quality scoring.
    
    Converts QualityMetrics to AgentSignal format.
    """

    def __init__(self):
        super().__init__(
            agent_id="external_fundamentals_quality",
            agent_name="Quality Score (Legacy)",
            external_agent_id="fundamentals_quality",
        )
        self._requires_llm = False  # Pure calculation

    async def run(
        self,
        symbol: str,
        data: MarketData,
        *,
        mode: LLMMode = LLMMode.REALTIME,
        run_id: str | None = None,
    ) -> AgentSignal:
        """Get quality score from fundamentals service."""
        from app.dipfinder.fundamentals import calculate_quality_metrics

        try:
            # Use raw_info from our data
            info = data.fundamentals.raw_info or {}

            # Calculate quality metrics
            metrics = calculate_quality_metrics(info, symbol)

            if metrics:
                # Convert composite score (0-100) to signal
                score = metrics.composite_score

                if score >= 80:
                    signal = Signal.STRONG_BUY
                    confidence = 0.85
                elif score >= 65:
                    signal = Signal.BUY
                    confidence = 0.7
                elif score >= 45:
                    signal = Signal.HOLD
                    confidence = 0.5
                elif score >= 30:
                    signal = Signal.SELL
                    confidence = 0.65
                else:
                    signal = Signal.STRONG_SELL
                    confidence = 0.8

                # Build key factors from component scores
                key_factors = []
                if hasattr(metrics, 'profitability_score'):
                    key_factors.append(f"Profitability: {metrics.profitability_score:.0f}/100")
                if hasattr(metrics, 'growth_score'):
                    key_factors.append(f"Growth: {metrics.growth_score:.0f}/100")
                if hasattr(metrics, 'financial_health_score'):
                    key_factors.append(f"Health: {metrics.financial_health_score:.0f}/100")
                if hasattr(metrics, 'valuation_score'):
                    key_factors.append(f"Valuation: {metrics.valuation_score:.0f}/100")

                return self._build_signal(
                    symbol=symbol,
                    signal=signal.value,
                    confidence=confidence,
                    reasoning=f"Quality composite score of {score:.0f}/100 based on profitability, growth, financial health, and valuation metrics.",
                    key_factors=key_factors,
                    metrics={
                        "composite_score": score,
                        "source": "dipfinder_fundamentals",
                    },
                )
            else:
                return self._build_signal(
                    symbol=symbol,
                    signal=Signal.HOLD.value,
                    confidence=0.3,
                    reasoning="Could not calculate quality metrics",
                    key_factors=["Insufficient data"],
                )

        except Exception as e:
            logger.error(f"Fundamentals service adapter failed: {e}")
            return self._build_signal(
                symbol=symbol,
                signal=Signal.HOLD.value,
                confidence=0.1,
                reasoning=f"Quality calculation failed: {e!s}",
                key_factors=["Error"],
            )


# =============================================================================
# OpenAI Rating Adapter
# =============================================================================


class OpenAIRatingAdapter(ExternalAgentAdapter):
    """
    Adapter for the existing openai_client rating functionality.
    
    Uses the RATING task type for structured investment opinions.
    """

    def __init__(self):
        super().__init__(
            agent_id="external_openai_rating",
            agent_name="AI Rating (Legacy)",
            external_agent_id="openai_rating",
        )

    # Rating to signal mapping
    RATING_MAP = {
        "strong_buy": Signal.STRONG_BUY,
        "buy": Signal.BUY,
        "hold": Signal.HOLD,
        "sell": Signal.SELL,
        "strong_sell": Signal.STRONG_SELL,
    }

    async def run(
        self,
        symbol: str,
        data: MarketData,
        *,
        mode: LLMMode = LLMMode.REALTIME,
        run_id: str | None = None,
    ) -> AgentSignal:
        """Get rating from OpenAI client."""
        from app.services.openai_client import TaskType, generate

        try:
            # Build context for rating
            context = {
                "symbol": symbol,
                "name": data.fundamentals.name,
                "sector": data.fundamentals.sector,
                "pe_ratio": data.fundamentals.pe_ratio,
                "roe": data.fundamentals.roe,
                "debt_to_equity": data.fundamentals.debt_to_equity,
                "revenue_growth": data.fundamentals.revenue_growth,
                "market_cap": data.fundamentals.market_cap,
            }

            result = await generate(
                task=TaskType.RATING,
                context=context,
            )

            if result and not result.get("error"):
                rating = result.get("rating", "hold")
                signal = self.RATING_MAP.get(rating, Signal.HOLD)
                confidence = result.get("confidence", 5) / 10.0
                reasoning = result.get("reasoning", "No reasoning provided")

                return self._build_signal(
                    symbol=symbol,
                    signal=signal.value,
                    confidence=confidence,
                    reasoning=reasoning,
                    key_factors=[f"AI Rating: {rating}"],
                    metrics={"source": "openai_rating"},
                )
            else:
                return self._build_signal(
                    symbol=symbol,
                    signal=Signal.HOLD.value,
                    confidence=0.3,
                    reasoning=result.get("error", "Rating generation failed") if result else "No response",
                    key_factors=["Rating failed"],
                )

        except Exception as e:
            logger.error(f"OpenAI rating adapter failed: {e}")
            return self._build_signal(
                symbol=symbol,
                signal=Signal.HOLD.value,
                confidence=0.1,
                reasoning=f"Rating failed: {e!s}",
                key_factors=["Error"],
            )


# =============================================================================
# Factory Functions
# =============================================================================


def get_legacy_persona_adapters() -> list[AIAgentsServiceAdapter]:
    """Get adapters for all legacy AI agents personas."""
    from app.services.ai_agents import AGENTS

    return [
        AIAgentsServiceAdapter(agent_id, config["name"])
        for agent_id, config in AGENTS.items()
    ]


def get_all_external_adapters() -> list[ExternalAgentAdapter]:
    """Get all available external adapters."""
    adapters = []

    # Legacy AI agents personas
    adapters.extend(get_legacy_persona_adapters())

    # Fundamentals quality adapter
    adapters.append(FundamentalsServiceAdapter())

    # OpenAI rating adapter
    adapters.append(OpenAIRatingAdapter())

    return adapters
