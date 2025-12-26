"""
Fundamentals analysis agent.

Analyzes financial statements and company fundamentals.
Pure calculation-based - no LLM required.
"""

import logging

from app.hedge_fund.agents.base import AgentSignal, CalculationAgentBase
from app.hedge_fund.schemas import (
    AgentType,
    Fundamentals,
    MarketData,
    Signal,
)


logger = logging.getLogger(__name__)


class FundamentalsAgent(CalculationAgentBase):
    """
    Analyzes company fundamentals to generate investment signals.
    
    Evaluates:
    - Profitability (ROE, ROA, margins)
    - Financial health (debt ratios, current ratio)
    - Growth (revenue, earnings)
    - Valuation relative to fundamentals
    """

    def __init__(self):
        super().__init__(
            agent_id="fundamentals",
            agent_name="Fundamentals Analyst",
            agent_type=AgentType.FUNDAMENTALS,
        )

    async def calculate(self, symbol: str, data: MarketData) -> AgentSignal:
        """Analyze fundamentals and return signal."""
        f = data.fundamentals

        # Calculate component scores
        profitability_score = self._score_profitability(f)
        health_score = self._score_financial_health(f)
        growth_score = self._score_growth(f)
        efficiency_score = self._score_efficiency(f)

        # Weight the scores
        weights = {
            "profitability": 0.30,
            "health": 0.25,
            "growth": 0.25,
            "efficiency": 0.20,
        }

        total_score = (
            profitability_score * weights["profitability"]
            + health_score * weights["health"]
            + growth_score * weights["growth"]
            + efficiency_score * weights["efficiency"]
        )

        # Map score to signal
        signal, confidence = self._score_to_signal(total_score)

        # Build key factors
        key_factors = self._identify_key_factors(f, {
            "profitability": profitability_score,
            "health": health_score,
            "growth": growth_score,
            "efficiency": efficiency_score,
        })

        # Build reasoning
        reasoning = self._build_reasoning(f, total_score, {
            "profitability": profitability_score,
            "health": health_score,
            "growth": growth_score,
            "efficiency": efficiency_score,
        })

        return self._build_signal(
            symbol=symbol,
            signal=signal.value,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            metrics={
                "total_score": round(total_score, 2),
                "profitability_score": round(profitability_score, 2),
                "health_score": round(health_score, 2),
                "growth_score": round(growth_score, 2),
                "efficiency_score": round(efficiency_score, 2),
                "pe_ratio": f.pe_ratio,
                "roe": f.roe,
                "debt_to_equity": f.debt_to_equity,
                "revenue_growth": f.revenue_growth,
            },
        )

    def _score_profitability(self, f: Fundamentals) -> float:
        """Score profitability metrics (0-100)."""
        scores = []

        # ROE: 15%+ is excellent
        if f.roe is not None:
            if f.roe >= 0.20:
                scores.append(100)
            elif f.roe >= 0.15:
                scores.append(80)
            elif f.roe >= 0.10:
                scores.append(60)
            elif f.roe >= 0.05:
                scores.append(40)
            else:
                scores.append(20)

        # ROA: 10%+ is excellent
        if f.roa is not None:
            if f.roa >= 0.15:
                scores.append(100)
            elif f.roa >= 0.10:
                scores.append(80)
            elif f.roa >= 0.05:
                scores.append(60)
            elif f.roa >= 0.02:
                scores.append(40)
            else:
                scores.append(20)

        # Profit margin
        if f.profit_margin is not None:
            if f.profit_margin >= 0.20:
                scores.append(100)
            elif f.profit_margin >= 0.10:
                scores.append(75)
            elif f.profit_margin >= 0.05:
                scores.append(50)
            elif f.profit_margin > 0:
                scores.append(30)
            else:
                scores.append(10)

        # Operating margin
        if f.operating_margin is not None:
            if f.operating_margin >= 0.25:
                scores.append(100)
            elif f.operating_margin >= 0.15:
                scores.append(75)
            elif f.operating_margin >= 0.10:
                scores.append(55)
            elif f.operating_margin > 0:
                scores.append(35)
            else:
                scores.append(15)

        return sum(scores) / len(scores) if scores else 50

    def _score_financial_health(self, f: Fundamentals) -> float:
        """Score financial health metrics (0-100)."""
        scores = []

        # Current ratio: 1.5-2.5 is ideal
        if f.current_ratio is not None:
            if 1.5 <= f.current_ratio <= 2.5:
                scores.append(100)
            elif 1.2 <= f.current_ratio <= 3.0:
                scores.append(75)
            elif 1.0 <= f.current_ratio <= 4.0:
                scores.append(50)
            else:
                scores.append(25)

        # Debt to equity: lower is better
        if f.debt_to_equity is not None:
            if f.debt_to_equity <= 0.3:
                scores.append(100)
            elif f.debt_to_equity <= 0.5:
                scores.append(85)
            elif f.debt_to_equity <= 1.0:
                scores.append(65)
            elif f.debt_to_equity <= 2.0:
                scores.append(40)
            else:
                scores.append(20)

        # Free cash flow positive
        if f.free_cash_flow is not None:
            if f.free_cash_flow > 0:
                if f.market_cap and f.free_cash_flow / f.market_cap > 0.05:
                    scores.append(100)
                else:
                    scores.append(70)
            else:
                scores.append(20)

        return sum(scores) / len(scores) if scores else 50

    def _score_growth(self, f: Fundamentals) -> float:
        """Score growth metrics (0-100)."""
        scores = []

        # Revenue growth
        if f.revenue_growth is not None:
            if f.revenue_growth >= 0.25:
                scores.append(100)
            elif f.revenue_growth >= 0.15:
                scores.append(80)
            elif f.revenue_growth >= 0.10:
                scores.append(65)
            elif f.revenue_growth >= 0.05:
                scores.append(50)
            elif f.revenue_growth >= 0:
                scores.append(35)
            else:
                scores.append(15)

        # Earnings growth
        if f.earnings_growth is not None:
            if f.earnings_growth >= 0.30:
                scores.append(100)
            elif f.earnings_growth >= 0.20:
                scores.append(80)
            elif f.earnings_growth >= 0.10:
                scores.append(60)
            elif f.earnings_growth >= 0:
                scores.append(40)
            else:
                scores.append(20)

        return sum(scores) / len(scores) if scores else 50

    def _score_efficiency(self, f: Fundamentals) -> float:
        """Score efficiency metrics (0-100)."""
        scores = []

        # Gross margin
        if f.gross_margin is not None:
            if f.gross_margin >= 0.50:
                scores.append(100)
            elif f.gross_margin >= 0.35:
                scores.append(75)
            elif f.gross_margin >= 0.25:
                scores.append(55)
            elif f.gross_margin >= 0.15:
                scores.append(35)
            else:
                scores.append(20)

        return sum(scores) / len(scores) if scores else 50

    def _score_to_signal(self, score: float) -> tuple[Signal, float]:
        """Convert score to signal and confidence."""
        if score >= 80:
            return Signal.STRONG_BUY, 0.9
        elif score >= 65:
            return Signal.BUY, 0.75
        elif score >= 45:
            return Signal.HOLD, 0.6
        elif score >= 30:
            return Signal.SELL, 0.7
        else:
            return Signal.STRONG_SELL, 0.85

    def _identify_key_factors(
        self,
        f: Fundamentals,
        scores: dict[str, float],
    ) -> list[str]:
        """Identify top factors driving the signal."""
        factors = []

        # Profitability factors
        if f.roe is not None:
            if f.roe >= 0.15:
                factors.append(f"Strong ROE of {f.roe:.1%}")
            elif f.roe < 0.05:
                factors.append(f"Weak ROE of {f.roe:.1%}")

        # Health factors
        if f.debt_to_equity is not None:
            if f.debt_to_equity <= 0.5:
                factors.append(f"Low debt-to-equity of {f.debt_to_equity:.2f}")
            elif f.debt_to_equity > 2.0:
                factors.append(f"High debt-to-equity of {f.debt_to_equity:.2f}")

        # Growth factors
        if f.revenue_growth is not None:
            if f.revenue_growth >= 0.15:
                factors.append(f"Strong revenue growth of {f.revenue_growth:.1%}")
            elif f.revenue_growth < 0:
                factors.append(f"Declining revenue of {f.revenue_growth:.1%}")

        # Free cash flow
        if f.free_cash_flow is not None:
            if f.free_cash_flow > 0:
                factors.append("Positive free cash flow")
            else:
                factors.append("Negative free cash flow")

        return factors[:5]

    def _build_reasoning(
        self,
        f: Fundamentals,
        total_score: float,
        scores: dict[str, float],
    ) -> str:
        """Build human-readable reasoning."""
        parts = [f"Overall fundamental score: {total_score:.0f}/100."]

        # Component breakdown
        parts.append(
            f"Profitability: {scores['profitability']:.0f}, "
            f"Financial Health: {scores['health']:.0f}, "
            f"Growth: {scores['growth']:.0f}, "
            f"Efficiency: {scores['efficiency']:.0f}."
        )

        # Key metrics
        if f.pe_ratio:
            parts.append(f"P/E ratio of {f.pe_ratio:.1f}.")
        if f.roe:
            parts.append(f"Return on equity of {f.roe:.1%}.")
        if f.revenue_growth:
            parts.append(f"Revenue growth of {f.revenue_growth:.1%}.")

        return " ".join(parts)


# Singleton instance
_fundamentals_agent: FundamentalsAgent | None = None


def get_fundamentals_agent() -> FundamentalsAgent:
    """Get singleton fundamentals agent."""
    global _fundamentals_agent
    if _fundamentals_agent is None:
        _fundamentals_agent = FundamentalsAgent()
    return _fundamentals_agent
