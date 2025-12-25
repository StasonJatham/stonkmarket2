"""
Valuation analysis agent.

Calculates intrinsic value using multiple methods.
Pure calculation-based - no LLM required.
"""

import logging
import math
from typing import Optional

from app.hedge_fund.agents.base import AgentSignal, CalculationAgentBase
from app.hedge_fund.schemas import (
    AgentType,
    Fundamentals,
    LLMMode,
    MarketData,
    Signal,
    ValuationMetrics,
)

logger = logging.getLogger(__name__)


# Default assumptions for DCF
DCF_DEFAULTS = {
    "discount_rate": 0.10,  # 10% required return
    "terminal_growth": 0.03,  # 3% terminal growth
    "projection_years": 5,
    "margin_of_safety": 0.25,  # 25% margin of safety
}


class ValuationAgent(CalculationAgentBase):
    """
    Analyzes valuation to generate investment signals.
    
    Methods:
    - DCF (Discounted Cash Flow)
    - Relative valuation (P/E, P/B, EV/EBITDA)
    - Owner earnings (Buffett method)
    - PEG ratio analysis
    """

    def __init__(
        self,
        discount_rate: float = DCF_DEFAULTS["discount_rate"],
        terminal_growth: float = DCF_DEFAULTS["terminal_growth"],
        margin_of_safety: float = DCF_DEFAULTS["margin_of_safety"],
    ):
        super().__init__(
            agent_id="valuation",
            agent_name="Valuation Analyst",
            agent_type=AgentType.VALUATION,
        )
        self.discount_rate = discount_rate
        self.terminal_growth = terminal_growth
        self.margin_of_safety = margin_of_safety

    async def calculate(self, symbol: str, data: MarketData) -> AgentSignal:
        """Run valuation analysis and return signal."""
        f = data.fundamentals
        current_price = data.prices.latest.close if data.prices.latest else None
        
        if not current_price:
            return self._build_signal(
                symbol=symbol,
                signal=Signal.HOLD.value,
                confidence=0.3,
                reasoning="No current price available for valuation.",
                key_factors=["Missing price data"],
            )
        
        # Calculate valuation metrics
        metrics = self._calculate_valuation_metrics(f, current_price)
        
        # Score each valuation method
        dcf_score = self._score_dcf(metrics)
        relative_score = self._score_relative_valuation(f)
        peg_score = self._score_peg(f)
        owner_earnings_score = self._score_owner_earnings(metrics)
        
        # Weight by reliability
        weights = {
            "dcf": 0.25 if metrics.dcf_value else 0,
            "relative": 0.35,
            "peg": 0.20 if f.peg_ratio else 0,
            "owner_earnings": 0.20 if metrics.owner_earnings else 0,
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = 1  # Fallback
        
        total_score = (
            dcf_score * weights["dcf"]
            + relative_score * weights["relative"]
            + peg_score * weights["peg"]
            + owner_earnings_score * weights["owner_earnings"]
        ) / total_weight * (total_weight / 1.0)  # Scale appropriately
        
        # Adjust based on margin of safety
        if metrics.margin_of_safety:
            if metrics.margin_of_safety > 0.30:
                total_score *= 1.15  # Boost for large margin
            elif metrics.margin_of_safety > 0.15:
                total_score *= 1.05
            elif metrics.margin_of_safety < -0.15:
                total_score *= 0.85  # Penalty for overvaluation
        
        total_score = min(100, max(0, total_score))
        
        # Map to signal
        signal, confidence = self._score_to_signal(total_score, metrics)
        
        # Key factors
        key_factors = self._identify_key_factors(f, metrics)
        
        # Reasoning
        reasoning = self._build_reasoning(f, metrics, total_score)
        
        return self._build_signal(
            symbol=symbol,
            signal=signal.value,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            metrics={
                "total_score": round(total_score, 2),
                "dcf_value": metrics.dcf_value,
                "intrinsic_value": metrics.intrinsic_value,
                "current_price": current_price,
                "margin_of_safety": metrics.margin_of_safety,
                "upside_potential": metrics.upside_potential,
                "pe_ratio": f.pe_ratio,
                "peg_ratio": f.peg_ratio,
                "price_to_book": f.price_to_book,
                "ev_to_ebitda": f.ev_to_ebitda,
            },
        )

    def _calculate_valuation_metrics(
        self,
        f: Fundamentals,
        current_price: float,
    ) -> ValuationMetrics:
        """Calculate comprehensive valuation metrics."""
        # DCF calculation
        dcf_value = self._calculate_dcf(f)
        
        # Owner earnings (Buffett)
        owner_earnings = self._calculate_owner_earnings(f)
        owner_earnings_yield = None
        if owner_earnings and f.market_cap:
            owner_earnings_yield = owner_earnings / f.market_cap
        
        # Intrinsic value (average of methods)
        intrinsic_values = [v for v in [dcf_value] if v and v > 0]
        intrinsic_value = sum(intrinsic_values) / len(intrinsic_values) if intrinsic_values else None
        
        # Margin of safety
        margin_of_safety = None
        upside_potential = None
        if intrinsic_value and current_price:
            margin_of_safety = (intrinsic_value - current_price) / intrinsic_value
            upside_potential = (intrinsic_value - current_price) / current_price
        
        # PEG assessment
        peg_assessment = None
        if f.peg_ratio:
            if f.peg_ratio < 1:
                peg_assessment = "Undervalued (PEG < 1)"
            elif f.peg_ratio < 1.5:
                peg_assessment = "Fairly valued"
            elif f.peg_ratio < 2:
                peg_assessment = "Slightly overvalued"
            else:
                peg_assessment = "Overvalued (PEG > 2)"
        
        # Valuation grade
        valuation_grade = self._calculate_valuation_grade(
            f, margin_of_safety, owner_earnings_yield
        )
        
        return ValuationMetrics(
            symbol=f.symbol,
            dcf_value=dcf_value,
            current_price=current_price,
            intrinsic_value=intrinsic_value,
            margin_of_safety=margin_of_safety,
            peg_assessment=peg_assessment,
            growth_rate_used=f.earnings_growth or f.revenue_growth,
            owner_earnings=owner_earnings,
            owner_earnings_yield=owner_earnings_yield,
            valuation_grade=valuation_grade,
            is_undervalued=margin_of_safety > self.margin_of_safety if margin_of_safety else None,
            upside_potential=upside_potential,
        )

    def _calculate_dcf(self, f: Fundamentals) -> Optional[float]:
        """Calculate DCF intrinsic value per share."""
        # Need free cash flow and shares
        if not f.free_cash_flow or not f.shares_outstanding:
            return None
        
        if f.free_cash_flow <= 0:
            return None  # Can't DCF negative cash flow
        
        fcf = f.free_cash_flow
        
        # Estimate growth rate
        growth_rate = f.earnings_growth or f.revenue_growth or 0.05
        growth_rate = max(0, min(0.25, growth_rate))  # Cap at 25%
        
        # Project cash flows
        projected_fcf = []
        for year in range(1, DCF_DEFAULTS["projection_years"] + 1):
            # Decay growth rate toward terminal
            year_growth = growth_rate * (1 - year / (DCF_DEFAULTS["projection_years"] + 2))
            year_growth = max(year_growth, self.terminal_growth)
            fcf = fcf * (1 + year_growth)
            projected_fcf.append(fcf)
        
        # Discount projected cash flows
        pv_fcf = sum(
            cf / ((1 + self.discount_rate) ** (i + 1))
            for i, cf in enumerate(projected_fcf)
        )
        
        # Terminal value
        terminal_value = (
            projected_fcf[-1] * (1 + self.terminal_growth)
            / (self.discount_rate - self.terminal_growth)
        )
        pv_terminal = terminal_value / (
            (1 + self.discount_rate) ** DCF_DEFAULTS["projection_years"]
        )
        
        # Total enterprise value
        enterprise_value = pv_fcf + pv_terminal
        
        # Subtract debt, add cash (simplified)
        # Using market cap as proxy if EV not available
        equity_value = enterprise_value
        
        # Per share
        intrinsic_per_share = equity_value / f.shares_outstanding
        
        return intrinsic_per_share

    def _calculate_owner_earnings(self, f: Fundamentals) -> Optional[float]:
        """
        Calculate owner earnings (Buffett method).
        
        Owner Earnings = Net Income + Depreciation - Maintenance CapEx
        Simplified: Free Cash Flow (as proxy)
        """
        return f.free_cash_flow

    def _calculate_valuation_grade(
        self,
        f: Fundamentals,
        margin_of_safety: Optional[float],
        owner_earnings_yield: Optional[float],
    ) -> str:
        """Calculate overall valuation grade (A-F)."""
        scores = []
        
        # P/E score
        if f.pe_ratio:
            if f.pe_ratio < 10:
                scores.append(95)
            elif f.pe_ratio < 15:
                scores.append(80)
            elif f.pe_ratio < 20:
                scores.append(65)
            elif f.pe_ratio < 30:
                scores.append(45)
            else:
                scores.append(25)
        
        # PEG score
        if f.peg_ratio:
            if f.peg_ratio < 0.8:
                scores.append(95)
            elif f.peg_ratio < 1.0:
                scores.append(80)
            elif f.peg_ratio < 1.5:
                scores.append(60)
            elif f.peg_ratio < 2.0:
                scores.append(40)
            else:
                scores.append(20)
        
        # Margin of safety score
        if margin_of_safety:
            if margin_of_safety > 0.4:
                scores.append(95)
            elif margin_of_safety > 0.25:
                scores.append(80)
            elif margin_of_safety > 0.10:
                scores.append(65)
            elif margin_of_safety > 0:
                scores.append(50)
            else:
                scores.append(30)
        
        # Owner earnings yield score
        if owner_earnings_yield:
            if owner_earnings_yield > 0.10:
                scores.append(90)
            elif owner_earnings_yield > 0.07:
                scores.append(75)
            elif owner_earnings_yield > 0.05:
                scores.append(60)
            elif owner_earnings_yield > 0.03:
                scores.append(45)
            else:
                scores.append(30)
        
        if not scores:
            return "N/A"
        
        avg_score = sum(scores) / len(scores)
        
        if avg_score >= 85:
            return "A"
        elif avg_score >= 70:
            return "B"
        elif avg_score >= 55:
            return "C"
        elif avg_score >= 40:
            return "D"
        else:
            return "F"

    def _score_dcf(self, metrics: ValuationMetrics) -> float:
        """Score based on DCF valuation (0-100)."""
        if not metrics.margin_of_safety:
            return 50
        
        mos = metrics.margin_of_safety
        
        if mos > 0.50:
            return 95
        elif mos > 0.35:
            return 85
        elif mos > 0.20:
            return 70
        elif mos > 0.10:
            return 60
        elif mos > 0:
            return 50
        elif mos > -0.15:
            return 40
        elif mos > -0.30:
            return 25
        else:
            return 10

    def _score_relative_valuation(self, f: Fundamentals) -> float:
        """Score relative valuation metrics (0-100)."""
        scores = []
        
        # P/E ratio
        if f.pe_ratio is not None:
            if f.pe_ratio < 0:
                scores.append(10)  # Negative earnings
            elif f.pe_ratio < 12:
                scores.append(90)
            elif f.pe_ratio < 18:
                scores.append(70)
            elif f.pe_ratio < 25:
                scores.append(50)
            elif f.pe_ratio < 35:
                scores.append(30)
            else:
                scores.append(15)
        
        # P/B ratio
        if f.price_to_book is not None:
            if f.price_to_book < 1:
                scores.append(90)
            elif f.price_to_book < 2:
                scores.append(70)
            elif f.price_to_book < 4:
                scores.append(50)
            elif f.price_to_book < 8:
                scores.append(30)
            else:
                scores.append(15)
        
        # EV/EBITDA
        if f.ev_to_ebitda is not None:
            if f.ev_to_ebitda < 6:
                scores.append(90)
            elif f.ev_to_ebitda < 10:
                scores.append(70)
            elif f.ev_to_ebitda < 15:
                scores.append(50)
            elif f.ev_to_ebitda < 20:
                scores.append(30)
            else:
                scores.append(15)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_peg(self, f: Fundamentals) -> float:
        """Score PEG ratio (0-100)."""
        if not f.peg_ratio:
            return 50
        
        peg = f.peg_ratio
        
        if peg < 0:
            return 20  # Negative growth
        elif peg < 0.5:
            return 95
        elif peg < 1.0:
            return 80
        elif peg < 1.5:
            return 60
        elif peg < 2.0:
            return 45
        elif peg < 3.0:
            return 30
        else:
            return 15

    def _score_owner_earnings(self, metrics: ValuationMetrics) -> float:
        """Score owner earnings yield (0-100)."""
        if not metrics.owner_earnings_yield:
            return 50
        
        oey = metrics.owner_earnings_yield
        
        if oey > 0.12:
            return 95
        elif oey > 0.08:
            return 80
        elif oey > 0.06:
            return 65
        elif oey > 0.04:
            return 50
        elif oey > 0.02:
            return 35
        else:
            return 20

    def _score_to_signal(
        self,
        score: float,
        metrics: ValuationMetrics,
    ) -> tuple[Signal, float]:
        """Convert score to signal with confidence."""
        # Adjust confidence based on data quality
        base_confidence = 0.7
        if metrics.dcf_value and metrics.owner_earnings:
            base_confidence = 0.85
        elif not metrics.dcf_value and not metrics.owner_earnings:
            base_confidence = 0.5
        
        if score >= 80:
            return Signal.STRONG_BUY, base_confidence
        elif score >= 65:
            return Signal.BUY, base_confidence * 0.9
        elif score >= 45:
            return Signal.HOLD, base_confidence * 0.75
        elif score >= 30:
            return Signal.SELL, base_confidence * 0.85
        else:
            return Signal.STRONG_SELL, base_confidence * 0.9

    def _identify_key_factors(
        self,
        f: Fundamentals,
        metrics: ValuationMetrics,
    ) -> list[str]:
        """Identify key valuation factors."""
        factors = []
        
        # Margin of safety
        if metrics.margin_of_safety:
            if metrics.margin_of_safety > 0.25:
                factors.append(f"Strong margin of safety: {metrics.margin_of_safety:.1%}")
            elif metrics.margin_of_safety < -0.15:
                factors.append(f"Trading above intrinsic value by {abs(metrics.margin_of_safety):.1%}")
        
        # P/E
        if f.pe_ratio:
            if f.pe_ratio < 15:
                factors.append(f"Attractive P/E of {f.pe_ratio:.1f}")
            elif f.pe_ratio > 30:
                factors.append(f"High P/E of {f.pe_ratio:.1f}")
        
        # PEG
        if f.peg_ratio:
            if f.peg_ratio < 1:
                factors.append(f"Low PEG ratio of {f.peg_ratio:.2f}")
            elif f.peg_ratio > 2:
                factors.append(f"High PEG ratio of {f.peg_ratio:.2f}")
        
        # Grade
        if metrics.valuation_grade:
            factors.append(f"Overall valuation grade: {metrics.valuation_grade}")
        
        return factors[:5]

    def _build_reasoning(
        self,
        f: Fundamentals,
        metrics: ValuationMetrics,
        score: float,
    ) -> str:
        """Build reasoning text."""
        parts = [f"Valuation score: {score:.0f}/100."]
        
        if metrics.intrinsic_value and metrics.current_price:
            parts.append(
                f"Intrinsic value estimate: ${metrics.intrinsic_value:.2f} "
                f"vs current price ${metrics.current_price:.2f}."
            )
        
        if metrics.margin_of_safety:
            if metrics.margin_of_safety > 0:
                parts.append(f"Margin of safety: {metrics.margin_of_safety:.1%}.")
            else:
                parts.append(f"Trading {abs(metrics.margin_of_safety):.1%} above estimated value.")
        
        if metrics.valuation_grade:
            parts.append(f"Valuation grade: {metrics.valuation_grade}.")
        
        return " ".join(parts)


# Singleton
_valuation_agent: Optional[ValuationAgent] = None


def get_valuation_agent() -> ValuationAgent:
    """Get singleton valuation agent."""
    global _valuation_agent
    if _valuation_agent is None:
        _valuation_agent = ValuationAgent()
    return _valuation_agent
