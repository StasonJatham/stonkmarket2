"""
Risk analysis agent.

Analyzes risk factors for investments.
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
    PriceSeries,
    RiskMetrics,
    Signal,
)

logger = logging.getLogger(__name__)


class RiskAgent(CalculationAgentBase):
    """
    Analyzes risk factors to generate risk-adjusted signals.
    
    Evaluates:
    - Volatility (historical, beta)
    - Drawdown analysis
    - Financial risk (debt, liquidity)
    - Concentration risk
    """

    def __init__(self, max_acceptable_risk: float = 0.7):
        super().__init__(
            agent_id="risk",
            agent_name="Risk Manager",
            agent_type=AgentType.RISK,
        )
        self.max_acceptable_risk = max_acceptable_risk

    async def calculate(self, symbol: str, data: MarketData) -> AgentSignal:
        """Analyze risk and return signal."""
        prices = data.prices
        f = data.fundamentals
        
        # Calculate risk metrics
        metrics = self._calculate_risk_metrics(prices, f)
        
        # Risk score determines signal inversely
        # High risk = SELL, Low risk = BUY (all else equal)
        risk_score = metrics.overall_risk_score
        
        # Invert for signal: low risk is good
        signal_score = (1 - risk_score) * 100
        
        # Map to signal
        signal, confidence = self._score_to_signal(signal_score, metrics)
        
        # Key factors
        key_factors = self._identify_key_factors(f, metrics)
        
        # Reasoning
        reasoning = self._build_reasoning(f, metrics)
        
        return self._build_signal(
            symbol=symbol,
            signal=signal.value,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            metrics={
                "overall_risk_score": round(metrics.overall_risk_score, 3),
                "risk_grade": metrics.risk_grade,
                "volatility_30d": metrics.volatility_30d,
                "volatility_90d": metrics.volatility_90d,
                "beta": metrics.beta,
                "max_drawdown_1y": metrics.max_drawdown_1y,
                "current_drawdown": metrics.current_drawdown,
                "debt_risk_score": metrics.debt_risk_score,
                "liquidity_risk_score": metrics.liquidity_risk_score,
            },
        )

    def _calculate_risk_metrics(
        self,
        prices: PriceSeries,
        f: Fundamentals,
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        closes = [p.close for p in prices.prices] if prices.prices else []
        
        # Volatility
        vol_30d = self._calculate_volatility(closes, 30)
        vol_90d = self._calculate_volatility(closes, 90)
        
        # Beta from fundamentals
        beta = f.beta
        
        # Drawdown
        max_dd, current_dd = self._calculate_drawdowns(closes)
        
        # Financial risk scores
        debt_risk = self._calculate_debt_risk(f)
        liquidity_risk = self._calculate_liquidity_risk(f)
        
        # Overall risk score (0-1)
        risk_components = []
        
        # Volatility risk (annualized vol > 40% is high)
        if vol_90d is not None:
            vol_risk = min(1.0, vol_90d / 0.50)  # 50% vol = max risk
            risk_components.append(("volatility", vol_risk, 0.30))
        
        # Beta risk
        if beta is not None:
            beta_risk = min(1.0, max(0, abs(beta) - 0.5) / 1.5)  # Beta > 2 = high
            risk_components.append(("beta", beta_risk, 0.15))
        
        # Drawdown risk
        if max_dd is not None:
            dd_risk = min(1.0, abs(max_dd) / 0.50)  # 50% drawdown = max risk
            risk_components.append(("drawdown", dd_risk, 0.20))
        
        # Debt risk
        risk_components.append(("debt", debt_risk, 0.20))
        
        # Liquidity risk
        risk_components.append(("liquidity", liquidity_risk, 0.15))
        
        # Weighted average
        if risk_components:
            total_weight = sum(w for _, _, w in risk_components)
            overall_risk = sum(r * w for _, r, w in risk_components) / total_weight
        else:
            overall_risk = 0.5
        
        # Risk grade
        if overall_risk < 0.25:
            risk_grade = "Low"
        elif overall_risk < 0.45:
            risk_grade = "Medium"
        elif overall_risk < 0.65:
            risk_grade = "High"
        else:
            risk_grade = "Very High"
        
        # Risk factors
        risk_factors = []
        for name, score, _ in risk_components:
            if score > 0.6:
                risk_factors.append(f"High {name} risk ({score:.0%})")
        
        return RiskMetrics(
            symbol=f.symbol,
            volatility_30d=vol_30d,
            volatility_90d=vol_90d,
            beta=beta,
            max_drawdown_1y=max_dd,
            current_drawdown=current_dd,
            debt_risk_score=debt_risk,
            liquidity_risk_score=liquidity_risk,
            overall_risk_score=overall_risk,
            risk_grade=risk_grade,
            risk_factors=risk_factors,
        )

    def _calculate_volatility(
        self,
        closes: list[float],
        period: int,
    ) -> Optional[float]:
        """Calculate annualized volatility."""
        if len(closes) < period + 1:
            return None
        
        # Daily returns
        returns = [
            (closes[i] - closes[i-1]) / closes[i-1]
            for i in range(1, len(closes))
        ]
        
        subset = returns[-period:]
        if not subset:
            return None
        
        mean = sum(subset) / len(subset)
        variance = sum((r - mean) ** 2 for r in subset) / len(subset)
        daily_vol = math.sqrt(variance)
        
        # Annualize
        return daily_vol * math.sqrt(252)

    def _calculate_drawdowns(
        self,
        closes: list[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate max drawdown and current drawdown."""
        if not closes:
            return None, None
        
        peak = closes[0]
        max_dd = 0.0
        
        for price in closes:
            if price > peak:
                peak = price
            dd = (price - peak) / peak
            if dd < max_dd:
                max_dd = dd
        
        # Current drawdown
        current_peak = max(closes)
        current_dd = (closes[-1] - current_peak) / current_peak
        
        return max_dd, current_dd

    def _calculate_debt_risk(self, f: Fundamentals) -> float:
        """Calculate debt risk score (0-1)."""
        scores = []
        
        # Debt to equity
        if f.debt_to_equity is not None:
            if f.debt_to_equity <= 0.3:
                scores.append(0.1)
            elif f.debt_to_equity <= 0.5:
                scores.append(0.25)
            elif f.debt_to_equity <= 1.0:
                scores.append(0.4)
            elif f.debt_to_equity <= 2.0:
                scores.append(0.6)
            elif f.debt_to_equity <= 3.0:
                scores.append(0.8)
            else:
                scores.append(1.0)
        
        # Interest coverage (if available in raw_info)
        if f.raw_info:
            int_cov = f.raw_info.get("interestCoverage")
            if int_cov:
                if int_cov > 10:
                    scores.append(0.1)
                elif int_cov > 5:
                    scores.append(0.25)
                elif int_cov > 3:
                    scores.append(0.4)
                elif int_cov > 1.5:
                    scores.append(0.6)
                elif int_cov > 1:
                    scores.append(0.8)
                else:
                    scores.append(1.0)
        
        return sum(scores) / len(scores) if scores else 0.5

    def _calculate_liquidity_risk(self, f: Fundamentals) -> float:
        """Calculate liquidity risk score (0-1)."""
        scores = []
        
        # Current ratio
        if f.current_ratio is not None:
            if f.current_ratio >= 2.0:
                scores.append(0.1)
            elif f.current_ratio >= 1.5:
                scores.append(0.25)
            elif f.current_ratio >= 1.2:
                scores.append(0.4)
            elif f.current_ratio >= 1.0:
                scores.append(0.6)
            elif f.current_ratio >= 0.8:
                scores.append(0.8)
            else:
                scores.append(1.0)
        
        # Quick ratio
        if f.quick_ratio is not None:
            if f.quick_ratio >= 1.5:
                scores.append(0.1)
            elif f.quick_ratio >= 1.0:
                scores.append(0.3)
            elif f.quick_ratio >= 0.7:
                scores.append(0.5)
            elif f.quick_ratio >= 0.5:
                scores.append(0.7)
            else:
                scores.append(0.9)
        
        return sum(scores) / len(scores) if scores else 0.5

    def _score_to_signal(
        self,
        score: float,
        metrics: RiskMetrics,
    ) -> tuple[Signal, float]:
        """Convert score to signal. Score is inverted risk (high score = low risk = good)."""
        # High risk threshold check
        if metrics.overall_risk_score > self.max_acceptable_risk:
            return Signal.SELL, 0.8
        
        if score >= 75:
            return Signal.BUY, 0.7  # Low risk
        elif score >= 55:
            return Signal.HOLD, 0.6  # Moderate risk
        elif score >= 40:
            return Signal.HOLD, 0.55  # Elevated risk
        elif score >= 25:
            return Signal.SELL, 0.65  # High risk
        else:
            return Signal.STRONG_SELL, 0.75  # Very high risk

    def _identify_key_factors(
        self,
        f: Fundamentals,
        metrics: RiskMetrics,
    ) -> list[str]:
        """Identify key risk factors."""
        factors = []
        
        # Risk grade
        factors.append(f"Risk grade: {metrics.risk_grade}")
        
        # Volatility
        if metrics.volatility_90d is not None:
            if metrics.volatility_90d > 0.40:
                factors.append(f"High volatility: {metrics.volatility_90d:.1%} annualized")
            elif metrics.volatility_90d < 0.20:
                factors.append(f"Low volatility: {metrics.volatility_90d:.1%} annualized")
        
        # Beta
        if metrics.beta is not None:
            if metrics.beta > 1.5:
                factors.append(f"High beta of {metrics.beta:.2f}")
            elif metrics.beta < 0.7:
                factors.append(f"Low beta of {metrics.beta:.2f}")
        
        # Drawdown
        if metrics.max_drawdown_1y is not None:
            if metrics.max_drawdown_1y < -0.30:
                factors.append(f"Significant max drawdown: {metrics.max_drawdown_1y:.1%}")
        
        # Debt
        if f.debt_to_equity is not None:
            if f.debt_to_equity > 1.5:
                factors.append(f"High debt-to-equity: {f.debt_to_equity:.2f}")
            elif f.debt_to_equity < 0.3:
                factors.append(f"Low debt-to-equity: {f.debt_to_equity:.2f}")
        
        # Add from risk factors
        factors.extend(metrics.risk_factors[:2])
        
        return factors[:5]

    def _build_reasoning(
        self,
        f: Fundamentals,
        metrics: RiskMetrics,
    ) -> str:
        """Build reasoning text."""
        parts = [
            f"Overall risk score: {metrics.overall_risk_score:.0%} ({metrics.risk_grade})."
        ]
        
        if metrics.volatility_90d:
            parts.append(f"90-day volatility: {metrics.volatility_90d:.1%}.")
        
        if metrics.beta:
            parts.append(f"Beta: {metrics.beta:.2f}.")
        
        if metrics.max_drawdown_1y:
            parts.append(f"Max drawdown (1Y): {metrics.max_drawdown_1y:.1%}.")
        
        if metrics.risk_factors:
            parts.append(f"Key risks: {', '.join(metrics.risk_factors[:3])}.")
        
        return " ".join(parts)


# Singleton
_risk_agent: Optional[RiskAgent] = None


def get_risk_agent() -> RiskAgent:
    """Get singleton risk agent."""
    global _risk_agent
    if _risk_agent is None:
        _risk_agent = RiskAgent()
    return _risk_agent
