"""
Sentiment analysis agent.

Analyzes market sentiment indicators.
Pure calculation-based - no LLM required.
"""

import logging
from typing import Optional

from app.hedge_fund.agents.base import AgentSignal, CalculationAgentBase
from app.hedge_fund.schemas import (
    AgentType,
    Fundamentals,
    LLMMode,
    MarketData,
    SentimentMetrics,
    Signal,
)

logger = logging.getLogger(__name__)


class SentimentAgent(CalculationAgentBase):
    """
    Analyzes sentiment indicators to generate investment signals.
    
    Evaluates:
    - Analyst ratings and price targets
    - Institutional ownership changes
    - Short interest
    - Insider activity
    """

    def __init__(self):
        super().__init__(
            agent_id="sentiment",
            agent_name="Sentiment Analyst",
            agent_type=AgentType.SENTIMENT,
        )

    async def calculate(self, symbol: str, data: MarketData) -> AgentSignal:
        """Analyze sentiment and return signal."""
        f = data.fundamentals
        current_price = data.prices.latest.close if data.prices.latest else None
        
        # Calculate sentiment metrics
        metrics = self._calculate_sentiment_metrics(f, current_price)
        
        # Score components
        analyst_score = self._score_analyst_sentiment(f, current_price)
        short_score = self._score_short_interest(f)
        institutional_score = self._score_institutional(f)
        
        # Weight the scores
        weights = {
            "analyst": 0.45,
            "short": 0.30,
            "institutional": 0.25,
        }
        
        total_score = (
            analyst_score * weights["analyst"]
            + short_score * weights["short"]
            + institutional_score * weights["institutional"]
        )
        
        # Map to signal
        signal, confidence = self._score_to_signal(total_score)
        
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
                "analyst_score": round(analyst_score, 2),
                "short_score": round(short_score, 2),
                "institutional_score": round(institutional_score, 2),
                "sentiment_score": metrics.sentiment_score,
                "overall_sentiment": metrics.overall_sentiment,
                "short_ratio": f.short_ratio,
                "short_percent_float": f.short_percent_of_float,
            },
        )

    def _calculate_sentiment_metrics(
        self,
        f: Fundamentals,
        current_price: Optional[float],
    ) -> SentimentMetrics:
        """Calculate comprehensive sentiment metrics."""
        # Analyst target upside
        target_upside = None
        if current_price and f.raw_info:
            target_mean = f.raw_info.get("targetMeanPrice")
            if target_mean:
                target_upside = (target_mean - current_price) / current_price
        
        # Calculate overall sentiment score (-1 to 1)
        sentiment_scores = []
        
        # Analyst sentiment
        if target_upside is not None:
            if target_upside > 0.30:
                sentiment_scores.append(0.8)
            elif target_upside > 0.15:
                sentiment_scores.append(0.5)
            elif target_upside > 0:
                sentiment_scores.append(0.2)
            elif target_upside > -0.15:
                sentiment_scores.append(-0.2)
            else:
                sentiment_scores.append(-0.6)
        
        # Short interest sentiment
        if f.short_percent_of_float is not None:
            if f.short_percent_of_float < 0.03:
                sentiment_scores.append(0.5)
            elif f.short_percent_of_float < 0.07:
                sentiment_scores.append(0.2)
            elif f.short_percent_of_float < 0.15:
                sentiment_scores.append(-0.2)
            else:
                sentiment_scores.append(-0.6)
        
        sentiment_score = (
            sum(sentiment_scores) / len(sentiment_scores)
            if sentiment_scores else 0.0
        )
        
        # Map to sentiment label
        if sentiment_score > 0.5:
            overall_sentiment = "Very Bullish"
        elif sentiment_score > 0.2:
            overall_sentiment = "Bullish"
        elif sentiment_score > -0.2:
            overall_sentiment = "Neutral"
        elif sentiment_score > -0.5:
            overall_sentiment = "Bearish"
        else:
            overall_sentiment = "Very Bearish"
        
        return SentimentMetrics(
            symbol=f.symbol,
            target_upside=target_upside,
            short_interest_ratio=f.short_ratio,
            short_percent_float=f.short_percent_of_float,
            overall_sentiment=overall_sentiment,
            sentiment_score=sentiment_score,
        )

    def _score_analyst_sentiment(
        self,
        f: Fundamentals,
        current_price: Optional[float],
    ) -> float:
        """Score analyst sentiment (0-100)."""
        scores = []
        
        # Target price upside
        if current_price and f.raw_info:
            target_mean = f.raw_info.get("targetMeanPrice")
            if target_mean:
                upside = (target_mean - current_price) / current_price
                if upside > 0.40:
                    scores.append(95)
                elif upside > 0.25:
                    scores.append(80)
                elif upside > 0.15:
                    scores.append(70)
                elif upside > 0.05:
                    scores.append(60)
                elif upside > 0:
                    scores.append(50)
                elif upside > -0.10:
                    scores.append(40)
                else:
                    scores.append(25)
        
        # Recommendation
        if f.raw_info:
            rec_key = f.raw_info.get("recommendationKey", "").lower()
            rec_map = {
                "strong_buy": 95,
                "buy": 75,
                "hold": 50,
                "underperform": 30,
                "sell": 15,
            }
            if rec_key in rec_map:
                scores.append(rec_map[rec_key])
        
        return sum(scores) / len(scores) if scores else 50

    def _score_short_interest(self, f: Fundamentals) -> float:
        """Score short interest (0-100). Lower short = higher score."""
        scores = []
        
        # Short percent of float
        if f.short_percent_of_float is not None:
            spf = f.short_percent_of_float
            if spf < 0.02:
                scores.append(90)
            elif spf < 0.05:
                scores.append(75)
            elif spf < 0.10:
                scores.append(55)
            elif spf < 0.20:
                scores.append(35)
            else:
                scores.append(20)  # Very high short interest
        
        # Short ratio (days to cover)
        if f.short_ratio is not None:
            sr = f.short_ratio
            if sr < 1:
                scores.append(85)
            elif sr < 3:
                scores.append(70)
            elif sr < 5:
                scores.append(50)
            elif sr < 10:
                scores.append(35)
            else:
                scores.append(20)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_institutional(self, f: Fundamentals) -> float:
        """Score institutional sentiment (0-100)."""
        # Limited data from yfinance, use what's available
        scores = []
        
        if f.raw_info:
            # Institutional holders percentage
            inst_pct = f.raw_info.get("heldPercentInstitutions")
            if inst_pct:
                if inst_pct > 0.80:
                    scores.append(70)  # Very institutionally held
                elif inst_pct > 0.60:
                    scores.append(65)
                elif inst_pct > 0.40:
                    scores.append(55)
                elif inst_pct > 0.20:
                    scores.append(50)
                else:
                    scores.append(45)  # Low institutional interest
        
        return sum(scores) / len(scores) if scores else 50

    def _score_to_signal(self, score: float) -> tuple[Signal, float]:
        """Convert score to signal and confidence."""
        # Sentiment is a softer signal - lower confidence
        if score >= 75:
            return Signal.BUY, 0.65
        elif score >= 60:
            return Signal.BUY, 0.55
        elif score >= 45:
            return Signal.HOLD, 0.50
        elif score >= 35:
            return Signal.SELL, 0.55
        else:
            return Signal.SELL, 0.65

    def _identify_key_factors(
        self,
        f: Fundamentals,
        metrics: SentimentMetrics,
    ) -> list[str]:
        """Identify key sentiment factors."""
        factors = []
        
        # Target upside
        if metrics.target_upside is not None:
            if metrics.target_upside > 0:
                factors.append(f"Analyst target implies {metrics.target_upside:.1%} upside")
            else:
                factors.append(f"Analyst target implies {abs(metrics.target_upside):.1%} downside")
        
        # Short interest
        if f.short_percent_of_float is not None:
            if f.short_percent_of_float > 0.10:
                factors.append(f"High short interest: {f.short_percent_of_float:.1%} of float")
            elif f.short_percent_of_float < 0.03:
                factors.append(f"Low short interest: {f.short_percent_of_float:.1%} of float")
        
        # Days to cover
        if f.short_ratio is not None:
            if f.short_ratio > 5:
                factors.append(f"High days to cover: {f.short_ratio:.1f}")
        
        # Overall sentiment
        factors.append(f"Overall sentiment: {metrics.overall_sentiment}")
        
        return factors[:5]

    def _build_reasoning(
        self,
        f: Fundamentals,
        metrics: SentimentMetrics,
        score: float,
    ) -> str:
        """Build reasoning text."""
        parts = [f"Sentiment score: {score:.0f}/100."]
        parts.append(f"Overall market sentiment: {metrics.overall_sentiment}.")
        
        if metrics.target_upside is not None:
            direction = "upside" if metrics.target_upside > 0 else "downside"
            parts.append(f"Analyst targets imply {abs(metrics.target_upside):.1%} {direction}.")
        
        if f.short_percent_of_float is not None:
            parts.append(f"Short interest at {f.short_percent_of_float:.1%} of float.")
        
        return " ".join(parts)


# Singleton
_sentiment_agent: Optional[SentimentAgent] = None


def get_sentiment_agent() -> SentimentAgent:
    """Get singleton sentiment agent."""
    global _sentiment_agent
    if _sentiment_agent is None:
        _sentiment_agent = SentimentAgent()
    return _sentiment_agent
