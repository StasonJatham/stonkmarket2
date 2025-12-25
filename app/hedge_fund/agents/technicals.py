"""
Technical analysis agent.

Analyzes price action and technical indicators.
Pure calculation-based - no LLM required.
"""

import logging
import math
from typing import Optional

from app.hedge_fund.agents.base import AgentSignal, CalculationAgentBase
from app.hedge_fund.schemas import (
    AgentType,
    LLMMode,
    MarketData,
    PriceSeries,
    Signal,
    TechnicalIndicators,
)

logger = logging.getLogger(__name__)


class TechnicalsAgent(CalculationAgentBase):
    """
    Analyzes technical indicators to generate investment signals.
    
    Evaluates:
    - Trend (moving averages, ADX)
    - Momentum (RSI, MACD)
    - Volatility (Bollinger Bands, ATR)
    - Volume patterns
    """

    def __init__(self):
        super().__init__(
            agent_id="technicals",
            agent_name="Technical Analyst",
            agent_type=AgentType.TECHNICALS,
        )

    async def calculate(self, symbol: str, data: MarketData) -> AgentSignal:
        """Analyze technicals and return signal."""
        prices = data.prices
        
        if not prices.prices or len(prices.prices) < 20:
            return self._build_signal(
                symbol=symbol,
                signal=Signal.HOLD.value,
                confidence=0.3,
                reasoning="Insufficient price data for technical analysis.",
                key_factors=["Need at least 20 days of price history"],
            )
        
        # Calculate indicators
        indicators = self._calculate_indicators(prices)
        
        # Score each component
        trend_score = self._score_trend(indicators)
        momentum_score = self._score_momentum(indicators)
        volatility_score = self._score_volatility(indicators)
        volume_score = self._score_volume(indicators)
        
        # Weight the scores
        weights = {
            "trend": 0.35,
            "momentum": 0.30,
            "volatility": 0.15,
            "volume": 0.20,
        }
        
        total_score = (
            trend_score * weights["trend"]
            + momentum_score * weights["momentum"]
            + volatility_score * weights["volatility"]
            + volume_score * weights["volume"]
        )
        
        # Map to signal
        signal, confidence = self._score_to_signal(total_score)
        
        # Key factors
        key_factors = self._identify_key_factors(indicators, {
            "trend": trend_score,
            "momentum": momentum_score,
            "volatility": volatility_score,
            "volume": volume_score,
        })
        
        # Reasoning
        reasoning = self._build_reasoning(indicators, total_score, {
            "trend": trend_score,
            "momentum": momentum_score,
            "volatility": volatility_score,
            "volume": volume_score,
        })
        
        return self._build_signal(
            symbol=symbol,
            signal=signal.value,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            metrics={
                "total_score": round(total_score, 2),
                "trend_score": round(trend_score, 2),
                "momentum_score": round(momentum_score, 2),
                "volatility_score": round(volatility_score, 2),
                "volume_score": round(volume_score, 2),
                "rsi_14": indicators.rsi_14,
                "macd": indicators.macd,
                "sma_50": indicators.sma_50,
                "sma_200": indicators.sma_200,
            },
        )

    def _calculate_indicators(self, prices: PriceSeries) -> TechnicalIndicators:
        """Calculate all technical indicators."""
        closes = [p.close for p in prices.prices]
        highs = [p.high for p in prices.prices]
        lows = [p.low for p in prices.prices]
        volumes = [p.volume for p in prices.prices]
        
        current_price = closes[-1] if closes else None
        
        # Moving averages
        sma_20 = self._sma(closes, 20)
        sma_50 = self._sma(closes, 50)
        sma_200 = self._sma(closes, 200)
        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)
        
        # MACD
        macd = ema_12 - ema_26 if ema_12 and ema_26 else None
        macd_signal = None
        macd_histogram = None
        if macd is not None:
            macd_values = self._calculate_macd_series(closes)
            if macd_values:
                macd_signal = self._ema(macd_values, 9)
                if macd_signal:
                    macd_histogram = macd - macd_signal
        
        # RSI
        rsi_14 = self._rsi(closes, 14)
        
        # Bollinger Bands
        bb_middle = sma_20
        bb_std = self._std(closes, 20)
        bb_upper = bb_middle + 2 * bb_std if bb_middle and bb_std else None
        bb_lower = bb_middle - 2 * bb_std if bb_middle and bb_std else None
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_upper and bb_lower and bb_middle else None
        
        # ATR
        atr_14 = self._atr(highs, lows, closes, 14)
        
        # Volatility
        volatility_20 = self._volatility(closes, 20)
        
        # Volume
        volume_sma_20 = self._sma(volumes, 20)
        volume_ratio = volumes[-1] / volume_sma_20 if volume_sma_20 and volumes else None
        
        # Price vs MAs
        price_vs_sma_20 = ((current_price - sma_20) / sma_20 * 100) if current_price and sma_20 else None
        price_vs_sma_50 = ((current_price - sma_50) / sma_50 * 100) if current_price and sma_50 else None
        price_vs_sma_200 = ((current_price - sma_200) / sma_200 * 100) if current_price and sma_200 else None
        
        return TechnicalIndicators(
            symbol=prices.symbol,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            rsi_14=rsi_14,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_width=bb_width,
            atr_14=atr_14,
            volatility_20=volatility_20,
            volume_sma_20=volume_sma_20,
            volume_ratio=volume_ratio,
            current_price=current_price,
            price_vs_sma_20=price_vs_sma_20,
            price_vs_sma_50=price_vs_sma_50,
            price_vs_sma_200=price_vs_sma_200,
        )

    def _sma(self, values: list[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _ema(self, values: list[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(values) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = sum(values[:period]) / period  # Start with SMA
        
        for price in values[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema

    def _std(self, values: list[float], period: int) -> Optional[float]:
        """Calculate standard deviation."""
        if len(values) < period:
            return None
        
        subset = values[-period:]
        mean = sum(subset) / period
        variance = sum((x - mean) ** 2 for x in subset) / period
        return math.sqrt(variance)

    def _rsi(self, closes: list[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(closes) < period + 1:
            return None
        
        # Calculate price changes
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        
        # Separate gains and losses
        gains = [max(0, c) for c in changes]
        losses = [abs(min(0, c)) for c in changes]
        
        # Initial averages
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Smooth averages
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd_series(self, closes: list[float]) -> list[float]:
        """Calculate MACD line series."""
        if len(closes) < 26:
            return []
        
        result = []
        for i in range(26, len(closes) + 1):
            subset = closes[:i]
            ema_12 = self._ema(subset, 12)
            ema_26 = self._ema(subset, 26)
            if ema_12 and ema_26:
                result.append(ema_12 - ema_26)
        
        return result

    def _atr(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> Optional[float]:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return None
        
        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        
        return sum(trs[-period:]) / period

    def _volatility(self, closes: list[float], period: int = 20) -> Optional[float]:
        """Calculate historical volatility (annualized)."""
        if len(closes) < period + 1:
            return None
        
        # Daily returns
        returns = [
            (closes[i] - closes[i-1]) / closes[i-1]
            for i in range(1, len(closes))
        ]
        
        if len(returns) < period:
            return None
        
        subset = returns[-period:]
        mean = sum(subset) / len(subset)
        variance = sum((r - mean) ** 2 for r in subset) / len(subset)
        daily_vol = math.sqrt(variance)
        
        # Annualize (252 trading days)
        return daily_vol * math.sqrt(252)

    def _score_trend(self, ind: TechnicalIndicators) -> float:
        """Score trend indicators (0-100)."""
        scores = []
        
        # Price vs 50 SMA
        if ind.price_vs_sma_50 is not None:
            if ind.price_vs_sma_50 > 10:
                scores.append(90)
            elif ind.price_vs_sma_50 > 5:
                scores.append(75)
            elif ind.price_vs_sma_50 > 0:
                scores.append(60)
            elif ind.price_vs_sma_50 > -5:
                scores.append(45)
            elif ind.price_vs_sma_50 > -10:
                scores.append(30)
            else:
                scores.append(15)
        
        # Price vs 200 SMA (golden/death cross area)
        if ind.price_vs_sma_200 is not None:
            if ind.price_vs_sma_200 > 15:
                scores.append(95)
            elif ind.price_vs_sma_200 > 5:
                scores.append(75)
            elif ind.price_vs_sma_200 > 0:
                scores.append(55)
            elif ind.price_vs_sma_200 > -10:
                scores.append(35)
            else:
                scores.append(15)
        
        # 50 vs 200 SMA (golden cross)
        if ind.sma_50 and ind.sma_200:
            if ind.sma_50 > ind.sma_200 * 1.05:
                scores.append(85)
            elif ind.sma_50 > ind.sma_200:
                scores.append(65)
            elif ind.sma_50 > ind.sma_200 * 0.95:
                scores.append(45)
            else:
                scores.append(25)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_momentum(self, ind: TechnicalIndicators) -> float:
        """Score momentum indicators (0-100)."""
        scores = []
        
        # RSI
        if ind.rsi_14 is not None:
            if 50 <= ind.rsi_14 <= 70:
                scores.append(80)  # Bullish but not overbought
            elif 30 <= ind.rsi_14 < 50:
                scores.append(50)  # Neutral to slightly bearish
            elif ind.rsi_14 < 30:
                scores.append(40)  # Oversold - potential bounce
            else:
                scores.append(30)  # Overbought
        
        # MACD
        if ind.macd is not None and ind.macd_signal is not None:
            if ind.macd > 0 and ind.macd > ind.macd_signal:
                scores.append(85)  # Bullish crossover
            elif ind.macd > 0:
                scores.append(65)  # Above zero but weakening
            elif ind.macd < 0 and ind.macd > ind.macd_signal:
                scores.append(45)  # Below zero but improving
            else:
                scores.append(25)  # Bearish
        
        # MACD histogram
        if ind.macd_histogram is not None:
            if ind.macd_histogram > 0:
                scores.append(70)
            else:
                scores.append(30)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_volatility(self, ind: TechnicalIndicators) -> float:
        """Score volatility indicators (0-100). Lower volatility = higher score for stability."""
        scores = []
        
        # Bollinger Band width (tighter = potential breakout)
        if ind.bb_width is not None:
            if ind.bb_width < 0.05:
                scores.append(70)  # Very tight - squeeze forming
            elif ind.bb_width < 0.10:
                scores.append(60)
            elif ind.bb_width < 0.15:
                scores.append(50)
            else:
                scores.append(40)  # Wide bands - high volatility
        
        # Price relative to Bollinger Bands
        if ind.current_price and ind.bb_upper and ind.bb_lower and ind.bb_middle:
            bb_position = (ind.current_price - ind.bb_lower) / (ind.bb_upper - ind.bb_lower)
            if 0.4 <= bb_position <= 0.7:
                scores.append(70)  # Healthy middle ground
            elif bb_position > 0.9:
                scores.append(35)  # Near upper band - overbought
            elif bb_position < 0.1:
                scores.append(45)  # Near lower band - potential bounce
            else:
                scores.append(55)
        
        return sum(scores) / len(scores) if scores else 50

    def _score_volume(self, ind: TechnicalIndicators) -> float:
        """Score volume indicators (0-100)."""
        scores = []
        
        # Volume ratio (current vs average)
        if ind.volume_ratio is not None:
            if ind.volume_ratio > 2.0:
                # High volume - could be good or bad, neutral-ish
                scores.append(60)
            elif ind.volume_ratio > 1.2:
                scores.append(70)  # Above average
            elif ind.volume_ratio > 0.8:
                scores.append(55)  # Normal
            else:
                scores.append(40)  # Low volume
        
        return sum(scores) / len(scores) if scores else 50

    def _score_to_signal(self, score: float) -> tuple[Signal, float]:
        """Convert score to signal and confidence."""
        if score >= 75:
            return Signal.STRONG_BUY, 0.85
        elif score >= 60:
            return Signal.BUY, 0.7
        elif score >= 45:
            return Signal.HOLD, 0.55
        elif score >= 35:
            return Signal.SELL, 0.65
        else:
            return Signal.STRONG_SELL, 0.8

    def _identify_key_factors(
        self,
        ind: TechnicalIndicators,
        scores: dict[str, float],
    ) -> list[str]:
        """Identify key technical factors."""
        factors = []
        
        # Trend factors
        if ind.sma_50 and ind.sma_200:
            if ind.sma_50 > ind.sma_200:
                factors.append("Golden cross (50 SMA > 200 SMA)")
            else:
                factors.append("Death cross (50 SMA < 200 SMA)")
        
        # RSI
        if ind.rsi_14 is not None:
            if ind.rsi_14 > 70:
                factors.append(f"Overbought RSI at {ind.rsi_14:.1f}")
            elif ind.rsi_14 < 30:
                factors.append(f"Oversold RSI at {ind.rsi_14:.1f}")
            else:
                factors.append(f"RSI at {ind.rsi_14:.1f}")
        
        # MACD
        if ind.macd is not None:
            if ind.macd > 0:
                factors.append("Positive MACD")
            else:
                factors.append("Negative MACD")
        
        # Price position
        if ind.price_vs_sma_200 is not None:
            if ind.price_vs_sma_200 > 0:
                factors.append(f"Trading {ind.price_vs_sma_200:.1f}% above 200 SMA")
            else:
                factors.append(f"Trading {abs(ind.price_vs_sma_200):.1f}% below 200 SMA")
        
        return factors[:5]

    def _build_reasoning(
        self,
        ind: TechnicalIndicators,
        total_score: float,
        scores: dict[str, float],
    ) -> str:
        """Build reasoning text."""
        parts = [f"Technical score: {total_score:.0f}/100."]
        
        parts.append(
            f"Trend: {scores['trend']:.0f}, "
            f"Momentum: {scores['momentum']:.0f}, "
            f"Volatility: {scores['volatility']:.0f}, "
            f"Volume: {scores['volume']:.0f}."
        )
        
        if ind.rsi_14:
            parts.append(f"RSI at {ind.rsi_14:.1f}.")
        
        if ind.macd is not None:
            direction = "positive" if ind.macd > 0 else "negative"
            parts.append(f"MACD is {direction}.")
        
        return " ".join(parts)


# Singleton
_technicals_agent: Optional[TechnicalsAgent] = None


def get_technicals_agent() -> TechnicalsAgent:
    """Get singleton technicals agent."""
    global _technicals_agent
    if _technicals_agent is None:
        _technicals_agent = TechnicalsAgent()
    return _technicals_agent
