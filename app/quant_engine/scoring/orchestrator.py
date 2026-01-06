"""
ScoringOrchestrator - Unified entry point for all stock scoring.

This replaces the fragmented scoring systems with a single orchestrator that:
1. Uses TechnicalService for all indicator calculations
2. Uses RegimeService for market regime detection
3. Uses DomainScoringAdapter for fundamental quality
4. Outputs a unified StockAnalysisDashboard

All other scoring files (scoring.py, scoring_v2.py, trade_engine.py)
should be deprecated in favor of this orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.quant_engine.core.technical_service import (
    TechnicalService,
    TechnicalSnapshot,
    get_technical_service,
)
from app.quant_engine.core.regime_service import (
    MarketRegime,
    RegimeService,
    RegimeState,
    get_regime_service,
)
from app.quant_engine.scoring.output import (
    StockAnalysisDashboard,
    ScoreComponents,
    EntryAnalysis,
    RiskAssessment,
)

# Import domain scoring - the source of truth for fundamental quality
from app.quant_engine.dipfinder.domain import Domain, classify_domain
from app.quant_engine.dipfinder.domain_scoring import (
    DomainScoreResult,
    get_adapter,
    compute_domain_score,
)

logger = logging.getLogger(__name__)

SCORING_VERSION = "3.0.0"


@dataclass
class OrchestratorConfig:
    """Configuration for the scoring orchestrator."""
    
    # Score weights (default for BULL market)
    technical_weight: float = 0.25
    fundamental_weight: float = 0.30
    regime_weight: float = 0.15
    entry_timing_weight: float = 0.20
    risk_weight: float = 0.10
    
    # Recommendation thresholds
    strong_buy_threshold: float = 75.0
    buy_threshold: float = 60.0
    accumulate_threshold: float = 50.0
    hold_threshold: float = 40.0
    avoid_threshold: float = 30.0
    
    # Risk limits
    max_drawdown_for_entry: float = -40.0  # Don't buy if > 40% down (falling knife)
    min_fundamental_for_bear: float = 55.0  # Minimum quality for bear market buys


class ScoringOrchestrator:
    """
    Main orchestrator for unified stock scoring.
    
    Usage:
        orchestrator = ScoringOrchestrator()
        dashboard = await orchestrator.analyze(
            symbol="AAPL",
            stock_prices=stock_df,
            spy_prices=spy_df,
            fundamentals=fund_dict,
        )
    """
    
    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        technical_service: TechnicalService | None = None,
        regime_service: RegimeService | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.tech_service = technical_service or get_technical_service()
        self.regime_service = regime_service or get_regime_service()
    
    async def analyze(
        self,
        symbol: str,
        stock_prices: pd.DataFrame,
        spy_prices: pd.DataFrame,
        fundamentals: dict[str, Any] | None = None,
        name: str | None = None,
        sector: str | None = None,
        vix_level: float | None = None,
    ) -> StockAnalysisDashboard:
        """
        Perform complete stock analysis.
        
        Args:
            symbol: Stock ticker
            stock_prices: OHLCV data for the stock
            spy_prices: OHLCV data for SPY (market benchmark)
            fundamentals: Dict of fundamental metrics (from yfinance or stored)
            name: Company name
            sector: Sector classification
            vix_level: Optional current VIX level
            
        Returns:
            StockAnalysisDashboard with complete analysis
        """
        logger.debug(f"Analyzing {symbol}")
        
        # Get technical snapshot
        technicals = self.tech_service.get_snapshot(stock_prices)
        
        # Get market regime
        regime = self.regime_service.get_current_regime(spy_prices, vix_level)
        
        # Get domain classification and quality score
        # Extract classification parameters from fundamentals dict
        info = fundamentals or {}
        domain_classification = classify_domain(
            symbol=symbol,
            sector=sector or info.get("sector"),
            industry=info.get("industry"),
            name=name or info.get("shortName") or info.get("longName"),
            quote_type=info.get("quoteType"),
        )
        domain_quality = compute_domain_score(
            classification=domain_classification,
            info=info,
            fundamentals=fundamentals,
        )
        
        # Compute component scores
        technical_score = self._compute_technical_score(technicals, regime)
        fundamental_score = domain_quality.final_score
        regime_score = self.regime_service.get_regime_score(regime)
        entry_analysis = self._compute_entry_analysis(stock_prices, technicals)
        entry_timing_score = self._compute_entry_timing_score(entry_analysis, technicals)
        risk_assessment = self._compute_risk_assessment(
            technicals, domain_quality, regime, entry_analysis
        )
        
        # Compute composite score with dynamic weights
        weights = self._get_dynamic_weights(regime)
        composite_score = (
            technical_score * weights["technical"] +
            fundamental_score * weights["fundamental"] +
            regime_score * weights["regime"] +
            entry_timing_score * weights["entry"] +
            risk_assessment.risk_score * weights["risk"]
        )
        
        # Build score components
        scores = ScoreComponents(
            technical=technical_score,
            fundamental=fundamental_score,
            regime=regime_score,
            entry_timing=entry_timing_score,
            risk=risk_assessment.risk_score,
            composite=composite_score,
        )
        
        # Determine recommendation
        recommendation, summary = self._determine_recommendation(
            scores=scores,
            regime=regime,
            entry_analysis=entry_analysis,
            risk_assessment=risk_assessment,
            domain_quality=domain_quality,
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            domain_quality.data_completeness,
            len(stock_prices),
            technicals.volatility_regime,
        )
        
        # Assess data quality
        data_quality = self._assess_data_quality(stock_prices, fundamentals)
        
        return StockAnalysisDashboard(
            symbol=symbol,
            name=name,
            sector=sector,
            domain=domain_quality.domain.value if domain_quality.domain else None,
            recommendation=recommendation,
            confidence=confidence,
            summary=summary,
            scores=scores,
            technicals=technicals,
            regime=regime,
            fundamental_score=fundamental_score,
            fundamental_notes=self._extract_fundamental_notes(domain_quality),
            entry=entry_analysis,
            risk=risk_assessment,
            data_quality=data_quality,
            scoring_version=SCORING_VERSION,
        )
    
    def _compute_technical_score(
        self,
        technicals: TechnicalSnapshot,
        regime: RegimeState,
    ) -> float:
        """Compute technical score (0-100) from indicators."""
        score = 50.0  # Start neutral
        
        # Momentum contribution (±20 points)
        score += technicals.momentum_score * 20
        
        # Trend contribution (±20 points)
        score += technicals.trend_score * 20
        
        # RSI extremes (±10 points)
        if technicals.rsi_14 < 30:
            score += 10  # Oversold = bullish potential
        elif technicals.rsi_14 > 70:
            score -= 10  # Overbought = caution
        
        # ADX trend strength bonus
        if technicals.adx > 25:
            # Strong trend - amplify direction
            if technicals.trend_direction == "UP":
                score += 5
            elif technicals.trend_direction == "DOWN":
                score -= 5
        
        # Regime adjustment
        if regime.regime == MarketRegime.BEAR:
            # In bear markets, lower technical expectations
            score = max(score - 10, 0)
        
        return float(np.clip(score, 0, 100))
    
    def _compute_entry_analysis(
        self,
        stock_prices: pd.DataFrame,
        technicals: TechnicalSnapshot,
    ) -> EntryAnalysis:
        """Compute entry timing analysis."""
        # Normalize columns
        prices = stock_prices.copy()
        prices.columns = [c.lower() for c in prices.columns]
        close = prices.get("close", prices.get("adj close", prices.iloc[:, 0]))
        volume = prices.get("volume", pd.Series([0] * len(close), index=close.index))
        
        current_price = float(close.iloc[-1])
        
        # Calculate drawdown from 52-week high
        if len(close) >= 252:
            high_52w = float(close.rolling(252).max().iloc[-1])
            # Find how many days since that high
            high_idx = close.rolling(252).apply(lambda x: x.argmax(), raw=True).iloc[-1]
            days_since_high = int(252 - high_idx) if not pd.isna(high_idx) else 0
        else:
            high_52w = float(close.max())
            days_since_high = 0
        
        drawdown_pct = ((current_price / high_52w) - 1) * 100
        
        # Is this a dip entry opportunity?
        rsi_oversold = technicals.rsi_14 < 35
        is_dip = drawdown_pct < -10 and rsi_oversold
        
        # Volume capitulation check
        volume_capitulation = False
        if volume.sum() > 0:
            volume_sma = volume.rolling(20).mean().iloc[-1] if len(volume) >= 20 else volume.mean()
            if float(volume.iloc[-1]) > volume_sma * 2 and drawdown_pct < -10:
                volume_capitulation = True
        
        # Calculate optimal entry price (based on further dip potential)
        # Simple heuristic: 5% below current price as buffer
        optimal_entry = current_price * 0.95 if drawdown_pct < -10 else None
        
        return EntryAnalysis(
            current_drawdown_pct=drawdown_pct,
            is_dip_entry=is_dip,
            optimal_entry_price=optimal_entry,
            days_since_high=days_since_high,
            rsi_oversold=rsi_oversold,
            volume_capitulation=volume_capitulation,
        )
    
    def _compute_entry_timing_score(
        self,
        entry: EntryAnalysis,
        technicals: TechnicalSnapshot,
    ) -> float:
        """Compute entry timing score (0-100)."""
        score = 50.0
        
        # Dip bonus (buying at discount)
        if entry.current_drawdown_pct < -5:
            score += min(abs(entry.current_drawdown_pct) * 1.5, 25)  # Up to +25 for deep dips
        
        # RSI oversold bonus
        if technicals.rsi_14 < 30:
            score += 15
        elif technicals.rsi_14 < 40:
            score += 5
        elif technicals.rsi_14 > 70:
            score -= 15  # Penalty for overbought
        
        # Volume capitulation bonus
        if entry.volume_capitulation:
            score += 10
        
        # Bollinger band bonus (at lower band = good entry)
        if technicals.bollinger_pct_b < 0.2:
            score += 10
        elif technicals.bollinger_pct_b > 0.9:
            score -= 10
        
        # Stochastic oversold
        if technicals.stoch_k < 20:
            score += 5
        
        return float(np.clip(score, 0, 100))
    
    def _compute_risk_assessment(
        self,
        technicals: TechnicalSnapshot,
        domain_quality: DomainScoreResult,
        regime: RegimeState,
        entry: EntryAnalysis,
    ) -> RiskAssessment:
        """Compute risk assessment (0-100, higher = safer)."""
        score = 100.0
        factors = []
        
        # Volatility risk
        vol_penalties = {
            "LOW": 0,
            "NORMAL": 5,
            "HIGH": 15,
            "EXTREME": 25,
        }
        penalty = vol_penalties.get(technicals.volatility_regime, 10)
        score -= penalty
        if penalty >= 15:
            factors.append(f"High volatility ({technicals.volatility_regime})")
        
        # Fundamental quality risk
        if domain_quality.final_score < 40:
            score -= 20
            factors.append("Low fundamental quality")
        elif domain_quality.final_score < 55:
            score -= 10
            factors.append("Below-average fundamentals")
        
        # Data completeness risk
        if domain_quality.data_completeness < 0.5:
            score -= 10
            factors.append("Limited fundamental data")
        
        # Regime risk
        if regime.regime == MarketRegime.CRASH:
            score -= 15
            factors.append("Market crash regime")
        elif regime.regime == MarketRegime.BEAR:
            score -= 10
            factors.append("Bear market")
        
        # Drawdown risk (potential falling knife)
        if entry.current_drawdown_pct < -35:
            score -= 20
            factors.append(f"Severe drawdown ({entry.current_drawdown_pct:.0f}%)")
        elif entry.current_drawdown_pct < -25:
            score -= 10
            factors.append(f"Deep drawdown ({entry.current_drawdown_pct:.0f}%)")
        
        # Trend risk
        if technicals.death_cross:
            score -= 10
            factors.append("Death cross (SMA50 < SMA200)")
        
        # ADX weak trend risk
        if technicals.adx < 15 and technicals.trend_direction == "DOWN":
            factors.append("Weak downtrend (low conviction)")
        
        # Calculate max position size based on risk
        risk_score = float(np.clip(score, 0, 100))
        
        if risk_score >= 80:
            max_position = 10.0
        elif risk_score >= 60:
            max_position = 7.5
        elif risk_score >= 40:
            max_position = 5.0
        elif risk_score >= 20:
            max_position = 2.5
        else:
            max_position = 1.0
        
        # Suggested stop loss
        if regime.strategy_config.stop_loss_pct:
            stop_loss = regime.strategy_config.stop_loss_pct
        else:
            # Dynamic based on volatility
            if technicals.volatility_regime == "HIGH":
                stop_loss = 15.0
            elif technicals.volatility_regime == "EXTREME":
                stop_loss = None  # Wide stops or no stops in extreme vol
            else:
                stop_loss = 10.0
        
        return RiskAssessment(
            risk_score=risk_score,
            risk_factors=factors,
            max_position_pct=max_position,
            suggested_stop_loss_pct=stop_loss,
        )
    
    def _get_dynamic_weights(self, regime: RegimeState) -> dict[str, float]:
        """Get dynamic weights based on regime."""
        if regime.regime in (MarketRegime.BEAR, MarketRegime.CRASH):
            # Bear/crash market: fundamentals matter most
            return {
                "technical": 0.15,
                "fundamental": 0.40,
                "regime": 0.10,
                "entry": 0.25,
                "risk": 0.10,
            }
        elif regime.regime == MarketRegime.BULL:
            # Bull market: technicals matter more
            return {
                "technical": 0.30,
                "fundamental": 0.25,
                "regime": 0.15,
                "entry": 0.20,
                "risk": 0.10,
            }
        elif regime.regime == MarketRegime.CORRECTION:
            # Correction: balanced with emphasis on entry timing
            return {
                "technical": 0.20,
                "fundamental": 0.30,
                "regime": 0.10,
                "entry": 0.30,
                "risk": 0.10,
            }
        else:
            # Default weights
            return {
                "technical": self.config.technical_weight,
                "fundamental": self.config.fundamental_weight,
                "regime": self.config.regime_weight,
                "entry": self.config.entry_timing_weight,
                "risk": self.config.risk_weight,
            }
    
    def _determine_recommendation(
        self,
        scores: ScoreComponents,
        regime: RegimeState,
        entry_analysis: EntryAnalysis,
        risk_assessment: RiskAssessment,
        domain_quality: DomainScoreResult,
    ) -> tuple[Literal["STRONG_BUY", "BUY", "ACCUMULATE", "HOLD", "AVOID", "SELL"], str]:
        """Determine final recommendation and summary."""
        cfg = self.config
        composite = scores.composite
        
        # Special case: Bear/crash market accumulation
        if regime.regime in (MarketRegime.BEAR, MarketRegime.CRASH):
            if scores.fundamental < cfg.min_fundamental_for_bear:
                return (
                    "AVOID",
                    f"Bear market requires quality fundamentals (score {scores.fundamental:.0f} < {cfg.min_fundamental_for_bear}). Pass."
                )
            
            if entry_analysis.is_dip_entry and scores.fundamental >= 60:
                if entry_analysis.current_drawdown_pct < cfg.max_drawdown_for_entry:
                    return (
                        "ACCUMULATE",
                        f"Bear market dip opportunity. Quality score: {scores.fundamental:.0f}. "
                        f"Down {abs(entry_analysis.current_drawdown_pct):.0f}% from highs. Scale in carefully."
                    )
        
        # Falling knife protection
        if entry_analysis.current_drawdown_pct < cfg.max_drawdown_for_entry:
            if scores.fundamental < 55:
                return (
                    "AVOID",
                    f"Severe drawdown ({entry_analysis.current_drawdown_pct:.0f}%) with weak fundamentals. "
                    f"Potential falling knife."
                )
        
        # Risk gate
        if risk_assessment.risk_score < 25:
            return (
                "AVOID",
                f"Too many risk factors: {', '.join(risk_assessment.risk_factors[:3])}. Pass."
            )
        
        # Standard scoring thresholds
        if composite >= cfg.strong_buy_threshold:
            summary = f"Strong buy signal. Composite: {composite:.0f}. "
            if entry_analysis.is_dip_entry:
                summary += f"Dip entry at {entry_analysis.current_drawdown_pct:.0f}% discount."
            elif scores.technical >= 70:
                summary += f"Strong technicals (momentum: {scores.technical:.0f})."
            else:
                summary += f"High quality with favorable conditions."
            return ("STRONG_BUY", summary)
        
        elif composite >= cfg.buy_threshold:
            summary = f"Buy signal. Composite: {composite:.0f}. "
            if regime.regime == MarketRegime.BULL:
                summary += "Bull market conditions."
            elif entry_analysis.is_dip_entry:
                summary += f"Dip entry opportunity."
            return ("BUY", summary)
        
        elif composite >= cfg.accumulate_threshold:
            if entry_analysis.is_dip_entry:
                return (
                    "ACCUMULATE",
                    f"Gradual accumulation zone. Composite: {composite:.0f}. "
                    f"Down {abs(entry_analysis.current_drawdown_pct):.0f}%."
                )
            return (
                "HOLD",
                f"Neutral. Composite: {composite:.0f}. No clear edge."
            )
        
        elif composite >= cfg.hold_threshold:
            return (
                "HOLD",
                f"Below average. Composite: {composite:.0f}. Wait for better setup."
            )
        
        elif composite >= cfg.avoid_threshold:
            return (
                "AVOID",
                f"Weak composite score ({composite:.0f}). Multiple negative factors."
            )
        
        else:
            reasons = []
            if scores.technical < 40:
                reasons.append("weak technicals")
            if scores.fundamental < 40:
                reasons.append("poor fundamentals")
            if scores.regime < 40:
                reasons.append("unfavorable regime")
            
            return (
                "SELL",
                f"Sell signal. Composite: {composite:.0f}. Issues: {', '.join(reasons) or 'multiple factors'}."
            )
    
    def _calculate_confidence(
        self,
        data_completeness: float,
        price_history_len: int,
        volatility_regime: str,
    ) -> float:
        """Calculate confidence in the analysis (0-100)."""
        confidence = 50.0
        
        # Data completeness contributes up to 30 points
        confidence += data_completeness * 30
        
        # Price history contributes up to 15 points
        if price_history_len >= 252 * 3:  # 3+ years
            confidence += 15
        elif price_history_len >= 252:  # 1+ year
            confidence += 10
        elif price_history_len >= 120:  # 6+ months
            confidence += 5
        
        # Volatility reduces confidence
        vol_penalties = {"LOW": -5, "NORMAL": 0, "HIGH": -10, "EXTREME": -20}
        confidence += vol_penalties.get(volatility_regime, 0)
        
        return float(np.clip(confidence, 10, 95))
    
    def _assess_data_quality(
        self,
        stock_prices: pd.DataFrame,
        fundamentals: dict[str, Any] | None,
    ) -> Literal["HIGH", "MEDIUM", "LOW"]:
        """Assess overall data quality."""
        price_quality = len(stock_prices) >= 252  # At least 1 year
        
        fundamental_quality = False
        if fundamentals:
            key_fields = ["market_cap", "pe_ratio", "profit_margin", "revenue_growth"]
            available = sum(1 for k in key_fields if fundamentals.get(k) is not None)
            fundamental_quality = available >= 2
        
        if price_quality and fundamental_quality:
            return "HIGH"
        elif price_quality or fundamental_quality:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_fundamental_notes(self, domain_quality: DomainScoreResult) -> list[str]:
        """Extract key notes from domain quality result."""
        notes = []
        
        # Add notes from sub-scores
        for sub in domain_quality.sub_scores:
            if not sub.available:
                notes.append(f"Missing: {sub.name}")
            elif sub.score >= 80:
                notes.append(f"Strong: {sub.name}")
            elif sub.score <= 30:
                notes.append(f"Weak: {sub.name}")
        
        if domain_quality.fallback_used:
            notes.append("Using generic scoring (domain-specific data unavailable)")
        
        if domain_quality.notes:
            notes.append(domain_quality.notes)
        
        return notes[:5]  # Limit to 5 notes


# Singleton instance
_orchestrator: ScoringOrchestrator | None = None


def get_scoring_orchestrator(config: OrchestratorConfig | None = None) -> ScoringOrchestrator:
    """Get singleton ScoringOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None or config is not None:
        _orchestrator = ScoringOrchestrator(config)
    return _orchestrator
