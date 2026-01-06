"""
ScoringOrchestrator - Unified entry point for all stock scoring.

V3 ARCHITECTURE:
1. DomainScoringAdapter: Quality score (0-100) from fundamentals
2. RegimeService: Market-wide regime (Bull/Bear/Crash)
3. SectorRegimeService: Sector-specific regime (NEW)
4. EntryTriggerService: BUY/WAIT signals (NEW)
5. EventRiskService: Earnings/dividend awareness (NEW)
6. LiquidityGate: Volume adequacy (NEW)

Output: StockAnalysisDashboard with UI-ready badges and chart markers.
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
# V3 Services
from app.quant_engine.core.sector_regime import (
    SectorRegimeService,
    SectorRegimeState,
    get_sector_regime_service,
)
from app.quant_engine.core.entry_trigger import (
    EntryTriggerService,
    EntryTriggerState,
    get_entry_trigger_service,
)
from app.quant_engine.core.event_risk import (
    EventRiskService,
    EventRiskState,
    get_event_risk_service,
)
from app.quant_engine.core.liquidity_gate import (
    LiquidityGate,
    LiquidityState,
    get_liquidity_gate,
)
from app.quant_engine.scoring.output import (
    StockAnalysisDashboard,
    ScoreComponents,
    EntryAnalysis,
    RiskAssessment,
    BadgeInfo,
    ChartMarker,
)

# Import domain scoring - the source of truth for fundamental quality
from app.quant_engine.dipfinder.domain import Domain, classify_domain
from app.quant_engine.dipfinder.domain_scoring import (
    DomainScoreResult,
    get_adapter,
    compute_domain_score,
)

logger = logging.getLogger(__name__)

SCORING_VERSION = "3.1.0"


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
    
    V3 Architecture integrates:
    - TechnicalService: Price/volume indicators
    - RegimeService: Market-wide regime
    - SectorRegimeService: Sector-specific momentum
    - EntryTriggerService: BUY/WAIT signals
    - EventRiskService: Earnings/dividend awareness
    - LiquidityGate: Volume adequacy
    
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
        sector_regime_service: SectorRegimeService | None = None,
        entry_trigger_service: EntryTriggerService | None = None,
        event_risk_service: EventRiskService | None = None,
        liquidity_gate: LiquidityGate | None = None,
    ):
        self.config = config or OrchestratorConfig()
        self.tech_service = technical_service or get_technical_service()
        self.regime_service = regime_service or get_regime_service()
        # V3 services
        self.sector_regime_service = sector_regime_service or get_sector_regime_service()
        self.entry_trigger_service = entry_trigger_service or get_entry_trigger_service()
        self.event_risk_service = event_risk_service or get_event_risk_service()
        self.liquidity_gate = liquidity_gate or get_liquidity_gate()
    
    async def analyze(
        self,
        symbol: str,
        stock_prices: pd.DataFrame,
        spy_prices: pd.DataFrame,
        fundamentals: dict[str, Any] | None = None,
        name: str | None = None,
        sector: str | None = None,
        vix_level: float | None = None,
        # V3 parameters
        sector_prices: pd.DataFrame | None = None,
        next_earnings_date: date | None = None,
        ex_dividend_date: date | None = None,
    ) -> StockAnalysisDashboard:
        """
        Perform complete stock analysis with V3 gates and triggers.
        
        Args:
            symbol: Stock ticker
            stock_prices: OHLCV data for the stock
            spy_prices: OHLCV data for SPY (market benchmark)
            fundamentals: Dict of fundamental metrics (from yfinance or stored)
            name: Company name
            sector: Sector classification
            vix_level: Optional current VIX level
            sector_prices: OHLCV data for sector ETF (e.g., XLK for tech)
            next_earnings_date: Next earnings announcement date
            ex_dividend_date: Next ex-dividend date
            
        Returns:
            StockAnalysisDashboard with complete analysis including V3 fields
        """
        logger.debug(f"Analyzing {symbol} (V3)")
        
        # Get technical snapshot
        technicals = self.tech_service.get_snapshot(stock_prices)
        
        # Get market regime
        regime = self.regime_service.get_current_regime(spy_prices, vix_level)
        
        # ===== V3: Sector Regime =====
        sector_regime: SectorRegimeState | None = None
        if sector and sector_prices is not None and len(sector_prices) > 0:
            sector_regime = self.sector_regime_service.get_sector_regime(
                sector=sector,
                sector_etf_prices=sector_prices,
                spy_prices=spy_prices,
            )
        
        # ===== V3: Event Risk =====
        event_risk = self.event_risk_service.analyze_event_risk(
            earnings_date=next_earnings_date,
            ex_dividend_date=ex_dividend_date,
        )
        
        # ===== V3: Liquidity Gate =====
        prices = stock_prices.copy()
        prices.columns = [c.lower() for c in prices.columns]
        close = prices.get("close", prices.get("adj close", prices.iloc[:, 0]))
        volume = prices.get("volume", pd.Series([0] * len(close), index=close.index))
        current_price = float(close.iloc[-1]) if len(close) > 0 else 0.0
        avg_volume = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
        
        liquidity = self.liquidity_gate.check_liquidity(
            volume_series=None,
            price=current_price,
            avg_volume=avg_volume,
        )
        
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
        
        # ===== V3: Entry Trigger =====
        entry_trigger = self.entry_trigger_service.analyze_entry(
            technicals=technicals,
            current_drawdown_pct=entry_analysis.current_drawdown_pct,
        )
        
        # Compute composite score with dynamic weights
        weights = self._get_dynamic_weights(regime)
        base_composite = (
            technical_score * weights["technical"] +
            fundamental_score * weights["fundamental"] +
            regime_score * weights["regime"] +
            entry_timing_score * weights["entry"] +
            risk_assessment.risk_score * weights["risk"]
        )
        
        # ===== V3: Apply multipliers =====
        # Sector regime multiplier (0.5-1.15)
        sector_multiplier = sector_regime.score_multiplier if sector_regime else 1.0
        
        # Event risk multiplier (0.0-1.0) - blocks trades before earnings
        event_multiplier = event_risk.score_multiplier
        
        # Final composite after V3 adjustments
        composite_score = base_composite * sector_multiplier * event_multiplier
        
        # Log V3 adjustments if they significantly impact score
        if sector_multiplier != 1.0 or event_multiplier != 1.0:
            logger.debug(
                f"{symbol}: base={base_composite:.1f}, "
                f"sector_mult={sector_multiplier:.2f}, "
                f"event_mult={event_multiplier:.2f}, "
                f"final={composite_score:.1f}"
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
        
        # ===== V3: Generate Badges =====
        badges = self._generate_badges(
            recommendation=None,  # Will be set after determination
            entry_trigger=entry_trigger,
            event_risk=event_risk,
            liquidity=liquidity,
            sector_regime=sector_regime,
            technicals=technicals,
        )
        
        # ===== V3: Generate Chart Markers =====
        chart_markers = self._generate_chart_markers(
            entry_trigger=entry_trigger,
            entry_analysis=entry_analysis,
            current_price=current_price,
        )
        
        # Determine recommendation
        recommendation, summary = self._determine_recommendation(
            scores=scores,
            regime=regime,
            entry_analysis=entry_analysis,
            risk_assessment=risk_assessment,
            domain_quality=domain_quality,
            entry_trigger=entry_trigger,  # V3: influence recommendation
            event_risk=event_risk,  # V3: block if earnings imminent
            liquidity=liquidity,  # V3: warn if illiquid
        )
        
        # Update badges with recommendation
        badges = self._generate_badges(
            recommendation=recommendation,
            entry_trigger=entry_trigger,
            event_risk=event_risk,
            liquidity=liquidity,
            sector_regime=sector_regime,
            technicals=technicals,
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
            # V3 fields - convert state objects to dicts
            sector_regime=sector_regime.to_dict() if sector_regime else None,
            entry_trigger=entry_trigger.to_dict() if entry_trigger else None,
            event_risk=event_risk.to_dict() if event_risk else None,
            liquidity=liquidity.to_dict() if liquidity else None,
            badges=badges,
            chart_markers=chart_markers,
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
        # V3 parameters
        entry_trigger: EntryTriggerState | None = None,
        event_risk: EventRiskState | None = None,
        liquidity: LiquidityState | None = None,
    ) -> tuple[Literal["STRONG_BUY", "BUY", "ACCUMULATE", "HOLD", "AVOID", "SELL"], str]:
        """Determine final recommendation and summary with V3 gates."""
        cfg = self.config
        composite = scores.composite
        
        # ===== V3: Event Risk Gate =====
        # Block trades before earnings (gambling prevention)
        if event_risk and event_risk.risk_level == "BLOCKED":
            return (
                "HOLD",
                f"⚠️ Earnings in {event_risk.days_to_earnings or 0} days. "
                f"Wait until after {event_risk.next_earnings_date} to avoid binary event risk."
            )
        
        # ===== V3: Liquidity Gate =====
        # Warn if stock is illiquid
        if liquidity and liquidity.liquidity_tier == "ILLIQUID":
            return (
                "AVOID",
                f"Insufficient liquidity. Daily volume: ${liquidity.dollar_volume_daily:,.0f}. "
                f"Minimum required: $500K. Risk of slippage on entry/exit."
            )
        
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
            # V3: Entry trigger influence
            if entry_trigger and entry_trigger.signal == "BUY_NOW":
                triggers = ", ".join(entry_trigger.active_triggers[:2])
                summary += f"🎯 BUY NOW ({triggers}). "
            elif entry_analysis.is_dip_entry:
                summary += f"Dip entry at {entry_analysis.current_drawdown_pct:.0f}% discount."
            elif scores.technical >= 70:
                summary += f"Strong technicals (momentum: {scores.technical:.0f})."
            else:
                summary += f"High quality with favorable conditions."
            return ("STRONG_BUY", summary)
        
        elif composite >= cfg.buy_threshold:
            summary = f"Buy signal. Composite: {composite:.0f}. "
            # V3: Entry trigger refinement
            if entry_trigger and entry_trigger.signal == "BUY_NOW":
                summary += f"🎯 Entry triggers active. "
            elif entry_trigger and entry_trigger.signal == "BUY_ZONE":
                summary += f"In buy zone (scale in). "
            elif regime.regime == MarketRegime.BULL:
                summary += "Bull market conditions."
            elif entry_analysis.is_dip_entry:
                summary += f"Dip entry opportunity."
            return ("BUY", summary)
        
        elif composite >= cfg.accumulate_threshold:
            # V3: Wait signal overrides accumulate
            if entry_trigger and entry_trigger.signal == "WAIT":
                return (
                    "HOLD",
                    f"Composite {composite:.0f} is decent, but entry timing is unfavorable. "
                    f"Wait for better setup (RSI: {entry_trigger.rsi:.0f})."
                )
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
    
    def _generate_badges(
        self,
        recommendation: str | None,
        entry_trigger: EntryTriggerState | None,
        event_risk: EventRiskState | None,
        liquidity: LiquidityState | None,
        sector_regime: SectorRegimeState | None,
        technicals: TechnicalSnapshot,
    ) -> list[BadgeInfo]:
        """Generate V3 UI badges based on analysis state."""
        badges: list[BadgeInfo] = []
        
        # Entry trigger badges
        if entry_trigger:
            if entry_trigger.signal == "BUY_NOW":
                badges.append(BadgeInfo(
                    text="BUY NOW",
                    color="green",
                    tooltip=f"Active triggers: {', '.join(entry_trigger.active_triggers)}",
                    icon="zap",
                ))
            elif entry_trigger.signal == "BUY_ZONE":
                badges.append(BadgeInfo(
                    text="BUY ZONE",
                    color="blue",
                    tooltip="Good entry zone - scale in",
                    icon="target",
                ))
            elif entry_trigger.signal == "WAIT":
                badges.append(BadgeInfo(
                    text="WAIT",
                    color="yellow",
                    tooltip="Wait for better entry",
                    icon="clock",
                ))
        
        # Event risk badges
        if event_risk:
            if event_risk.risk_level == "BLOCKED":
                badges.append(BadgeInfo(
                    text=f"EARNINGS {event_risk.days_to_earnings}D",
                    color="red",
                    tooltip=f"Earnings on {event_risk.next_earnings_date}. Hold until after.",
                    icon="alert-triangle",
                ))
            elif event_risk.risk_level == "CAUTION":
                badges.append(BadgeInfo(
                    text="EARNINGS SOON",
                    color="yellow",
                    tooltip=f"Earnings in {event_risk.days_to_earnings} days. Consider position size.",
                    icon="calendar",
                ))
            if event_risk.ex_dividend_date and event_risk.days_to_ex_dividend is not None:
                if event_risk.days_to_ex_dividend <= 7:
                    badges.append(BadgeInfo(
                        text="EX-DIV SOON",
                        color="purple",
                        tooltip=f"Ex-dividend: {event_risk.ex_dividend_date}",
                        icon="dollar-sign",
                    ))
        
        # Liquidity badges
        if liquidity:
            if liquidity.liquidity_tier == "ILLIQUID":
                badges.append(BadgeInfo(
                    text="LOW VOLUME",
                    color="red",
                    tooltip=f"Daily volume: ${liquidity.dollar_volume_daily:,.0f}. Exit risk.",
                    icon="alert-circle",
                ))
            elif liquidity.liquidity_tier == "POOR":
                badges.append(BadgeInfo(
                    text="THIN VOLUME",
                    color="yellow",
                    tooltip=f"Limited liquidity. Max position: ${liquidity.max_position_dollars:,.0f}",
                    icon="droplet",
                ))
        
        # Sector regime badges
        if sector_regime:
            if sector_regime.regime == "CRISIS":
                badges.append(BadgeInfo(
                    text="SECTOR CRISIS",
                    color="red",
                    tooltip=f"Sector in crisis mode. Score reduced by {(1-sector_regime.score_multiplier)*100:.0f}%",
                    icon="trending-down",
                ))
            elif sector_regime.regime == "WEAK":
                badges.append(BadgeInfo(
                    text="WEAK SECTOR",
                    color="yellow",
                    tooltip=f"Sector underperforming market",
                    icon="arrow-down",
                ))
            elif sector_regime.regime == "STRONG":
                badges.append(BadgeInfo(
                    text="STRONG SECTOR",
                    color="green",
                    tooltip=f"Sector outperforming market (+{(sector_regime.score_multiplier-1)*100:.0f}% boost)",
                    icon="arrow-up",
                ))
        
        # Technical badges
        if technicals.death_cross:
            badges.append(BadgeInfo(
                text="DEATH CROSS",
                color="red",
                tooltip="SMA50 crossed below SMA200",
                icon="x-circle",
            ))
        elif technicals.golden_cross:
            badges.append(BadgeInfo(
                text="GOLDEN CROSS",
                color="green",
                tooltip="SMA50 crossed above SMA200",
                icon="sun",
            ))
        
        if technicals.volatility_regime == "EXTREME":
            badges.append(BadgeInfo(
                text="HIGH VOL",
                color="orange",
                tooltip=f"Extreme volatility regime",
                icon="activity",
            ))
        
        return badges[:6]  # Limit to 6 badges
    
    def _generate_chart_markers(
        self,
        entry_trigger: EntryTriggerState | None,
        entry_analysis: EntryAnalysis,
        current_price: float,
    ) -> list[ChartMarker]:
        """Generate V3 chart markers for UI visualization."""
        markers: list[ChartMarker] = []
        
        # Entry zone markers
        if entry_trigger and entry_trigger.entry_zone_low and entry_trigger.entry_zone_high:
            markers.append(ChartMarker(
                price=entry_trigger.entry_zone_low,
                marker_type="entry_zone_low",
                label=f"Entry Low: ${entry_trigger.entry_zone_low:.2f}",
                color="green",
            ))
            markers.append(ChartMarker(
                price=entry_trigger.entry_zone_high,
                marker_type="entry_zone_high",
                label=f"Entry High: ${entry_trigger.entry_zone_high:.2f}",
                color="blue",
            ))
        
        # Optimal entry marker
        if entry_analysis.optimal_entry_price:
            markers.append(ChartMarker(
                price=entry_analysis.optimal_entry_price,
                marker_type="optimal_entry",
                label=f"Target: ${entry_analysis.optimal_entry_price:.2f}",
                color="gold",
            ))
        
        # Current price marker
        if current_price > 0:
            markers.append(ChartMarker(
                price=current_price,
                marker_type="current_price",
                label=f"Now: ${current_price:.2f}",
                color="white",
            ))
        
        return markers


# Singleton instance
_orchestrator: ScoringOrchestrator | None = None


def get_scoring_orchestrator(config: OrchestratorConfig | None = None) -> ScoringOrchestrator:
    """Get singleton ScoringOrchestrator instance."""
    global _orchestrator
    if _orchestrator is None or config is not None:
        _orchestrator = ScoringOrchestrator(config)
    return _orchestrator
