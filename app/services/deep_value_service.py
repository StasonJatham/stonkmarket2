"""
Deep Value Alert Service

This service identifies and alerts when quality stocks drop significantly below
their intrinsic value, especially during market downturns (BEAR/CRASH regimes).

Key features:
1. Multiple intrinsic value methods (analyst, PEG, Graham, DCF)
2. Quality score assessment (fundamentals, moat, balance sheet)
3. Market regime awareness (BEAR/CRASH = opportunity)
4. Deep value alert when all conditions align

The key insight: "Buy wonderful companies at fair prices" (Buffett)
But even better: "Buy wonderful companies at UNFAIR prices during panic"
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class ValueStatus(str, Enum):
    """Valuation status relative to intrinsic value."""
    DEEPLY_UNDERVALUED = "deeply_undervalued"  # >40% upside
    UNDERVALUED = "undervalued"                # 20-40% upside
    FAIR_VALUE = "fair_value"                  # -10% to +20%
    OVERVALUED = "overvalued"                  # -10% to -30%
    EXTREMELY_OVERVALUED = "extremely_overvalued"  # <-30%


class QualityTier(str, Enum):
    """Quality tier based on fundamentals."""
    EXCEPTIONAL = "exceptional"  # Score >= 80
    HIGH = "high"                # Score 65-79
    MODERATE = "moderate"        # Score 50-64
    LOW = "low"                  # Score 30-49
    POOR = "poor"                # Score < 30


class AlertPriority(str, Enum):
    """Alert priority level."""
    CRITICAL = "critical"  # Exceptional quality + deeply undervalued + crash regime
    HIGH = "high"          # High quality + undervalued + bear/crash
    MEDIUM = "medium"      # Moderate quality + undervalued
    LOW = "low"            # Some discount but not compelling
    NONE = "none"          # No alert


class IntrinsicValueMethod(str, Enum):
    """Method used to calculate intrinsic value."""
    ANALYST = "analyst"
    PEG = "peg"
    GRAHAM = "graham"
    DCF = "dcf"
    COMPOSITE = "composite"


class IntrinsicValueEstimate(BaseModel):
    """Individual intrinsic value estimate from one method."""
    method: IntrinsicValueMethod
    value: float
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")
    reasoning: str
    
    
class DeepValueAlert(BaseModel):
    """Alert for a deep value opportunity."""
    symbol: str
    current_price: float
    
    # Intrinsic value
    intrinsic_value: float
    intrinsic_value_method: IntrinsicValueMethod
    upside_pct: float
    value_status: ValueStatus
    all_estimates: list[IntrinsicValueEstimate] = []
    
    # Quality assessment
    quality_score: float = Field(ge=0.0, le=100.0)
    quality_tier: QualityTier
    quality_factors: dict[str, Any] = {}
    
    # Market context
    market_regime: str = "UNKNOWN"
    regime_context: str = ""
    
    # Alert info
    priority: AlertPriority
    alert_reason: str
    action_recommendation: str
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_notification(self) -> str:
        """Generate human-readable notification text."""
        emoji = {
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.MEDIUM: "📊",
            AlertPriority.LOW: "📌",
            AlertPriority.NONE: "",
        }.get(self.priority, "")
        
        return (
            f"{emoji} DEEP VALUE ALERT: {self.symbol}\n"
            f"Price: ${self.current_price:.2f} | Target: ${self.intrinsic_value:.2f} ({self.upside_pct:+.0f}%)\n"
            f"Quality: {self.quality_tier.value.upper()} ({self.quality_score:.0f}/100)\n"
            f"Regime: {self.market_regime}\n"
            f"Action: {self.action_recommendation}"
        )


class DeepValueService:
    """
    Service for identifying deep value opportunities.
    
    Combines:
    - Multiple intrinsic value estimation methods
    - Quality/moat analysis
    - Market regime awareness
    - Alert generation
    """
    
    def __init__(
        self,
        min_upside_for_alert: float = 20.0,
        min_quality_for_alert: float = 50.0,
        discount_rate: float = 0.10,  # For DCF
    ) -> None:
        self.min_upside_for_alert = min_upside_for_alert
        self.min_quality_for_alert = min_quality_for_alert
        self.discount_rate = discount_rate
        
    def calculate_intrinsic_value(
        self,
        fundamentals: dict[str, Any],
        current_price: float,
    ) -> tuple[float | None, IntrinsicValueMethod | None, list[IntrinsicValueEstimate]]:
        """
        Calculate intrinsic value using multiple methods.
        
        Returns:
            (composite_value, primary_method, all_estimates)
        """
        import math
        
        estimates: list[IntrinsicValueEstimate] = []
        
        if not current_price or current_price <= 0:
            return None, None, estimates
        
        # Method 1: Analyst Target Price
        target_price = fundamentals.get("target_mean_price")
        num_analysts = fundamentals.get("num_analyst_opinions", 0) or 0
        if target_price and num_analysts >= 3:
            # Confidence scales with analyst count
            confidence = min(num_analysts / 20, 1.0)  # 20+ analysts = 100% confidence
            estimates.append(IntrinsicValueEstimate(
                method=IntrinsicValueMethod.ANALYST,
                value=round(target_price, 2),
                confidence=confidence,
                reasoning=f"Consensus of {num_analysts} analysts"
            ))
        
        # Method 2: PEG-based Fair Value
        peg_ratio = fundamentals.get("peg_ratio")
        pe_ratio = fundamentals.get("pe_ratio")
        if peg_ratio and pe_ratio and 0 < peg_ratio < 5 and 0 < pe_ratio < 100:
            try:
                growth_rate = pe_ratio / peg_ratio
                fair_pe = growth_rate  # Fair P/E at PEG = 1
                eps = current_price / pe_ratio
                fair_value = eps * fair_pe
                if 0 < fair_value < current_price * 5:
                    estimates.append(IntrinsicValueEstimate(
                        method=IntrinsicValueMethod.PEG,
                        value=round(fair_value, 2),
                        confidence=0.6,  # Medium confidence
                        reasoning=f"PEG-based with growth rate {growth_rate:.1f}%"
                    ))
            except (ValueError, ZeroDivisionError):
                pass
        
        # Method 3: Graham Number
        price_to_book = fundamentals.get("price_to_book")
        if pe_ratio and price_to_book and pe_ratio > 0 and price_to_book > 0:
            try:
                eps = current_price / pe_ratio
                bvps = current_price / price_to_book
                if eps > 0 and bvps > 0:
                    graham = math.sqrt(22.5 * eps * bvps)
                    if 0 < graham < current_price * 5:
                        estimates.append(IntrinsicValueEstimate(
                            method=IntrinsicValueMethod.GRAHAM,
                            value=round(graham, 2),
                            confidence=0.5,  # Conservative estimate
                            reasoning=f"Graham Number formula"
                        ))
            except (ValueError, ZeroDivisionError):
                pass
        
        # Method 4: Simple DCF (owner earnings model)
        fcf = fundamentals.get("free_cash_flow")
        if fcf:
            try:
                # Parse FCF string like "1.2B" or numeric
                fcf_val = self._parse_financial_value(fcf)
                shares_outstanding = fundamentals.get("shares_outstanding")
                if fcf_val and fcf_val > 0 and shares_outstanding and shares_outstanding > 0:
                    # Simple 10-year DCF with terminal value
                    growth_rate = 0.05  # Conservative 5% growth
                    terminal_multiple = 15  # 15x terminal FCF
                    
                    fcf_per_share = fcf_val / shares_outstanding
                    
                    # NPV of 10 years of FCF
                    npv = sum(
                        (fcf_per_share * (1 + growth_rate) ** i) / (1 + self.discount_rate) ** i
                        for i in range(1, 11)
                    )
                    
                    # Terminal value
                    terminal_fcf = fcf_per_share * (1 + growth_rate) ** 10
                    terminal_value = (terminal_fcf * terminal_multiple) / (1 + self.discount_rate) ** 10
                    
                    dcf_value = npv + terminal_value
                    if 0 < dcf_value < current_price * 5:
                        estimates.append(IntrinsicValueEstimate(
                            method=IntrinsicValueMethod.DCF,
                            value=round(dcf_value, 2),
                            confidence=0.4,  # Lower confidence - many assumptions
                            reasoning=f"10-year DCF with 5% growth, 10% discount rate"
                        ))
            except (ValueError, ZeroDivisionError, TypeError):
                pass
        
        # No estimates possible
        if not estimates:
            return None, None, estimates
        
        # Sort by confidence (highest first)
        sorted_estimates = sorted(estimates, key=lambda e: e.confidence, reverse=True)
        primary = sorted_estimates[0]
        
        # If we have a high-confidence estimate (analyst with many analysts), use it directly
        # Otherwise, use weighted average of estimates within reasonable range
        if primary.confidence >= 0.8:
            # Use the highest confidence estimate directly
            return primary.value, primary.method, estimates
        
        # Filter out extreme outliers (>3x or <0.3x median)
        values = [e.value for e in estimates]
        median_val = sorted(values)[len(values) // 2]
        
        valid_estimates = [
            e for e in estimates 
            if 0.3 * median_val <= e.value <= 3 * median_val
        ]
        
        if not valid_estimates:
            # All estimates are outliers, use primary
            return primary.value, primary.method, estimates
        
        # Calculate weighted average of valid estimates
        total_weight = sum(e.confidence for e in valid_estimates)
        if total_weight > 0:
            weighted_value = sum(e.value * e.confidence for e in valid_estimates) / total_weight
            return round(weighted_value, 2), primary.method, estimates
        
        return primary.value, primary.method, estimates
    
    def calculate_quality_score(
        self,
        fundamentals: dict[str, Any],
    ) -> tuple[float, QualityTier, dict[str, Any]]:
        """
        Calculate quality score (0-100) based on fundamentals.
        
        Factors:
        - Profitability (margins, ROE)
        - Financial health (debt, current ratio)
        - Growth (revenue, earnings)
        - Moat indicators (margins stability, market position)
        """
        score = 50.0  # Start at neutral
        factors: dict[str, Any] = {}
        
        # 1. Profitability (max 25 points)
        profit_margin = self._parse_percentage(fundamentals.get("profit_margin"))
        if profit_margin is not None:
            if profit_margin > 20:
                score += 10
                factors["profit_margin"] = {"value": profit_margin, "signal": "excellent"}
            elif profit_margin > 10:
                score += 5
                factors["profit_margin"] = {"value": profit_margin, "signal": "good"}
            elif profit_margin < 0:
                score -= 15
                factors["profit_margin"] = {"value": profit_margin, "signal": "negative"}
            else:
                factors["profit_margin"] = {"value": profit_margin, "signal": "fair"}
        
        roe = self._parse_percentage(fundamentals.get("return_on_equity"))
        if roe is not None:
            if roe > 20:
                score += 10
                factors["roe"] = {"value": roe, "signal": "excellent"}
            elif roe > 15:
                score += 5
                factors["roe"] = {"value": roe, "signal": "good"}
            elif roe < 5:
                score -= 10
                factors["roe"] = {"value": roe, "signal": "poor"}
            else:
                factors["roe"] = {"value": roe, "signal": "fair"}
        
        gross_margin = self._parse_percentage(fundamentals.get("gross_margin"))
        if gross_margin is not None:
            if gross_margin > 50:
                score += 5
                factors["gross_margin"] = {"value": gross_margin, "signal": "strong moat"}
            elif gross_margin > 30:
                score += 2
                factors["gross_margin"] = {"value": gross_margin, "signal": "good"}
        
        # 2. Financial Health (max 25 points)
        debt_to_equity = self._parse_ratio(fundamentals.get("debt_to_equity"))
        if debt_to_equity is not None:
            if debt_to_equity < 0.3:
                score += 10
                factors["debt_to_equity"] = {"value": debt_to_equity, "signal": "fortress balance sheet"}
            elif debt_to_equity < 0.7:
                score += 5
                factors["debt_to_equity"] = {"value": debt_to_equity, "signal": "healthy"}
            elif debt_to_equity > 2:
                score -= 15
                factors["debt_to_equity"] = {"value": debt_to_equity, "signal": "high leverage"}
            else:
                factors["debt_to_equity"] = {"value": debt_to_equity, "signal": "moderate"}
        
        current_ratio = self._parse_ratio(fundamentals.get("current_ratio"))
        if current_ratio is not None:
            if current_ratio > 2:
                score += 5
                factors["current_ratio"] = {"value": current_ratio, "signal": "very liquid"}
            elif current_ratio > 1.5:
                score += 3
                factors["current_ratio"] = {"value": current_ratio, "signal": "healthy"}
            elif current_ratio < 1:
                score -= 10
                factors["current_ratio"] = {"value": current_ratio, "signal": "liquidity risk"}
        
        # 3. Growth (max 20 points)
        rev_growth = self._parse_percentage(fundamentals.get("revenue_growth"))
        if rev_growth is not None:
            if rev_growth > 20:
                score += 10
                factors["revenue_growth"] = {"value": rev_growth, "signal": "high growth"}
            elif rev_growth > 10:
                score += 5
                factors["revenue_growth"] = {"value": rev_growth, "signal": "solid growth"}
            elif rev_growth < 0:
                score -= 5
                factors["revenue_growth"] = {"value": rev_growth, "signal": "declining"}
        
        earnings_growth = self._parse_percentage(fundamentals.get("earnings_growth"))
        if earnings_growth is not None:
            if earnings_growth > 25:
                score += 10
                factors["earnings_growth"] = {"value": earnings_growth, "signal": "excellent"}
            elif earnings_growth > 10:
                score += 5
                factors["earnings_growth"] = {"value": earnings_growth, "signal": "good"}
            elif earnings_growth < 0:
                score -= 5
                factors["earnings_growth"] = {"value": earnings_growth, "signal": "declining"}
        
        # 4. Valuation sanity check (max 10 points)
        pe_ratio = fundamentals.get("pe_ratio")
        if pe_ratio:
            if 5 < pe_ratio < 15:
                score += 5
                factors["pe_ratio"] = {"value": pe_ratio, "signal": "value"}
            elif 15 <= pe_ratio <= 25:
                score += 2
                factors["pe_ratio"] = {"value": pe_ratio, "signal": "fair"}
            elif pe_ratio > 50:
                score -= 10
                factors["pe_ratio"] = {"value": pe_ratio, "signal": "expensive"}
            elif pe_ratio < 0:
                score -= 15
                factors["pe_ratio"] = {"value": pe_ratio, "signal": "negative earnings"}
        
        # Clamp score
        score = max(0, min(100, score))
        
        # Determine tier
        if score >= 80:
            tier = QualityTier.EXCEPTIONAL
        elif score >= 65:
            tier = QualityTier.HIGH
        elif score >= 50:
            tier = QualityTier.MODERATE
        elif score >= 30:
            tier = QualityTier.LOW
        else:
            tier = QualityTier.POOR
        
        return score, tier, factors
    
    def get_market_regime(
        self,
        spy_prices: pd.Series | None = None,
    ) -> tuple[str, str]:
        """
        Detect current market regime using SPY data.
        
        Returns:
            (regime_name, context_description)
        """
        if spy_prices is None or len(spy_prices) < 200:
            return "UNKNOWN", "Insufficient data"
        
        try:
            from app.quant_engine.core import get_regime_service
            
            # Use unified RegimeService
            regime_service = get_regime_service()
            
            # Get current regime from SPY data
            spy_df = pd.DataFrame({"Close": spy_prices})
            regime_state = regime_service.get_current_regime(spy_df)
            
            context = {
                "BULL": "Uptrend - be selective, quality over discount",
                "BEAR": "Downtrend - opportunities emerging, wait for capitulation",
                "CRASH": "Crisis - maximum opportunity, deploy capital aggressively",
                "RECOVERY": "Early recovery - best risk/reward window",
            }.get(regime_state.regime.value, "Unknown regime")
            
            return regime_state.regime.value, context
        except Exception:
            return "UNKNOWN", "Could not detect regime"
    
    def generate_alert(
        self,
        symbol: str,
        current_price: float,
        fundamentals: dict[str, Any],
        spy_prices: pd.Series | None = None,
    ) -> DeepValueAlert | None:
        """
        Generate a deep value alert if conditions are met.
        
        Returns:
            DeepValueAlert if opportunity exists, None otherwise
        """
        # Calculate intrinsic value
        intrinsic_value, method, estimates = self.calculate_intrinsic_value(
            fundamentals, current_price
        )
        
        if not intrinsic_value or not method:
            return None
        
        # Calculate upside
        upside_pct = ((intrinsic_value / current_price) - 1) * 100
        
        # Determine value status
        if upside_pct >= 40:
            value_status = ValueStatus.DEEPLY_UNDERVALUED
        elif upside_pct >= 20:
            value_status = ValueStatus.UNDERVALUED
        elif upside_pct >= -10:
            value_status = ValueStatus.FAIR_VALUE
        elif upside_pct >= -30:
            value_status = ValueStatus.OVERVALUED
        else:
            value_status = ValueStatus.EXTREMELY_OVERVALUED
        
        # Calculate quality
        quality_score, quality_tier, quality_factors = self.calculate_quality_score(fundamentals)
        
        # Get market regime
        regime, regime_context = self.get_market_regime(spy_prices)
        
        # Determine alert priority
        priority = self._determine_priority(
            upside_pct, quality_score, quality_tier, regime
        )
        
        # Generate alert reason
        alert_reason = self._generate_alert_reason(
            symbol, upside_pct, quality_tier, regime, value_status
        )
        
        # Action recommendation
        action = self._generate_action(
            priority, upside_pct, quality_tier, regime
        )
        
        return DeepValueAlert(
            symbol=symbol,
            current_price=current_price,
            intrinsic_value=intrinsic_value,
            intrinsic_value_method=method,
            upside_pct=round(upside_pct, 1),
            value_status=value_status,
            all_estimates=estimates,
            quality_score=quality_score,
            quality_tier=quality_tier,
            quality_factors=quality_factors,
            market_regime=regime,
            regime_context=regime_context,
            priority=priority,
            alert_reason=alert_reason,
            action_recommendation=action,
        )
    
    def _determine_priority(
        self,
        upside_pct: float,
        quality_score: float,
        quality_tier: QualityTier,
        regime: str,
    ) -> AlertPriority:
        """Determine alert priority based on all factors."""
        
        # No alert if not undervalued
        if upside_pct < self.min_upside_for_alert:
            return AlertPriority.NONE
        
        # No alert if poor quality
        if quality_score < self.min_quality_for_alert:
            return AlertPriority.NONE
        
        # CRITICAL: Exceptional quality + deeply undervalued + crash/bear
        if (
            quality_tier == QualityTier.EXCEPTIONAL and
            upside_pct >= 40 and
            regime in ["CRASH", "BEAR"]
        ):
            return AlertPriority.CRITICAL
        
        # HIGH: High quality + significant upside + favorable regime
        if (
            quality_tier in [QualityTier.EXCEPTIONAL, QualityTier.HIGH] and
            upside_pct >= 30 and
            regime in ["CRASH", "BEAR", "RECOVERY"]
        ):
            return AlertPriority.HIGH
        
        # MEDIUM: Decent quality + undervalued
        if (
            quality_tier in [QualityTier.EXCEPTIONAL, QualityTier.HIGH, QualityTier.MODERATE] and
            upside_pct >= 20
        ):
            return AlertPriority.MEDIUM
        
        # LOW: Some discount
        if upside_pct >= self.min_upside_for_alert:
            return AlertPriority.LOW
        
        return AlertPriority.NONE
    
    def _generate_alert_reason(
        self,
        symbol: str,
        upside_pct: float,
        quality_tier: QualityTier,
        regime: str,
        value_status: ValueStatus,
    ) -> str:
        """Generate human-readable alert reason."""
        
        quality_desc = {
            QualityTier.EXCEPTIONAL: "exceptional fundamentals",
            QualityTier.HIGH: "strong fundamentals",
            QualityTier.MODERATE: "decent fundamentals",
            QualityTier.LOW: "weak fundamentals",
            QualityTier.POOR: "poor fundamentals",
        }[quality_tier]
        
        regime_desc = {
            "CRASH": "market crash creating opportunity",
            "BEAR": "bear market providing discount",
            "RECOVERY": "early recovery phase",
            "BULL": "bull market",
            "UNKNOWN": "unknown market conditions",
        }.get(regime, "")
        
        return (
            f"{symbol} is {value_status.value.replace('_', ' ')} "
            f"({upside_pct:+.0f}% upside) with {quality_desc}. "
            f"Market: {regime_desc}."
        )
    
    def _generate_action(
        self,
        priority: AlertPriority,
        upside_pct: float,
        quality_tier: QualityTier,
        regime: str,
    ) -> str:
        """Generate action recommendation."""
        
        if priority == AlertPriority.CRITICAL:
            return "STRONG BUY - Deploy 5-10% of portfolio. Historic opportunity."
        elif priority == AlertPriority.HIGH:
            return "BUY - Consider 2-5% position. Excellent risk/reward."
        elif priority == AlertPriority.MEDIUM:
            return "WATCH/ACCUMULATE - Good value. Start small position."
        elif priority == AlertPriority.LOW:
            return "MONITOR - Add to watchlist. Wait for better entry."
        else:
            return "NO ACTION - Does not meet criteria."
    
    def _parse_financial_value(self, value: Any) -> float | None:
        """Parse financial value like '1.2B' or '500M' to float."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).strip().upper()
        multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
        
        for suffix, mult in multipliers.items():
            if value_str.endswith(suffix):
                try:
                    return float(value_str[:-1]) * mult
                except ValueError:
                    return None
        
        try:
            return float(value_str.replace(",", ""))
        except ValueError:
            return None
    
    def _parse_percentage(self, value: Any) -> float | None:
        """Parse percentage value like '15.5%' or 0.155 to percentage float."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            # Values > 1 are likely already percentages (e.g., 15.5 means 15.5%)
            # Values <= 1 are likely decimals (e.g., 0.155 means 15.5%)
            # Exception: ratios like ROE can be > 1 (e.g., 1.71 = 171%)
            if abs(value) <= 1:
                return float(value) * 100
            else:
                return float(value)  # Already a percentage
        
        value_str = str(value).strip().rstrip("%")
        try:
            return float(value_str)
        except ValueError:
            return None
    
    def _parse_ratio(self, value: Any) -> float | None:
        """Parse ratio value."""
        if value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        try:
            return float(str(value).strip())
        except ValueError:
            return None


# Convenience function for quick checks
def check_deep_value(
    symbol: str,
    fundamentals: dict[str, Any],
    current_price: float,
    spy_prices: pd.Series | None = None,
) -> DeepValueAlert | None:
    """
    Quick check for deep value opportunity.
    
    Usage:
        alert = check_deep_value("AAPL", fundamentals, 170.0)
        if alert and alert.priority != AlertPriority.NONE:
            print(alert.to_notification())
    """
    service = DeepValueService()
    return service.generate_alert(symbol, current_price, fundamentals, spy_prices)
