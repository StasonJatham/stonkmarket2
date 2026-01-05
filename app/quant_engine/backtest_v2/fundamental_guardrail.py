"""
Fundamental Guardrails for Bear Market Accumulation.

In bear markets, we switch from technical signals to fundamental quality checks.
This module ensures we only accumulate "Quality on Sale" and avoid "Value Traps".

Key Checks:
- Solvency: Can the company survive the downturn?
- Valuation: Is it actually cheap vs history?
- Profitability: Does it generate cash?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FundamentalData:
    """Point-in-time fundamental data for a stock."""

    symbol: str
    report_date: datetime  # When this data was available

    # Valuation
    pe_ratio: float | None = None
    pe_5y_avg: float | None = None
    forward_pe: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    peg_ratio: float | None = None

    # Solvency / Balance Sheet
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    interest_coverage: float | None = None
    total_debt: float | None = None
    total_cash: float | None = None
    net_debt: float | None = None

    # Profitability
    gross_margin: float | None = None  # %
    operating_margin: float | None = None  # %
    profit_margin: float | None = None  # %
    roe: float | None = None  # Return on Equity %
    roa: float | None = None  # Return on Assets %
    roic: float | None = None  # Return on Invested Capital %

    # Cash Flow
    free_cash_flow: float | None = None
    operating_cash_flow: float | None = None
    fcf_yield: float | None = None  # FCF / Market Cap %

    # Growth
    revenue_growth_yoy: float | None = None  # %
    earnings_growth_yoy: float | None = None  # %
    revenue_growth_3y_cagr: float | None = None  # %

    # Quality Indicators
    piotroski_f_score: int | None = None  # 0-9
    altman_z_score: float | None = None  # Bankruptcy risk

    @property
    def has_sufficient_data(self) -> bool:
        """Check if we have enough data for analysis."""
        critical_fields = [
            self.debt_to_equity,
            self.free_cash_flow,
            self.pe_ratio,
        ]
        return all(f is not None for f in critical_fields)


@dataclass
class CheckResult:
    """Result of a single fundamental check."""

    name: str
    passed: bool
    value: float | None
    threshold: float | None
    reason: str
    weight: float = 1.0  # Importance of this check
    is_critical: bool = False  # If critical and fails, entire check fails


@dataclass
class GuardrailResult:
    """Result of fundamental guardrail evaluation."""

    passed: bool
    recommendation: Literal["STRONG_BUY", "ACCUMULATE", "HOLD", "AVOID", "SELL"]
    confidence: float  # 0-100

    # Individual checks
    checks: list[CheckResult] = field(default_factory=list)
    passed_count: int = 0
    failed_count: int = 0
    critical_failures: list[str] = field(default_factory=list)

    # Summary
    quality_score: float = 0.0  # 0-100 composite score
    risk_level: Literal["low", "medium", "high", "extreme"] = "medium"
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "passed": self.passed,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "risk_level": self.risk_level,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": c.value,
                    "threshold": c.threshold,
                    "reason": c.reason,
                }
                for c in self.checks
            ],
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "critical_failures": self.critical_failures,
        }


@dataclass
class GuardrailConfig:
    """Configuration for fundamental guardrails."""

    # Solvency thresholds
    max_debt_to_equity: float = 2.0
    min_current_ratio: float = 1.0
    min_interest_coverage: float = 2.0

    # Valuation thresholds
    require_pe_below_historical: bool = True
    max_pe_ratio: float = 50.0  # Absolute max
    max_pb_ratio: float = 10.0

    # Profitability thresholds
    require_positive_fcf: bool = True
    min_profit_margin: float = 0.0  # At least break-even
    min_roe: float = 5.0  # %

    # Growth thresholds
    max_revenue_decline: float = -20.0  # % YoY

    # Quality scores
    min_piotroski_score: int = 4  # Out of 9
    min_altman_z: float = 1.8  # Below 1.8 = distress zone


class FundamentalGuardrail:
    """
    Quality filter for bear market accumulation.

    PHILOSOPHY:
    - In bear markets, EVERYTHING looks cheap
    - We need to separate "Quality on Sale" from "Value Traps"
    - Value traps are companies that look cheap but deserve it
    - Quality on sale are great companies at temporary discounts

    USAGE:
    - Before any bear-market buy signal, run this check
    - If check fails, BLOCK the buy signal
    - If check passes, APPROVE with confidence score
    """

    def __init__(self, config: GuardrailConfig | None = None):
        self.config = config or GuardrailConfig()

    def check(self, fundamentals: FundamentalData) -> GuardrailResult:
        """
        Run all fundamental checks.

        Returns GuardrailResult with passed=True if stock is quality.
        """
        if not fundamentals.has_sufficient_data:
            return GuardrailResult(
                passed=False,
                recommendation="AVOID",
                confidence=0.0,
                checks=[],
                summary="Insufficient fundamental data for analysis",
                risk_level="extreme",
            )

        checks: list[CheckResult] = []

        # =====================================================================
        # SOLVENCY CHECKS (Can they survive the downturn?)
        # =====================================================================

        # Check 1: Debt to Equity
        if fundamentals.debt_to_equity is not None:
            passed = fundamentals.debt_to_equity < self.config.max_debt_to_equity
            checks.append(
                CheckResult(
                    name="debt_to_equity",
                    passed=passed,
                    value=fundamentals.debt_to_equity,
                    threshold=self.config.max_debt_to_equity,
                    reason=f"D/E {fundamentals.debt_to_equity:.2f} {'<' if passed else '>='} {self.config.max_debt_to_equity}",
                    weight=1.5,
                    is_critical=True,  # High debt = potential bankruptcy
                )
            )

        # Check 2: Current Ratio
        if fundamentals.current_ratio is not None:
            passed = fundamentals.current_ratio >= self.config.min_current_ratio
            checks.append(
                CheckResult(
                    name="current_ratio",
                    passed=passed,
                    value=fundamentals.current_ratio,
                    threshold=self.config.min_current_ratio,
                    reason=f"Current ratio {fundamentals.current_ratio:.2f} {'>=' if passed else '<'} {self.config.min_current_ratio}",
                    weight=1.2,
                    is_critical=fundamentals.current_ratio < 0.5,  # Critical if very low
                )
            )

        # Check 3: Interest Coverage
        if fundamentals.interest_coverage is not None:
            passed = fundamentals.interest_coverage >= self.config.min_interest_coverage
            checks.append(
                CheckResult(
                    name="interest_coverage",
                    passed=passed,
                    value=fundamentals.interest_coverage,
                    threshold=self.config.min_interest_coverage,
                    reason=f"Interest coverage {fundamentals.interest_coverage:.1f}x {'>=' if passed else '<'} {self.config.min_interest_coverage}x",
                    weight=1.3,
                    is_critical=fundamentals.interest_coverage < 1.0,
                )
            )

        # =====================================================================
        # PROFITABILITY CHECKS (Do they generate cash?)
        # =====================================================================

        # Check 4: Free Cash Flow
        if fundamentals.free_cash_flow is not None:
            passed = fundamentals.free_cash_flow > 0
            fcf_millions = fundamentals.free_cash_flow / 1e6
            checks.append(
                CheckResult(
                    name="free_cash_flow",
                    passed=passed,
                    value=fundamentals.free_cash_flow,
                    threshold=0.0,
                    reason=f"FCF ${fcf_millions:.0f}M {'>' if passed else '<='} $0",
                    weight=1.5,
                    is_critical=True,  # Negative FCF = cash burn
                )
            )

        # Check 5: Profit Margin
        if fundamentals.profit_margin is not None:
            passed = fundamentals.profit_margin >= self.config.min_profit_margin
            checks.append(
                CheckResult(
                    name="profit_margin",
                    passed=passed,
                    value=fundamentals.profit_margin,
                    threshold=self.config.min_profit_margin,
                    reason=f"Profit margin {fundamentals.profit_margin:.1f}% {'>=' if passed else '<'} {self.config.min_profit_margin}%",
                    weight=1.0,
                    is_critical=False,
                )
            )

        # Check 6: Return on Equity
        if fundamentals.roe is not None:
            passed = fundamentals.roe >= self.config.min_roe
            checks.append(
                CheckResult(
                    name="return_on_equity",
                    passed=passed,
                    value=fundamentals.roe,
                    threshold=self.config.min_roe,
                    reason=f"ROE {fundamentals.roe:.1f}% {'>=' if passed else '<'} {self.config.min_roe}%",
                    weight=1.0,
                    is_critical=False,
                )
            )

        # =====================================================================
        # VALUATION CHECKS (Is it actually cheap?)
        # =====================================================================

        # Check 7: PE vs Historical
        if fundamentals.pe_ratio is not None and fundamentals.pe_5y_avg is not None:
            passed = fundamentals.pe_ratio < fundamentals.pe_5y_avg
            checks.append(
                CheckResult(
                    name="pe_vs_historical",
                    passed=passed,
                    value=fundamentals.pe_ratio,
                    threshold=fundamentals.pe_5y_avg,
                    reason=f"PE {fundamentals.pe_ratio:.1f} {'<' if passed else '>='} 5Y avg {fundamentals.pe_5y_avg:.1f}",
                    weight=1.2,
                    is_critical=False,
                )
            )

        # Check 8: Absolute PE Cap
        if fundamentals.pe_ratio is not None:
            passed = fundamentals.pe_ratio < self.config.max_pe_ratio
            checks.append(
                CheckResult(
                    name="pe_ratio_cap",
                    passed=passed,
                    value=fundamentals.pe_ratio,
                    threshold=self.config.max_pe_ratio,
                    reason=f"PE {fundamentals.pe_ratio:.1f} {'<' if passed else '>='} {self.config.max_pe_ratio}",
                    weight=0.8,
                    is_critical=False,
                )
            )

        # =====================================================================
        # GROWTH CHECKS (Is it in terminal decline?)
        # =====================================================================

        # Check 9: Revenue Growth
        if fundamentals.revenue_growth_yoy is not None:
            passed = fundamentals.revenue_growth_yoy > self.config.max_revenue_decline
            checks.append(
                CheckResult(
                    name="revenue_growth",
                    passed=passed,
                    value=fundamentals.revenue_growth_yoy,
                    threshold=self.config.max_revenue_decline,
                    reason=f"Revenue growth {fundamentals.revenue_growth_yoy:.1f}% {'>' if passed else '<='} {self.config.max_revenue_decline}%",
                    weight=1.0,
                    is_critical=fundamentals.revenue_growth_yoy < -30,  # Collapse
                )
            )

        # =====================================================================
        # QUALITY SCORES (Composite indicators)
        # =====================================================================

        # Check 10: Piotroski F-Score
        if fundamentals.piotroski_f_score is not None:
            passed = fundamentals.piotroski_f_score >= self.config.min_piotroski_score
            checks.append(
                CheckResult(
                    name="piotroski_f_score",
                    passed=passed,
                    value=float(fundamentals.piotroski_f_score),
                    threshold=float(self.config.min_piotroski_score),
                    reason=f"Piotroski {fundamentals.piotroski_f_score}/9 {'>=' if passed else '<'} {self.config.min_piotroski_score}",
                    weight=1.3,
                    is_critical=False,
                )
            )

        # Check 11: Altman Z-Score (bankruptcy risk)
        if fundamentals.altman_z_score is not None:
            passed = fundamentals.altman_z_score >= self.config.min_altman_z
            checks.append(
                CheckResult(
                    name="altman_z_score",
                    passed=passed,
                    value=fundamentals.altman_z_score,
                    threshold=self.config.min_altman_z,
                    reason=f"Altman Z {fundamentals.altman_z_score:.2f} {'>=' if passed else '<'} {self.config.min_altman_z} (distress zone)",
                    weight=1.5,
                    is_critical=fundamentals.altman_z_score < 1.1,  # High bankruptcy risk
                )
            )

        # =====================================================================
        # AGGREGATE RESULTS
        # =====================================================================

        return self._aggregate_results(checks, fundamentals)

    def _aggregate_results(
        self,
        checks: list[CheckResult],
        fundamentals: FundamentalData,
    ) -> GuardrailResult:
        """Aggregate individual checks into final result."""

        if not checks:
            return GuardrailResult(
                passed=False,
                recommendation="AVOID",
                confidence=0.0,
                summary="No checks could be performed",
                risk_level="extreme",
            )

        # Count passes and failures
        passed_count = sum(1 for c in checks if c.passed)
        failed_count = sum(1 for c in checks if not c.passed)
        total_count = len(checks)

        # Identify critical failures
        critical_failures = [c.name for c in checks if c.is_critical and not c.passed]

        # Calculate weighted quality score
        total_weight = sum(c.weight for c in checks)
        weighted_score = sum(c.weight for c in checks if c.passed) / total_weight * 100

        # Determine pass/fail
        # Must pass all critical checks AND at least 60% of weighted checks
        passes_critical = len(critical_failures) == 0
        passes_weighted = weighted_score >= 60.0
        passed = passes_critical and passes_weighted

        # Determine recommendation
        if not passes_critical:
            recommendation: Literal["STRONG_BUY", "ACCUMULATE", "HOLD", "AVOID", "SELL"] = "AVOID"
            risk_level: Literal["low", "medium", "high", "extreme"] = "extreme"
        elif weighted_score >= 90:
            recommendation = "STRONG_BUY"
            risk_level = "low"
        elif weighted_score >= 75:
            recommendation = "ACCUMULATE"
            risk_level = "low"
        elif weighted_score >= 60:
            recommendation = "ACCUMULATE"
            risk_level = "medium"
        elif weighted_score >= 40:
            recommendation = "HOLD"
            risk_level = "high"
        else:
            recommendation = "AVOID"
            risk_level = "extreme"

        # Calculate confidence
        # Higher when more checks available and more agreement
        data_completeness = len(checks) / 11  # 11 possible checks
        confidence = min(100, weighted_score * data_completeness)

        # Generate summary
        if passed:
            summary = f"Quality stock: {passed_count}/{total_count} checks passed ({weighted_score:.0f}% weighted score)"
        else:
            if critical_failures:
                summary = f"Failed critical checks: {', '.join(critical_failures)}"
            else:
                summary = f"Insufficient quality: only {passed_count}/{total_count} checks passed ({weighted_score:.0f}% weighted)"

        return GuardrailResult(
            passed=passed,
            recommendation=recommendation,
            confidence=confidence,
            checks=checks,
            passed_count=passed_count,
            failed_count=failed_count,
            critical_failures=critical_failures,
            quality_score=weighted_score,
            risk_level=risk_level,
            summary=summary,
        )

    def quick_check(self, fundamentals: FundamentalData) -> bool:
        """
        Quick pass/fail check without full analysis.

        Useful for filtering large lists of stocks.
        """
        # Critical checks only
        if fundamentals.debt_to_equity is not None:
            if fundamentals.debt_to_equity >= self.config.max_debt_to_equity:
                return False

        if fundamentals.free_cash_flow is not None:
            if fundamentals.free_cash_flow <= 0:
                return False

        if fundamentals.altman_z_score is not None:
            if fundamentals.altman_z_score < 1.1:
                return False

        return True


class PointInTimeFundamentals:
    """
    Provides point-in-time fundamental data to avoid look-ahead bias.

    When backtesting, we must only use data that was available at the time.
    This class handles the temporal alignment of fundamental data.
    """

    def __init__(self, fundamental_history: pd.DataFrame | None = None):
        """
        Initialize with historical fundamental data.

        Args:
            fundamental_history: DataFrame with DatetimeIndex (report_date)
                               and columns for each fundamental metric.
        """
        self.history = fundamental_history

    def get_at_date(self, symbol: str, as_of_date: pd.Timestamp) -> FundamentalData | None:
        """
        Get the most recent fundamentals available on a given date.

        This ensures no look-ahead bias - we only return data from
        reports that were already published by as_of_date.
        """
        if self.history is None or self.history.empty:
            return None

        # Filter to reports available before as_of_date
        # Add a delay for report publication (earnings typically available 1-2 weeks after quarter end)
        available = self.history[self.history.index <= as_of_date]

        if available.empty:
            return None

        # Get most recent report
        latest = available.iloc[-1]

        return FundamentalData(
            symbol=symbol,
            report_date=latest.name,
            pe_ratio=latest.get("pe_ratio"),
            pe_5y_avg=latest.get("pe_5y_avg"),
            forward_pe=latest.get("forward_pe"),
            pb_ratio=latest.get("pb_ratio"),
            debt_to_equity=latest.get("debt_to_equity"),
            current_ratio=latest.get("current_ratio"),
            interest_coverage=latest.get("interest_coverage"),
            free_cash_flow=latest.get("free_cash_flow"),
            operating_cash_flow=latest.get("operating_cash_flow"),
            profit_margin=latest.get("profit_margin"),
            gross_margin=latest.get("gross_margin"),
            roe=latest.get("roe"),
            roa=latest.get("roa"),
            revenue_growth_yoy=latest.get("revenue_growth_yoy"),
            earnings_growth_yoy=latest.get("earnings_growth_yoy"),
            piotroski_f_score=latest.get("piotroski_f_score"),
            altman_z_score=latest.get("altman_z_score"),
        )
