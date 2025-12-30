"""
Domain-Specific Scoring Adapters.

Each adapter customizes quality scoring for its domain:
- Banks: Skip D/E, focus on ROE and book value
- REITs: Skip FCF, focus on dividend yield
- ETFs: Minimal quality scoring (focus on dip/stability)
- etc.

Adapters compute sub-scores and weights appropriate for their domain.
When key data is missing, they fall back to generic scoring with a confidence penalty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.dipfinder.domain import Domain, DomainClassification


logger = get_logger("dipfinder.domain_scoring")


@dataclass
class SubScore:
    """A component score with its weight and explanation."""

    name: str
    score: float  # 0-100
    weight: float  # 0.0-1.0
    reason: str
    available: bool = True  # False if data was missing

    @property
    def weighted_score(self) -> float:
        """Return score * weight."""
        return self.score * self.weight if self.available else 0.0


@dataclass
class DomainScoreResult:
    """Result from domain-specific scoring."""

    domain: Domain
    domain_confidence: float
    final_score: float  # 0-100
    sub_scores: list[SubScore]
    data_completeness: float  # 0.0-1.0, fraction of expected data available
    fallback_used: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for API responses."""
        return {
            "domain": self.domain.value,
            "domain_confidence": self.domain_confidence,
            "final_score": self.final_score,
            "sub_scores": [
                {
                    "name": s.name,
                    "score": s.score,
                    "weight": s.weight,
                    "weighted": s.weighted_score,
                    "available": s.available,
                    "reason": s.reason,
                }
                for s in self.sub_scores
            ],
            "data_completeness": self.data_completeness,
            "fallback_used": self.fallback_used,
            "notes": self.notes,
        }


def _normalize_score(value: float, optimal: float, bad: float, higher_better: bool = True) -> float:
    """
    Normalize a metric to 0-100 scale.
        Args:
        value: The metric value
        optimal: The "best" value (score = 100)
        bad: The "worst" value (score = 0)
        higher_better: True if higher values are better
    
    Returns:
        Score from 0 to 100
    """
    if higher_better:
        if value >= optimal:
            return 100.0
        if value <= bad:
            return 0.0
        return ((value - bad) / (optimal - bad)) * 100.0
    else:
        if value <= optimal:
            return 100.0
        if value >= bad:
            return 0.0
        return ((bad - value) / (bad - optimal)) * 100.0


def _safe_float(value: Any) -> float | None:
    """Safely convert to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _latest_value(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in sorted(value.keys(), reverse=True):
            v = value.get(key)
            if v is not None:
                return _safe_float(v)
        return None
    if isinstance(value, list):
        for item in value:
            if item is not None:
                return _safe_float(item)
        return None
    return _safe_float(value)


# Key mappings: normalized key -> list of possible yfinance/alternative keys
_KEY_ALIASES: dict[str, list[str]] = {
    "profit_margin": ["profit_margin", "profitMargins", "profitMargin"],
    "operating_margin": ["operating_margin", "operatingMargins", "operatingMargin"],
    "return_on_equity": ["return_on_equity", "returnOnEquity", "roe"],
    "return_on_assets": ["return_on_assets", "returnOnAssets", "roa"],
    "price_to_book": ["price_to_book", "priceToBook"],
    "book_value": ["book_value", "bookValue"],
    "dividend_yield": ["dividend_yield", "dividendYield"],
    "debt_to_equity": ["debt_to_equity", "debtToEquity"],
    "current_ratio": ["current_ratio", "currentRatio"],
    "free_cash_flow": ["free_cash_flow", "freeCashflow", "freeCashFlow"],
    "operating_cash_flow": ["operating_cash_flow", "operatingCashflow", "operatingCashFlow"],
    "market_cap": ["market_cap", "marketCap"],
    "pe_ratio": ["pe_ratio", "trailingPE", "pe"],
    "forward_pe": ["forward_pe", "forwardPE"],
    "peg_ratio": ["peg_ratio", "trailingPegRatio", "pegRatio"],
    "revenue_growth": ["revenue_growth", "revenueGrowth"],
    "earnings_growth": ["earnings_growth", "earningsGrowth"],
    "recommendation_mean": ["recommendation_mean", "recommendationMean"],
    "target_mean_price": ["target_mean_price", "targetMeanPrice"],
    "current_price": ["current_price", "regularMarketPrice", "previousClose"],
    "total_cash": ["total_cash", "totalCash"],
    "beta": ["beta"],
    "short_percent_of_float": ["short_percent_of_float", "shortPercentOfFloat"],
    "num_analyst_opinions": ["num_analyst_opinions", "numberOfAnalystOpinions"],
}


def _get_metric(info: dict, key: str) -> float | None:
    """
    Get a metric from info dict, trying multiple possible key names.
    
    Handles both normalized keys (e.g., 'profit_margin') and
    yfinance raw keys (e.g., 'profitMargins').
    """
    aliases = _KEY_ALIASES.get(key, [key])
    for alias in aliases:
        value = info.get(alias)
        if value is not None:
            return _safe_float(value)
    return None


# =============================================================================
# Financial Statement Helpers
# =============================================================================

def _get_financial_metric(
    info: dict,
    statement: str,
    metric: str,
    period: str = "quarterly",
) -> float | None:
    """
    Get a metric from financial statements embedded in info.
    
    Args:
        info: Dict from yfinance with optional 'financials' key
        statement: 'income_statement', 'balance_sheet', or 'cash_flow'
        metric: The metric name as it appears in yfinance (e.g., 'Net Interest Income')
        period: 'quarterly' or 'annual'
    
    Returns:
        The metric value or None if not found
    """
    financials = info.get("financials")
    if not financials:
        return None

    period_data = financials.get(period, {})
    statement_data = period_data.get(statement, {})

    value = statement_data.get(metric)
    if value is not None:
        return _latest_value(value)
    return None


def _get_ffo(info: dict, period: str = "quarterly") -> float | None:
    """
    Calculate Funds From Operations (FFO) for REITs.
    
    FFO = Net Income + Depreciation & Amortization
    This is a simplified calculation; full FFO also excludes gains on property sales.
    """
    net_income = _get_financial_metric(info, "income_statement", "Net Income", period)
    depreciation = _get_financial_metric(info, "cash_flow", "Depreciation Amortization Depletion", period)

    # Try alternate depreciation keys
    if depreciation is None:
        depreciation = _get_financial_metric(info, "cash_flow", "Depreciation And Amortization", period)

    if net_income is not None and depreciation is not None:
        return net_income + depreciation
    return None


def _get_net_interest_income(info: dict, period: str = "quarterly") -> float | None:
    """Get Net Interest Income from income statement (for banks)."""
    return _get_financial_metric(info, "income_statement", "Net Interest Income", period)


def _get_provision_for_credit_losses(info: dict, period: str = "quarterly") -> float | None:
    """Get Provision for Credit Losses from income statement (for banks)."""
    # Try different possible names
    for metric_name in [
        "Provision For Credit Losses",
        "Credit Losses Provision",
        "Loan Loss Provision",
    ]:
        val = _get_financial_metric(info, "income_statement", metric_name, period)
        if val is not None:
            return val
    return None


def _get_loss_adjustment_expense(info: dict, period: str = "quarterly") -> float | None:
    """Get Loss Adjustment Expense from income statement (for insurers)."""
    for metric_name in [
        "Loss Adjustment Expense",
        "Net Policyholder Benefits And Claims",
        "Policyholder Benefits And Claims Incurred",
    ]:
        val = _get_financial_metric(info, "income_statement", metric_name, period)
        if val is not None:
            return val
    return None


def _get_total_revenue(info: dict, period: str = "quarterly") -> float | None:
    """Get Total Revenue from income statement."""
    for metric_name in ["Total Revenue", "Operating Revenue", "Revenue"]:
        val = _get_financial_metric(info, "income_statement", metric_name, period)
        if val is not None:
            return val
    return None


def _get_total_assets(info: dict, period: str = "quarterly") -> float | None:
    """Get Total Assets from balance sheet."""
    return _get_financial_metric(info, "balance_sheet", "Total Assets", period)


def _calculate_nim(info: dict, period: str = "quarterly") -> float | None:
    """
    Calculate Net Interest Margin proxy for banks.
    
    True NIM = Net Interest Income / Average Earning Assets
    We approximate with: NIM ≈ Net Interest Income / Total Assets (annualized)
    """
    nii = _get_net_interest_income(info, period)
    total_assets = _get_total_assets(info, period)

    if nii is not None and total_assets is not None and total_assets > 0:
        # Annualize quarterly NII
        annual_nii = nii * 4 if period == "quarterly" else nii
        return annual_nii / total_assets
    return None


# =============================================================================
# Base Adapter
# =============================================================================

class DomainScoringAdapter(ABC):
    """Base class for domain-specific scoring."""

    domain: Domain

    @abstractmethod
    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        """
        Compute quality score for this domain.
        
        Args:
            info: Dict from yfinance_service.get_ticker_info()
            fundamentals: Optional additional fundamentals data
        
        Returns:
            DomainScoreResult with score and breakdown
        """
        pass

    def _count_available(self, info: dict, keys: list[str]) -> tuple[int, int]:
        """Count how many of the expected keys have non-None values using key aliases."""
        available = sum(1 for k in keys if _get_metric(info, k) is not None)
        return available, len(keys)


# =============================================================================
# Operating Company Adapter (Default)
# =============================================================================

class OperatingCompanyAdapter(DomainScoringAdapter):
    """Default adapter for standard operating companies."""

    domain = Domain.OPERATING_COMPANY

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []

        # 1. Profitability (20%)
        profit_margin = _get_metric(info, "profit_margin")
        operating_margin = _get_metric(info, "operating_margin")

        expected_keys.extend(["profit_margin", "operating_margin"])

        if profit_margin is not None or operating_margin is not None:
            pm_score = _normalize_score(profit_margin or 0, 0.20, 0.0) if profit_margin else 50
            om_score = _normalize_score(operating_margin or 0, 0.25, 0.0) if operating_margin else 50
            score = (pm_score + om_score) / 2
            sub_scores.append(SubScore(
                name="profitability",
                score=score,
                weight=0.20,
                reason=f"profit_margin={profit_margin:.1%}, operating_margin={operating_margin:.1%}" if profit_margin and operating_margin else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="profitability",
                score=50.0,
                weight=0.20,
                reason="no margin data",
                available=False,
            ))

        # 2. Balance Sheet (15%)
        debt_to_equity = _get_metric(info, "debt_to_equity")
        current_ratio = _get_metric(info, "current_ratio")

        expected_keys.extend(["debt_to_equity", "current_ratio"])

        if debt_to_equity is not None or current_ratio is not None:
            # D/E: lower is better (in percentage form from yfinance)
            de_score = _normalize_score(debt_to_equity or 100, 0, 200, higher_better=False) if debt_to_equity is not None else 50
            # Current ratio: higher is better
            cr_score = _normalize_score(current_ratio or 1, 2.0, 0.5) if current_ratio is not None else 50
            score = (de_score + cr_score) / 2
            sub_scores.append(SubScore(
                name="balance_sheet",
                score=score,
                weight=0.15,
                reason=f"D/E={debt_to_equity:.0f}%, current_ratio={current_ratio:.2f}" if debt_to_equity and current_ratio else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="balance_sheet",
                score=50.0,
                weight=0.15,
                reason="no balance sheet data",
                available=False,
            ))

        # 3. Cash Generation (20%)
        free_cash_flow = _get_metric(info, "free_cash_flow")
        operating_cash_flow = _get_metric(info, "operating_cash_flow")
        market_cap = _get_metric(info, "market_cap")

        expected_keys.extend(["free_cash_flow", "operating_cash_flow", "market_cap"])

        if free_cash_flow is not None and market_cap and market_cap > 0:
            fcf_yield = free_cash_flow / market_cap
            score = _normalize_score(fcf_yield, 0.10, -0.05)
            sub_scores.append(SubScore(
                name="cash_generation",
                score=score,
                weight=0.20,
                reason=f"FCF yield={fcf_yield:.1%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="cash_generation",
                score=50.0,
                weight=0.20,
                reason="no FCF data",
                available=False,
            ))

        # 4. Valuation (20%)
        pe_ratio = _get_metric(info, "pe_ratio")
        forward_pe = _get_metric(info, "forward_pe")
        peg_ratio = _get_metric(info, "peg_ratio")

        expected_keys.extend(["pe_ratio", "forward_pe", "peg_ratio"])

        val_scores = []
        if pe_ratio is not None and 0 < pe_ratio < 100:
            val_scores.append(_normalize_score(pe_ratio, 10, 40, higher_better=False))
        if peg_ratio is not None and 0 < peg_ratio < 5:
            val_scores.append(_normalize_score(peg_ratio, 0.5, 2.5, higher_better=False))

        if val_scores:
            score = sum(val_scores) / len(val_scores)
            sub_scores.append(SubScore(
                name="valuation",
                score=score,
                weight=0.20,
                reason=f"P/E={pe_ratio:.1f}, PEG={peg_ratio:.2f}" if pe_ratio and peg_ratio else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="valuation",
                score=50.0,
                weight=0.20,
                reason="no valuation data",
                available=False,
            ))

        # 5. Growth (15%)
        revenue_growth = _get_metric(info, "revenue_growth")
        earnings_growth = _get_metric(info, "earnings_growth")

        expected_keys.extend(["revenue_growth", "earnings_growth"])

        if revenue_growth is not None or earnings_growth is not None:
            rg_score = _normalize_score(revenue_growth or 0, 0.25, -0.10) if revenue_growth is not None else 50
            eg_score = _normalize_score(earnings_growth or 0, 0.30, -0.20) if earnings_growth is not None else 50
            score = (rg_score + eg_score) / 2
            sub_scores.append(SubScore(
                name="growth",
                score=score,
                weight=0.15,
                reason=f"rev_growth={revenue_growth:.1%}, earn_growth={earnings_growth:.1%}" if revenue_growth and earnings_growth else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="growth",
                score=50.0,
                weight=0.15,
                reason="no growth data",
                available=False,
            ))

        # 6. Analyst (10%)
        recommendation_mean = _get_metric(info, "recommendation_mean")
        target_mean_price = _get_metric(info, "target_mean_price")
        current_price = _get_metric(info, "current_price")

        expected_keys.extend(["recommendation_mean", "target_mean_price"])

        if recommendation_mean is not None:
            # 1=strong buy, 5=strong sell
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            upside = ""
            if target_mean_price and current_price and current_price > 0:
                upside_pct = (target_mean_price - current_price) / current_price
                upside = f", upside={upside_pct:.0%}"
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.10,
                reason=f"recommendation={recommendation_mean:.1f}/5{upside}",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.10,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0  # No data at all

        # Data completeness
        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
        )


# =============================================================================
# Bank Adapter
# =============================================================================

class BankAdapter(DomainScoringAdapter):
    """Adapter for banks and financial institutions.
    
    Banks are different:
    - D/E is meaningless (leverage IS the business)
    - FCF is not a good metric
    - ROE and ROA are key profitability metrics
    - Book value matters more than enterprise value
    - Dividend yield is often important
    - Net Interest Margin (NIM) from financial statements is key
    """

    domain = Domain.BANK

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []
        has_financials = info.get("financials") is not None

        # 1. Returns (30%) - ROE and ROA are the key metrics for banks
        roe = _get_metric(info, "return_on_equity")
        roa = _get_metric(info, "return_on_assets")

        expected_keys.extend(["return_on_equity", "return_on_assets"])

        if roe is not None or roa is not None:
            # ROE: 12%+ is good, 8% is ok, <5% is poor
            roe_score = _normalize_score(roe or 0, 0.15, 0.03) if roe is not None else 50
            # ROA: 1%+ is good for banks (they have huge asset bases)
            roa_score = _normalize_score(roa or 0, 0.015, 0.003) if roa is not None else 50
            score = (roe_score * 0.7 + roa_score * 0.3)  # ROE weighted more
            sub_scores.append(SubScore(
                name="returns",
                score=score,
                weight=0.30,
                reason=f"ROE={roe:.1%}, ROA={roa:.2%}" if roe and roa else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="returns",
                score=50.0,
                weight=0.30,
                reason="no ROE/ROA data",
                available=False,
            ))

        # 2. Net Interest Margin (15%) - From financial statements
        nim = _calculate_nim(info)
        nii = _get_net_interest_income(info)

        if nim is not None:
            # NIM: 2.5%+ is good, 2% is ok, <1.5% is poor
            score = _normalize_score(nim, 0.035, 0.015)
            nii_display = f", NII=${nii/1e9:.1f}B" if nii and nii > 1e9 else ""
            sub_scores.append(SubScore(
                name="net_interest_margin",
                score=score,
                weight=0.15,
                reason=f"NIM≈{nim:.2%}{nii_display}",
            ))
        elif nii is not None:
            # We have NII but can't calculate NIM (missing total assets)
            # Still give some credit for having positive NII
            score = 60.0 if nii > 0 else 40.0
            sub_scores.append(SubScore(
                name="net_interest_margin",
                score=score,
                weight=0.15,
                reason=f"NII=${nii/1e9:.1f}B (NIM unavailable)" if nii > 1e9 else f"NII=${nii/1e6:.0f}M",
            ))
        else:
            sub_scores.append(SubScore(
                name="net_interest_margin",
                score=50.0,
                weight=0.15 if has_financials else 0.0,  # Zero weight if no financials
                reason="no NII data from financial statements",
                available=False,
            ))

        # 3. Book Value (20%) - Price to book matters for banks
        price_to_book = _get_metric(info, "price_to_book")
        book_value = _get_metric(info, "book_value")

        expected_keys.extend(["price_to_book", "book_value"])

        if price_to_book is not None:
            # P/B: <1.0 is cheap, 1.0-1.5 is fair, >2.0 is expensive
            score = _normalize_score(price_to_book, 0.8, 2.5, higher_better=False)
            sub_scores.append(SubScore(
                name="book_value",
                score=score,
                weight=0.20,
                reason=f"P/B={price_to_book:.2f}, book=${book_value:.2f}" if book_value else f"P/B={price_to_book:.2f}",
            ))
        else:
            sub_scores.append(SubScore(
                name="book_value",
                score=50.0,
                weight=0.20,
                reason="no P/B data",
                available=False,
            ))

        # 4. Dividend (20%) - Income is important for bank investors
        dividend_yield = _get_metric(info, "dividend_yield")

        expected_keys.append("dividend_yield")

        if dividend_yield is not None:
            # 3%+ is good, 1-3% is ok, <1% or >8% is concerning
            if dividend_yield > 0.08:  # Suspiciously high
                score = 40.0
            else:
                score = _normalize_score(dividend_yield, 0.04, 0.0)
            sub_scores.append(SubScore(
                name="dividend",
                score=score,
                weight=0.20,
                reason=f"yield={dividend_yield:.2%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="dividend",
                score=50.0,
                weight=0.20,
                reason="no dividend data",
                available=False,
            ))

        # 5. Analyst (15%)
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.15,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.15,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        notes = "Bank scoring: D/E and FCF ignored; ROE, ROA, P/B, dividend weighted"
        if has_financials:
            notes += "; using financial statements for NIM"

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes=notes,
        )


# =============================================================================
# Insurance Adapter
# =============================================================================

class InsuranceAdapter(DomainScoringAdapter):
    """Adapter for insurance companies.
    
    Insurers are different:
    - Combined ratio is the key metric (can calculate from financial statements)
    - ROE is very important for capital efficiency
    - Book value matters (similar to banks)
    - Investment income is a major component
    - D/E is less meaningful (reserves are liabilities)
    """

    domain = Domain.INSURER

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []
        has_financials = info.get("financials") is not None

        # 1. Loss Ratio / Underwriting Quality (20%)
        # Try to get loss adjustment expense from financial statements
        loss_expense = _get_loss_adjustment_expense(info)
        total_revenue = _get_total_revenue(info)

        if loss_expense is not None and total_revenue is not None and total_revenue > 0:
            # Loss ratio = Losses / Premiums (approximated by revenue)
            loss_ratio = loss_expense / total_revenue
            # Loss ratio: 60-70% is healthy, >80% is concerning
            score = _normalize_score(loss_ratio, 0.55, 0.85, higher_better=False)
            sub_scores.append(SubScore(
                name="loss_ratio",
                score=score,
                weight=0.20,
                reason=f"loss_ratio≈{loss_ratio:.1%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="loss_ratio",
                score=50.0,
                weight=0.20 if has_financials else 0.0,
                reason="no loss data from financial statements",
                available=False,
            ))

        # 2. Returns (30%) - ROE is critical for insurers
        roe = _get_metric(info, "return_on_equity")
        roa = _get_metric(info, "return_on_assets")

        expected_keys.extend(["return_on_equity", "return_on_assets"])

        if roe is not None or roa is not None:
            # ROE: 12%+ is excellent for insurers, 8-12% is good
            roe_score = _normalize_score(roe or 0, 0.15, 0.05) if roe is not None else 50
            roa_score = _normalize_score(roa or 0, 0.02, 0.005) if roa is not None else 50
            score = (roe_score * 0.8 + roa_score * 0.2)
            sub_scores.append(SubScore(
                name="returns",
                score=score,
                weight=0.30,
                reason=f"ROE={roe:.1%}, ROA={roa:.2%}" if roe and roa else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="returns",
                score=50.0,
                weight=0.30,
                reason="no ROE/ROA data",
                available=False,
            ))

        # 3. Book Value (20%) - Important for insurers
        price_to_book = _get_metric(info, "price_to_book")
        book_value = _get_metric(info, "book_value")

        expected_keys.extend(["price_to_book", "book_value"])

        if price_to_book is not None:
            # P/B: 1.0-1.5 is fair, <1.0 is cheap, >2.5 is expensive
            score = _normalize_score(price_to_book, 0.9, 2.5, higher_better=False)
            sub_scores.append(SubScore(
                name="book_value",
                score=score,
                weight=0.20,
                reason=f"P/B={price_to_book:.2f}",
            ))
        else:
            sub_scores.append(SubScore(
                name="book_value",
                score=50.0,
                weight=0.20,
                reason="no P/B data",
                available=False,
            ))

        # 4. Dividend (15%) - Many insurers pay dividends
        dividend_yield = _get_metric(info, "dividend_yield")

        expected_keys.append("dividend_yield")

        if dividend_yield is not None:
            if dividend_yield > 0.08:  # Suspiciously high
                score = 40.0
            else:
                score = _normalize_score(dividend_yield, 0.03, 0.0)
            sub_scores.append(SubScore(
                name="dividend",
                score=score,
                weight=0.15,
                reason=f"yield={dividend_yield:.2%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="dividend",
                score=50.0,
                weight=0.15,
                reason="no dividend data",
                available=False,
            ))

        # 5. Analyst (15%)
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.15,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.15,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        notes = "Insurance scoring: D/E ignored; ROE, P/B, loss ratio weighted"
        if has_financials:
            notes += "; using financial statements for loss ratio"

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes=notes,
        )


# =============================================================================
# REIT Adapter
# =============================================================================

class REITAdapter(DomainScoringAdapter):
    """Adapter for Real Estate Investment Trusts.
    
    REITs are different:
    - Required to distribute 90%+ of income as dividends
    - FFO/AFFO are the key metrics (calculated from financial statements)
    - P/E is misleading due to depreciation
    - Dividend yield is critical
    - Some debt is normal and expected
    """

    domain = Domain.REIT

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []
        has_financials = info.get("financials") is not None

        # 1. FFO / P/FFO (25%) - The real earnings metric for REITs
        ffo = _get_ffo(info)
        shares_outstanding = info.get("shares_outstanding") or info.get("sharesOutstanding")
        current_price = _get_metric(info, "current_price")

        if ffo is not None and shares_outstanding and current_price:
            # Annualize quarterly FFO
            annual_ffo = ffo * 4
            ffo_per_share = annual_ffo / shares_outstanding
            p_ffo = current_price / ffo_per_share if ffo_per_share > 0 else None

            if p_ffo is not None:
                # P/FFO: 12-16x is fair, <10x is cheap, >20x is expensive
                score = _normalize_score(p_ffo, 10, 22, higher_better=False)
                sub_scores.append(SubScore(
                    name="p_ffo",
                    score=score,
                    weight=0.25,
                    reason=f"P/FFO={p_ffo:.1f}x, FFO/sh=${ffo_per_share:.2f}",
                ))
            else:
                sub_scores.append(SubScore(
                    name="p_ffo",
                    score=50.0,
                    weight=0.25,
                    reason=f"FFO=${ffo/1e6:.0f}M (P/FFO unavailable)",
                    available=False,
                ))
        else:
            sub_scores.append(SubScore(
                name="p_ffo",
                score=50.0,
                weight=0.25 if has_financials else 0.0,  # Zero weight if no financials
                reason="no FFO data from financial statements",
                available=False,
            ))

        # 2. Dividend (30%) - This is the main reason to own REITs
        dividend_yield = _get_metric(info, "dividend_yield")

        expected_keys.append("dividend_yield")

        if dividend_yield is not None:
            # 4-7% is healthy, <3% is low, >10% may be distressed
            if dividend_yield > 0.12:
                score = 30.0  # Suspiciously high, possible distress
            elif dividend_yield > 0.10:
                score = 50.0  # Elevated, proceed with caution
            else:
                score = _normalize_score(dividend_yield, 0.06, 0.02)
            sub_scores.append(SubScore(
                name="dividend",
                score=score,
                weight=0.30,
                reason=f"yield={dividend_yield:.2%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="dividend",
                score=50.0,
                weight=0.30,
                reason="no dividend data (unusual for REIT)",
                available=False,
            ))

        # 3. Price to Book (20%) - NAV proxy
        price_to_book = _get_metric(info, "price_to_book")

        expected_keys.append("price_to_book")

        if price_to_book is not None:
            # P/B near 1.0 is fair value, <0.9 is discount to NAV
            score = _normalize_score(price_to_book, 0.8, 2.0, higher_better=False)
            sub_scores.append(SubScore(
                name="price_to_book",
                score=score,
                weight=0.20,
                reason=f"P/B={price_to_book:.2f}",
            ))
        else:
            sub_scores.append(SubScore(
                name="price_to_book",
                score=50.0,
                weight=0.20,
                reason="no P/B data",
                available=False,
            ))

        # 4. Leverage (10%) - Some debt is normal, but not too much
        debt_to_equity = _get_metric(info, "debt_to_equity")

        expected_keys.append("debt_to_equity")

        if debt_to_equity is not None:
            # D/E 50-100% is normal for REITs, >200% is concerning
            score = _normalize_score(debt_to_equity, 50, 250, higher_better=False)
            sub_scores.append(SubScore(
                name="leverage",
                score=score,
                weight=0.10,
                reason=f"D/E={debt_to_equity:.0f}%",
            ))
        else:
            sub_scores.append(SubScore(
                name="leverage",
                score=50.0,
                weight=0.10,
                reason="no D/E data",
                available=False,
            ))

        # 5. Analyst (15%)
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.15,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.15,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        notes = "REIT scoring: P/E and FCF ignored; dividend yield and P/B weighted heavily"
        if has_financials:
            notes += "; using financial statements for FFO"

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes=notes,
        )


# =============================================================================
# ETF Adapter
# =============================================================================

class ETFAdapter(DomainScoringAdapter):
    """Adapter for ETFs and Index Funds.
    
    ETFs are fundamentally different:
    - No company-level fundamentals apply
    - Quality score should be minimal/neutral
    - Focus is purely on dip magnitude and stability
    """

    domain = Domain.ETF

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        # ETFs get a neutral quality score - quality doesn't really apply
        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=70.0,  # Neutral-positive (well-managed ETFs are generally fine)
            sub_scores=[
                SubScore(
                    name="etf_neutral",
                    score=70.0,
                    weight=1.0,
                    reason="ETF: company-level quality metrics not applicable",
                ),
            ],
            data_completeness=1.0,  # Nothing expected
            fallback_used=False,
            notes="ETF scoring bypassed. Focus on dip magnitude and price stability only.",
        )


# =============================================================================
# Utility Adapter
# =============================================================================

class UtilityAdapter(DomainScoringAdapter):
    """Adapter for regulated utilities.
    
    Utilities are different:
    - Stable, regulated earnings
    - High dividend yields expected
    - Higher debt is normal due to capital intensity
    - Growth is low but predictable
    """

    domain = Domain.UTILITY

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []

        # 1. Dividend (35%) - Income is primary
        dividend_yield = _get_metric(info, "dividend_yield")

        expected_keys.append("dividend_yield")

        if dividend_yield is not None:
            # 3-5% is standard for utilities
            score = _normalize_score(dividend_yield, 0.045, 0.015)
            sub_scores.append(SubScore(
                name="dividend",
                score=score,
                weight=0.35,
                reason=f"yield={dividend_yield:.2%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="dividend",
                score=50.0,
                weight=0.35,
                reason="no dividend data",
                available=False,
            ))

        # 2. Operating Margin (25%) - Efficiency matters
        operating_margin = _get_metric(info, "operating_margin")

        expected_keys.append("operating_margin")

        if operating_margin is not None:
            # 15-25% is typical for utilities
            score = _normalize_score(operating_margin, 0.25, 0.10)
            sub_scores.append(SubScore(
                name="operating_margin",
                score=score,
                weight=0.25,
                reason=f"op_margin={operating_margin:.1%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="operating_margin",
                score=50.0,
                weight=0.25,
                reason="no operating margin data",
                available=False,
            ))

        # 3. Debt Coverage (20%) - Important but high debt is normal
        debt_to_equity = _get_metric(info, "debt_to_equity")

        expected_keys.append("debt_to_equity")

        if debt_to_equity is not None:
            # Utilities often have D/E 100-150%, >200% is concerning
            score = _normalize_score(debt_to_equity, 80, 250, higher_better=False)
            sub_scores.append(SubScore(
                name="debt",
                score=score,
                weight=0.20,
                reason=f"D/E={debt_to_equity:.0f}%",
            ))
        else:
            sub_scores.append(SubScore(
                name="debt",
                score=50.0,
                weight=0.20,
                reason="no D/E data",
                available=False,
            ))

        # 4. Analyst (20%)
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.20,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.20,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes="Utility scoring: growth de-emphasized; dividend, margin, and moderate debt accepted",
        )


# =============================================================================
# Biotech Adapter
# =============================================================================

class BiotechAdapter(DomainScoringAdapter):
    """Adapter for pre-revenue or R&D-heavy biotech.
    
    Biotech is different:
    - Many are pre-revenue, so profitability metrics don't apply
    - Cash runway is critical
    - Pipeline value isn't in fundamentals
    - High beta is expected
    """

    domain = Domain.BIOTECH

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []

        # 1. Cash Position (40%) - Cash runway is critical
        total_cash = _get_metric(info, "total_cash")
        market_cap = _get_metric(info, "market_cap")

        expected_keys.extend(["total_cash", "market_cap"])

        if total_cash is not None and market_cap and market_cap > 0:
            cash_ratio = total_cash / market_cap
            # Cash = 30%+ of market cap is strong, <10% is concerning
            score = _normalize_score(cash_ratio, 0.40, 0.05)
            sub_scores.append(SubScore(
                name="cash_position",
                score=score,
                weight=0.40,
                reason=f"cash/mcap={cash_ratio:.0%}, cash=${total_cash/1e9:.1f}B",
            ))
        else:
            sub_scores.append(SubScore(
                name="cash_position",
                score=50.0,
                weight=0.40,
                reason="no cash data",
                available=False,
            ))

        # 2. Analyst (35%) - Analyst coverage is important for biotech
        recommendation_mean = _get_metric(info, "recommendation_mean")
        num_analysts = _get_metric(info, "num_analyst_opinions")

        expected_keys.extend(["recommendation_mean", "num_analyst_opinions"])

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            # Bonus for more analyst coverage
            if num_analysts and num_analysts > 10:
                score = min(100, score + 10)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.35,
                reason=f"recommendation={recommendation_mean:.1f}/5, n={int(num_analysts) if num_analysts else 0}",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.35,
                reason="no analyst data",
                available=False,
            ))

        # 3. Risk Profile (25%) - Beta and short interest
        beta = _get_metric(info, "beta")
        short_percent = _get_metric(info, "short_percent_of_float")

        expected_keys.extend(["beta", "short_percent_of_float"])

        risk_scores = []
        if beta is not None:
            # Beta: 1.0-2.0 is normal for biotech, >3.0 is very high
            beta_score = _normalize_score(beta, 1.0, 3.0, higher_better=False)
            risk_scores.append(beta_score)
        if short_percent is not None:
            # Short interest: <10% is ok, >30% is high risk
            short_score = _normalize_score(short_percent, 0.05, 0.40, higher_better=False)
            risk_scores.append(short_score)

        if risk_scores:
            score = sum(risk_scores) / len(risk_scores)
            sub_scores.append(SubScore(
                name="risk",
                score=score,
                weight=0.25,
                reason=f"beta={beta:.2f}, short={short_percent:.1%}" if beta and short_percent else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="risk",
                score=50.0,
                weight=0.25,
                reason="no risk data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes="Biotech scoring: profitability ignored; cash runway, analyst coverage, and risk profile emphasized",
        )


# =============================================================================
# Energy Adapter
# =============================================================================

class EnergyAdapter(DomainScoringAdapter):
    """Adapter for energy companies (E&P, midstream, refiners).
    
    Energy is different:
    - Highly cyclical, commodity-price dependent
    - High capex intensity
    - Debt levels vary by sub-sector (midstream higher)
    - Free cash flow matters after capex
    """

    domain = Domain.ENERGY

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []

        # 1. Profitability (25%) - Margins matter in commodity business
        operating_margin = _get_metric(info, "operating_margin")
        profit_margin = _get_metric(info, "profit_margin")

        expected_keys.extend(["operating_margin", "profit_margin"])

        if operating_margin is not None or profit_margin is not None:
            # Energy can have wide margin swings
            om_score = _normalize_score(operating_margin or 0, 0.20, 0.0) if operating_margin else 50
            pm_score = _normalize_score(profit_margin or 0, 0.15, 0.0) if profit_margin else 50
            score = (om_score + pm_score) / 2
            sub_scores.append(SubScore(
                name="profitability",
                score=score,
                weight=0.25,
                reason=f"op_margin={operating_margin:.1%}" if operating_margin else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="profitability",
                score=50.0,
                weight=0.25,
                reason="no margin data",
                available=False,
            ))

        # 2. Cash Flow (25%) - FCF yield after capex
        free_cash_flow = _get_metric(info, "free_cash_flow")
        market_cap = _get_metric(info, "market_cap")

        expected_keys.extend(["free_cash_flow", "market_cap"])

        if free_cash_flow is not None and market_cap and market_cap > 0:
            fcf_yield = free_cash_flow / market_cap
            # Energy often has negative FCF in growth periods
            score = _normalize_score(fcf_yield, 0.10, -0.05)
            sub_scores.append(SubScore(
                name="cash_flow",
                score=score,
                weight=0.25,
                reason=f"FCF yield={fcf_yield:.1%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="cash_flow",
                score=50.0,
                weight=0.25,
                reason="no FCF data",
                available=False,
            ))

        # 3. Leverage (20%) - Debt matters especially for midstream
        debt_to_equity = _get_metric(info, "debt_to_equity")

        expected_keys.append("debt_to_equity")

        if debt_to_equity is not None:
            # Energy can handle more debt than most, but not too much
            score = _normalize_score(debt_to_equity, 50, 200, higher_better=False)
            sub_scores.append(SubScore(
                name="leverage",
                score=score,
                weight=0.20,
                reason=f"D/E={debt_to_equity:.0f}%",
            ))
        else:
            sub_scores.append(SubScore(
                name="leverage",
                score=50.0,
                weight=0.20,
                reason="no D/E data",
                available=False,
            ))

        # 4. Dividend (15%) - Many energy names are income plays
        dividend_yield = _get_metric(info, "dividend_yield")

        expected_keys.append("dividend_yield")

        if dividend_yield is not None:
            score = _normalize_score(dividend_yield, 0.05, 0.0)
            sub_scores.append(SubScore(
                name="dividend",
                score=score,
                weight=0.15,
                reason=f"yield={dividend_yield:.2%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="dividend",
                score=50.0,
                weight=0.15,
                reason="no dividend data",
                available=False,
            ))

        # 5. Analyst (15%)
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.15,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.15,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes="Energy scoring: commodity-cycle aware; FCF and leverage emphasized",
        )


# =============================================================================
# Retail Adapter
# =============================================================================

class RetailAdapter(DomainScoringAdapter):
    """Adapter for retail and consumer companies.
    
    Retail is different:
    - Thin margins are normal
    - Inventory turnover matters (not in yfinance)
    - Consumer discretionary vs staples dynamics
    - Growth expectations vary widely
    """

    domain = Domain.RETAIL

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []

        # 1. Margins (25%) - Thin margins are acceptable
        profit_margin = _get_metric(info, "profit_margin")
        operating_margin = _get_metric(info, "operating_margin")

        expected_keys.extend(["profit_margin", "operating_margin"])

        if profit_margin is not None or operating_margin is not None:
            # Retail margins: 2-5% net is good, >10% is exceptional
            pm_score = _normalize_score(profit_margin or 0, 0.08, 0.0) if profit_margin else 50
            om_score = _normalize_score(operating_margin or 0, 0.12, 0.0) if operating_margin else 50
            score = (pm_score + om_score) / 2
            sub_scores.append(SubScore(
                name="margins",
                score=score,
                weight=0.25,
                reason=f"profit_margin={profit_margin:.1%}" if profit_margin else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="margins",
                score=50.0,
                weight=0.25,
                reason="no margin data",
                available=False,
            ))

        # 2. Growth (25%) - Growth is key for retail
        revenue_growth = _get_metric(info, "revenue_growth")
        earnings_growth = _get_metric(info, "earnings_growth")

        expected_keys.extend(["revenue_growth", "earnings_growth"])

        if revenue_growth is not None or earnings_growth is not None:
            rg_score = _normalize_score(revenue_growth or 0, 0.15, -0.05) if revenue_growth is not None else 50
            eg_score = _normalize_score(earnings_growth or 0, 0.20, -0.10) if earnings_growth is not None else 50
            score = (rg_score + eg_score) / 2
            sub_scores.append(SubScore(
                name="growth",
                score=score,
                weight=0.25,
                reason=f"rev_growth={revenue_growth:.1%}" if revenue_growth else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="growth",
                score=50.0,
                weight=0.25,
                reason="no growth data",
                available=False,
            ))

        # 3. Valuation (20%)
        pe_ratio = _get_metric(info, "pe_ratio")
        peg_ratio = _get_metric(info, "peg_ratio")

        expected_keys.extend(["pe_ratio", "peg_ratio"])

        val_scores = []
        if pe_ratio is not None and 0 < pe_ratio < 100:
            val_scores.append(_normalize_score(pe_ratio, 12, 35, higher_better=False))
        if peg_ratio is not None and 0 < peg_ratio < 5:
            val_scores.append(_normalize_score(peg_ratio, 0.8, 2.5, higher_better=False))

        if val_scores:
            score = sum(val_scores) / len(val_scores)
            sub_scores.append(SubScore(
                name="valuation",
                score=score,
                weight=0.20,
                reason=f"P/E={pe_ratio:.1f}" if pe_ratio else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="valuation",
                score=50.0,
                weight=0.20,
                reason="no valuation data",
                available=False,
            ))

        # 4. Balance Sheet (15%)
        debt_to_equity = _get_metric(info, "debt_to_equity")
        current_ratio = _get_metric(info, "current_ratio")

        expected_keys.extend(["debt_to_equity", "current_ratio"])

        if debt_to_equity is not None or current_ratio is not None:
            de_score = _normalize_score(debt_to_equity or 100, 0, 150, higher_better=False) if debt_to_equity is not None else 50
            cr_score = _normalize_score(current_ratio or 1, 1.5, 0.8) if current_ratio is not None else 50
            score = (de_score + cr_score) / 2
            sub_scores.append(SubScore(
                name="balance_sheet",
                score=score,
                weight=0.15,
                reason=f"D/E={debt_to_equity:.0f}%" if debt_to_equity else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="balance_sheet",
                score=50.0,
                weight=0.15,
                reason="no balance sheet data",
                available=False,
            ))

        # 5. Analyst (15%)
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.15,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.15,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes="Retail scoring: thin margins accepted; growth and valuation emphasized",
        )


# =============================================================================
# Semiconductor Adapter
# =============================================================================

class SemiconductorAdapter(DomainScoringAdapter):
    """Adapter for semiconductor and hardware companies.
    
    Semis are different:
    - Highly cyclical
    - High R&D and capex intensity
    - Inventory cycles matter (not in yfinance)
    - Strong margins when times are good
    """

    domain = Domain.SEMICONDUCTOR

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []

        # 1. Margins (30%) - High margins are key for semis
        operating_margin = _get_metric(info, "operating_margin")
        profit_margin = _get_metric(info, "profit_margin")

        expected_keys.extend(["operating_margin", "profit_margin"])

        if operating_margin is not None or profit_margin is not None:
            # Semis can have very high margins: 30%+ is great
            om_score = _normalize_score(operating_margin or 0, 0.35, 0.05) if operating_margin else 50
            pm_score = _normalize_score(profit_margin or 0, 0.25, 0.0) if profit_margin else 50
            score = (om_score + pm_score) / 2
            sub_scores.append(SubScore(
                name="margins",
                score=score,
                weight=0.30,
                reason=f"op_margin={operating_margin:.1%}" if operating_margin else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="margins",
                score=50.0,
                weight=0.30,
                reason="no margin data",
                available=False,
            ))

        # 2. Growth (25%) - Semis need to grow
        revenue_growth = _get_metric(info, "revenue_growth")
        earnings_growth = _get_metric(info, "earnings_growth")

        expected_keys.extend(["revenue_growth", "earnings_growth"])

        if revenue_growth is not None or earnings_growth is not None:
            rg_score = _normalize_score(revenue_growth or 0, 0.20, -0.15) if revenue_growth is not None else 50
            eg_score = _normalize_score(earnings_growth or 0, 0.25, -0.20) if earnings_growth is not None else 50
            score = (rg_score + eg_score) / 2
            sub_scores.append(SubScore(
                name="growth",
                score=score,
                weight=0.25,
                reason=f"rev_growth={revenue_growth:.1%}" if revenue_growth else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="growth",
                score=50.0,
                weight=0.25,
                reason="no growth data",
                available=False,
            ))

        # 3. Balance Sheet (15%) - Low debt is good, cash is king
        debt_to_equity = _get_metric(info, "debt_to_equity")
        total_cash = _get_metric(info, "total_cash")
        market_cap = _get_metric(info, "market_cap")

        expected_keys.extend(["debt_to_equity", "total_cash", "market_cap"])

        balance_scores = []
        if debt_to_equity is not None:
            balance_scores.append(_normalize_score(debt_to_equity, 0, 100, higher_better=False))
        if total_cash is not None and market_cap and market_cap > 0:
            cash_ratio = total_cash / market_cap
            balance_scores.append(_normalize_score(cash_ratio, 0.20, 0.02))

        if balance_scores:
            score = sum(balance_scores) / len(balance_scores)
            sub_scores.append(SubScore(
                name="balance_sheet",
                score=score,
                weight=0.15,
                reason=f"D/E={debt_to_equity:.0f}%" if debt_to_equity else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="balance_sheet",
                score=50.0,
                weight=0.15,
                reason="no balance sheet data",
                available=False,
            ))

        # 4. Valuation (15%) - Can trade at high multiples
        pe_ratio = _get_metric(info, "pe_ratio")
        forward_pe = _get_metric(info, "forward_pe")

        expected_keys.extend(["pe_ratio", "forward_pe"])

        if pe_ratio is not None and 0 < pe_ratio < 100:
            # Semis can trade at higher multiples
            score = _normalize_score(pe_ratio, 15, 50, higher_better=False)
            sub_scores.append(SubScore(
                name="valuation",
                score=score,
                weight=0.15,
                reason=f"P/E={pe_ratio:.1f}",
            ))
        else:
            sub_scores.append(SubScore(
                name="valuation",
                score=50.0,
                weight=0.15,
                reason="no P/E data",
                available=False,
            ))

        # 5. Analyst (15%)
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.15,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.15,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes="Semiconductor scoring: cyclical-aware; high margins and growth emphasized",
        )


# =============================================================================
# Capital Intensive Adapter (Airlines, Shipping)
# =============================================================================

class CapitalIntensiveAdapter(DomainScoringAdapter):
    """Adapter for capital-intensive transport (airlines, shipping).
    
    These businesses are different:
    - Very capital intensive (fleet/vessels)
    - High operating leverage
    - Fuel/commodity exposure
    - Often thin margins, high debt
    """

    domain = Domain.AIRLINE

    def compute_quality_score(
        self,
        info: dict,
        fundamentals: dict | None = None,
    ) -> DomainScoreResult:
        sub_scores = []
        expected_keys = []

        # 1. Margins (30%) - Thin margins are normal
        operating_margin = _get_metric(info, "operating_margin")
        profit_margin = _get_metric(info, "profit_margin")

        expected_keys.extend(["operating_margin", "profit_margin"])

        if operating_margin is not None or profit_margin is not None:
            # Airlines: 5-10% operating margin is decent
            om_score = _normalize_score(operating_margin or 0, 0.12, -0.05) if operating_margin else 50
            pm_score = _normalize_score(profit_margin or 0, 0.08, -0.05) if profit_margin else 50
            score = (om_score + pm_score) / 2
            sub_scores.append(SubScore(
                name="margins",
                score=score,
                weight=0.30,
                reason=f"op_margin={operating_margin:.1%}" if operating_margin else "partial data",
            ))
        else:
            sub_scores.append(SubScore(
                name="margins",
                score=50.0,
                weight=0.30,
                reason="no margin data",
                available=False,
            ))

        # 2. Leverage (25%) - High debt is common but risky
        debt_to_equity = _get_metric(info, "debt_to_equity")

        expected_keys.append("debt_to_equity")

        if debt_to_equity is not None:
            # Airlines often have high D/E, but >300% is dangerous
            score = _normalize_score(debt_to_equity, 50, 350, higher_better=False)
            sub_scores.append(SubScore(
                name="leverage",
                score=score,
                weight=0.25,
                reason=f"D/E={debt_to_equity:.0f}%",
            ))
        else:
            sub_scores.append(SubScore(
                name="leverage",
                score=50.0,
                weight=0.25,
                reason="no D/E data",
                available=False,
            ))

        # 3. Cash Flow (20%) - FCF matters for capex coverage
        free_cash_flow = _get_metric(info, "free_cash_flow")
        market_cap = _get_metric(info, "market_cap")

        expected_keys.extend(["free_cash_flow", "market_cap"])

        if free_cash_flow is not None and market_cap and market_cap > 0:
            fcf_yield = free_cash_flow / market_cap
            score = _normalize_score(fcf_yield, 0.08, -0.08)
            sub_scores.append(SubScore(
                name="cash_flow",
                score=score,
                weight=0.20,
                reason=f"FCF yield={fcf_yield:.1%}",
            ))
        else:
            sub_scores.append(SubScore(
                name="cash_flow",
                score=50.0,
                weight=0.20,
                reason="no FCF data",
                available=False,
            ))

        # 4. Analyst (25%) - Analyst view matters for cyclicals
        recommendation_mean = _get_metric(info, "recommendation_mean")

        expected_keys.append("recommendation_mean")

        if recommendation_mean is not None:
            score = _normalize_score(recommendation_mean, 1.0, 4.0, higher_better=False)
            sub_scores.append(SubScore(
                name="analyst",
                score=score,
                weight=0.25,
                reason=f"recommendation={recommendation_mean:.1f}/5",
            ))
        else:
            sub_scores.append(SubScore(
                name="analyst",
                score=50.0,
                weight=0.25,
                reason="no analyst data",
                available=False,
            ))

        # Calculate final score
        total_weight = sum(s.weight for s in sub_scores if s.available)
        if total_weight > 0:
            final_score = sum(s.weighted_score for s in sub_scores) / total_weight
        else:
            final_score = 50.0

        available, total = self._count_available(info, expected_keys)
        data_completeness = available / total if total > 0 else 0.0

        return DomainScoreResult(
            domain=self.domain,
            domain_confidence=1.0,
            final_score=final_score,
            sub_scores=sub_scores,
            data_completeness=data_completeness,
            fallback_used=False,
            notes="Capital-intensive transport: high debt tolerance; margins and analyst sentiment emphasized",
        )


# =============================================================================
# Adapter Registry
# =============================================================================

_ADAPTERS: dict[Domain, type[DomainScoringAdapter]] = {
    Domain.OPERATING_COMPANY: OperatingCompanyAdapter,
    Domain.BANK: BankAdapter,
    Domain.INSURER: InsuranceAdapter,  # Dedicated insurance adapter
    Domain.ASSET_MANAGER: BankAdapter,  # Use bank adapter
    Domain.REIT: REITAdapter,
    Domain.ETF: ETFAdapter,
    Domain.MUTUAL_FUND: ETFAdapter,
    Domain.INDEX: ETFAdapter,
    Domain.UTILITY: UtilityAdapter,
    Domain.TELECOM: UtilityAdapter,  # Similar characteristics
    Domain.BIOTECH: BiotechAdapter,
    Domain.PHARMA: OperatingCompanyAdapter,  # Established pharma uses standard metrics
    Domain.ENERGY: EnergyAdapter,
    Domain.MINING: EnergyAdapter,  # Similar commodity-based characteristics
    Domain.RETAIL: RetailAdapter,
    Domain.SEMICONDUCTOR: SemiconductorAdapter,
    Domain.AIRLINE: CapitalIntensiveAdapter,
    Domain.SHIPPING: CapitalIntensiveAdapter,
}


def get_adapter(domain: Domain) -> DomainScoringAdapter:
    """Get the scoring adapter for a domain."""
    adapter_class = _ADAPTERS.get(domain, OperatingCompanyAdapter)
    return adapter_class()


def compute_domain_score(
    classification: DomainClassification,
    info: dict,
    fundamentals: dict | None = None,
) -> DomainScoreResult:
    """
    Compute quality score using the appropriate domain adapter.
    
    If domain confidence is low and we have a fallback, compute both
    and blend the scores.
    
    Args:
        classification: Result from classify_domain()
        info: Dict from yfinance_service.get_ticker_info()
        fundamentals: Optional additional data
    
    Returns:
        DomainScoreResult with domain-appropriate scoring
    """
    adapter = get_adapter(classification.domain)
    result = adapter.compute_quality_score(info, fundamentals)

    # Update result with classification confidence
    result = DomainScoreResult(
        domain=result.domain,
        domain_confidence=classification.confidence,
        final_score=result.final_score,
        sub_scores=result.sub_scores,
        data_completeness=result.data_completeness,
        fallback_used=result.fallback_used,
        notes=result.notes,
    )

    # If low confidence and we have a fallback, blend scores
    if classification.confidence < 0.70 and classification.fallback_domain:
        fallback_adapter = get_adapter(classification.fallback_domain)
        fallback_result = fallback_adapter.compute_quality_score(info, fundamentals)

        # Blend: use classification confidence as weight
        blended_score = (
            result.final_score * classification.confidence +
            fallback_result.final_score * (1 - classification.confidence)
        )

        result = DomainScoreResult(
            domain=result.domain,
            domain_confidence=classification.confidence,
            final_score=blended_score,
            sub_scores=result.sub_scores,  # Keep primary domain breakdown
            data_completeness=result.data_completeness,
            fallback_used=True,
            notes=f"Blended with {classification.fallback_domain.value} fallback ({1-classification.confidence:.0%} weight)",
        )

    return result
