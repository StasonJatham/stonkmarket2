"""Sector-aware risk highlights for portfolio holdings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from app.core.logging import get_logger
from app.dipfinder.structural_analysis import _extract_quarterly_series
from app.quant_engine.core import Sector, normalize_sector
from app.repositories import symbols_orm
from app.services.fundamentals import get_fundamentals_with_status


logger = get_logger("quant_engine.risk_highlights")

TREND_PERIODS = 3
MAX_HIGHLIGHTS_PER_SYMBOL = 3

_SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass
class RiskHighlight:
    title: str
    detail: str
    severity: str
    confidence: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "detail": self.detail,
            "severity": self.severity,
            "confidence": self.confidence,
        }


def _format_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def _series(statement: dict[str, Any] | None, names: list[str]) -> list[float | None]:
    if not statement:
        return []
    for name in names:
        series = _extract_quarterly_series(statement, name)
        if series:
            return series
    return []


def _ratio_series(
    numerators: list[float | None],
    denominators: list[float | None],
) -> list[float | None]:
    length = min(len(numerators), len(denominators))
    output: list[float | None] = []
    for i in range(length):
        num = numerators[i]
        denom = denominators[i]
        if num is None or denom is None or denom == 0:
            output.append(None)
        else:
            output.append(num / denom)
    return output


def _scale_series(series: list[float | None], factor: float) -> list[float | None]:
    return [None if value is None else value * factor for value in series]


def _trend(
    series: list[float | None],
    direction: str,
    *,
    periods: int = TREND_PERIODS,
    min_change: float = 0.0,
) -> bool:
    if len(series) < periods + 1:
        return False
    for idx in range(periods):
        current = series[idx]
        previous = series[idx + 1]
        if current is None or previous is None:
            return False
        if direction == "up":
            if current <= previous * (1 + min_change):
                return False
        else:
            if current >= previous * (1 - min_change):
                return False
    return True


def _pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return (current - previous) / abs(previous)


def _append_highlight(
    highlights: list[RiskHighlight],
    *,
    title: str,
    detail: str,
    severity: str,
    confidence: str,
    score: float,
) -> None:
    highlights.append(
        RiskHighlight(
            title=title,
            detail=detail,
            severity=severity,
            confidence=confidence,
            score=score,
        )
    )


def _general_highlights(
    income_stmt: dict[str, Any],
    balance_sheet: dict[str, Any],
    _cash_flow: dict[str, Any],
    sector: Sector,
) -> list[RiskHighlight]:
    highlights: list[RiskHighlight] = []

    debt = _series(balance_sheet, ["TotalDebt", "Total Debt", "LongTermDebt", "Long Term Debt"])
    equity = _series(
        balance_sheet,
        [
            "TotalStockholderEquity",
            "Total Stockholder Equity",
            "StockholdersEquity",
            "TotalEquity",
            "Total Equity",
        ],
    )
    debt_to_equity = _ratio_series(debt, equity)
    if _trend(debt_to_equity, "up", min_change=0.02):
        latest = debt_to_equity[0]
        prior = debt_to_equity[TREND_PERIODS]
        change = (latest - prior) if latest is not None and prior is not None else None
        if latest is not None and (latest >= 1.5 or (change is not None and change >= 0.3)):
            severity = "high" if latest >= 2.0 or (change is not None and change >= 0.5) else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = (
                f"Debt-to-equity climbed for {TREND_PERIODS} straight quarters "
                f"({prior:.2f} -> {latest:.2f})."
            )
            score = abs(change) * 100 if change is not None else 0
            _append_highlight(
                highlights,
                title="Leverage building",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    operating_income = _series(income_stmt, ["OperatingIncome", "Operating Income", "EBIT"])
    interest_expense = _series(income_stmt, ["InterestExpense", "Interest Expense"])
    interest_expense = [None if v is None else abs(v) for v in interest_expense]
    coverage = _ratio_series(operating_income, interest_expense)
    coverage_threshold = 2.5 if sector == Sector.UTILITIES else 3.0
    if _trend(coverage, "down", min_change=0.05):
        latest = coverage[0]
        prior = coverage[TREND_PERIODS]
        if latest is not None and latest < coverage_threshold:
            severity = "high" if latest < 2 else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = (
                f"Interest coverage fell to {_format_ratio(latest)} after "
                f"{TREND_PERIODS} declines."
            )
            score = abs(latest - prior) * 10 if prior is not None else 0
            _append_highlight(
                highlights,
                title="Interest coverage weakening",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    revenue = _series(income_stmt, ["TotalRevenue", "Total Revenue"])
    operating_margin = _ratio_series(operating_income, revenue) if revenue else []
    gross_profit = _series(income_stmt, ["GrossProfit", "Gross Profit"])
    gross_margin = _ratio_series(gross_profit, revenue) if revenue else []
    margin_series = operating_margin or gross_margin
    margin_label = "Operating margin" if operating_margin else "Gross margin"

    if margin_series and _trend(margin_series, "down", min_change=0.01):
        latest = margin_series[0]
        prior = margin_series[TREND_PERIODS]
        if latest is not None and prior is not None:
            drop = prior - latest
            if drop >= 0.05:
                severity = "high" if drop >= 0.1 or latest < 0.1 else "medium"
                confidence = "high" if severity == "high" else "medium"
                detail = (
                    f"{margin_label} fell {drop * 100:.1f}pp over {TREND_PERIODS} quarters "
                    f"({_format_pct(prior)} -> {_format_pct(latest)})."
                )
                score = drop * 100
                _append_highlight(
                    highlights,
                    title="Margin compression",
                    detail=detail,
                    severity=severity,
                    confidence=confidence,
                    score=score,
                )

    return highlights


def _bank_highlights(
    income_stmt: dict[str, Any],
    balance_sheet: dict[str, Any],
) -> list[RiskHighlight]:
    highlights: list[RiskHighlight] = []

    net_interest_income = _series(income_stmt, ["NetInterestIncome", "Net Interest Income"])
    total_assets = _series(balance_sheet, ["TotalAssets", "Total Assets"])
    nim_series = _ratio_series(_scale_series(net_interest_income, 4), total_assets)

    if _trend(nim_series, "down", min_change=0.01):
        latest = nim_series[0]
        prior = nim_series[TREND_PERIODS]
        if latest is not None and prior is not None and latest < 0.03:
            drop = prior - latest
            severity = "high" if latest < 0.02 or drop >= 0.005 else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = (
                f"Net interest margin compressed for {TREND_PERIODS} quarters "
                f"({_format_pct(prior, 2)} -> {_format_pct(latest, 2)})."
            )
            score = drop * 1000
            _append_highlight(
                highlights,
                title="Net interest margin pressure",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    provisions = _series(
        income_stmt,
        [
            "ProvisionForCreditLosses",
            "Provision For Credit Losses",
            "ProvisionForLoanLosses",
            "Provision For Loan Losses",
        ],
    )
    if _trend(provisions, "up", min_change=0.05):
        change = _pct_change(provisions[0], provisions[TREND_PERIODS])
        if change is not None and change >= 0.2:
            severity = "high" if change >= 0.5 else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = (
                f"Credit loss provisions rose {_format_pct(change)} vs {TREND_PERIODS} quarters ago."
            )
            score = change * 100
            _append_highlight(
                highlights,
                title="Credit costs rising",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    if _trend(net_interest_income, "down", min_change=0.02):
        change = _pct_change(net_interest_income[0], net_interest_income[TREND_PERIODS])
        if change is not None and change <= -0.1:
            severity = "high" if change <= -0.2 else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = (
                f"Net interest income down {_format_pct(abs(change))} "
                f"over {TREND_PERIODS} quarters."
            )
            score = abs(change) * 100
            _append_highlight(
                highlights,
                title="Net interest income declining",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    return highlights


def _insurer_highlights(
    income_stmt: dict[str, Any],
) -> list[RiskHighlight]:
    highlights: list[RiskHighlight] = []

    benefits = _series(
        income_stmt,
        [
            "NetPolicyholderBenefitsAndClaims",
            "Net Policyholder Benefits And Claims",
            "PolicyholderBenefitsAndClaims",
            "Policyholder Benefits And Claims",
            "NetPolicyholderBenefits",
            "Net Policyholder Benefits",
        ],
    )
    revenue = _series(
        income_stmt,
        [
            "TotalRevenue",
            "Total Revenue",
            "OperatingRevenue",
            "Operating Revenue",
            "PremiumsEarned",
            "NetPremiumsEarned",
        ],
    )

    loss_ratio = _ratio_series(benefits, revenue)
    if _trend(loss_ratio, "up", min_change=0.02):
        latest = loss_ratio[0]
        prior = loss_ratio[TREND_PERIODS]
        if latest is not None and prior is not None and latest > 0.6:
            change = latest - prior
            severity = "high" if latest > 0.8 or change >= 0.1 else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = (
                f"Loss ratio increased {_format_pct(change)} over {TREND_PERIODS} quarters "
                f"({_format_pct(prior)} -> {_format_pct(latest)})."
            )
            score = change * 100
            _append_highlight(
                highlights,
                title="Claims pressure building",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    return highlights


def _reit_highlights(
    income_stmt: dict[str, Any],
    balance_sheet: dict[str, Any],
    cash_flow: dict[str, Any],
) -> list[RiskHighlight]:
    highlights: list[RiskHighlight] = []

    net_income = _series(income_stmt, ["NetIncome", "Net Income"])
    depreciation = _series(
        cash_flow,
        [
            "DepreciationAmortizationDepletion",
            "Depreciation Amortization Depletion",
            "Depreciation And Amortization",
        ],
    )
    ffo_series = []
    for idx in range(min(len(net_income), len(depreciation))):
        ni = net_income[idx]
        dep = depreciation[idx]
        ffo_series.append(None if ni is None or dep is None else ni + dep)

    if _trend(ffo_series, "down", min_change=0.02):
        change = _pct_change(ffo_series[0], ffo_series[TREND_PERIODS])
        if change is not None and change <= -0.1:
            severity = "high" if change <= -0.2 else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = f"FFO down {_format_pct(abs(change))} over {TREND_PERIODS} quarters."
            score = abs(change) * 100
            _append_highlight(
                highlights,
                title="FFO trending lower",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    debt = _series(balance_sheet, ["TotalDebt", "Total Debt", "LongTermDebt", "Long Term Debt"])
    assets = _series(balance_sheet, ["TotalAssets", "Total Assets"])
    debt_to_assets = _ratio_series(debt, assets)
    if _trend(debt_to_assets, "up", min_change=0.02):
        latest = debt_to_assets[0]
        prior = debt_to_assets[TREND_PERIODS]
        if latest is not None and prior is not None and latest > 0.6:
            change = latest - prior
            severity = "high" if latest > 0.7 or change >= 0.1 else "medium"
            confidence = "high" if severity == "high" else "medium"
            detail = (
                f"Leverage rising ({prior:.2f} -> {latest:.2f} debt/assets)."
            )
            score = change * 100
            _append_highlight(
                highlights,
                title="Leverage rising",
                detail=detail,
                severity=severity,
                confidence=confidence,
                score=score,
            )

    return highlights


def _select_highlights(highlights: list[RiskHighlight]) -> list[RiskHighlight]:
    highlights.sort(
        key=lambda h: (_SEVERITY_ORDER.get(h.severity, 0), h.score),
        reverse=True,
    )
    return highlights[:MAX_HIGHLIGHTS_PER_SYMBOL]


def _build_highlights_for_symbol(
    fundamentals: dict[str, Any],
    sector_name: str | None,
) -> list[RiskHighlight]:
    income_stmt = fundamentals.get("income_stmt_quarterly") or {}
    balance_sheet = fundamentals.get("balance_sheet_quarterly") or {}
    cash_flow = fundamentals.get("cash_flow_quarterly") or {}

    domain = (fundamentals.get("domain") or "stock").lower()
    sector = normalize_sector(sector_name)

    if domain == "bank":
        highlights = _bank_highlights(income_stmt, balance_sheet)
    elif domain == "insurer":
        highlights = _insurer_highlights(income_stmt)
    elif domain == "reit":
        highlights = _reit_highlights(income_stmt, balance_sheet, cash_flow)
    else:
        highlights = _general_highlights(income_stmt, balance_sheet, cash_flow, sector)

    return _select_highlights(highlights)


async def build_portfolio_risk_highlights(symbols: list[str]) -> list[dict[str, Any]]:
    """Compute sector-aware risk highlights for a list of symbols."""
    async def _build(symbol: str) -> dict[str, Any] | None:
        try:
            fundamentals, _stale = await get_fundamentals_with_status(symbol, allow_stale=True)
            if not fundamentals:
                return None
            symbol_row = await symbols_orm.get_symbol(symbol)
            sector_name = symbol_row.sector if symbol_row else None
            highlights = _build_highlights_for_symbol(fundamentals, sector_name)
            if not highlights:
                return None
            return {
                "symbol": symbol,
                "sector": sector_name,
                "domain": fundamentals.get("domain"),
                "highlights": [h.to_dict() for h in highlights],
            }
        except Exception as exc:
            logger.debug(f"Risk highlights failed for {symbol}: {exc}")
            return None

    results = await asyncio.gather(*[_build(symbol) for symbol in symbols])
    return [item for item in results if item]
