"""
Centralized Stock Classification Utilities.

This module provides stock/ETF classification logic used across the codebase.
All stock type detection should be imported from here to avoid duplication.

Usage:
    from app.core.stock_classification import is_etf_or_index, detect_domain
"""

from __future__ import annotations

from typing import Any

from app.core.data_helpers import latest_value


def is_etf_or_index(symbol: str, quote_type: str | None = None) -> bool:
    """
    Check if symbol is an ETF, index, or fund (not a stock).
    
    Args:
        symbol: Ticker symbol
        quote_type: Optional quote type from yfinance
        
    Returns:
        True if ETF/index/fund, False if stock
    """
    # Index symbols start with ^
    if symbol.startswith("^"):
        return True
    # Check quote type
    if quote_type:
        return quote_type.upper() in ("ETF", "INDEX", "MUTUALFUND", "TRUST")
    return False


def detect_domain(info: dict[str, Any]) -> str | None:
    """
    Detect the domain type from ticker info.
    
    Domains are specialized stock categories that require different
    analysis approaches:
    - bank: Use Net Interest Margin, not standard P/E
    - reit: Use FFO and P/FFO, not earnings
    - insurer: Use combined ratio, loss ratio
    - utility: Capital-intensive, regulated returns
    - biotech: R&D focused, often no earnings
    - etf: Fund, not individual company
    
    Args:
        info: Ticker info dict from yfinance
        
    Returns:
        Domain string or None for standard stocks
    """
    quote_type = (info.get("quote_type") or info.get("quoteType") or "").upper()
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    name = (info.get("name") or info.get("shortName") or "").lower()

    # ETFs and Funds
    if quote_type in ("ETF", "MUTUALFUND", "INDEX"):
        return "etf"

    # Banks
    if "bank" in industry or "bank" in name:
        return "bank"
    if sector == "financial services" and any(
        term in industry for term in ["banks", "credit", "savings"]
    ):
        return "bank"

    # REITs
    if "reit" in industry or "reit" in name or "real estate" in industry:
        return "reit"

    # Insurance
    if "insurance" in industry:
        return "insurer"

    # Utilities
    if sector == "utilities" or "utility" in industry:
        return "utility"

    # Biotech
    if "biotechnology" in industry:
        return "biotech"

    return None


def calculate_domain_metrics(
    info: dict[str, Any],
    financials: dict[str, Any] | None,
    domain: str | None,
) -> dict[str, Any]:
    """
    Calculate domain-specific metrics from financial statements.
    
    Different stock domains require specialized metrics:
    - Banks: Net Interest Margin, NII
    - REITs: FFO, P/FFO
    - Insurers: Loss Ratio
    
    Args:
        info: Ticker info dict
        financials: Financial statements dict with quarterly/annual data
        domain: Domain type from detect_domain()
        
    Returns:
        Dict of domain-specific metrics
    """
    metrics: dict[str, Any] = {}

    if not financials or not domain:
        return metrics

    quarterly = financials.get("quarterly", {})
    income = quarterly.get("income_statement", {})
    balance = quarterly.get("balance_sheet", {})
    cashflow = quarterly.get("cash_flow", {})

    if domain == "bank":
        metrics.update(_calculate_bank_metrics(income, balance))
    elif domain == "reit":
        metrics.update(_calculate_reit_metrics(info, income, cashflow))
    elif domain == "insurer":
        metrics.update(_calculate_insurer_metrics(income))

    return metrics


def _calculate_bank_metrics(
    income: dict[str, Any],
    balance: dict[str, Any],
) -> dict[str, Any]:
    """Calculate bank-specific metrics."""
    metrics: dict[str, Any] = {}
    
    # Net Interest Income
    nii = latest_value(income.get("Net Interest Income"))
    if nii:
        metrics["net_interest_income"] = nii

        # Calculate NIM proxy: NII (annualized) / Total Assets
        total_assets = latest_value(balance.get("Total Assets"))
        if total_assets and total_assets > 0:
            metrics["net_interest_margin"] = (nii * 4) / total_assets

    return metrics


def _calculate_reit_metrics(
    info: dict[str, Any],
    income: dict[str, Any],
    cashflow: dict[str, Any],
) -> dict[str, Any]:
    """Calculate REIT-specific metrics (FFO, P/FFO)."""
    metrics: dict[str, Any] = {}
    
    # FFO = Net Income + Depreciation
    net_income = latest_value(income.get("Net Income"))
    depreciation = (
        cashflow.get("Depreciation Amortization Depletion") or
        cashflow.get("Depreciation And Amortization")
    )
    depreciation = latest_value(depreciation)

    if net_income is not None and depreciation is not None:
        ffo = net_income + depreciation
        metrics["ffo"] = ffo * 4  # Annualize

        shares = latest_value(
            info.get("shares_outstanding") or info.get("sharesOutstanding")
        )
        current_price = latest_value(
            info.get("current_price") or info.get("regularMarketPrice")
        )

        if shares and shares > 0:
            ffo_per_share = (ffo * 4) / shares
            metrics["ffo_per_share"] = ffo_per_share

            if current_price and ffo_per_share > 0:
                metrics["p_ffo"] = current_price / ffo_per_share

    return metrics


def _calculate_insurer_metrics(income: dict[str, Any]) -> dict[str, Any]:
    """Calculate insurance company-specific metrics."""
    metrics: dict[str, Any] = {}
    
    # Loss Ratio = Losses / Premiums (approximated by revenue)
    loss_expense = latest_value(
        income.get("Net Policyholder Benefits And Claims") or
        income.get("Loss Adjustment Expense")
    )
    revenue = latest_value(
        income.get("Total Revenue") or
        income.get("Operating Revenue")
    )

    if loss_expense and revenue and revenue > 0:
        metrics["loss_ratio"] = loss_expense / revenue

    return metrics


__all__ = [
    "is_etf_or_index",
    "detect_domain",
    "calculate_domain_metrics",
]
