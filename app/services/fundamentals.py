"""Stock fundamentals service - fetches and stores financial metrics.

MIGRATED: Now uses unified YFinanceService for yfinance calls.
Fundamentals storage to stock_fundamentals table is kept for longer-term caching.

Fundamentals are refreshed monthly (vs prices which update daily).
This data is used for AI analysis to make better buy/hold/sell recommendations.

Usage:
    from app.services.fundamentals import (
        get_fundamentals,
        refresh_fundamentals,
        get_fundamentals_for_analysis,
    )
    
    # Get cached fundamentals (fetches if expired)
    fundamentals = await get_fundamentals("AAPL")
    
    # Force refresh
    fundamentals = await refresh_fundamentals("AAPL")
    
    # Get formatted data for AI prompts
    context = await get_fundamentals_for_analysis("AAPL")
"""

from __future__ import annotations

import asyncio
from datetime import datetime, date, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import StockFundamentals, Symbol
from app.services.data_providers import get_yfinance_service

logger = get_logger("services.fundamentals")


def _is_etf_or_index(symbol: str, quote_type: Optional[str] = None) -> bool:
    """Check if symbol is an ETF or index (no fundamentals available)."""
    if symbol.startswith("^"):
        return True
    if quote_type:
        quote_type_upper = quote_type.upper()
        return quote_type_upper in ("ETF", "INDEX", "MUTUALFUND", "TRUST")
    return False


def _safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float, returning None for invalid values."""
    if value is None:
        return None
    try:
        f = float(value)
        if f != f or f == float('inf') or f == float('-inf'):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Safely convert value to int, returning None for invalid values."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_date(value: Any) -> Optional[date]:
    """Safely convert value to date, returning None for invalid values.
    
    Handles:
    - datetime objects
    - date objects  
    - pandas Timestamp
    - ISO format strings (YYYY-MM-DD)
    """
    if value is None:
        return None
    try:
        # Already a date
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        # datetime -> date
        if isinstance(value, datetime):
            return value.date()
        # pandas Timestamp
        if hasattr(value, 'date') and callable(value.date):
            return value.date()
        # String in ISO format
        if isinstance(value, str):
            # Handle 'YYYY-MM-DD' format
            return datetime.strptime(value.split('T')[0], '%Y-%m-%d').date()
        return None
    except (ValueError, TypeError, AttributeError):
        return None


# =============================================================================
# Domain Detection and Metrics
# =============================================================================


def _detect_domain(info: dict[str, Any]) -> str:
    """
    Detect the domain type of a security from its info.
    
    Returns: bank, reit, insurer, utility, biotech, etf, stock
    """
    quote_type = (info.get("quote_type") or info.get("quoteType") or "").upper()
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    name = (info.get("name") or info.get("shortName") or "").lower()
    
    # ETFs and funds
    if quote_type in ("ETF", "MUTUALFUND", "INDEX"):
        return "etf"
    
    # Banks
    bank_keywords = ["bank", "bancorp", "bancshares", "banking"]
    if sector == "financial services":
        if any(kw in industry for kw in ["bank", "credit"]):
            return "bank"
        if any(kw in name for kw in bank_keywords):
            return "bank"
    
    # REITs
    if "reit" in industry or "real estate investment" in industry:
        return "reit"
    if quote_type == "REIT":
        return "reit"
    if "reit" in name:
        return "reit"
    
    # Insurers
    insurer_keywords = ["insurance", "insurer", "assurance", "reinsurance"]
    if any(kw in industry for kw in insurer_keywords):
        return "insurer"
    if any(kw in name for kw in insurer_keywords):
        return "insurer"
    
    # Utilities
    if sector == "utilities":
        return "utility"
    
    # Biotech
    if "biotech" in industry or "pharmaceutical" in industry:
        return "biotech"
    if sector == "healthcare" and "drug" in industry:
        return "biotech"
    
    return "stock"


def _calculate_domain_metrics(
    info: dict[str, Any],
    financials: Optional[dict[str, Any]],
    domain: str,
) -> dict[str, Any]:
    """
    Calculate domain-specific metrics from financial statements.
    
    Returns dict with:
    - Banks: net_interest_income, net_interest_margin
    - REITs: ffo, ffo_per_share, p_ffo
    - Insurers: loss_ratio, combined_ratio
    """
    result = {}
    
    if not financials:
        return result
    
    # Get quarterly income statement for most recent data
    quarterly = financials.get("quarterly", {})
    income_stmt = quarterly.get("income_statement", {})
    balance_sheet = quarterly.get("balance_sheet", {})
    
    if domain == "bank":
        # Net Interest Income
        nii = income_stmt.get("Net Interest Income")
        if nii:
            result["net_interest_income"] = int(nii)
        
        # Net Interest Margin (NII / Interest-earning assets)
        total_assets = balance_sheet.get("Total Assets")
        if nii and total_assets and total_assets > 0:
            # Rough approximation - real NIM uses interest-earning assets
            result["net_interest_margin"] = (nii * 4) / total_assets  # Annualized
    
    elif domain == "reit":
        # FFO = Net Income + Depreciation & Amortization
        net_income = income_stmt.get("Net Income") or info.get("net_income")
        depreciation = income_stmt.get("Depreciation And Amortization In Income Statement")
        if not depreciation:
            depreciation = income_stmt.get("Depreciation Amortization Depletion")
        
        if net_income:
            ffo = int(net_income)
            if depreciation:
                ffo += int(depreciation)
            result["ffo"] = ffo
            
            # FFO per share
            shares = info.get("shares_outstanding") or info.get("sharesOutstanding")
            if shares and shares > 0:
                ffo_per_share = ffo / shares
                result["ffo_per_share"] = round(ffo_per_share, 4)
                
                # P/FFO
                price = info.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice")
                if price and ffo_per_share > 0:
                    result["p_ffo"] = round(price / ffo_per_share, 4)
    
    elif domain == "insurer":
        # Loss ratio = Loss Adjustment Expense / Premiums Earned
        loss_expense = income_stmt.get("Net Policyholder Benefits And Claims")
        if not loss_expense:
            loss_expense = income_stmt.get("Policyholder Benefits And Claims")
        
        premiums = income_stmt.get("Total Revenue") or info.get("revenue")
        
        if loss_expense and premiums and premiums > 0:
            result["loss_ratio"] = abs(loss_expense) / premiums
        
        # Combined ratio (would need expense ratio too, but often not available)
        # Just use loss ratio for now
    
    return result


async def _fetch_fundamentals_from_service(symbol: str) -> Optional[dict[str, Any]]:
    """
    Fetch fundamentals via unified YFinanceService.
    
    Fetches:
    - Basic fundamentals from ticker info
    - Earnings calendar
    - Financial statements (quarterly/annual income stmt, balance sheet, cash flow)
    - Domain-specific metrics (NIM, FFO, loss ratio based on domain type)
    """
    if symbol.startswith("^"):
        logger.debug(f"Skipping fundamentals for index: {symbol}")
        return None
    
    service = get_yfinance_service()
    
    # Fetch ticker info and financials concurrently
    info_task = service.get_ticker_info(symbol)
    financials_task = service.get_financials(symbol)
    calendar_task = service.get_calendar(symbol)
    
    info, financials, (calendar, _) = await asyncio.gather(
        info_task,
        financials_task,
        calendar_task,
    )
    
    if not info:
        return None
    
    # Defensive: parse JSON if info came back as string (DB cache issue)
    if isinstance(info, str):
        import json
        try:
            info = json.loads(info)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Failed to parse info for {symbol}: not valid JSON")
            return None
    
    # Check if ETF/fund
    quote_type = info.get("quote_type")
    if _is_etf_or_index(symbol, quote_type):
        logger.debug(f"Skipping fundamentals for {quote_type}: {symbol}")
        return None
    
    # Detect domain
    domain = _detect_domain(info)
    
    # Calculate domain-specific metrics
    domain_metrics = _calculate_domain_metrics(info, financials, domain)
    
    # Parse earnings dates
    next_earnings_date = None
    earnings_estimate_high = None
    earnings_estimate_low = None
    earnings_estimate_avg = None
    
    if calendar:
        next_earnings_date = _safe_date(calendar.get("next_earnings_date"))
        earnings_estimate_high = calendar.get("earnings_estimate_high")
        earnings_estimate_low = calendar.get("earnings_estimate_low")
        earnings_estimate_avg = calendar.get("earnings_estimate_avg")
    
    # Build result with all data
    result = {
        "symbol": symbol.upper(),
        # Domain
        "domain": domain,
        # Valuation
        "pe_ratio": _safe_float(info.get("pe_ratio")),
        "forward_pe": _safe_float(info.get("forward_pe")),
        "peg_ratio": _safe_float(info.get("peg_ratio")),
        "price_to_book": _safe_float(info.get("price_to_book")),
        "price_to_sales": _safe_float(info.get("price_to_sales")),
        "enterprise_value": _safe_int(info.get("enterprise_value")),
        "ev_to_ebitda": _safe_float(info.get("ev_to_ebitda")),
        "ev_to_revenue": _safe_float(info.get("ev_to_revenue")),
        # Profitability
        "profit_margin": _safe_float(info.get("profit_margin")),
        "operating_margin": _safe_float(info.get("operating_margin")),
        "gross_margin": _safe_float(info.get("gross_margin")),
        "ebitda_margin": _safe_float(info.get("ebitda_margin")),
        "return_on_equity": _safe_float(info.get("return_on_equity")),
        "return_on_assets": _safe_float(info.get("return_on_assets")),
        # Financial Health
        "debt_to_equity": _safe_float(info.get("debt_to_equity")),
        "current_ratio": _safe_float(info.get("current_ratio")),
        "quick_ratio": _safe_float(info.get("quick_ratio")),
        "total_cash": _safe_int(info.get("total_cash")),
        "total_debt": _safe_int(info.get("total_debt")),
        "free_cash_flow": _safe_int(info.get("free_cash_flow")),
        "operating_cash_flow": _safe_int(info.get("operating_cash_flow")),
        # Per Share
        "book_value": _safe_float(info.get("book_value")),
        "eps_trailing": _safe_float(info.get("eps_trailing")),
        "eps_forward": _safe_float(info.get("eps_forward")),
        "revenue_per_share": _safe_float(info.get("revenue_per_share")),
        # Growth
        "revenue_growth": _safe_float(info.get("revenue_growth")),
        "earnings_growth": _safe_float(info.get("earnings_growth")),
        "earnings_quarterly_growth": _safe_float(info.get("earnings_quarterly_growth")),
        # Shares & Ownership
        "shares_outstanding": _safe_int(info.get("shares_outstanding")),
        "float_shares": _safe_int(info.get("float_shares")),
        "held_percent_insiders": _safe_float(info.get("held_percent_insiders")),
        "held_percent_institutions": _safe_float(info.get("held_percent_institutions")),
        "short_ratio": _safe_float(info.get("short_ratio")),
        "short_percent_of_float": _safe_float(info.get("short_percent_of_float")),
        # Risk
        "beta": _safe_float(info.get("beta")),
        # Analyst Ratings
        "recommendation": info.get("recommendation"),
        "recommendation_mean": _safe_float(info.get("recommendation_mean")),
        "num_analyst_opinions": _safe_int(info.get("num_analyst_opinions")),
        "target_high_price": _safe_float(info.get("target_high_price")),
        "target_low_price": _safe_float(info.get("target_low_price")),
        "target_mean_price": _safe_float(info.get("target_mean_price")),
        "target_median_price": _safe_float(info.get("target_median_price")),
        # Revenue & Earnings
        "revenue": _safe_int(info.get("revenue")),
        "ebitda": _safe_int(info.get("ebitda")),
        "net_income": _safe_int(info.get("net_income")),
        # Earnings Calendar
        "next_earnings_date": next_earnings_date,
        "earnings_estimate_high": _safe_float(earnings_estimate_high),
        "earnings_estimate_low": _safe_float(earnings_estimate_low),
        "earnings_estimate_avg": _safe_float(earnings_estimate_avg),
        # Financial Statements (store as JSONB)
        "income_stmt_quarterly": financials.get("quarterly", {}).get("income_statement") if financials else None,
        "income_stmt_annual": financials.get("annual", {}).get("income_statement") if financials else None,
        "balance_sheet_quarterly": financials.get("quarterly", {}).get("balance_sheet") if financials else None,
        "balance_sheet_annual": financials.get("annual", {}).get("balance_sheet") if financials else None,
        "cash_flow_quarterly": financials.get("quarterly", {}).get("cash_flow") if financials else None,
        "cash_flow_annual": financials.get("annual", {}).get("cash_flow") if financials else None,
        # Domain-Specific Metrics
        "net_interest_income": domain_metrics.get("net_interest_income"),
        "net_interest_margin": domain_metrics.get("net_interest_margin"),
        "ffo": domain_metrics.get("ffo"),
        "ffo_per_share": domain_metrics.get("ffo_per_share"),
        "p_ffo": domain_metrics.get("p_ffo"),
        "loss_ratio": domain_metrics.get("loss_ratio"),
        "combined_ratio": domain_metrics.get("combined_ratio"),
        # Timestamp for financials
        "financials_fetched_at": datetime.now(timezone.utc) if financials else None,
    }
    
    return result


async def _store_fundamentals(data: dict[str, Any]) -> None:
    """Store fundamentals in database including financial statements and domain metrics."""
    symbol = data["symbol"]
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    now = datetime.now(timezone.utc)
    
    # Build values dict for upsert
    values = {
        "symbol": symbol,
        "domain": data.get("domain"),
        "pe_ratio": data.get("pe_ratio"),
        "forward_pe": data.get("forward_pe"),
        "peg_ratio": data.get("peg_ratio"),
        "price_to_book": data.get("price_to_book"),
        "price_to_sales": data.get("price_to_sales"),
        "enterprise_value": data.get("enterprise_value"),
        "ev_to_ebitda": data.get("ev_to_ebitda"),
        "ev_to_revenue": data.get("ev_to_revenue"),
        "profit_margin": data.get("profit_margin"),
        "operating_margin": data.get("operating_margin"),
        "gross_margin": data.get("gross_margin"),
        "ebitda_margin": data.get("ebitda_margin"),
        "return_on_equity": data.get("return_on_equity"),
        "return_on_assets": data.get("return_on_assets"),
        "debt_to_equity": data.get("debt_to_equity"),
        "current_ratio": data.get("current_ratio"),
        "quick_ratio": data.get("quick_ratio"),
        "total_cash": data.get("total_cash"),
        "total_debt": data.get("total_debt"),
        "free_cash_flow": data.get("free_cash_flow"),
        "operating_cash_flow": data.get("operating_cash_flow"),
        "book_value": data.get("book_value"),
        "eps_trailing": data.get("eps_trailing"),
        "eps_forward": data.get("eps_forward"),
        "revenue_per_share": data.get("revenue_per_share"),
        "revenue_growth": data.get("revenue_growth"),
        "earnings_growth": data.get("earnings_growth"),
        "earnings_quarterly_growth": data.get("earnings_quarterly_growth"),
        "shares_outstanding": data.get("shares_outstanding"),
        "float_shares": data.get("float_shares"),
        "held_percent_insiders": data.get("held_percent_insiders"),
        "held_percent_institutions": data.get("held_percent_institutions"),
        "short_ratio": data.get("short_ratio"),
        "short_percent_of_float": data.get("short_percent_of_float"),
        "beta": data.get("beta"),
        "recommendation": data.get("recommendation"),
        "recommendation_mean": data.get("recommendation_mean"),
        "num_analyst_opinions": data.get("num_analyst_opinions"),
        "target_high_price": data.get("target_high_price"),
        "target_low_price": data.get("target_low_price"),
        "target_mean_price": data.get("target_mean_price"),
        "target_median_price": data.get("target_median_price"),
        "revenue": data.get("revenue"),
        "ebitda": data.get("ebitda"),
        "net_income": data.get("net_income"),
        "next_earnings_date": data.get("next_earnings_date"),
        "earnings_estimate_high": data.get("earnings_estimate_high"),
        "earnings_estimate_low": data.get("earnings_estimate_low"),
        "earnings_estimate_avg": data.get("earnings_estimate_avg"),
        "income_stmt_quarterly": data.get("income_stmt_quarterly"),
        "income_stmt_annual": data.get("income_stmt_annual"),
        "balance_sheet_quarterly": data.get("balance_sheet_quarterly"),
        "balance_sheet_annual": data.get("balance_sheet_annual"),
        "cash_flow_quarterly": data.get("cash_flow_quarterly"),
        "cash_flow_annual": data.get("cash_flow_annual"),
        "net_interest_income": data.get("net_interest_income"),
        "net_interest_margin": data.get("net_interest_margin"),
        "ffo": data.get("ffo"),
        "ffo_per_share": data.get("ffo_per_share"),
        "p_ffo": data.get("p_ffo"),
        "loss_ratio": data.get("loss_ratio"),
        "combined_ratio": data.get("combined_ratio"),
        "fetched_at": now,
        "expires_at": expires_at,
        "financials_fetched_at": data.get("financials_fetched_at"),
    }
    
    # Build update dict (exclude symbol as it's the conflict key)
    update_values = {k: v for k, v in values.items() if k != "symbol"}
    
    async with get_session() as session:
        stmt = insert(StockFundamentals).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_=update_values,
        )
        await session.execute(stmt)
        await session.commit()
    
    logger.debug(f"Stored fundamentals for {symbol} (domain={data.get('domain')})")


async def fetch_fundamentals_live(symbol: str) -> Optional[dict[str, Any]]:
    """
    Fetch fundamentals from Yahoo Finance without storing to database.
    
    Use this for symbols that aren't tracked (not in symbols table).
    For tracked symbols, use get_fundamentals() which caches data.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dict with formatted fundamental metrics, or None if not available
    """
    symbol = symbol.upper()
    
    if symbol.startswith("^"):
        return None
    
    data = await _fetch_fundamentals_from_service(symbol)
    
    if not data:
        return None
    
    # Format the data the same way as get_fundamentals_for_analysis
    def fmt_pct(val: Optional[float]) -> Optional[str]:
        if val is None:
            return None
        return f"{val * 100:.1f}%"
    
    def fmt_ratio(val: Optional[float]) -> Optional[str]:
        if val is None:
            return None
        return f"{val:.2f}"
    
    def fmt_large_num(val: Optional[int]) -> Optional[str]:
        if val is None:
            return None
        if val >= 1e12:
            return f"${val / 1e12:.1f}T"
        if val >= 1e9:
            return f"${val / 1e9:.1f}B"
        if val >= 1e6:
            return f"${val / 1e6:.1f}M"
        return f"${val:,.0f}"
    
    return {
        # Valuation
        "pe_ratio": data.get("pe_ratio"),
        "forward_pe": data.get("forward_pe"),
        "peg_ratio": fmt_ratio(data.get("peg_ratio")),
        "price_to_book": fmt_ratio(data.get("price_to_book")),
        "ev_to_ebitda": fmt_ratio(data.get("ev_to_ebitda")),
        # Profitability
        "profit_margin": fmt_pct(data.get("profit_margin")),
        "operating_margin": fmt_pct(data.get("operating_margin")),
        "gross_margin": fmt_pct(data.get("gross_margin")),
        "return_on_equity": fmt_pct(data.get("return_on_equity")),
        "return_on_assets": fmt_pct(data.get("return_on_assets")),
        # Financial Health
        "debt_to_equity": fmt_ratio(data.get("debt_to_equity")),
        "current_ratio": fmt_ratio(data.get("current_ratio")),
        "free_cash_flow": fmt_large_num(data.get("free_cash_flow")),
        "total_debt": fmt_large_num(data.get("total_debt")),
        "total_cash": fmt_large_num(data.get("total_cash")),
        # Growth
        "revenue_growth": fmt_pct(data.get("revenue_growth")),
        "earnings_growth": fmt_pct(data.get("earnings_growth")),
        # Analyst
        "recommendation": data.get("recommendation"),
        "num_analyst_opinions": data.get("num_analyst_opinions"),
        "target_mean_price": data.get("target_mean_price"),
        # Risk
        "beta": fmt_ratio(data.get("beta")),
        "short_percent_of_float": fmt_pct(data.get("short_percent_of_float")),
        # Earnings
        "next_earnings_date": str(data.get("next_earnings_date")) if data.get("next_earnings_date") else None,
        "eps_trailing": data.get("eps_trailing"),
        "eps_forward": data.get("eps_forward"),
    }


def _fundamentals_to_dict(f: StockFundamentals) -> dict[str, Any]:
    """Convert StockFundamentals ORM object to dictionary."""
    return {
        "symbol": f.symbol,
        "domain": f.domain,
        "pe_ratio": float(f.pe_ratio) if f.pe_ratio else None,
        "forward_pe": float(f.forward_pe) if f.forward_pe else None,
        "peg_ratio": float(f.peg_ratio) if f.peg_ratio else None,
        "price_to_book": float(f.price_to_book) if f.price_to_book else None,
        "price_to_sales": float(f.price_to_sales) if f.price_to_sales else None,
        "enterprise_value": f.enterprise_value,
        "ev_to_ebitda": float(f.ev_to_ebitda) if f.ev_to_ebitda else None,
        "ev_to_revenue": float(f.ev_to_revenue) if f.ev_to_revenue else None,
        "profit_margin": float(f.profit_margin) if f.profit_margin else None,
        "operating_margin": float(f.operating_margin) if f.operating_margin else None,
        "gross_margin": float(f.gross_margin) if f.gross_margin else None,
        "ebitda_margin": float(f.ebitda_margin) if f.ebitda_margin else None,
        "return_on_equity": float(f.return_on_equity) if f.return_on_equity else None,
        "return_on_assets": float(f.return_on_assets) if f.return_on_assets else None,
        "debt_to_equity": float(f.debt_to_equity) if f.debt_to_equity else None,
        "current_ratio": float(f.current_ratio) if f.current_ratio else None,
        "quick_ratio": float(f.quick_ratio) if f.quick_ratio else None,
        "total_cash": f.total_cash,
        "total_debt": f.total_debt,
        "free_cash_flow": f.free_cash_flow,
        "operating_cash_flow": f.operating_cash_flow,
        "book_value": float(f.book_value) if f.book_value else None,
        "eps_trailing": float(f.eps_trailing) if f.eps_trailing else None,
        "eps_forward": float(f.eps_forward) if f.eps_forward else None,
        "revenue_per_share": float(f.revenue_per_share) if f.revenue_per_share else None,
        "revenue_growth": float(f.revenue_growth) if f.revenue_growth else None,
        "earnings_growth": float(f.earnings_growth) if f.earnings_growth else None,
        "earnings_quarterly_growth": float(f.earnings_quarterly_growth) if f.earnings_quarterly_growth else None,
        "shares_outstanding": f.shares_outstanding,
        "float_shares": f.float_shares,
        "held_percent_insiders": float(f.held_percent_insiders) if f.held_percent_insiders else None,
        "held_percent_institutions": float(f.held_percent_institutions) if f.held_percent_institutions else None,
        "short_ratio": float(f.short_ratio) if f.short_ratio else None,
        "short_percent_of_float": float(f.short_percent_of_float) if f.short_percent_of_float else None,
        "beta": float(f.beta) if f.beta else None,
        "recommendation": f.recommendation,
        "recommendation_mean": float(f.recommendation_mean) if f.recommendation_mean else None,
        "num_analyst_opinions": f.num_analyst_opinions,
        "target_high_price": float(f.target_high_price) if f.target_high_price else None,
        "target_low_price": float(f.target_low_price) if f.target_low_price else None,
        "target_mean_price": float(f.target_mean_price) if f.target_mean_price else None,
        "target_median_price": float(f.target_median_price) if f.target_median_price else None,
        "revenue": f.revenue,
        "ebitda": f.ebitda,
        "net_income": f.net_income,
        "next_earnings_date": f.next_earnings_date,
        "earnings_estimate_high": float(f.earnings_estimate_high) if f.earnings_estimate_high else None,
        "earnings_estimate_low": float(f.earnings_estimate_low) if f.earnings_estimate_low else None,
        "earnings_estimate_avg": float(f.earnings_estimate_avg) if f.earnings_estimate_avg else None,
        "income_stmt_quarterly": f.income_stmt_quarterly,
        "income_stmt_annual": f.income_stmt_annual,
        "balance_sheet_quarterly": f.balance_sheet_quarterly,
        "balance_sheet_annual": f.balance_sheet_annual,
        "cash_flow_quarterly": f.cash_flow_quarterly,
        "cash_flow_annual": f.cash_flow_annual,
        "net_interest_income": f.net_interest_income,
        "net_interest_margin": float(f.net_interest_margin) if f.net_interest_margin else None,
        "ffo": f.ffo,
        "ffo_per_share": float(f.ffo_per_share) if f.ffo_per_share else None,
        "p_ffo": float(f.p_ffo) if f.p_ffo else None,
        "loss_ratio": float(f.loss_ratio) if f.loss_ratio else None,
        "combined_ratio": float(f.combined_ratio) if f.combined_ratio else None,
        "fetched_at": f.fetched_at,
        "expires_at": f.expires_at,
        "financials_fetched_at": f.financials_fetched_at,
    }


async def get_fundamentals_from_db(symbol: str) -> Optional[dict[str, Any]]:
    """
    Get fundamentals from database cache only (no fetching).
    
    Use this when you want to check if data exists without triggering
    a yfinance fetch. Useful for untracked symbols.
    """
    symbol = symbol.upper()
    
    if symbol.startswith("^"):
        return None
    
    async with get_session() as session:
        result = await session.execute(
            select(StockFundamentals).where(
                StockFundamentals.symbol == symbol,
                StockFundamentals.expires_at > datetime.now(timezone.utc),
            )
        )
        row = result.scalar_one_or_none()
        return _fundamentals_to_dict(row) if row else None


async def get_fundamentals(symbol: str, force_refresh: bool = False) -> Optional[dict[str, Any]]:
    """
    Get fundamentals for a symbol, fetching from Yahoo Finance if expired.
    
    Args:
        symbol: Stock symbol
        force_refresh: If True, fetch fresh data regardless of cache
        
    Returns:
        Dict with fundamental metrics, or None for ETFs/indexes
    """
    symbol = symbol.upper()
    
    if symbol.startswith("^"):
        return None
    
    # Check database cache
    if not force_refresh:
        async with get_session() as session:
            result = await session.execute(
                select(StockFundamentals).where(
                    StockFundamentals.symbol == symbol,
                    StockFundamentals.expires_at > datetime.now(timezone.utc),
                )
            )
            row = result.scalar_one_or_none()
            if row:
                return _fundamentals_to_dict(row)
    
    # Fetch via unified service
    data = await _fetch_fundamentals_from_service(symbol)
    
    if data:
        await _store_fundamentals(data)
        return data
    
    return None


async def refresh_fundamentals(symbol: str) -> Optional[dict[str, Any]]:
    """Force refresh fundamentals for a symbol."""
    return await get_fundamentals(symbol, force_refresh=True)


async def refresh_all_fundamentals(
    symbols: list[str] | None = None, 
    batch_size: int = 10
) -> dict[str, int]:
    """
    Refresh fundamentals for specified symbols or all symbols with expired/missing data.
    
    Args:
        symbols: Optional list of specific symbols to refresh. If None, auto-detects.
        batch_size: Number of symbols to process in parallel
        
    Returns:
        Dict with counts: {"refreshed": N, "failed": M, "skipped": K}
    """
    if symbols is None:
        # Get symbols needing refresh (expired or missing)
        async with get_session() as session:
            from sqlalchemy import or_
            result = await session.execute(
                select(Symbol.symbol)
                .outerjoin(StockFundamentals, Symbol.symbol == StockFundamentals.symbol)
                .where(
                    Symbol.symbol_type == "stock",
                    Symbol.is_active == True,
                    or_(
                        StockFundamentals.symbol == None,
                        StockFundamentals.expires_at < datetime.now(timezone.utc),
                    ),
                )
                .order_by(StockFundamentals.expires_at.asc().nulls_first())
                .limit(100)
            )
            symbols = [r[0] for r in result.all()]
    
    logger.info(f"Refreshing fundamentals for {len(symbols)} symbols")
    
    refreshed = 0
    failed = 0
    skipped = 0
    
    # Process in batches to avoid overwhelming yfinance
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        
        # Process batch concurrently
        tasks = [get_fundamentals(s, force_refresh=True) for s in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for symbol, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to refresh {symbol}: {result}")
                failed += 1
            elif result is None:
                skipped += 1
            else:
                refreshed += 1
        
        # Small delay between batches
        if i + batch_size < len(symbols):
            await asyncio.sleep(1)
    
    logger.info(f"Fundamentals refresh complete: {refreshed} refreshed, {failed} failed, {skipped} skipped")
    return {"refreshed": refreshed, "failed": failed, "skipped": skipped}


async def get_fundamentals_for_analysis(symbol: str) -> dict[str, Any]:
    """
    Get fundamentals formatted for AI analysis context.
    
    Returns a dict with key metrics formatted for prompt injection,
    including human-readable labels and formatted numbers.
    """
    data = await get_fundamentals(symbol)
    
    if not data:
        return {}
    
    def fmt_pct(val: Optional[float]) -> Optional[str]:
        """Format decimal as percentage."""
        if val is None:
            return None
        return f"{val * 100:.1f}%"
    
    def fmt_ratio(val: Optional[float]) -> Optional[str]:
        """Format ratio."""
        if val is None:
            return None
        return f"{val:.2f}"
    
    def fmt_large_num(val: Optional[int]) -> Optional[str]:
        """Format large number (billions/millions)."""
        if val is None:
            return None
        if val >= 1e12:
            return f"${val / 1e12:.1f}T"
        if val >= 1e9:
            return f"${val / 1e9:.1f}B"
        if val >= 1e6:
            return f"${val / 1e6:.1f}M"
        return f"${val:,.0f}"
    
    return {
        # Valuation
        "pe_ratio": data.get("pe_ratio"),
        "forward_pe": data.get("forward_pe"),
        "peg_ratio": fmt_ratio(data.get("peg_ratio")),
        "price_to_book": fmt_ratio(data.get("price_to_book")),
        "ev_to_ebitda": fmt_ratio(data.get("ev_to_ebitda")),
        
        # Profitability
        "profit_margin": fmt_pct(data.get("profit_margin")),
        "operating_margin": fmt_pct(data.get("operating_margin")),
        "gross_margin": fmt_pct(data.get("gross_margin")),
        "return_on_equity": fmt_pct(data.get("return_on_equity")),
        "return_on_assets": fmt_pct(data.get("return_on_assets")),
        
        # Financial Health
        "debt_to_equity": fmt_ratio(data.get("debt_to_equity")),
        "current_ratio": fmt_ratio(data.get("current_ratio")),
        "free_cash_flow": fmt_large_num(data.get("free_cash_flow")),
        "total_debt": fmt_large_num(data.get("total_debt")),
        "total_cash": fmt_large_num(data.get("total_cash")),
        
        # Growth
        "revenue_growth": fmt_pct(data.get("revenue_growth")),
        "earnings_growth": fmt_pct(data.get("earnings_growth")),
        
        # Analyst
        "recommendation": data.get("recommendation"),
        "num_analyst_opinions": data.get("num_analyst_opinions"),
        "target_mean_price": data.get("target_mean_price"),
        
        # Risk
        "beta": fmt_ratio(data.get("beta")),
        "short_percent_of_float": fmt_pct(data.get("short_percent_of_float")),
        
        # Earnings
        "next_earnings_date": str(data.get("next_earnings_date")) if data.get("next_earnings_date") else None,
        "eps_trailing": data.get("eps_trailing"),
        "eps_forward": data.get("eps_forward"),
    }


async def get_pe_ratio(symbol: str) -> Optional[float]:
    """Quick helper to get just the P/E ratio."""
    # First check symbols table (fast)
    async with get_session() as session:
        result = await session.execute(
            select(Symbol.pe_ratio).where(Symbol.symbol == symbol.upper())
        )
        row = result.scalar_one_or_none()
        if row:
            return float(row)
    
    # Fall back to fundamentals
    data = await get_fundamentals(symbol)
    return data.get("pe_ratio") if data else None


# Backward compatibility - export for tests that import _fetch_fundamentals_sync
# Tests should be updated to use the async version instead
async def _fetch_fundamentals_sync_compat(symbol: str) -> Optional[dict[str, Any]]:
    """Backward compatible sync-style function (actually async)."""
    return await _fetch_fundamentals_from_service(symbol)


# For tests importing the old name
_fetch_fundamentals_sync = _fetch_fundamentals_sync_compat
