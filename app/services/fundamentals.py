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
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.cache.cache import Cache
from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import StockFundamentals, Symbol
from app.services.data_providers import get_yfinance_service


logger = get_logger("services.fundamentals")

# Cache for live (untracked) fundamentals lookups
LIVE_FUNDAMENTALS_CACHE_TTL = 900  # 15 minutes
_live_fundamentals_cache = Cache(prefix="fundamentals_live", default_ttl=LIVE_FUNDAMENTALS_CACHE_TTL)


def _is_etf_or_index(symbol: str, quote_type: str | None = None) -> bool:
    """Check if symbol is an ETF or index (no fundamentals available)."""
    if symbol.startswith("^"):
        return True
    if quote_type:
        quote_type_upper = quote_type.upper()
        return quote_type_upper in ("ETF", "INDEX", "MUTUALFUND", "TRUST")
    return False


def _safe_float(value: Any) -> float | None:
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


def _safe_int(value: Any) -> int | None:
    """Safely convert value to int, returning None for invalid values."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_date(value: Any) -> date | None:
    """Safely convert value to date, returning None for invalid values.
    
    Handles:
    - datetime objects
    - date objects  
    - pandas Timestamp
    - unix timestamps (seconds or ms)
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
        # Unix timestamp
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=UTC).date()
        # String in ISO format
        if isinstance(value, str):
            # Handle 'YYYY-MM-DD' format
            return datetime.strptime(value.split('T')[0], '%Y-%m-%d').date()
        return None
    except (ValueError, TypeError, AttributeError):
        return None


def _sorted_period_keys(keys: list[Any]) -> list[Any]:
    def _key(item: Any) -> datetime:
        if isinstance(item, datetime):
            return item
        if isinstance(item, date):
            return datetime.combine(item, datetime.min.time(), tzinfo=UTC)
        try:
            return datetime.fromisoformat(str(item))
        except (ValueError, TypeError):
            return datetime.min
    return sorted(keys, key=_key, reverse=True)


def _latest_value(value: Any) -> float | None:
    """Return most recent numeric value from a scalar or date-keyed dict."""
    if isinstance(value, dict):
        for key in _sorted_period_keys(list(value.keys())):
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
    financials: dict[str, Any] | None,
    domain: str,
) -> dict[str, Any]:
    """
    Calculate domain-specific metrics from financial statements.
    
    Supports both yfinance and yahooquery field names.
    
    Returns dict with:
    - Banks: net_interest_income, net_interest_margin, interest_income, interest_expense
    - REITs: ffo, ffo_per_share, p_ffo
    - Insurers: loss_ratio, combined_ratio, expense_ratio
    """
    result = {}

    if not financials:
        return result

    # Get quarterly income statement for most recent data
    quarterly = financials.get("quarterly", {})
    income_stmt = quarterly.get("income_statement", {})
    balance_sheet = quarterly.get("balance_sheet", {})
    cash_flow = quarterly.get("cash_flow", {})

    if domain == "bank":
        # Net Interest Income - try multiple field names (yahooquery vs yfinance)
        nii = _latest_value(
            income_stmt.get("NetInterestIncome") or  # yahooquery
            income_stmt.get("Net Interest Income")    # yfinance
        )
        if nii and nii > 0:
            result["net_interest_income"] = int(nii)
        
        # Interest Income and Expense
        int_income = _latest_value(
            income_stmt.get("InterestIncome") or 
            income_stmt.get("Interest Income")
        )
        int_expense = _latest_value(
            income_stmt.get("InterestExpense") or 
            income_stmt.get("Interest Expense")
        )
        if int_income:
            result["interest_income"] = int(int_income)
        if int_expense:
            result["interest_expense"] = int(int_expense)

        # Net Interest Margin (NII / Interest-earning assets)
        total_assets = _latest_value(
            balance_sheet.get("TotalAssets") or
            balance_sheet.get("Total Assets")
        )
        if nii and nii > 0 and total_assets and total_assets > 0:
            # Rough approximation - real NIM uses interest-earning assets
            result["net_interest_margin"] = (nii * 4) / total_assets  # Annualized

    elif domain == "reit":
        # FFO = Net Income + Depreciation & Amortization
        net_income = _latest_value(
            income_stmt.get("NetIncome") or
            income_stmt.get("Net Income") or 
            info.get("net_income")
        )
        
        # Depreciation from cash flow (more reliable) or income statement
        depreciation = _latest_value(
            cash_flow.get("DepreciationAndAmortization") or  # yahooquery
            income_stmt.get("DepreciationAndAmortizationInIncomeStatement") or
            income_stmt.get("Depreciation And Amortization In Income Statement") or
            income_stmt.get("Depreciation Amortization Depletion")
        )

        if net_income:
            ffo = int(net_income)
            if depreciation:
                ffo += int(depreciation)
            result["ffo"] = ffo

            # FFO per share
            shares = _latest_value(info.get("shares_outstanding") or info.get("sharesOutstanding"))
            if shares and shares > 0:
                ffo_per_share = (ffo * 4) / shares  # Annualized FFO per share
                result["ffo_per_share"] = round(ffo_per_share, 4)

                # P/FFO
                price = _latest_value(info.get("current_price") or info.get("currentPrice") or info.get("regularMarketPrice"))
                if price and ffo_per_share > 0:
                    result["p_ffo"] = round(price / ffo_per_share, 4)

    elif domain == "insurer":
        # For insurers, calculate expense ratio as proxy for efficiency
        total_revenue = _latest_value(
            income_stmt.get("TotalRevenue") or
            income_stmt.get("Total Revenue") or
            income_stmt.get("OperatingRevenue") or
            info.get("revenue")
        )
        
        total_expenses = _latest_value(
            income_stmt.get("TotalExpenses") or
            income_stmt.get("Total Expenses")
        )
        
        # Loss ratio from policyholder benefits if available
        loss_expense = _latest_value(
            income_stmt.get("NetPolicyholderBenefitsAndClaims") or
            income_stmt.get("Net Policyholder Benefits And Claims") or
            income_stmt.get("PolicyholderBenefitsAndClaims") or
            income_stmt.get("Policyholder Benefits And Claims")
        )

        if loss_expense and total_revenue and total_revenue > 0:
            result["loss_ratio"] = abs(loss_expense) / total_revenue
            
        # Expense ratio as alternative
        if total_expenses and total_revenue and total_revenue > 0:
            result["expense_ratio"] = abs(total_expenses) / total_revenue

    return result


async def _fetch_fundamentals_from_service(
    symbol: str,
    include_financials: bool = True,
    include_calendar: bool = True,
    skip_cache: bool = False,
) -> dict[str, Any] | None:
    """
    Fetch fundamentals via unified YFinanceService.
    
    Fetches:
    - Basic fundamentals from ticker info
    - Earnings calendar
    - Financial statements (quarterly/annual income stmt, balance sheet, cash flow)
    - Domain-specific metrics (NIM, FFO, loss ratio based on domain type)
    
    Args:
        symbol: Stock ticker symbol
        include_financials: Whether to fetch financial statements
        include_calendar: Whether to fetch earnings calendar
        skip_cache: If True, bypass Redis cache and fetch fresh from yfinance
    """
    if symbol.startswith("^"):
        logger.debug(f"Skipping fundamentals for index: {symbol}")
        return None

    service = get_yfinance_service()

    # Fetch ticker info and financials concurrently
    info_task = service.get_ticker_info(symbol, skip_cache=skip_cache)
    financials_task = (
        service.get_financials(symbol, skip_cache=skip_cache)
        if include_financials
        else asyncio.sleep(0, result=None)
    )
    calendar_task = (
        service.get_calendar(symbol, skip_cache=skip_cache)
        if include_calendar
        else asyncio.sleep(0, result=(None, None))
    )

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
    earnings_date = _safe_date(info.get("earnings_date")) or _safe_date(
        info.get("most_recent_quarter")
    )
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
        "earnings_date": earnings_date,
        # Financial Statements (store as JSONB)
        "income_stmt_quarterly": (
            financials.get("quarterly", {}).get("income_statement_series")
            or financials.get("quarterly", {}).get("income_statement")
        ) if financials else None,
        "income_stmt_annual": (
            financials.get("annual", {}).get("income_statement_series")
            or financials.get("annual", {}).get("income_statement")
        ) if financials else None,
        "balance_sheet_quarterly": (
            financials.get("quarterly", {}).get("balance_sheet_series")
            or financials.get("quarterly", {}).get("balance_sheet")
        ) if financials else None,
        "balance_sheet_annual": (
            financials.get("annual", {}).get("balance_sheet_series")
            or financials.get("annual", {}).get("balance_sheet")
        ) if financials else None,
        "cash_flow_quarterly": (
            financials.get("quarterly", {}).get("cash_flow_series")
            or financials.get("quarterly", {}).get("cash_flow")
        ) if financials else None,
        "cash_flow_annual": (
            financials.get("annual", {}).get("cash_flow_series")
            or financials.get("annual", {}).get("cash_flow")
        ) if financials else None,
        # Domain-Specific Metrics
        "net_interest_income": domain_metrics.get("net_interest_income"),
        "net_interest_margin": domain_metrics.get("net_interest_margin"),
        "interest_income": domain_metrics.get("interest_income"),
        "interest_expense": domain_metrics.get("interest_expense"),
        "ffo": domain_metrics.get("ffo"),
        "ffo_per_share": domain_metrics.get("ffo_per_share"),
        "p_ffo": domain_metrics.get("p_ffo"),
        "loss_ratio": domain_metrics.get("loss_ratio"),
        "expense_ratio": domain_metrics.get("expense_ratio"),
        "combined_ratio": domain_metrics.get("combined_ratio"),
        # Timestamp for financials
        "financials_fetched_at": datetime.now(UTC) if financials else None,
    }

    return result


async def _store_fundamentals(
    data: dict[str, Any],
    update_financials: bool = True,
    update_calendar: bool = True,
) -> None:
    """Store fundamentals in database including financial statements and domain metrics."""
    symbol = data["symbol"]
    expires_at = datetime.now(UTC) + timedelta(days=30)
    now = datetime.now(UTC)
    has_financials = data.get("financials_fetched_at") is not None

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
        "earnings_date": data.get("earnings_date"),
        "dividend_date": data.get("dividend_date"),
        "ex_dividend_date": data.get("ex_dividend_date"),
        "income_stmt_quarterly": data.get("income_stmt_quarterly"),
        "income_stmt_annual": data.get("income_stmt_annual"),
        "balance_sheet_quarterly": data.get("balance_sheet_quarterly"),
        "balance_sheet_annual": data.get("balance_sheet_annual"),
        "cash_flow_quarterly": data.get("cash_flow_quarterly"),
        "cash_flow_annual": data.get("cash_flow_annual"),
        "net_interest_income": data.get("net_interest_income"),
        "net_interest_margin": data.get("net_interest_margin"),
        "interest_income": data.get("interest_income"),
        "interest_expense": data.get("interest_expense"),
        "ffo": data.get("ffo"),
        "ffo_per_share": data.get("ffo_per_share"),
        "p_ffo": data.get("p_ffo"),
        "loss_ratio": data.get("loss_ratio"),
        "expense_ratio": data.get("expense_ratio"),
        "combined_ratio": data.get("combined_ratio"),
        "fetched_at": now,
        "expires_at": expires_at,
        "financials_fetched_at": data.get("financials_fetched_at"),
    }

    # Build update dict (exclude symbol as it's the conflict key)
    update_values = {}
    for key, value in values.items():
        if key == "symbol":
            continue
        if key in {
            "income_stmt_quarterly",
            "income_stmt_annual",
            "balance_sheet_quarterly",
            "balance_sheet_annual",
            "cash_flow_quarterly",
            "cash_flow_annual",
            "net_interest_income",
            "net_interest_margin",
            "interest_income",
            "interest_expense",
            "ffo",
            "ffo_per_share",
            "p_ffo",
            "loss_ratio",
            "expense_ratio",
            "combined_ratio",
            "financials_fetched_at",
        }:
            if update_financials and has_financials:
                update_values[key] = value
            continue
        if key in {
            "next_earnings_date",
            "earnings_estimate_high",
            "earnings_estimate_low",
            "earnings_estimate_avg",
            "dividend_date",
            "ex_dividend_date",
        }:
            if update_calendar and value is not None:
                update_values[key] = value
            continue
        if key == "earnings_date":
            if value is not None:
                update_values[key] = value
            continue
        update_values[key] = value

    async with get_session() as session:
        stmt = insert(StockFundamentals).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_=update_values,
        )
        await session.execute(stmt)
        await session.commit()

    logger.debug(f"Stored fundamentals for {symbol} (domain={data.get('domain')})")


async def fetch_fundamentals_live(
    symbol: str,
    use_cache: bool = True,
) -> dict[str, Any] | None:
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

    cache_key = f"live:{symbol}"
    if use_cache:
        cached = await _live_fundamentals_cache.get(cache_key)
        if cached:
            return cached

    data = await _fetch_fundamentals_from_service(
        symbol,
        include_financials=False,
        include_calendar=True,
    )

    if not data:
        return None

    # Format the data the same way as get_fundamentals_for_analysis
    def fmt_pct(val: float | None) -> str | None:
        if val is None:
            return None
        return f"{val * 100:.1f}%"

    def fmt_ratio(val: float | None) -> str | None:
        if val is None:
            return None
        return f"{val:.2f}"

    def fmt_large_num(val: int | None) -> str | None:
        if val is None:
            return None
        if val >= 1e12:
            return f"${val / 1e12:.1f}T"
        if val >= 1e9:
            return f"${val / 1e9:.1f}B"
        if val >= 1e6:
            return f"${val / 1e6:.1f}M"
        return f"${val:,.0f}"

    formatted = {
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

    if use_cache:
        await _live_fundamentals_cache.set(cache_key, formatted)

    return formatted


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
        "earnings_date": f.earnings_date,
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
        "interest_income": f.interest_income,
        "interest_expense": f.interest_expense,
        "ffo": f.ffo,
        "ffo_per_share": float(f.ffo_per_share) if f.ffo_per_share else None,
        "p_ffo": float(f.p_ffo) if f.p_ffo else None,
        "loss_ratio": float(f.loss_ratio) if f.loss_ratio else None,
        "expense_ratio": float(f.expense_ratio) if f.expense_ratio else None,
        "combined_ratio": float(f.combined_ratio) if f.combined_ratio else None,
        "fetched_at": f.fetched_at,
        "expires_at": f.expires_at,
        "financials_fetched_at": f.financials_fetched_at,
    }


async def get_fundamentals_from_db(symbol: str) -> dict[str, Any] | None:
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
                StockFundamentals.expires_at > datetime.now(UTC),
            )
        )
        row = result.scalar_one_or_none()
        return _fundamentals_to_dict(row) if row else None


async def get_fundamentals_with_status(
    symbol: str,
    allow_stale: bool = True,
) -> tuple[dict[str, Any] | None, bool]:
    """
    Get fundamentals from database with stale status.

    Returns:
        (data, is_stale)
    """
    symbol = symbol.upper()

    if symbol.startswith("^"):
        return None, False

    async with get_session() as session:
        result = await session.execute(
            select(StockFundamentals).where(StockFundamentals.symbol == symbol)
        )
        row = result.scalar_one_or_none()
        if not row:
            return None, False

    data = _fundamentals_to_dict(row)
    is_stale = bool(
        data.get("expires_at") and data["expires_at"] <= datetime.now(UTC)
    )
    if is_stale and not allow_stale:
        return None, True
    return data, is_stale


async def get_fundamentals(
    symbol: str,
    force_refresh: bool = False,
    include_financials: bool = True,
    include_calendar: bool = True,
    allow_stale: bool = False,
) -> dict[str, Any] | None:
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
                    StockFundamentals.expires_at > datetime.now(UTC),
                )
            )
            row = result.scalar_one_or_none()
            if row:
                return _fundamentals_to_dict(row)
            if allow_stale:
                result = await session.execute(
                    select(StockFundamentals).where(StockFundamentals.symbol == symbol)
                )
                row = result.scalar_one_or_none()
                if row:
                    return _fundamentals_to_dict(row)

    # Fetch via unified service
    data = await _fetch_fundamentals_from_service(
        symbol,
        include_financials=include_financials,
        include_calendar=include_calendar,
    )

    if data:
        await _store_fundamentals(
            data,
            update_financials=include_financials,
            update_calendar=include_calendar,
        )
        return data

    return None


async def refresh_fundamentals(
    symbol: str,
    include_financials: bool = True,
    include_calendar: bool = True,
) -> dict[str, Any] | None:
    """Force refresh fundamentals for a symbol."""
    return await get_fundamentals(
        symbol,
        force_refresh=True,
        include_financials=include_financials,
        include_calendar=include_calendar,
    )


async def refresh_all_fundamentals(
    symbols: list[str] | None = None,
    batch_size: int = 10,
    include_financials_for: set[str] | None = None,
    include_calendar_for: set[str] | None = None,
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
                        StockFundamentals.expires_at < datetime.now(UTC),
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
        tasks = []
        for symbol in batch:
            include_financials = (
                include_financials_for is None or symbol in include_financials_for
            )
            include_calendar = (
                include_calendar_for is None or symbol in include_calendar_for
            )
            tasks.append(
                get_fundamentals(
                    symbol,
                    force_refresh=True,
                    include_financials=include_financials,
                    include_calendar=include_calendar,
                )
            )
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


async def update_price_based_metrics(symbol: str, current_price: float) -> bool:
    """
    Update price-based ratios using the latest price and stored fundamentals.

    This does NOT touch expires_at or financial statements.
    """
    if current_price <= 0:
        return False

    symbol = symbol.upper()
    price = Decimal(str(current_price))

    async with get_session() as session:
        result = await session.execute(
            select(StockFundamentals).where(StockFundamentals.symbol == symbol)
        )
        row = result.scalar_one_or_none()
        if not row:
            return False

    updates: dict[str, Any] = {}

    if row.eps_trailing and row.eps_trailing > 0:
        updates["pe_ratio"] = price / row.eps_trailing
    if row.eps_forward and row.eps_forward > 0:
        updates["forward_pe"] = price / row.eps_forward
    if row.book_value and row.book_value > 0:
        updates["price_to_book"] = price / row.book_value
    market_cap: Decimal | None = None
    if row.shares_outstanding and row.shares_outstanding > 0:
        market_cap = price * Decimal(row.shares_outstanding)
        total_debt = Decimal(row.total_debt or 0)
        total_cash = Decimal(row.total_cash or 0)
        enterprise_value = market_cap + total_debt - total_cash
        updates["enterprise_value"] = int(enterprise_value)
        if row.ebitda and row.ebitda > 0:
            updates["ev_to_ebitda"] = enterprise_value / Decimal(row.ebitda)
        if row.revenue and row.revenue > 0:
            updates["ev_to_revenue"] = enterprise_value / Decimal(row.revenue)

    if row.revenue_per_share and row.revenue_per_share > 0:
        updates["price_to_sales"] = price / row.revenue_per_share
    elif market_cap is not None and row.revenue and row.revenue > 0:
        updates["price_to_sales"] = market_cap / Decimal(row.revenue)

    if row.earnings_growth and row.earnings_growth > 0 and "pe_ratio" in updates:
        updates["peg_ratio"] = updates["pe_ratio"] / row.earnings_growth

    if not updates:
        return False

    async with get_session() as session:
        await session.execute(
            update(StockFundamentals)
            .where(StockFundamentals.symbol == symbol)
            .values(**updates)
        )
        await session.commit()

    return True


async def get_fundamentals_for_analysis(symbol: str) -> dict[str, Any]:
    """
    Get fundamentals formatted for AI analysis context.
    
    Returns a dict with key metrics formatted for prompt injection,
    including human-readable labels and formatted numbers.
    """
    data = await get_fundamentals(symbol)

    if not data:
        return {}

    def fmt_pct(val: float | None) -> str | None:
        """Format decimal as percentage."""
        if val is None:
            return None
        return f"{val * 100:.1f}%"

    def fmt_ratio(val: float | None) -> str | None:
        """Format ratio."""
        if val is None:
            return None
        return f"{val:.2f}"

    def fmt_large_num(val: int | None) -> str | None:
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


async def get_pe_ratio(symbol: str) -> float | None:
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
async def _fetch_fundamentals_sync_compat(symbol: str) -> dict[str, Any] | None:
    """Backward compatible sync-style function (actually async)."""
    return await _fetch_fundamentals_from_service(symbol)


# For tests importing the old name
_fetch_fundamentals_sync = _fetch_fundamentals_sync_compat
