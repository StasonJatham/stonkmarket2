"""
yfinance data service for fetching market data.

MIGRATED: Now uses unified YFinanceService for all yfinance calls.
Provides a clean interface with typed Pydantic models for the hedge_fund module.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

from app.services.data_providers import get_yfinance_service
from app.hedge_fund.schemas import (
    CalendarEvents,
    Fundamentals,
    MarketData,
    PricePoint,
    PriceSeries,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Domain Detection
# =============================================================================


def _detect_domain(info: dict[str, Any]) -> Optional[str]:
    """Detect the domain type from ticker info."""
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


def _calculate_domain_metrics(
    info: dict[str, Any], 
    financials: Optional[dict[str, Any]],
    domain: Optional[str],
) -> dict[str, Any]:
    """Calculate domain-specific metrics from financial statements."""
    metrics = {}
    
    if not financials or not domain:
        return metrics
    
    quarterly = financials.get("quarterly", {})
    income = quarterly.get("income_statement", {})
    balance = quarterly.get("balance_sheet", {})
    cashflow = quarterly.get("cash_flow", {})
    
    if domain == "bank":
        # Net Interest Income
        nii = income.get("Net Interest Income")
        if nii:
            metrics["net_interest_income"] = nii
            
            # Calculate NIM proxy: NII (annualized) / Total Assets
            total_assets = balance.get("Total Assets")
            if total_assets and total_assets > 0:
                metrics["net_interest_margin"] = (nii * 4) / total_assets
    
    elif domain == "reit":
        # FFO = Net Income + Depreciation
        net_income = income.get("Net Income")
        depreciation = (
            cashflow.get("Depreciation Amortization Depletion") or
            cashflow.get("Depreciation And Amortization")
        )
        
        if net_income is not None and depreciation is not None:
            ffo = net_income + depreciation
            metrics["ffo"] = ffo * 4  # Annualize
            
            shares = info.get("shares_outstanding") or info.get("sharesOutstanding")
            current_price = info.get("current_price") or info.get("regularMarketPrice")
            
            if shares and shares > 0:
                ffo_per_share = (ffo * 4) / shares
                metrics["ffo_per_share"] = ffo_per_share
                
                if current_price and ffo_per_share > 0:
                    metrics["p_ffo"] = current_price / ffo_per_share
    
    elif domain == "insurer":
        # Loss Ratio = Losses / Premiums (approximated by revenue)
        loss_expense = (
            income.get("Net Policyholder Benefits And Claims") or
            income.get("Loss Adjustment Expense")
        )
        revenue = (
            income.get("Total Revenue") or
            income.get("Operating Revenue")
        )
        
        if loss_expense and revenue and revenue > 0:
            metrics["loss_ratio"] = loss_expense / revenue
    
    return metrics


# =============================================================================
# Async wrappers using unified YFinanceService
# =============================================================================


async def get_ticker_info(symbol: str) -> dict[str, Any]:
    """Get ticker info asynchronously via unified service."""
    service = get_yfinance_service()
    info = await service.get_ticker_info(symbol)
    return info or {}


async def get_price_history(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> PriceSeries:
    """Get price history as PriceSeries."""
    service = get_yfinance_service()
    
    df, _ = await service.get_price_history(
        symbol,
        period=period,
        start_date=start_date,
        end_date=end_date,
    )
    
    prices = []
    if df is not None and not df.empty:
        for idx, row in df.iterrows():
            prices.append(PricePoint(
                date=idx.date() if hasattr(idx, "date") else idx,
                open=_safe_float(row.get("Open"), 0.0),
                high=_safe_float(row.get("High"), 0.0),
                low=_safe_float(row.get("Low"), 0.0),
                close=_safe_float(row.get("Close"), 0.0),
                volume=_safe_int(row.get("Volume"), 0),
                adj_close=_safe_float(row.get("Adj Close")),
            ))
    
    return PriceSeries(symbol=symbol, prices=prices)


async def get_calendar_events(symbol: str) -> CalendarEvents:
    """Get upcoming calendar events."""
    service = get_yfinance_service()
    calendar, _ = await service.get_calendar(symbol)
    
    # Parse calendar data
    next_earnings = None
    ex_dividend = None
    dividend_date = None
    
    if calendar:
        # Try to extract earnings date
        earnings = calendar.get("next_earnings_date")
        if earnings:
            if isinstance(earnings, list) and len(earnings) > 0:
                next_earnings = earnings[0]
            elif hasattr(earnings, "date"):
                next_earnings = earnings.date() if callable(getattr(earnings, "date", None)) else earnings
            else:
                next_earnings = earnings
        
        # Dividend dates
        ex_div = calendar.get("ex_dividend_date")
        if ex_div:
            if hasattr(ex_div, "date"):
                ex_dividend = ex_div.date() if callable(getattr(ex_div, "date", None)) else ex_div
            else:
                ex_dividend = ex_div
        
        div_date = calendar.get("dividend_date")
        if div_date:
            if hasattr(div_date, "date"):
                dividend_date = div_date.date() if callable(getattr(div_date, "date", None)) else div_date
            else:
                dividend_date = div_date
    
    return CalendarEvents(
        symbol=symbol,
        next_earnings_date=next_earnings,
        ex_dividend_date=ex_dividend,
        dividend_date=dividend_date,
    )


async def get_fundamentals(symbol: str, include_financials: bool = True) -> Fundamentals:
    """
    Get fundamental data for a symbol, including domain-specific metrics.
    
    Reads from database first (stock_fundamentals table), falls back to yfinance
    if not found. This minimizes API calls since fundamentals change quarterly.
    """
    from app.services.fundamentals import get_fundamentals as get_db_fundamentals
    
    # Try to get from DB first (already cached with domain metrics)
    db_data = await get_db_fundamentals(symbol)
    
    if db_data:
        # Map DB fields to Fundamentals schema
        return Fundamentals(
            symbol=symbol,
            name=db_data.get("name") or symbol,
            sector=db_data.get("sector"),
            industry=db_data.get("industry"),
            market_cap=_safe_float(db_data.get("market_cap")),
            enterprise_value=_safe_float(db_data.get("enterprise_value")),
            
            # Valuation
            pe_ratio=_safe_float(db_data.get("pe_ratio")),
            forward_pe=_safe_float(db_data.get("forward_pe")),
            peg_ratio=_safe_float(db_data.get("peg_ratio")),
            price_to_book=_safe_float(db_data.get("price_to_book")),
            price_to_sales=_safe_float(db_data.get("price_to_sales")),
            ev_to_ebitda=_safe_float(db_data.get("ev_to_ebitda")),
            ev_to_revenue=_safe_float(db_data.get("ev_to_revenue")),
            
            # Profitability
            profit_margin=_safe_float(db_data.get("profit_margin")),
            operating_margin=_safe_float(db_data.get("operating_margin")),
            gross_margin=_safe_float(db_data.get("gross_margin")),
            roe=_safe_float(db_data.get("return_on_equity")),
            roa=_safe_float(db_data.get("return_on_assets")),
            
            # Growth
            revenue_growth=_safe_float(db_data.get("revenue_growth")),
            earnings_growth=_safe_float(db_data.get("earnings_growth")),
            
            # Financial health
            current_ratio=_safe_float(db_data.get("current_ratio")),
            quick_ratio=_safe_float(db_data.get("quick_ratio")),
            debt_to_equity=_safe_float(db_data.get("debt_to_equity")),
            free_cash_flow=_safe_float(db_data.get("free_cash_flow")),
            
            # Dividends
            dividend_yield=_safe_float(db_data.get("dividend_yield")),
            payout_ratio=_safe_float(db_data.get("payout_ratio")),
            
            # Per-share
            eps=_safe_float(db_data.get("eps_trailing")),
            eps_forward=_safe_float(db_data.get("eps_forward")),
            book_value_per_share=_safe_float(db_data.get("book_value")),
            
            # Risk
            beta=_safe_float(db_data.get("beta")),
            shares_outstanding=_safe_float(db_data.get("shares_outstanding")),
            float_shares=_safe_float(db_data.get("float_shares")),
            short_ratio=_safe_float(db_data.get("short_ratio")),
            short_percent_of_float=_safe_float(db_data.get("short_percent_of_float")),
            
            # Domain-specific from DB
            domain=db_data.get("domain"),
            financials={
                "quarterly": {
                    "income_statement": db_data.get("income_stmt_quarterly"),
                    "balance_sheet": db_data.get("balance_sheet_quarterly"),
                    "cash_flow": db_data.get("cash_flow_quarterly"),
                },
                "annual": {
                    "income_statement": db_data.get("income_stmt_annual"),
                    "balance_sheet": db_data.get("balance_sheet_annual"),
                    "cash_flow": db_data.get("cash_flow_annual"),
                },
            } if db_data.get("income_stmt_quarterly") else None,
            net_interest_income=_safe_float(db_data.get("net_interest_income")),
            net_interest_margin=_safe_float(db_data.get("net_interest_margin")),
            ffo=_safe_float(db_data.get("ffo")),
            ffo_per_share=_safe_float(db_data.get("ffo_per_share")),
            p_ffo=_safe_float(db_data.get("p_ffo")),
            loss_ratio=_safe_float(db_data.get("loss_ratio")),
            
            raw_info=db_data,
        )
    
    # Fallback to yfinance if not in DB (and store the result)
    logger.info(f"No DB data for {symbol}, fetching from yfinance")
    service = get_yfinance_service()
    
    # Fetch info and optionally financials concurrently
    if include_financials:
        info, financials = await asyncio.gather(
            get_ticker_info(symbol),
            service.get_financials(symbol),
        )
    else:
        info = await get_ticker_info(symbol)
        financials = None
    
    if not info:
        return Fundamentals(symbol=symbol, name=symbol)
    
    # Detect domain and calculate domain-specific metrics
    domain = _detect_domain(info)
    domain_metrics = _calculate_domain_metrics(info, financials, domain)
    
    # Store in DB for future requests (via the fundamentals service)
    try:
        from app.services.fundamentals import _fetch_fundamentals_from_service, _store_fundamentals
        fetched = await _fetch_fundamentals_from_service(symbol)
        if fetched:
            await _store_fundamentals(fetched)
    except Exception as e:
        logger.warning(f"Failed to store fundamentals for {symbol}: {e}")
    
    return Fundamentals(
        symbol=symbol,
        name=info.get("name") or info.get("shortName") or info.get("longName") or symbol,
        sector=info.get("sector"),
        industry=info.get("industry"),
        market_cap=_safe_float(info.get("market_cap") or info.get("marketCap")),
        enterprise_value=_safe_float(info.get("enterprise_value") or info.get("enterpriseValue")),
        
        # Valuation
        pe_ratio=_safe_float(info.get("pe_ratio") or info.get("trailingPE")),
        forward_pe=_safe_float(info.get("forward_pe") or info.get("forwardPE")),
        peg_ratio=_safe_float(info.get("peg_ratio") or info.get("pegRatio")),
        price_to_book=_safe_float(info.get("price_to_book") or info.get("priceToBook")),
        price_to_sales=_safe_float(info.get("price_to_sales") or info.get("priceToSalesTrailing12Months")),
        ev_to_ebitda=_safe_float(info.get("ev_to_ebitda") or info.get("enterpriseToEbitda")),
        ev_to_revenue=_safe_float(info.get("ev_to_revenue") or info.get("enterpriseToRevenue")),
        
        # Profitability
        profit_margin=_safe_float(info.get("profit_margin") or info.get("profitMargins")),
        operating_margin=_safe_float(info.get("operating_margin") or info.get("operatingMargins")),
        gross_margin=_safe_float(info.get("gross_margin") or info.get("grossMargins")),
        roe=_safe_float(info.get("return_on_equity") or info.get("returnOnEquity")),
        roa=_safe_float(info.get("return_on_assets") or info.get("returnOnAssets")),
        
        # Growth
        revenue_growth=_safe_float(info.get("revenue_growth") or info.get("revenueGrowth")),
        earnings_growth=_safe_float(info.get("earnings_growth") or info.get("earningsGrowth")),
        
        # Financial health
        current_ratio=_safe_float(info.get("current_ratio") or info.get("currentRatio")),
        quick_ratio=_safe_float(info.get("quick_ratio") or info.get("quickRatio")),
        debt_to_equity=_safe_float(info.get("debt_to_equity") or info.get("debtToEquity")),
        free_cash_flow=_safe_float(info.get("free_cash_flow") or info.get("freeCashflow")),
        
        # Dividends
        dividend_yield=_safe_float(info.get("dividend_yield") or info.get("dividendYield")),
        payout_ratio=_safe_float(info.get("payoutRatio")),
        
        # Per-share
        eps=_safe_float(info.get("eps_trailing") or info.get("trailingEps")),
        eps_forward=_safe_float(info.get("eps_forward") or info.get("forwardEps")),
        book_value_per_share=_safe_float(info.get("book_value") or info.get("bookValue")),
        
        # Risk
        beta=_safe_float(info.get("beta")),
        shares_outstanding=_safe_float(info.get("shares_outstanding") or info.get("sharesOutstanding")),
        float_shares=_safe_float(info.get("float_shares") or info.get("floatShares")),
        short_ratio=_safe_float(info.get("short_ratio") or info.get("shortRatio")),
        short_percent_of_float=_safe_float(info.get("short_percent_of_float") or info.get("shortPercentOfFloat")),
        
        # Domain-specific
        domain=domain,
        financials=financials,
        net_interest_income=domain_metrics.get("net_interest_income"),
        net_interest_margin=domain_metrics.get("net_interest_margin"),
        ffo=domain_metrics.get("ffo"),
        ffo_per_share=domain_metrics.get("ffo_per_share"),
        p_ffo=domain_metrics.get("p_ffo"),
        loss_ratio=domain_metrics.get("loss_ratio"),
        
        raw_info=info,
    )


async def get_market_data(
    symbol: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> MarketData:
    """Get complete market data for a symbol."""
    # Fetch all data concurrently
    prices_task = get_price_history(symbol, start_date, end_date, period)
    fundamentals_task = get_fundamentals(symbol)
    calendar_task = get_calendar_events(symbol)
    
    prices, fundamentals, calendar = await asyncio.gather(
        prices_task,
        fundamentals_task,
        calendar_task,
    )
    
    return MarketData(
        symbol=symbol,
        prices=prices,
        fundamentals=fundamentals,
        calendar=calendar,
    )


async def get_market_data_batch(
    symbols: list[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    period: str = "1y",
) -> dict[str, MarketData]:
    """Get market data for multiple symbols concurrently."""
    tasks = [
        get_market_data(symbol, start_date, end_date, period)
        for symbol in symbols
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    data = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, Exception):
            logger.error(f"Failed to fetch data for {symbol}: {result}")
            # Return minimal data
            data[symbol] = MarketData(
                symbol=symbol,
                prices=PriceSeries(symbol=symbol, prices=[]),
                fundamentals=Fundamentals(symbol=symbol, name=symbol),
            )
        else:
            data[symbol] = result
    
    return data


# =============================================================================
# Helpers
# =============================================================================


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert value to float."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:  # NaN check
            return default
        return f
    except (ValueError, TypeError):
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely convert value to int."""
    if value is None:
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default


# =============================================================================
# Cache management (delegated to unified service)
# =============================================================================


def clear_cache():
    """Clear cache - unified service manages its own cache."""
    # The unified service handles cache management
    pass


def get_cache_stats() -> dict[str, int]:
    """Get cache statistics."""
    # Return empty stats - unified service manages cache
    return {
        "info_cache_size": 0,
        "price_cache_size": 0,
        "note": "Cache managed by unified YFinanceService",
    }
