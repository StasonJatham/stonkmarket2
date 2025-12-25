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
from typing import Any, Optional

from app.core.logging import get_logger
from app.database.connection import fetch_one, fetch_all, execute
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


async def _fetch_fundamentals_from_service(symbol: str) -> Optional[dict[str, Any]]:
    """
    Fetch fundamentals via unified YFinanceService.
    
    Converts the ticker info response to fundamentals format.
    """
    if symbol.startswith("^"):
        logger.debug(f"Skipping fundamentals for index: {symbol}")
        return None
    
    service = get_yfinance_service()
    
    # Get ticker info - includes all fundamental data
    info = await service.get_ticker_info(symbol)
    if not info:
        return None
    
    # Check if ETF/fund
    quote_type = info.get("quote_type")
    if _is_etf_or_index(symbol, quote_type):
        logger.debug(f"Skipping fundamentals for {quote_type}: {symbol}")
        return None
    
    # Get calendar for earnings dates
    calendar, _ = await service.get_calendar(symbol)
    
    next_earnings_date = None
    earnings_estimate_high = None
    earnings_estimate_low = None
    earnings_estimate_avg = None
    
    if calendar:
        next_earnings_date = calendar.get("next_earnings_date")
        earnings_estimate_high = calendar.get("earnings_estimate_high")
        earnings_estimate_low = calendar.get("earnings_estimate_low")
        earnings_estimate_avg = calendar.get("earnings_estimate_avg")
    
    return {
        "symbol": symbol.upper(),
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
    }


async def _store_fundamentals(data: dict[str, Any]) -> None:
    """Store fundamentals in database."""
    symbol = data["symbol"]
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    
    await execute(
        """
        INSERT INTO stock_fundamentals (
            symbol,
            pe_ratio, forward_pe, peg_ratio, price_to_book, price_to_sales,
            enterprise_value, ev_to_ebitda, ev_to_revenue,
            profit_margin, operating_margin, gross_margin, ebitda_margin,
            return_on_equity, return_on_assets,
            debt_to_equity, current_ratio, quick_ratio,
            total_cash, total_debt, free_cash_flow, operating_cash_flow,
            book_value, eps_trailing, eps_forward, revenue_per_share,
            revenue_growth, earnings_growth, earnings_quarterly_growth,
            shares_outstanding, float_shares,
            held_percent_insiders, held_percent_institutions,
            short_ratio, short_percent_of_float,
            beta,
            recommendation, recommendation_mean, num_analyst_opinions,
            target_high_price, target_low_price, target_mean_price, target_median_price,
            revenue, ebitda, net_income,
            next_earnings_date, earnings_estimate_high, earnings_estimate_low, earnings_estimate_avg,
            fetched_at, expires_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
            $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
            $31, $32, $33, $34, $35, $36, $37, $38, $39, $40,
            $41, $42, $43, $44, $45, $46, $47, $48, $49, $50,
            NOW(), $51
        )
        ON CONFLICT (symbol) DO UPDATE SET
            pe_ratio = EXCLUDED.pe_ratio,
            forward_pe = EXCLUDED.forward_pe,
            peg_ratio = EXCLUDED.peg_ratio,
            price_to_book = EXCLUDED.price_to_book,
            price_to_sales = EXCLUDED.price_to_sales,
            enterprise_value = EXCLUDED.enterprise_value,
            ev_to_ebitda = EXCLUDED.ev_to_ebitda,
            ev_to_revenue = EXCLUDED.ev_to_revenue,
            profit_margin = EXCLUDED.profit_margin,
            operating_margin = EXCLUDED.operating_margin,
            gross_margin = EXCLUDED.gross_margin,
            ebitda_margin = EXCLUDED.ebitda_margin,
            return_on_equity = EXCLUDED.return_on_equity,
            return_on_assets = EXCLUDED.return_on_assets,
            debt_to_equity = EXCLUDED.debt_to_equity,
            current_ratio = EXCLUDED.current_ratio,
            quick_ratio = EXCLUDED.quick_ratio,
            total_cash = EXCLUDED.total_cash,
            total_debt = EXCLUDED.total_debt,
            free_cash_flow = EXCLUDED.free_cash_flow,
            operating_cash_flow = EXCLUDED.operating_cash_flow,
            book_value = EXCLUDED.book_value,
            eps_trailing = EXCLUDED.eps_trailing,
            eps_forward = EXCLUDED.eps_forward,
            revenue_per_share = EXCLUDED.revenue_per_share,
            revenue_growth = EXCLUDED.revenue_growth,
            earnings_growth = EXCLUDED.earnings_growth,
            earnings_quarterly_growth = EXCLUDED.earnings_quarterly_growth,
            shares_outstanding = EXCLUDED.shares_outstanding,
            float_shares = EXCLUDED.float_shares,
            held_percent_insiders = EXCLUDED.held_percent_insiders,
            held_percent_institutions = EXCLUDED.held_percent_institutions,
            short_ratio = EXCLUDED.short_ratio,
            short_percent_of_float = EXCLUDED.short_percent_of_float,
            beta = EXCLUDED.beta,
            recommendation = EXCLUDED.recommendation,
            recommendation_mean = EXCLUDED.recommendation_mean,
            num_analyst_opinions = EXCLUDED.num_analyst_opinions,
            target_high_price = EXCLUDED.target_high_price,
            target_low_price = EXCLUDED.target_low_price,
            target_mean_price = EXCLUDED.target_mean_price,
            target_median_price = EXCLUDED.target_median_price,
            revenue = EXCLUDED.revenue,
            ebitda = EXCLUDED.ebitda,
            net_income = EXCLUDED.net_income,
            next_earnings_date = EXCLUDED.next_earnings_date,
            earnings_estimate_high = EXCLUDED.earnings_estimate_high,
            earnings_estimate_low = EXCLUDED.earnings_estimate_low,
            earnings_estimate_avg = EXCLUDED.earnings_estimate_avg,
            fetched_at = NOW(),
            expires_at = EXCLUDED.expires_at
        """,
        symbol,
        data["pe_ratio"], data["forward_pe"], data["peg_ratio"],
        data["price_to_book"], data["price_to_sales"],
        data["enterprise_value"], data["ev_to_ebitda"], data["ev_to_revenue"],
        data["profit_margin"], data["operating_margin"],
        data["gross_margin"], data["ebitda_margin"],
        data["return_on_equity"], data["return_on_assets"],
        data["debt_to_equity"], data["current_ratio"], data["quick_ratio"],
        data["total_cash"], data["total_debt"],
        data["free_cash_flow"], data["operating_cash_flow"],
        data["book_value"], data["eps_trailing"], data["eps_forward"],
        data["revenue_per_share"],
        data["revenue_growth"], data["earnings_growth"], data["earnings_quarterly_growth"],
        data["shares_outstanding"], data["float_shares"],
        data["held_percent_insiders"], data["held_percent_institutions"],
        data["short_ratio"], data["short_percent_of_float"],
        data["beta"],
        data["recommendation"], data["recommendation_mean"], data["num_analyst_opinions"],
        data["target_high_price"], data["target_low_price"],
        data["target_mean_price"], data["target_median_price"],
        data["revenue"], data["ebitda"], data["net_income"],
        data["next_earnings_date"],
        data["earnings_estimate_high"], data["earnings_estimate_low"], data["earnings_estimate_avg"],
        expires_at,
    )
    
    logger.debug(f"Stored fundamentals for {symbol}")


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


async def get_fundamentals_from_db(symbol: str) -> Optional[dict[str, Any]]:
    """
    Get fundamentals from database cache only (no fetching).
    
    Use this when you want to check if data exists without triggering
    a yfinance fetch. Useful for untracked symbols.
    """
    symbol = symbol.upper()
    
    if symbol.startswith("^"):
        return None
    
    row = await fetch_one(
        """
        SELECT * FROM stock_fundamentals
        WHERE symbol = $1 AND expires_at > NOW()
        """,
        symbol,
    )
    if row:
        return dict(row)
    return None


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
        row = await fetch_one(
            """
            SELECT * FROM stock_fundamentals
            WHERE symbol = $1 AND expires_at > NOW()
            """,
            symbol,
        )
        if row:
            return dict(row)
    
    # Fetch via unified service
    data = await _fetch_fundamentals_from_service(symbol)
    
    if data:
        await _store_fundamentals(data)
        return data
    
    return None


async def refresh_fundamentals(symbol: str) -> Optional[dict[str, Any]]:
    """Force refresh fundamentals for a symbol."""
    return await get_fundamentals(symbol, force_refresh=True)


async def refresh_all_fundamentals(batch_size: int = 10) -> dict[str, int]:
    """
    Refresh fundamentals for all symbols with expired or missing data.
    
    Args:
        batch_size: Number of symbols to process in parallel
        
    Returns:
        Dict with counts: {"refreshed": N, "failed": M, "skipped": K}
    """
    # Get symbols needing refresh (expired or missing)
    rows = await fetch_all(
        """
        SELECT s.symbol
        FROM symbols s
        LEFT JOIN stock_fundamentals f ON s.symbol = f.symbol
        WHERE s.symbol_type = 'stock'
          AND s.is_active = TRUE
          AND (f.symbol IS NULL OR f.expires_at < NOW())
        ORDER BY f.expires_at NULLS FIRST
        LIMIT 100
        """
    )
    
    symbols = [r["symbol"] for r in rows]
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
    row = await fetch_one(
        "SELECT pe_ratio FROM symbols WHERE symbol = $1",
        symbol.upper(),
    )
    if row and row["pe_ratio"]:
        return float(row["pe_ratio"])
    
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
