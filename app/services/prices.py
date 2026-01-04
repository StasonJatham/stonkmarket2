"""
Unified Price Service.

Single source of truth for all price data operations.
This is the ONLY module that should be used for fetching/saving price data.

Architecture:
    1. Database first - always check local DB
    2. yfinance fallback - fetch missing data from yfinance  
    3. Validate before save - reject corrupt data
    4. No corrupt data returns - never return unvalidated yfinance data

Usage:
    from app.services.prices import get_price_service
    
    service = get_price_service()
    
    # Get prices (DB first, yfinance fills gaps)
    df = await service.get_prices("AAPL", start_date, end_date)
    
    # Get prices for multiple symbols
    results = await service.get_prices_batch(["AAPL", "MSFT"], start, end)
    
    # Refresh prices (fetch from yfinance, validate, save)
    count = await service.refresh_prices(["AAPL", "MSFT"])
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pandas as pd
import yfinance as yf

from app.core.logging import get_logger
from app.core.rate_limiter import get_yfinance_limiter
from app.repositories import price_history_orm


if TYPE_CHECKING:
    from collections.abc import Sequence


logger = get_logger("services.prices")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Validation: reject data with day-over-day change exceeding this
MAX_VALID_DAILY_CHANGE = 0.50  # 50% - anything higher is likely corrupt

# Minimum trading days expected per calendar week
TRADING_DAYS_PER_WEEK = 5

# yfinance batch size limit
MAX_BATCH_SIZE = 50


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Result of price data validation."""
    
    valid: bool
    error: str | None = None
    

def validate_prices(
    symbol: str,
    df: pd.DataFrame,
    existing_last_price: Decimal | None = None,
) -> ValidationResult:
    """
    Validate price data before saving.
    
    Checks:
    1. DataFrame is not empty
    2. Has required Close column
    3. No impossible day-over-day price changes (>50%)
    4. Continuity with existing data (if provided)
    
    Args:
        symbol: Ticker symbol (for logging)
        df: Price DataFrame with Close column
        existing_last_price: Last known price in DB for continuity check
    
    Returns:
        ValidationResult with valid=True or error message
    """
    if df.empty:
        return ValidationResult(valid=False, error="Empty DataFrame")
    
    if "Close" not in df.columns:
        return ValidationResult(valid=False, error="Missing Close column")
    
    close = df["Close"].dropna()
    if close.empty:
        return ValidationResult(valid=False, error="No valid Close values")
    
    # Check for impossible daily changes within the data
    if len(close) > 1:
        pct_changes = close.pct_change().abs().dropna()
        max_change = pct_changes.max()
        if max_change > MAX_VALID_DAILY_CHANGE:
            bad_idx = pct_changes.idxmax()
            return ValidationResult(
                valid=False,
                error=f"{max_change:.1%} change at {bad_idx} exceeds {MAX_VALID_DAILY_CHANGE:.0%} threshold",
            )
    
    # Check continuity with existing data
    if existing_last_price is not None and existing_last_price > 0:
        first_new_price = float(close.iloc[0])
        existing_float = float(existing_last_price)
        gap_pct = abs(first_new_price - existing_float) / existing_float
        
        if gap_pct > MAX_VALID_DAILY_CHANGE:
            return ValidationResult(
                valid=False,
                error=f"Gap from existing ${existing_float:.2f} to new ${first_new_price:.2f} ({gap_pct:.1%})",
            )
    
    return ValidationResult(valid=True)


# =============================================================================
# YFINANCE HELPERS
# =============================================================================

def _fetch_from_yfinance(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame | None:
    """
    Fetch price history from yfinance.
    
    Uses auto_adjust=True so Close column contains split-adjusted prices.
    
    Returns:
        DataFrame with OHLCV data, or None if fetch failed
    """
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning(f"Rate limit for {symbol}")
        return None
    
    try:
        # Add 1 day to end_date because yfinance end is exclusive
        df = yf.download(
            symbol,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            progress=False,
            timeout=30,
        )
        
        if df.empty:
            return None
        
        # Handle MultiIndex columns (newer yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except Exception:
                pass
        
        return df
        
    except Exception as e:
        logger.warning(f"yfinance fetch failed for {symbol}: {e}")
        return None


def _fetch_batch_from_yfinance(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    """
    Batch fetch price history from yfinance.
    
    Returns:
        Dict mapping symbol to DataFrame
    """
    if not symbols:
        return {}
    
    limiter = get_yfinance_limiter()
    if not limiter.acquire_sync():
        logger.warning("Rate limit for batch fetch")
        return {}
    
    try:
        # Single symbol - simpler path
        if len(symbols) == 1:
            df = _fetch_from_yfinance(symbols[0], start_date, end_date)
            return {symbols[0]: df} if df is not None else {}
        
        # Multi-symbol batch
        df = yf.download(
            " ".join(symbols),
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            timeout=60,
        )
        
        if df.empty:
            return {}
        
        results = {}
        for symbol in symbols:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    if symbol in df.columns.get_level_values(0):
                        ticker_df = df[symbol].dropna(how="all")
                    elif symbol.upper() in df.columns.get_level_values(0):
                        ticker_df = df[symbol.upper()].dropna(how="all")
                    else:
                        continue
                else:
                    ticker_df = df
                
                if not ticker_df.empty:
                    results[symbol] = ticker_df
            except Exception as e:
                logger.debug(f"Failed to extract {symbol}: {e}")
        
        return results
        
    except Exception as e:
        logger.warning(f"yfinance batch fetch failed: {e}")
        return {}


# =============================================================================
# PRICE SERVICE
# =============================================================================

class PriceService:
    """
    Unified price service.
    
    Single entry point for all price data operations.
    Database-first with yfinance fallback.
    """
    
    async def get_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame | None:
        """
        Get price data for a symbol.
        
        1. Fetch from database
        2. If data is incomplete, fetch missing from yfinance
        3. Validate yfinance data before merging
        4. Return only validated data
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            DataFrame with OHLCV data, or None if no data available
        """
        symbol = symbol.upper()
        
        # Get from database
        db_df = await price_history_orm.get_prices_as_dataframe(
            symbol, start_date, end_date
        )
        
        # Check if we have complete data
        if db_df is not None and not db_df.empty:
            last_idx = db_df.index.max()
            last_date = last_idx.date() if hasattr(last_idx, "date") else last_idx
            
            # Calculate days since last data, accounting for weekends
            # If it's weekend (Sat=5, Sun=6), last trading day was Friday
            today = date.today()
            days_since = (today - last_date).days
            
            # Allow up to 3 calendar days gap (covers weekends + 1 day)
            # This means: data from Friday is valid until Monday
            if days_since <= 3:
                return db_df
            
            # Need to fetch more recent data
            fetch_start = last_date + timedelta(days=1)
        else:
            # No DB data - fetch everything
            fetch_start = start_date
            db_df = None
        
        # Fetch from yfinance
        yf_df = _fetch_from_yfinance(symbol, fetch_start, end_date)
        
        if yf_df is None or yf_df.empty:
            # yfinance failed - return whatever we have from DB
            return db_df
        
        # Get existing last price for validation
        existing_prices = await price_history_orm.get_latest_prices([symbol])
        existing_price = existing_prices.get(symbol)
        
        # Validate yfinance data
        validation = validate_prices(symbol, yf_df, existing_price)
        
        if not validation.valid:
            logger.warning(f"Rejecting {symbol} yfinance data: {validation.error}")
            # Return only DB data - don't merge corrupt yfinance data
            return db_df
        
        # Save valid yfinance data
        try:
            count = await price_history_orm.save_prices(symbol, yf_df)
            logger.debug(f"Saved {count} prices for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to save {symbol} prices: {e}")
        
        # Merge and return
        if db_df is not None and not db_df.empty:
            merged = pd.concat([db_df, yf_df], axis=0)
            merged.index = pd.to_datetime(merged.index)
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            return merged
        
        return yf_df
    
    async def get_prices_batch(
        self,
        symbols: Sequence[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """
        Get price data for multiple symbols.
        
        Optimizes by:
        1. Checking DB for all symbols first
        2. Grouping symbols by how much data is missing
        3. Batch fetching from yfinance
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            Dict mapping symbol to DataFrame
        """
        if not symbols:
            return {}
        
        normalized = [s.upper() for s in symbols]
        results: dict[str, pd.DataFrame] = {}
        
        # Get all DB data in one query
        db_data = await price_history_orm.get_prices_batch(
            normalized, start_date, end_date
        )
        
        # Get last prices for validation
        all_last_prices = await price_history_orm.get_latest_prices(normalized)
        
        # Determine which symbols need yfinance fetch
        symbols_to_fetch: list[str] = []
        
        for symbol in normalized:
            prices = db_data.get(symbol, [])
            
            if prices:
                # Convert to DataFrame
                data = []
                for p in prices:
                    data.append({
                        "date": p.date,
                        "Open": float(p.open) if p.open else None,
                        "High": float(p.high) if p.high else None,
                        "Low": float(p.low) if p.low else None,
                        "Close": float(p.close) if p.close else None,
                        "Volume": int(p.volume) if p.volume else None,
                    })
                df = pd.DataFrame(data)
                df.set_index("date", inplace=True)
                df.index = pd.to_datetime(df.index)
                
                last_date = df.index.max().date()
                if last_date >= end_date - timedelta(days=1):
                    # Complete data
                    results[symbol] = df
                else:
                    # Need more recent data
                    results[symbol] = df
                    symbols_to_fetch.append(symbol)
            else:
                # No DB data
                symbols_to_fetch.append(symbol)
        
        # Batch fetch from yfinance
        if symbols_to_fetch:
            for i in range(0, len(symbols_to_fetch), MAX_BATCH_SIZE):
                batch = symbols_to_fetch[i : i + MAX_BATCH_SIZE]
                yf_data = _fetch_batch_from_yfinance(batch, start_date, end_date)
                
                for symbol, yf_df in yf_data.items():
                    existing_price = all_last_prices.get(symbol)
                    validation = validate_prices(symbol, yf_df, existing_price)
                    
                    if not validation.valid:
                        logger.warning(f"Rejecting {symbol}: {validation.error}")
                        continue
                    
                    # Save to DB
                    try:
                        await price_history_orm.save_prices(symbol, yf_df)
                    except Exception as e:
                        logger.warning(f"Failed to save {symbol}: {e}")
                    
                    # Merge with existing
                    if symbol in results:
                        merged = pd.concat([results[symbol], yf_df], axis=0)
                        merged.index = pd.to_datetime(merged.index)
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                        results[symbol] = merged
                    else:
                        results[symbol] = yf_df
        
        return results
    
    async def refresh_prices(
        self,
        symbols: Sequence[str],
        days: int = 5,
    ) -> dict[str, int]:
        """
        Refresh recent prices for symbols.
        
        Fetches last N days from yfinance and updates DB.
        
        Args:
            symbols: List of stock ticker symbols
            days: Number of days to refresh (default 5)
        
        Returns:
            Dict mapping symbol to number of rows saved
        """
        if not symbols:
            return {}
        
        normalized = [s.upper() for s in symbols]
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        # Get existing last prices for validation
        all_last_prices = await price_history_orm.get_latest_prices(normalized)
        
        saved_counts: dict[str, int] = {}
        
        # Batch fetch
        for i in range(0, len(normalized), MAX_BATCH_SIZE):
            batch = normalized[i : i + MAX_BATCH_SIZE]
            yf_data = _fetch_batch_from_yfinance(batch, start_date, end_date)
            
            for symbol, yf_df in yf_data.items():
                existing_price = all_last_prices.get(symbol)
                validation = validate_prices(symbol, yf_df, existing_price)
                
                if not validation.valid:
                    logger.warning(f"Rejecting {symbol}: {validation.error}")
                    saved_counts[symbol] = 0
                    continue
                
                try:
                    count = await price_history_orm.save_prices(symbol, yf_df)
                    saved_counts[symbol] = count
                    logger.debug(f"Refreshed {count} prices for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to save {symbol}: {e}")
                    saved_counts[symbol] = 0
        
        return saved_counts
    
    async def get_latest_prices(
        self,
        symbols: Sequence[str],
    ) -> dict[str, Decimal]:
        """
        Get most recent close price for symbols.
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            Dict mapping symbol to latest close price
        """
        return await price_history_orm.get_latest_prices(symbols)
    
    async def get_latest_price_dates(
        self,
        symbols: Sequence[str],
    ) -> dict[str, date]:
        """
        Get most recent price date for symbols.
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            Dict mapping symbol to latest price date
        """
        return await price_history_orm.get_latest_price_dates(symbols)

    async def get_gaps_summary(
        self,
        symbols: Sequence[str],
    ) -> dict:
        """
        Get summary of price data gaps for reporting.
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            Dict with gap statistics
        """
        if not symbols:
            return {"total_symbols": 0}
        
        normalized = [s.upper() for s in symbols]
        latest_dates = await price_history_orm.get_latest_price_dates(normalized)
        today = date.today()
        
        gaps: list[dict] = []
        up_to_date = 0
        slight_gaps = 0
        moderate_gaps = 0
        large_gaps = 0
        no_data = 0
        
        for symbol in normalized:
            last_date = latest_dates.get(symbol)
            
            if last_date is None:
                missing_days = 365
            else:
                calendar_days = (today - last_date).days
                missing_days = max(0, int(calendar_days * 5 / 7))  # Trading days
            
            if missing_days == 0:
                up_to_date += 1
            elif missing_days <= 7:
                slight_gaps += 1
            elif missing_days <= 30:
                moderate_gaps += 1
            elif missing_days <= 120:
                large_gaps += 1
            else:
                no_data += 1
            
            if missing_days > 0:
                gaps.append({
                    "symbol": symbol,
                    "last_date": str(last_date) if last_date else None,
                    "missing_days": missing_days,
                })
        
        return {
            "total_symbols": len(normalized),
            "up_to_date": up_to_date,
            "slight_gaps_1_7_days": slight_gaps,
            "moderate_gaps_8_30_days": moderate_gaps,
            "large_gaps_31_120_days": large_gaps,
            "no_data_or_stale": no_data,
            "gaps": sorted(gaps, key=lambda x: x["missing_days"], reverse=True)[:50],
        }

    async def check_data_integrity(
        self,
        symbols: Sequence[str] | None = None,
        max_daily_change_pct: float = 40.0,
    ) -> dict:
        """
        Check price data integrity for all or specified symbols.
        
        Detects:
        1. Suspicious daily price changes (>40% by default)
        2. Negative prices
        3. High < Low violations
        
        Args:
            symbols: Symbols to check. If None, check all tracked symbols.
            max_daily_change_pct: Threshold for suspicious daily changes.
        
        Returns:
            Dictionary with integrity report and list of anomalies.
        """
        from app.database.connection import get_session
        from sqlalchemy import text
        
        # Get all symbols if not specified
        if symbols is None:
            symbols_repo = await price_history_orm.get_distinct_symbols()
            symbols = list(symbols_repo)
        
        async with get_session() as session:
            # Query for suspicious price jumps
            result = await session.execute(text("""
                WITH price_changes AS (
                    SELECT 
                        symbol,
                        date,
                        close,
                        LAG(close) OVER (PARTITION BY symbol ORDER BY date) as prev_close,
                        (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) / 
                        NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) * 100 as pct_change
                    FROM price_history
                    WHERE symbol = ANY(:symbols)
                )
                SELECT symbol, date, prev_close, close, pct_change
                FROM price_changes
                WHERE ABS(pct_change) > :threshold
                ORDER BY ABS(pct_change) DESC
                LIMIT 100
            """), {"symbols": list(symbols), "threshold": max_daily_change_pct})
            
            anomalies = [
                {
                    "symbol": row.symbol,
                    "date": str(row.date),
                    "prev_close": float(row.prev_close) if row.prev_close else None,
                    "close": float(row.close),
                    "pct_change": round(float(row.pct_change), 2),
                    "type": "suspicious_jump",
                }
                for row in result.fetchall()
            ]
            
            # Query for negative prices
            result = await session.execute(text("""
                SELECT symbol, date, close
                FROM price_history
                WHERE symbol = ANY(:symbols)
                AND (close < 0 OR open < 0 OR high < 0 OR low < 0)
                ORDER BY date DESC
                LIMIT 50
            """), {"symbols": list(symbols)})
            
            for row in result.fetchall():
                anomalies.append({
                    "symbol": row.symbol,
                    "date": str(row.date),
                    "close": float(row.close),
                    "type": "negative_price",
                })
            
            # Query for High < Low violations
            result = await session.execute(text("""
                SELECT symbol, date, high, low
                FROM price_history
                WHERE symbol = ANY(:symbols)
                AND high < low
                ORDER BY date DESC
                LIMIT 50
            """), {"symbols": list(symbols)})
            
            for row in result.fetchall():
                anomalies.append({
                    "symbol": row.symbol,
                    "date": str(row.date),
                    "high": float(row.high),
                    "low": float(row.low),
                    "type": "high_low_violation",
                })
        
        # Group anomalies by symbol
        symbols_affected = set(a["symbol"] for a in anomalies)
        
        return {
            "total_symbols_checked": len(symbols),
            "symbols_with_anomalies": len(symbols_affected),
            "total_anomalies": len(anomalies),
            "affected_symbols": sorted(symbols_affected),
            "anomalies": anomalies,
            "recommendation": (
                "Run repair for affected symbols" if anomalies
                else "All data looks healthy"
            ),
        }

    async def repair_symbol_data(
        self,
        symbol: str,
        delete_from_date: date | None = None,
    ) -> dict:
        """
        Repair price data for a symbol by deleting corrupt data and re-fetching.
        
        Args:
            symbol: Symbol to repair
            delete_from_date: Delete all data from this date onwards.
                              If None, deletes last 30 days.
        
        Returns:
            Dictionary with repair results.
        """
        from app.database.connection import get_session
        from sqlalchemy import text
        
        symbol = symbol.upper()
        
        if delete_from_date is None:
            delete_from_date = date.today() - timedelta(days=30)
        
        # Delete corrupt data
        async with get_session() as session:
            result = await session.execute(text("""
                DELETE FROM price_history
                WHERE symbol = :symbol
                AND date >= :from_date
            """), {"symbol": symbol, "from_date": delete_from_date})
            deleted_count = result.rowcount
            await session.commit()
        
        logger.info(f"Deleted {deleted_count} rows for {symbol} from {delete_from_date}")
        
        # Re-fetch data
        end_date = date.today()
        start_date = delete_from_date - timedelta(days=7)  # Overlap for continuity
        
        df = await self.get_prices(symbol, start_date, end_date)
        
        new_count = len(df) if df is not None else 0
        
        return {
            "symbol": symbol,
            "deleted_rows": deleted_count,
            "new_rows_fetched": new_count,
            "from_date": str(delete_from_date),
            "success": new_count > 0,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_price_service: PriceService | None = None


def get_price_service() -> PriceService:
    """Get singleton PriceService instance."""
    global _price_service
    if _price_service is None:
        _price_service = PriceService()
    return _price_service
