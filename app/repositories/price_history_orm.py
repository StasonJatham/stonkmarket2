"""Price history repository using SQLAlchemy ORM.

Repository for managing price history data.

Usage:
    from app.repositories import price_history_orm as price_history_repo
    
    prices = await price_history_repo.get_prices("AAPL", start_date, end_date)
    await price_history_repo.save_prices("AAPL", df)
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from decimal import Decimal

import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import PriceHistory


logger = get_logger("repositories.price_history_orm")


async def get_prices(
    symbol: str,
    start_date: date,
    end_date: date,
) -> Sequence[PriceHistory]:
    """Get price history for a symbol within date range.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        Sequence of PriceHistory objects ordered by date
    """
    async with get_session() as session:
        result = await session.execute(
            select(PriceHistory)
            .where(
                and_(
                    PriceHistory.symbol == symbol.upper(),
                    PriceHistory.date >= start_date,
                    PriceHistory.date <= end_date,
                )
            )
            .order_by(PriceHistory.date.asc())
        )
        return result.scalars().all()


async def get_latest_price_date(symbol: str) -> date | None:
    """Get the most recent price date for a symbol."""
    async with get_session() as session:
        result = await session.execute(
            select(func.max(PriceHistory.date)).where(
                PriceHistory.symbol == symbol.upper()
            )
        )
        return result.scalar_one_or_none()


async def has_price_history(symbol: str, min_days: int = 30) -> bool:
    """Check if a symbol has sufficient price history.
    
    Args:
        symbol: Stock ticker symbol
        min_days: Minimum number of days required (default 30)
    
    Returns:
        True if symbol has at least min_days of price data
    """
    async with get_session() as session:
        result = await session.execute(
            select(func.count(PriceHistory.id)).where(
                PriceHistory.symbol == symbol.upper()
            )
        )
        count = result.scalar_one()
        return count >= min_days


async def get_prices_batch(
    symbols: Sequence[str],
    start_date: date,
    end_date: date,
) -> dict[str, list[PriceHistory]]:
    """Get price history for multiple symbols in a single query.
    
    Args:
        symbols: List of stock ticker symbols
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        Dict mapping symbol to list of PriceHistory objects ordered by date
    """
    if not symbols:
        return {}
    
    normalized = [s.upper() for s in symbols]
    
    async with get_session() as session:
        result = await session.execute(
            select(PriceHistory)
            .where(
                and_(
                    PriceHistory.symbol.in_(normalized),
                    PriceHistory.date >= start_date,
                    PriceHistory.date <= end_date,
                )
            )
            .order_by(PriceHistory.symbol, PriceHistory.date.asc())
        )
        
        # Group by symbol
        prices_by_symbol: dict[str, list[PriceHistory]] = {s: [] for s in normalized}
        for price in result.scalars().all():
            prices_by_symbol[price.symbol].append(price)
        
        return prices_by_symbol


async def get_latest_price_dates(symbols: Sequence[str]) -> dict[str, date]:
    """Get most recent price dates for multiple symbols."""
    normalized = [s.upper() for s in symbols]
    if not normalized:
        return {}

    async with get_session() as session:
        result = await session.execute(
            select(PriceHistory.symbol, func.max(PriceHistory.date))
            .where(PriceHistory.symbol.in_(normalized))
            .group_by(PriceHistory.symbol)
        )
        return {row[0]: row[1] for row in result.all() if row[1] is not None}


async def get_latest_prices(symbols: Sequence[str]) -> dict[str, Decimal]:
    """Get most recent close price for multiple symbols.

    Uses a subquery to find the max date per symbol, then joins
    to get the close price for that date. Single efficient query.

    Args:
        symbols: List of stock ticker symbols

    Returns:
        Dict mapping symbol to latest close price
    """
    normalized = [s.upper() for s in symbols]
    if not normalized:
        return {}

    async with get_session() as session:
        # Subquery: get max date per symbol
        subq = (
            select(
                PriceHistory.symbol,
                func.max(PriceHistory.date).label("max_date"),
            )
            .where(PriceHistory.symbol.in_(normalized))
            .group_by(PriceHistory.symbol)
            .subquery()
        )

        # Join to get close price at max date
        result = await session.execute(
            select(PriceHistory.symbol, PriceHistory.close)
            .join(
                subq,
                and_(
                    PriceHistory.symbol == subq.c.symbol,
                    PriceHistory.date == subq.c.max_date,
                ),
            )
        )

        return {
            row[0]: row[1]
            for row in result.all()
            if row[1] is not None
        }


async def get_prices_as_dataframe(
    symbol: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame | None:
    """Get price history as a pandas DataFrame.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
    
    Returns:
        DataFrame with OHLCV data indexed by date, or None if no data
    """
    prices = await get_prices(symbol, start_date, end_date)

    if not prices:
        return None

    data = []
    for p in prices:
        data.append({
            "date": p.date,
            "Open": float(p.open) if p.open else None,
            "High": float(p.high) if p.high else None,
            "Low": float(p.low) if p.low else None,
            "Close": float(p.close) if p.close else None,
            "Adj Close": float(p.adj_close) if p.adj_close else None,
            "Volume": int(p.volume) if p.volume else None,
        })

    df = pd.DataFrame(data)
    df.set_index("date", inplace=True)
    # Convert date index to DatetimeIndex for compatibility with datetime comparisons
    df.index = pd.to_datetime(df.index)
    return df


async def save_prices(symbol: str, df: pd.DataFrame, auto_adjusted: bool = True) -> int:
    """Save price data to database.
    
    IMPORTANT: When using yfinance with auto_adjust=True (the default), the Close
    column already contains split-adjusted prices but there is no separate Adj Close
    column. In this case, we store Close in BOTH the close and adj_close columns
    to maintain consistency with the read path (get_close_column prefers Adj Close).
    
    Args:
        symbol: Stock ticker symbol
        df: DataFrame with OHLCV data
        auto_adjusted: If True, Close is already adjusted (use for adj_close too)
    
    Returns:
        Number of rows saved
    """
    if df.empty:
        return 0

    async with get_session() as session:
        count = 0
        skipped = 0
        for idx, row in df.iterrows():
            # Skip rows with NaN close values - they're not usable
            if pd.isna(row["Close"]):
                skipped += 1
                continue
                
            dt = idx.date() if hasattr(idx, "date") else idx
            close_val = float(row["Close"])
            
            # Determine adj_close value:
            # 1. If Adj Close column exists and has value, use it
            # 2. If auto_adjusted=True and no Adj Close, Close IS the adjusted price
            # 3. Otherwise, set to None
            raw_adj = row.get("Adj Close")
            if pd.notna(raw_adj):
                adj_val = float(raw_adj)
            elif auto_adjusted:
                # With auto_adjust=True, Close is already adjusted
                adj_val = close_val
            else:
                adj_val = None
            
            adj_decimal = Decimal(str(adj_val)) if adj_val is not None else None

            stmt = insert(PriceHistory).values(
                symbol=symbol.upper(),
                date=dt,
                open=Decimal(str(row.get("Open", 0))) if pd.notna(row.get("Open")) else None,
                high=Decimal(str(row.get("High", 0))) if pd.notna(row.get("High")) else None,
                low=Decimal(str(row.get("Low", 0))) if pd.notna(row.get("Low")) else None,
                close=Decimal(str(close_val)),
                adj_close=adj_decimal,
                volume=int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else None,
            ).on_conflict_do_update(
                index_elements=["symbol", "date"],
                set_={
                    "open": Decimal(str(row.get("Open", 0))) if pd.notna(row.get("Open")) else None,
                    "high": Decimal(str(row.get("High", 0))) if pd.notna(row.get("High")) else None,
                    "low": Decimal(str(row.get("Low", 0))) if pd.notna(row.get("Low")) else None,
                    "close": Decimal(str(close_val)),
                    "adj_close": adj_decimal,
                    "volume": int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else None,
                }
            )

            await session.execute(stmt)
            count += 1

        await session.commit()
        if skipped > 0:
            logger.warning(f"Skipped {skipped} rows with NaN close for {symbol}")
        logger.debug(f"Saved {count} price records for {symbol}")
        return count


async def fix_null_adj_close(symbols: list[str] | None = None) -> int:
    """Fix corrupted price data by copying close to adj_close where adj_close is NULL.
    
    This fixes the issue where yfinance auto_adjust=True returns adjusted OHLC
    but no Adj Close column, causing adj_close to be stored as NULL while older
    rows have adj_close set. The read path (get_close_column) prefers Adj Close
    if any row has it, causing a mismatch between old and new data.
    
    Args:
        symbols: List of symbols to fix, or None for all symbols
    
    Returns:
        Number of rows fixed
    """
    from sqlalchemy import update
    
    async with get_session() as session:
        if symbols:
            # Fix specific symbols
            normalized = [s.upper() for s in symbols]
            stmt = (
                update(PriceHistory)
                .where(
                    and_(
                        PriceHistory.symbol.in_(normalized),
                        PriceHistory.adj_close.is_(None),
                        PriceHistory.close.isnot(None),
                    )
                )
                .values(adj_close=PriceHistory.close)
            )
        else:
            # Fix all symbols
            stmt = (
                update(PriceHistory)
                .where(
                    and_(
                        PriceHistory.adj_close.is_(None),
                        PriceHistory.close.isnot(None),
                    )
                )
                .values(adj_close=PriceHistory.close)
            )
        
        result = await session.execute(stmt)
        await session.commit()
        
        fixed = result.rowcount
        logger.info(f"Fixed {fixed} rows with NULL adj_close")
        return fixed


async def get_distinct_symbols() -> Sequence[str]:
    """Get list of all symbols with price history data.
    
    Returns:
        List of unique symbol names
    """
    async with get_session() as session:
        result = await session.execute(
            select(PriceHistory.symbol)
            .distinct()
            .order_by(PriceHistory.symbol)
        )
        return [row[0] for row in result.fetchall()]