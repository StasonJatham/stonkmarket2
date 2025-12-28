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


async def save_prices(symbol: str, df: pd.DataFrame) -> int:
    """Save price data to database.
    
    Args:
        symbol: Stock ticker symbol
        df: DataFrame with OHLCV data
    
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

            stmt = insert(PriceHistory).values(
                symbol=symbol.upper(),
                date=dt,
                open=Decimal(str(row.get("Open", 0))) if pd.notna(row.get("Open")) else None,
                high=Decimal(str(row.get("High", 0))) if pd.notna(row.get("High")) else None,
                low=Decimal(str(row.get("Low", 0))) if pd.notna(row.get("Low")) else None,
                close=Decimal(str(row["Close"])),
                adj_close=Decimal(str(row.get("Adj Close", row["Close"]))) if pd.notna(row.get("Adj Close", row["Close"])) else None,
                volume=int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else None,
            ).on_conflict_do_update(
                index_elements=["symbol", "date"],
                set_={
                    "open": Decimal(str(row.get("Open", 0))) if pd.notna(row.get("Open")) else None,
                    "high": Decimal(str(row.get("High", 0))) if pd.notna(row.get("High")) else None,
                    "low": Decimal(str(row.get("Low", 0))) if pd.notna(row.get("Low")) else None,
                    "close": Decimal(str(row["Close"])),
                    "adj_close": Decimal(str(row.get("Adj Close", row["Close"]))) if pd.notna(row.get("Adj Close", row["Close"])) else None,
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
