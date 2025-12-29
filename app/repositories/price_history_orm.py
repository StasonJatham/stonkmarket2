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


# Maximum allowed deviation between Close and Adj Close (10%)
# yfinance occasionally returns anomalous adj_close values that corrupt data
MAX_ADJ_CLOSE_DEVIATION = 0.10


def _validate_adj_close(close: float, adj_close: float | None, symbol: str, dt: date) -> float | None:
    """Validate adjusted close price against close price.
    
    yfinance can return anomalous adjusted close values (e.g., 315 instead of 618).
    This validation rejects adj_close values that differ by more than MAX_ADJ_CLOSE_DEVIATION.
    
    Args:
        close: Regular close price
        adj_close: Adjusted close price from yfinance
        symbol: Stock ticker (for logging)
        dt: Date of the price (for logging)
    
    Returns:
        adj_close if valid, close if invalid (fallback), None if both missing
    """
    if adj_close is None or pd.isna(adj_close):
        return None
    
    if close <= 0:
        return None
    
    deviation = abs(adj_close - close) / close
    
    if deviation > MAX_ADJ_CLOSE_DEVIATION:
        logger.warning(
            f"Rejected anomalous adj_close for {symbol} on {dt}: "
            f"adj_close={adj_close:.2f} vs close={close:.2f} (deviation={deviation:.1%}). "
            f"Using close price instead."
        )
        return close
    
    return adj_close


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
        rejected_adj = 0
        for idx, row in df.iterrows():
            # Skip rows with NaN close values - they're not usable
            if pd.isna(row["Close"]):
                skipped += 1
                continue
                
            dt = idx.date() if hasattr(idx, "date") else idx
            close_val = float(row["Close"])
            raw_adj = row.get("Adj Close")
            raw_adj_float = float(raw_adj) if pd.notna(raw_adj) else None
            
            # Validate adj_close - reject anomalous values from yfinance
            validated_adj = _validate_adj_close(close_val, raw_adj_float, symbol, dt)
            if validated_adj != raw_adj_float and raw_adj_float is not None:
                rejected_adj += 1
            
            adj_decimal = Decimal(str(validated_adj)) if validated_adj is not None else None

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
        if rejected_adj > 0:
            logger.warning(f"Rejected {rejected_adj} anomalous adj_close values for {symbol}")
        logger.debug(f"Saved {count} price records for {symbol}")
        return count
