"""Price domain models.

Type-safe representations of price history data.
"""

from __future__ import annotations

from datetime import date as DateType
from datetime import datetime, timezone
from typing import Iterator, Literal

import pandas as pd
from pydantic import BaseModel, Field, computed_field


PriceInterval = Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"]


class PriceBar(BaseModel):
    """Single OHLCV price bar.
    
    Represents one candlestick of price data.
    """
    
    date: DateType = Field(..., description="Trading date")
    open: float = Field(..., ge=0, description="Opening price")
    high: float = Field(..., ge=0, description="High price")
    low: float = Field(..., ge=0, description="Low price")
    close: float = Field(..., ge=0, description="Closing price")
    volume: int = Field(default=0, ge=0, description="Trading volume")
    
    model_config = {
        "from_attributes": True,
    }
    
    @computed_field
    @property
    def change(self) -> float:
        """Daily price change (close - open)."""
        return self.close - self.open
    
    @computed_field
    @property
    def change_pct(self) -> float:
        """Daily price change percentage."""
        if self.open > 0:
            return (self.close - self.open) / self.open
        return 0.0
    
    @computed_field
    @property
    def range(self) -> float:
        """High-low range."""
        return self.high - self.low


class PriceHistory(BaseModel):
    """Price history for a symbol.
    
    Contains a series of OHLCV price bars with metadata.
    """
    
    symbol: str = Field(..., description="Ticker symbol")
    bars: list[PriceBar] = Field(default_factory=list, description="Price bars (chronological)")
    interval: PriceInterval = Field(default="1d", description="Bar interval")
    currency: str = Field(default="USD", description="Price currency")
    version_hash: str | None = Field(None, description="Data version hash for change detection")
    fetched_at: datetime | None = Field(None, description="When data was fetched")
    
    model_config = {
        "from_attributes": True,
    }
    
    def __len__(self) -> int:
        """Number of price bars."""
        return len(self.bars)
    
    def __iter__(self) -> Iterator[PriceBar]:
        """Iterate over price bars."""
        return iter(self.bars)
    
    def __getitem__(self, index: int) -> PriceBar:
        """Get price bar by index."""
        return self.bars[index]
    
    @computed_field
    @property
    def start_date(self) -> DateType | None:
        """First date in history."""
        return self.bars[0].date if self.bars else None
    
    @computed_field
    @property
    def end_date(self) -> DateType | None:
        """Last date in history."""
        return self.bars[-1].date if self.bars else None
    
    @computed_field
    @property
    def latest_close(self) -> float | None:
        """Most recent closing price."""
        return self.bars[-1].close if self.bars else None
    
    @computed_field
    @property
    def high_52w(self) -> float | None:
        """52-week high (or all-time if less than 52 weeks)."""
        if not self.bars:
            return None
        # Take up to 252 trading days (~52 weeks)
        recent = self.bars[-252:] if len(self.bars) > 252 else self.bars
        return max(bar.high for bar in recent)
    
    @computed_field
    @property
    def low_52w(self) -> float | None:
        """52-week low (or all-time if less than 52 weeks)."""
        if not self.bars:
            return None
        recent = self.bars[-252:] if len(self.bars) > 252 else self.bars
        return min(bar.low for bar in recent)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.
        
        Returns:
            DataFrame with DatetimeIndex and OHLCV columns.
        """
        if not self.bars:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        
        data = [
            {
                "Date": bar.date,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume,
            }
            for bar in self.bars
        ]
        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        return df
    
    @classmethod
    def from_dataframe(
        cls,
        symbol: str,
        df: pd.DataFrame,
        interval: PriceInterval = "1d",
        version_hash: str | None = None,
    ) -> "PriceHistory":
        """Create PriceHistory from pandas DataFrame.
        
        Args:
            symbol: Ticker symbol
            df: DataFrame with OHLCV columns and date index
            interval: Price bar interval
            version_hash: Data version hash
            
        Returns:
            PriceHistory instance
        """
        if df is None or df.empty:
            return cls(symbol=symbol, bars=[], interval=interval, version_hash=version_hash)
        
        bars = []
        for idx, row in df.iterrows():
            # Handle both DatetimeIndex and date column
            if hasattr(idx, "date"):
                bar_date = idx.date() if callable(idx.date) else idx.date
            else:
                bar_date = idx
            
            # Normalize column names (yfinance uses capitalized)
            open_col = "Open" if "Open" in df.columns else "open"
            high_col = "High" if "High" in df.columns else "high"
            low_col = "Low" if "Low" in df.columns else "low"
            close_col = "Close" if "Close" in df.columns else "close"
            volume_col = "Volume" if "Volume" in df.columns else "volume"
            
            try:
                bar = PriceBar(
                    date=bar_date,
                    open=float(row[open_col]),
                    high=float(row[high_col]),
                    low=float(row[low_col]),
                    close=float(row[close_col]),
                    volume=int(row.get(volume_col, 0) or 0),
                )
                bars.append(bar)
            except (ValueError, TypeError, KeyError):
                # Skip invalid rows
                continue
        
        return cls(
            symbol=symbol,
            bars=bars,
            interval=interval,
            version_hash=version_hash,
            fetched_at=datetime.now(timezone.utc),
        )
    
    def slice(self, days: int) -> "PriceHistory":
        """Get most recent N days.
        
        Args:
            days: Number of recent bars to include
            
        Returns:
            New PriceHistory with sliced data
        """
        sliced_bars = self.bars[-days:] if len(self.bars) > days else self.bars
        return PriceHistory(
            symbol=self.symbol,
            bars=sliced_bars,
            interval=self.interval,
            currency=self.currency,
            version_hash=self.version_hash,
            fetched_at=self.fetched_at,
        )
