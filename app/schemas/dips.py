"""Dip and stock data schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DipStateResponse(BaseModel):
    """Dip state response schema."""

    symbol: str = Field(..., description="Stock ticker symbol")
    ref_high: float = Field(..., description="Reference high price")
    days_below: int = Field(..., description="Days below threshold")
    last_price: float = Field(..., description="Last known price")
    dip_depth: float = Field(..., description="Current dip depth (negative = down)")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    model_config = {"from_attributes": True}


class RankingEntry(BaseModel):
    """Ranking entry schema."""

    symbol: str = Field(..., description="Stock ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    depth: float = Field(..., description="Dip depth as positive fraction (0.15 = 15% dip)")
    last_price: float = Field(..., description="Last known price")
    previous_close: Optional[float] = Field(None, description="Previous closing price")
    change_percent: Optional[float] = Field(None, description="Daily change percentage")
    days_since_dip: Optional[int] = Field(None, description="Days since dip started")
    high_52w: Optional[float] = Field(None, description="52-week high")
    low_52w: Optional[float] = Field(None, description="52-week low")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    sector: Optional[str] = Field(None, description="Sector")
    pe_ratio: Optional[float] = Field(None, description="P/E ratio")
    volume: Optional[int] = Field(None, description="Trading volume")
    symbol_type: Optional[str] = Field("stock", description="Type: 'stock' or 'index'")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    model_config = {"from_attributes": True}


class ChartPoint(BaseModel):
    """Chart data point schema."""

    date: str = Field(..., description="Date (YYYY-MM-DD)")
    close: float = Field(..., description="Closing price")
    threshold: Optional[float] = Field(None, description="Dip threshold price")
    ref_high: Optional[float] = Field(None, description="Reference high price")
    drawdown: Optional[float] = Field(None, description="Drawdown from high")
    since_dip: Optional[float] = Field(None, description="Change since dip started")
    ref_high_date: Optional[str] = Field(None, description="Reference high date")
    dip_start_date: Optional[str] = Field(None, description="Dip start date")

    model_config = {"from_attributes": True}


class StockInfo(BaseModel):
    """Detailed stock information schema."""

    symbol: str = Field(..., description="Stock ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    sector: Optional[str] = Field(None, description="Sector")
    industry: Optional[str] = Field(None, description="Industry")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    pe_ratio: Optional[float] = Field(None, description="Trailing P/E ratio")
    forward_pe: Optional[float] = Field(None, description="Forward P/E ratio")
    dividend_yield: Optional[float] = Field(None, description="Dividend yield")
    beta: Optional[float] = Field(None, description="Beta")
    avg_volume: Optional[int] = Field(None, description="Average volume")
    summary: Optional[str] = Field(None, description="Business summary")
    summary_ai: Optional[str] = Field(None, description="AI-generated short summary (~300 chars)")
    website: Optional[str] = Field(None, description="Company website")
    recommendation: Optional[str] = Field(None, description="Analyst recommendation")

    model_config = {"from_attributes": True}
