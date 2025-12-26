"""Dip and stock data schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DipStateResponse(BaseModel):
    """Dip state response schema."""

    symbol: str = Field(..., description="Stock ticker symbol")
    ref_high: float = Field(..., description="Reference high price")
    days_below: int = Field(..., description="Days below threshold")
    last_price: float = Field(..., description="Last known price")
    dip_depth: float = Field(..., description="Current dip depth (negative = down)")
    updated_at: datetime | None = Field(None, description="Last update timestamp")

    model_config = {"from_attributes": True}


class RankingEntry(BaseModel):
    """Ranking entry schema."""

    symbol: str = Field(..., description="Stock ticker symbol")
    name: str | None = Field(None, description="Company name")
    depth: float = Field(..., description="Dip depth as positive fraction (0.15 = 15% dip)")
    last_price: float = Field(..., description="Last known price")
    previous_close: float | None = Field(None, description="Previous closing price")
    change_percent: float | None = Field(None, description="Daily change percentage")
    days_since_dip: int | None = Field(None, description="Days since dip started")
    high_52w: float | None = Field(None, description="52-week high")
    low_52w: float | None = Field(None, description="52-week low")
    market_cap: float | None = Field(None, description="Market capitalization")
    sector: str | None = Field(None, description="Sector")
    pe_ratio: float | None = Field(None, description="P/E ratio")
    volume: int | None = Field(None, description="Trading volume")
    symbol_type: str | None = Field("stock", description="Type: 'stock' or 'index'")
    updated_at: datetime | None = Field(None, description="Last update timestamp")

    model_config = {"from_attributes": True}


class ChartPoint(BaseModel):
    """Chart data point schema."""

    date: str = Field(..., description="Date (YYYY-MM-DD)")
    close: float = Field(..., description="Closing price")
    threshold: float | None = Field(None, description="Dip threshold price")
    ref_high: float | None = Field(None, description="Reference high price")
    drawdown: float | None = Field(None, description="Drawdown from high")
    since_dip: float | None = Field(None, description="Change since dip started")
    ref_high_date: str | None = Field(None, description="Reference high date")
    dip_start_date: str | None = Field(None, description="Dip start date")

    model_config = {"from_attributes": True}


class StockInfo(BaseModel):
    """Detailed stock information schema."""

    symbol: str = Field(..., description="Stock ticker symbol")
    name: str | None = Field(None, description="Company name")
    sector: str | None = Field(None, description="Sector")
    industry: str | None = Field(None, description="Industry")
    market_cap: float | None = Field(None, description="Market capitalization")
    pe_ratio: float | None = Field(None, description="Trailing P/E ratio")
    forward_pe: float | None = Field(None, description="Forward P/E ratio")
    peg_ratio: float | None = Field(None, description="PEG ratio (P/E to Growth)")
    dividend_yield: float | None = Field(None, description="Dividend yield")
    beta: float | None = Field(None, description="Beta")
    avg_volume: int | None = Field(None, description="Average volume")
    summary: str | None = Field(None, description="Business summary")
    summary_ai: str | None = Field(None, description="AI-generated short summary (~300 chars)")
    website: str | None = Field(None, description="Company website")
    recommendation: str | None = Field(None, description="Analyst recommendation")
    # Extended fundamentals
    profit_margin: float | None = Field(None, description="Profit margin (decimal)")
    gross_margin: float | None = Field(None, description="Gross margin (decimal)")
    return_on_equity: float | None = Field(None, description="Return on equity (decimal)")
    debt_to_equity: float | None = Field(None, description="Debt to equity ratio")
    current_ratio: float | None = Field(None, description="Current ratio")
    revenue_growth: float | None = Field(None, description="Revenue growth (decimal)")
    free_cash_flow: int | None = Field(None, description="Free cash flow")
    target_mean_price: float | None = Field(None, description="Analyst target mean price")
    num_analyst_opinions: int | None = Field(None, description="Number of analyst opinions")

    model_config = {"from_attributes": True}
