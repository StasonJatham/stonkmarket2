"""Ticker domain models.

Type-safe representations of stock/ETF ticker data from yfinance.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field


QuoteType = Literal["EQUITY", "ETF", "INDEX", "MUTUALFUND", "TRUST", "CRYPTOCURRENCY", "CURRENCY"]


class TickerInfo(BaseModel):
    """Complete ticker information from yfinance.
    
    This is the canonical representation of stock/ETF metadata.
    All fields are optional except symbol to handle partial data gracefully.
    """
    
    # Identity (required)
    symbol: str = Field(..., description="Ticker symbol (uppercase)")
    
    # Identity (optional)
    name: str | None = Field(None, description="Company/ETF name")
    quote_type: QuoteType = Field(default="EQUITY", description="Asset type")
    exchange: str | None = Field(None, description="Exchange code")
    currency: str | None = Field(None, description="Trading currency")
    website: str | None = Field(None, description="Company website")
    country: str | None = Field(None, description="Country of incorporation")
    sector: str | None = Field(None, description="GICS sector (stocks only)")
    industry: str | None = Field(None, description="GICS industry (stocks only)")
    summary: str | None = Field(None, description="Business description")
    ipo_year: int | None = Field(None, description="IPO year")
    
    @computed_field
    @property
    def is_etf(self) -> bool:
        """Check if this is an ETF, index, or fund (not a stock)."""
        return self.quote_type in ("ETF", "INDEX", "MUTUALFUND", "TRUST")
    
    # Price
    current_price: float | None = Field(None, ge=0, description="Current/last price")
    previous_close: float | None = Field(None, ge=0, description="Previous close")
    fifty_two_week_high: float | None = Field(None, ge=0, description="52-week high")
    fifty_two_week_low: float | None = Field(None, ge=0, description="52-week low")
    fifty_day_average: float | None = Field(None, ge=0, description="50-day moving average")
    two_hundred_day_average: float | None = Field(None, ge=0, description="200-day moving average")
    
    # Market
    market_cap: int | None = Field(None, ge=0, description="Market capitalization")
    avg_volume: int | None = Field(None, ge=0, description="Average daily volume")
    volume: int | None = Field(None, ge=0, description="Current volume")
    
    # Valuation (stocks only)
    pe_ratio: float | None = Field(None, description="Trailing P/E ratio")
    forward_pe: float | None = Field(None, description="Forward P/E ratio")
    peg_ratio: float | None = Field(None, description="PEG ratio")
    price_to_book: float | None = Field(None, description="Price to book value")
    price_to_sales: float | None = Field(None, description="Price to sales (TTM)")
    enterprise_value: int | None = Field(None, description="Enterprise value")
    ev_to_ebitda: float | None = Field(None, description="EV/EBITDA ratio")
    ev_to_revenue: float | None = Field(None, description="EV/Revenue ratio")
    
    # Profitability (stocks only)
    profit_margin: float | None = Field(None, description="Net profit margin")
    operating_margin: float | None = Field(None, description="Operating margin")
    gross_margin: float | None = Field(None, description="Gross margin")
    ebitda_margin: float | None = Field(None, description="EBITDA margin")
    return_on_equity: float | None = Field(None, description="Return on equity (ROE)")
    return_on_assets: float | None = Field(None, description="Return on assets (ROA)")
    
    # Financial Health (stocks only)
    debt_to_equity: float | None = Field(None, description="Debt to equity ratio")
    stockholders_equity: float | None = Field(None, description="Total stockholders equity")
    current_ratio: float | None = Field(None, description="Current ratio")
    quick_ratio: float | None = Field(None, description="Quick ratio")
    total_cash: int | None = Field(None, description="Total cash")
    total_debt: int | None = Field(None, description="Total debt")
    free_cash_flow: int | None = Field(None, description="Free cash flow")
    operating_cash_flow: int | None = Field(None, description="Operating cash flow")
    
    # Per Share (stocks only)
    book_value: float | None = Field(None, description="Book value per share")
    eps_trailing: float | None = Field(None, description="Trailing EPS")
    eps_forward: float | None = Field(None, description="Forward EPS")
    revenue_per_share: float | None = Field(None, description="Revenue per share")
    dividend_yield: float | None = Field(None, ge=0, le=1, description="Dividend yield (0-1)")
    payout_ratio: float | None = Field(None, description="Dividend payout ratio")
    shares_outstanding: int | None = Field(None, ge=0, description="Shares outstanding")
    float_shares: int | None = Field(None, ge=0, description="Float shares")
    
    # Growth (stocks only)
    revenue_growth: float | None = Field(None, description="Revenue growth (YoY)")
    earnings_growth: float | None = Field(None, description="Earnings growth (YoY)")
    earnings_quarterly_growth: float | None = Field(None, description="Quarterly earnings growth")
    
    # Analyst
    recommendation: str | None = Field(None, description="Analyst recommendation key")
    recommendation_mean: float | None = Field(None, ge=1, le=5, description="Mean recommendation (1=buy, 5=sell)")
    target_mean_price: float | None = Field(None, ge=0, description="Mean analyst target price")
    target_high_price: float | None = Field(None, ge=0, description="High analyst target")
    target_low_price: float | None = Field(None, ge=0, description="Low analyst target")
    num_analyst_opinions: int | None = Field(None, ge=0, description="Number of analysts")
    
    # Earnings
    earnings_date: str | None = Field(None, description="Next earnings date (ISO)")
    most_recent_quarter: str | None = Field(None, description="Most recent quarter end (ISO)")
    
    # Revenue & Earnings
    revenue: int | None = Field(None, description="Total revenue (TTM)")
    ebitda: int | None = Field(None, description="EBITDA")
    net_income: int | None = Field(None, description="Net income")
    
    # Risk
    beta: float | None = Field(None, description="Beta (market correlation)")
    short_ratio: float | None = Field(None, ge=0, description="Days to cover short interest")
    short_percent_of_float: float | None = Field(None, ge=0, le=1, description="Short interest % of float")
    held_percent_insiders: float | None = Field(None, ge=0, le=1, description="Insider ownership %")
    held_percent_institutions: float | None = Field(None, ge=0, le=1, description="Institutional ownership %")
    
    # Metadata
    fetched_at: datetime | None = Field(None, description="When data was fetched")
    
    model_config = {
        "from_attributes": True,
        "extra": "ignore",  # Ignore unknown fields from yfinance
    }
    
    @computed_field
    @property
    def target_upside(self) -> float | None:
        """Calculate upside to mean analyst target price."""
        if self.target_mean_price and self.current_price and self.current_price > 0:
            return (self.target_mean_price - self.current_price) / self.current_price
        return None
    
    @computed_field
    @property
    def fcf_to_market_cap(self) -> float | None:
        """Calculate free cash flow yield."""
        if self.free_cash_flow and self.market_cap and self.market_cap > 0:
            return self.free_cash_flow / self.market_cap
        return None


class TickerSearchResult(BaseModel):
    """Search result for a ticker symbol."""
    
    symbol: str = Field(..., description="Ticker symbol")
    name: str | None = Field(None, description="Company/ETF name")
    exchange: str | None = Field(None, description="Exchange code")
    quote_type: str | None = Field(None, description="Asset type")
    score: float = Field(default=0.0, ge=0, le=1, description="Search relevance score")
    
    model_config = {
        "from_attributes": True,
    }
