"""Fundamentals domain models.

Type-safe representations of company fundamentals and quality metrics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field


class FundamentalsData(BaseModel):
    """Raw fundamental data extracted from ticker info.
    
    This is the data used to compute quality scores.
    """
    
    symbol: str = Field(..., description="Ticker symbol")
    
    # Profitability
    profit_margin: float | None = Field(None, description="Net profit margin")
    operating_margin: float | None = Field(None, description="Operating margin")
    gross_margin: float | None = Field(None, description="Gross margin")
    return_on_equity: float | None = Field(None, description="Return on equity (ROE)")
    return_on_assets: float | None = Field(None, description="Return on assets (ROA)")
    
    # Balance Sheet
    debt_to_equity: float | None = Field(None, description="Debt to equity ratio")
    current_ratio: float | None = Field(None, description="Current ratio")
    quick_ratio: float | None = Field(None, description="Quick ratio")
    
    # Cash Flow
    free_cash_flow: int | None = Field(None, description="Free cash flow")
    operating_cash_flow: int | None = Field(None, description="Operating cash flow")
    
    # Market
    market_cap: int | None = Field(None, description="Market capitalization")
    avg_volume: int | None = Field(None, description="Average daily volume")
    
    # Growth
    revenue_growth: float | None = Field(None, description="Revenue growth (YoY)")
    earnings_growth: float | None = Field(None, description="Earnings growth (YoY)")
    
    # Valuation
    pe_ratio: float | None = Field(None, description="Trailing P/E ratio")
    forward_pe: float | None = Field(None, description="Forward P/E ratio")
    peg_ratio: float | None = Field(None, description="PEG ratio")
    ev_to_ebitda: float | None = Field(None, description="EV/EBITDA ratio")
    price_to_book: float | None = Field(None, description="Price to book value")
    
    # Analyst
    recommendation: str | None = Field(None, description="Analyst recommendation key")
    target_mean_price: float | None = Field(None, description="Mean analyst target")
    current_price: float | None = Field(None, description="Current price")
    num_analyst_opinions: int | None = Field(None, description="Number of analysts")
    
    # Risk
    short_percent_of_float: float | None = Field(None, description="Short interest %")
    held_percent_institutions: float | None = Field(None, description="Institutional ownership %")
    
    model_config = {
        "from_attributes": True,
        "extra": "ignore",
    }
    
    @computed_field
    @property
    def fcf_to_market_cap(self) -> float | None:
        """Free cash flow yield."""
        if self.free_cash_flow and self.market_cap and self.market_cap > 0:
            return self.free_cash_flow / self.market_cap
        return None
    
    @computed_field
    @property
    def target_upside(self) -> float | None:
        """Upside to analyst target."""
        if self.target_mean_price and self.current_price and self.current_price > 0:
            return (self.target_mean_price - self.current_price) / self.current_price
        return None
    
    @computed_field
    @property
    def fields_available(self) -> int:
        """Count of available fundamental fields."""
        fields = [
            self.profit_margin, self.operating_margin, self.debt_to_equity,
            self.current_ratio, self.free_cash_flow, self.revenue_growth,
            self.earnings_growth, self.market_cap, self.avg_volume,
            self.pe_ratio, self.forward_pe, self.peg_ratio,
            self.return_on_equity, self.return_on_assets,
            self.short_percent_of_float, self.held_percent_institutions,
        ]
        return sum(1 for f in fields if f is not None)
    
    @classmethod
    def from_ticker_info(cls, info: dict) -> "FundamentalsData":
        """Create from yfinance ticker info dict.
        
        Args:
            info: Raw ticker info dictionary from yfinance
            
        Returns:
            FundamentalsData instance
        """
        return cls(
            symbol=info.get("symbol", ""),
            profit_margin=info.get("profit_margin"),
            operating_margin=info.get("operating_margin"),
            gross_margin=info.get("gross_margin"),
            return_on_equity=info.get("return_on_equity"),
            return_on_assets=info.get("return_on_assets"),
            debt_to_equity=info.get("debt_to_equity"),
            current_ratio=info.get("current_ratio"),
            quick_ratio=info.get("quick_ratio"),
            free_cash_flow=info.get("free_cash_flow"),
            operating_cash_flow=info.get("operating_cash_flow"),
            market_cap=info.get("market_cap"),
            avg_volume=info.get("avg_volume"),
            revenue_growth=info.get("revenue_growth"),
            earnings_growth=info.get("earnings_growth"),
            pe_ratio=info.get("pe_ratio"),
            forward_pe=info.get("forward_pe"),
            peg_ratio=info.get("peg_ratio"),
            ev_to_ebitda=info.get("ev_to_ebitda"),
            price_to_book=info.get("price_to_book"),
            recommendation=info.get("recommendation"),
            target_mean_price=info.get("target_mean_price"),
            current_price=info.get("current_price"),
            num_analyst_opinions=info.get("num_analyst_opinions"),
            short_percent_of_float=info.get("short_percent_of_float"),
            held_percent_institutions=info.get("held_percent_institutions"),
        )


SubScoreType = Literal[
    "profitability",
    "balance_sheet",
    "cash_generation",
    "growth",
    "liquidity",
    "valuation",
    "analyst",
    "risk",
]


class QualityMetrics(BaseModel):
    """Quality score and contributing factors.
    
    Computed from FundamentalsData, provides a 0-100 quality score
    with detailed sub-scores for each factor category.
    """
    
    symbol: str = Field(..., alias="ticker", description="Ticker symbol")
    score: float = Field(..., ge=0, le=100, description="Overall quality score (0-100)")
    
    # Raw metrics (from fundamentals)
    profit_margin: float | None = None
    operating_margin: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    free_cash_flow: float | None = None
    fcf_to_market_cap: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    market_cap: float | None = None
    avg_volume: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    peg_ratio: float | None = None
    ev_to_ebitda: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    recommendation: str | None = None
    target_upside: float | None = None
    short_percent_of_float: float | None = None
    institutional_ownership: float | None = None
    
    # Sub-scores (0-100)
    profitability_score: float = Field(default=50.0, ge=0, le=100)
    balance_sheet_score: float = Field(default=50.0, ge=0, le=100)
    cash_generation_score: float = Field(default=50.0, ge=0, le=100)
    growth_score: float = Field(default=50.0, ge=0, le=100)
    liquidity_score: float = Field(default=50.0, ge=0, le=100)
    valuation_score: float = Field(default=50.0, ge=0, le=100)
    analyst_score: float = Field(default=50.0, ge=0, le=100)
    risk_score: float = Field(default=50.0, ge=0, le=100)
    
    # Data quality
    fields_available: int = Field(default=0, ge=0, description="Number of fields with data")
    fields_total: int = Field(default=16, ge=0, description="Total possible fields")
    
    model_config = {
        "from_attributes": True,
        "populate_by_name": True,  # Allow both 'symbol' and 'ticker'
    }
    
    @computed_field
    @property
    def data_completeness(self) -> float:
        """Percentage of available fundamental data (0-1)."""
        if self.fields_total == 0:
            return 0.0
        return self.fields_available / self.fields_total
    
    def get_sub_score(self, category: SubScoreType) -> float:
        """Get sub-score by category name."""
        return getattr(self, f"{category}_score", 50.0)
    
    def get_weakest_category(self) -> tuple[SubScoreType, float]:
        """Find the weakest sub-score category."""
        categories: list[SubScoreType] = [
            "profitability", "balance_sheet", "cash_generation", "growth",
            "liquidity", "valuation", "analyst", "risk"
        ]
        scores = [(cat, self.get_sub_score(cat)) for cat in categories]
        return min(scores, key=lambda x: x[1])
    
    def get_strongest_category(self) -> tuple[SubScoreType, float]:
        """Find the strongest sub-score category."""
        categories: list[SubScoreType] = [
            "profitability", "balance_sheet", "cash_generation", "growth",
            "liquidity", "valuation", "analyst", "risk"
        ]
        scores = [(cat, self.get_sub_score(cat)) for cat in categories]
        return max(scores, key=lambda x: x[1])
