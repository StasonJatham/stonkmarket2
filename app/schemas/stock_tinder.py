"""Stock tinder schemas for dip voting and AI analysis."""

from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, Field


class DipVoteRequest(BaseModel):
    """Request to vote on a dip."""

    vote_type: Literal["buy", "sell"] = Field(..., description="Vote type: buy or sell")

    model_config = {"json_schema_extra": {"example": {"vote_type": "buy"}}}


class DipVoteResponse(BaseModel):
    """Response after voting on a dip."""

    symbol: str
    vote_type: str
    message: str


class VoteCounts(BaseModel):
    """Vote counts for a dip with weighted totals."""

    buy: int = 0
    sell: int = 0
    buy_weighted: int = Field(
        default=0, description="Weighted buy votes (API key holders get 10x)"
    )
    sell_weighted: int = Field(default=0, description="Weighted sell votes")
    net_score: int = Field(
        default=0, description="Net score (buy_weighted - sell_weighted)"
    )


class DipCard(BaseModel):
    """A stock dip card for the tinder interface."""

    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    website: Optional[str] = Field(None, description="Company website URL for logo")
    ipo_year: Optional[int] = Field(None, description="Year of IPO/first trade")

    # Price data
    current_price: float
    ref_high: float
    dip_pct: float = Field(..., description="Dip percentage from high")
    days_below: int = Field(..., description="Days below dip threshold")
    min_dip_pct: Optional[float] = Field(None, description="Configured dip threshold")

    # AI content
    summary_ai: Optional[str] = Field(None, description="AI-generated company summary from finance description")
    tinder_bio: Optional[str] = Field(None, description="AI-generated tinder-style bio")
    ai_rating: Optional[str] = Field(
        None, description="AI rating: strong_buy, buy, hold, sell, strong_sell"
    )
    ai_reasoning: Optional[str] = Field(None, description="AI explanation for rating")
    ai_confidence: Optional[int] = Field(
        None, ge=1, le=10, description="AI confidence score"
    )

    # Voting
    vote_counts: VoteCounts = Field(default_factory=VoteCounts)

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "current_price": 175.50,
                "ref_high": 198.23,
                "dip_pct": 11.47,
                "days_below": 5,
                "min_dip_pct": 10.0,
                "tinder_bio": "Tech giant going through a rough patch. Looking for patient investors who appreciate my services ecosystem. Swipe right if you believe in the long game 📱💪",
                "ai_rating": "buy",
                "ai_reasoning": "Strong fundamentals despite short-term weakness. Services revenue growing. Good entry point.",
                "ai_confidence": 7,
                "vote_counts": {"buy": 42, "sell": 8, "skip": 15},
            }
        }
    }


class DipCardList(BaseModel):
    """List of dip cards."""

    cards: list[DipCard]
    total: int


class DipStats(BaseModel):
    """Voting statistics for a dip."""

    symbol: str
    vote_counts: VoteCounts
    total_votes: int
    weighted_total: int = Field(default=0, description="Total weighted votes")
    buy_pct: float
    sell_pct: float
    sentiment: str = Field(
        default="neutral",
        description="Overall sentiment: very_bullish, bullish, neutral, bearish, very_bearish",
    )
    sentiment: str = Field(
        ...,
        description="Overall sentiment: very_bullish, bullish, neutral, bearish, very_bearish",
    )
