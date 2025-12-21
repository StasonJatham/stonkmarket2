"""Stock suggestion schemas."""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SuggestionStatus(str, Enum):
    """Status of a stock suggestion."""
    
    PENDING = "pending"        # Awaiting admin review
    APPROVED = "approved"      # Admin approved, added to tracked symbols
    REJECTED = "rejected"      # Admin rejected (with reason)
    REMOVED = "removed"        # Was tracked but removed by admin
    FETCHING = "fetching"      # Background job is fetching data
    FETCH_FAILED = "fetch_failed"  # Failed to fetch from Yahoo Finance


# Yahoo Finance symbol pattern:
# - US stocks: 1-5 uppercase letters (AAPL, MSFT)
# - International: letters/numbers + dot + exchange (7203.T, BHP.AX)
# - Crypto: XXX-YYY (BTC-USD)
# - Indices: ^XXX (^GSPC)
YAHOO_SYMBOL_PATTERN = re.compile(
    r"^(?:"
    r"\^[A-Z0-9]{1,20}|"              # Index: ^GSPC, ^DJI
    r"[A-Z]{1,5}|"                     # US stock: AAPL, MSFT
    r"[A-Z0-9]{1,10}\.[A-Z]{1,4}|"    # International: 7203.T, BHP.AX
    r"[A-Z]{2,10}-[A-Z]{2,10}"        # Crypto: BTC-USD
    r")$"
)


class SuggestionCreate(BaseModel):
    """Request to suggest a new stock."""
    
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Yahoo Finance symbol (e.g., AAPL, BHP.AX, BTC-USD)"
    )
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize Yahoo Finance symbol."""
        symbol = v.strip().upper()
        if not YAHOO_SYMBOL_PATTERN.match(symbol):
            raise ValueError(
                "Invalid Yahoo Finance symbol format. Examples: "
                "AAPL (US), 7203.T (Japan), BHP.AX (Australia), BTC-USD (crypto), ^GSPC (index)"
            )
        return symbol


class SuggestionVote(BaseModel):
    """Request to vote for a suggestion."""
    
    symbol: str = Field(..., min_length=1, max_length=20)
    
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        return v.strip().upper()


class SuggestionResponse(BaseModel):
    """Response for a single suggestion."""
    
    id: int
    symbol: str
    status: SuggestionStatus
    vote_count: int = 0
    name: Optional[str] = None
    sector: Optional[str] = None
    summary: Optional[str] = None
    last_price: Optional[float] = None
    price_change_90d: Optional[float] = None  # Percentage change over 90 days
    created_at: datetime
    updated_at: Optional[datetime] = None
    fetched_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    
    model_config = {"from_attributes": True}


class SuggestionListResponse(BaseModel):
    """Response for listing suggestions."""
    
    items: list[SuggestionResponse]
    total: int
    page: int
    page_size: int


class SuggestionAdminAction(BaseModel):
    """Admin action on a suggestion (for approve/reject)."""
    
    reason: Optional[str] = Field(None, max_length=500, description="Reason for rejection (required for reject)")


class SuggestionApprove(BaseModel):
    """Request to approve a suggestion."""
    pass  # No additional data needed, symbol is already fetched


class SuggestionReject(BaseModel):
    """Request to reject a suggestion."""
    
    reason: str = Field(..., min_length=1, max_length=500, description="Reason for rejection")


class TopSuggestion(BaseModel):
    """A suggestion ranked by votes."""
    
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    vote_count: int
    status: SuggestionStatus
    last_price: Optional[float] = None
    summary: Optional[str] = None
