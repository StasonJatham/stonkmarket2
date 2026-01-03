"""
Typed context models for each task type.

These dataclasses define the expected input data for each task type,
providing type safety and clear documentation of required/optional fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BioContext:
    """Context for BIO task - dating-style stock bio."""
    symbol: str
    name: str | None = None
    sector: str | None = None
    summary: str | None = None
    dip_pct: float | None = None  # Only used as mood indicator


@dataclass
class RatingContext:
    """Context for RATING task - dip-buy opportunity rating."""
    symbol: str
    current_price: float | None = None
    ref_high: float | None = None
    dip_pct: float | None = None
    days_in_dip: int | None = None
    dip_type: str | None = None  # MARKET_DIP, STOCK_SPECIFIC, MIXED
    quality_score: float | None = None
    stability_score: float | None = None
    # Valuation metrics
    pe_ratio: float | None = None
    forward_pe: float | None = None
    ev_to_ebitda: float | None = None
    market_cap: float | None = None
    # Additional context
    name: str | None = None
    sector: str | None = None


@dataclass
class SummaryContext:
    """Context for SUMMARY task - company description summary."""
    symbol: str
    name: str | None = None
    description: str = ""


@dataclass
class AgentContext:
    """Context for AGENT task - persona investor analysis."""
    prompt: str  # Full prompt with persona instructions and financial data


@dataclass
class PortfolioContext:
    """Context for PORTFOLIO task - portfolio advisor analysis."""
    portfolio_name: str | None = None
    total_value: float | None = None
    total_gain: float | None = None
    total_gain_pct: float | None = None
    performance: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] = field(default_factory=dict)
    holdings: list[dict[str, Any]] = field(default_factory=list)
    sector_allocation: dict[str, float] = field(default_factory=dict)
    custom_id: str | None = None  # For batch processing


# Union type for all contexts
TaskContext = BioContext | RatingContext | SummaryContext | AgentContext | PortfolioContext


def context_to_dict(context: TaskContext | dict[str, Any]) -> dict[str, Any]:
    """Convert a typed context to dictionary for prompt building."""
    if isinstance(context, dict):
        return context
    
    from dataclasses import asdict
    return asdict(context)
