"""
Pydantic models for OpenAI structured outputs.

These models define the exact structure of AI responses for each task type.
The models generate JSON schemas used for OpenAI's Structured Outputs feature,
guaranteeing the AI returns data matching these schemas exactly.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# =============================================================================
# BIO Task Output
# =============================================================================

class BioOutput(BaseModel):
    """Dating-app style stock bio output."""
    bio: str = Field(
        description="Tinder-style dating bio for the stock (150-200 chars, 3 sentences max, 1-2 emojis)"
    )


# =============================================================================
# RATING Task Output
# =============================================================================

class RatingValue(str, Enum):
    """Valid rating values for dip-buy opportunity."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class RatingOutput(BaseModel):
    """Dip-buy opportunity rating output."""
    rating: RatingValue = Field(
        description="The dip-buy opportunity rating"
    )
    reasoning: str = Field(
        description="Brief explanation under 400 chars citing at least 2 concrete context facts"
    )
    confidence: int = Field(
        ge=1, le=10,
        description="Confidence level from 1-10"
    )


# =============================================================================
# SUMMARY Task Output
# =============================================================================

class SummaryOutput(BaseModel):
    """Company description summary output."""
    summary: str = Field(
        description="Plain-English company summary (300-400 chars)"
    )


# =============================================================================
# AGENT Task Output
# =============================================================================

class AgentOutput(BaseModel):
    """Persona investor analysis output."""
    rating: RatingValue = Field(
        description="Investment signal rating"
    )
    reasoning: str = Field(
        description="2-3 sentence explanation in the investor persona's voice"
    )
    confidence: int = Field(
        ge=1, le=10,
        description="Confidence level from 1-10"
    )
    key_factors: list[str] = Field(
        description="3-5 key factors that influenced the rating"
    )


# =============================================================================
# PORTFOLIO Task Output
# =============================================================================

class InsightType(str, Enum):
    """Insight category types."""
    POSITIVE = "positive"
    WARNING = "warning"
    NEUTRAL = "neutral"


class Insight(BaseModel):
    """Portfolio insight item."""
    type: InsightType = Field(description="Insight category")
    text: str = Field(description="Insight text (max 200 chars)")


class ActionItem(BaseModel):
    """Portfolio action recommendation."""
    priority: int = Field(
        ge=1, le=3,
        description="Priority level: 1=high, 2=medium, 3=low"
    )
    action: str = Field(description="Action recommendation (max 200 chars)")


class RiskSeverity(str, Enum):
    """Risk severity levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskAlert(BaseModel):
    """Portfolio risk alert."""
    severity: RiskSeverity = Field(description="Risk severity level")
    alert: str = Field(description="Risk alert text (max 200 chars)")


class HealthRating(str, Enum):
    """Portfolio health ratings."""
    STRONG = "strong"
    GOOD = "good"
    FAIR = "fair"
    WEAK = "weak"


class PortfolioOutput(BaseModel):
    """Portfolio advisor analysis output."""
    health: HealthRating = Field(
        description="Overall portfolio health rating"
    )
    headline: str = Field(
        description="One-sentence summary with key metric (max 120 chars)"
    )
    insights: list[Insight] = Field(
        description="2-4 key observations about the portfolio"
    )
    actions: list[ActionItem] = Field(
        description="0-3 specific actionable recommendations"
    )
    risks: list[RiskAlert] = Field(
        description="0-3 risk alerts (empty array if none)"
    )


# =============================================================================
# Schema Generation
# =============================================================================

def get_json_schema(output_model: type[BaseModel], name: str) -> dict:
    """
    Generate OpenAI Structured Outputs compatible JSON schema from Pydantic model.
    
    OpenAI requires:
    - type: "json_schema"
    - name: schema name
    - strict: True
    - schema: the actual JSON schema with additionalProperties: false
    """
    schema = output_model.model_json_schema()
    
    # OpenAI requires additionalProperties: false at all levels
    _add_additional_properties_false(schema)
    
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": schema,
    }


def _add_additional_properties_false(schema: dict) -> None:
    """Recursively add additionalProperties: false to all object types."""
    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        
    # Handle properties
    if "properties" in schema:
        for prop in schema["properties"].values():
            _add_additional_properties_false(prop)
    
    # Handle array items
    if "items" in schema:
        _add_additional_properties_false(schema["items"])
    
    # Handle $defs (Pydantic uses this for nested models)
    if "$defs" in schema:
        for definition in schema["$defs"].values():
            _add_additional_properties_false(definition)
    
    # Handle anyOf, oneOf, allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            for item in schema[key]:
                _add_additional_properties_false(item)


# Pre-built schemas for each task type
from app.services.openai.config import TaskType

TASK_SCHEMAS: dict[TaskType, dict] = {
    TaskType.BIO: get_json_schema(BioOutput, "stock_bio"),
    TaskType.RATING: get_json_schema(RatingOutput, "stock_rating"),
    TaskType.SUMMARY: get_json_schema(SummaryOutput, "company_summary"),
    TaskType.AGENT: get_json_schema(AgentOutput, "agent_analysis"),
    TaskType.PORTFOLIO: get_json_schema(PortfolioOutput, "portfolio_analysis"),
}


# Type alias for all possible output types
TaskOutput = BioOutput | RatingOutput | SummaryOutput | AgentOutput | PortfolioOutput


def get_output_model(task: TaskType) -> type[BaseModel]:
    """Get the Pydantic output model for a task type."""
    models: dict[TaskType, type[BaseModel]] = {
        TaskType.BIO: BioOutput,
        TaskType.RATING: RatingOutput,
        TaskType.SUMMARY: SummaryOutput,
        TaskType.AGENT: AgentOutput,
        TaskType.PORTFOLIO: PortfolioOutput,
    }
    return models[task]
