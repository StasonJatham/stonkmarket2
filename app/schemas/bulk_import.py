"""Schemas for bulk portfolio import from image/screenshot."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExtractionConfidence(str, Enum):
    """Confidence level for extracted data."""
    
    HIGH = "high"       # Clear, unambiguous data
    MEDIUM = "medium"   # Likely correct but may need review
    LOW = "low"         # Uncertain, user should verify


class ExtractedPosition(BaseModel):
    """A single position extracted from an image."""
    
    # Identification
    symbol: str | None = Field(
        default=None,
        description="Stock ticker symbol (e.g., AAPL). May be None if only name was found.",
        max_length=20,
    )
    name: str | None = Field(
        default=None,
        description="Company name as shown in the image",
        max_length=200,
    )
    isin: str | None = Field(
        default=None,
        description="ISIN if visible in the image",
        max_length=12,
    )
    
    # Position details
    quantity: float | None = Field(
        default=None,
        ge=0,
        description="Number of shares/units",
    )
    avg_cost: float | None = Field(
        default=None,
        ge=0,
        description="Average purchase price per share",
    )
    current_price: float | None = Field(
        default=None,
        ge=0,
        description="Current market price if shown",
    )
    total_value: float | None = Field(
        default=None,
        ge=0,
        description="Total position value if shown",
    )
    
    # Additional metadata
    currency: str = Field(
        default="USD",
        max_length=3,
        description="Currency code (USD, EUR, etc.)",
    )
    exchange: str | None = Field(
        default=None,
        max_length=50,
        description="Exchange or market (NYSE, NASDAQ, XETRA, etc.)",
    )
    
    # Extraction metadata
    confidence: ExtractionConfidence = Field(
        default=ExtractionConfidence.MEDIUM,
        description="AI confidence in this extraction",
    )
    raw_text: str | None = Field(
        default=None,
        description="Original text from image for this row",
        max_length=500,
    )
    notes: str | None = Field(
        default=None,
        description="Any issues or notes about extraction",
        max_length=500,
    )
    
    # UI state
    skip: bool = Field(
        default=False,
        description="Whether user wants to skip importing this row",
    )
    
    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.upper().strip()
    
    @field_validator("currency", mode="before")
    @classmethod
    def normalize_currency(cls, v: str) -> str:
        return v.upper().strip()[:3]


class ImageExtractionResponse(BaseModel):
    """Response from image extraction endpoint."""
    
    success: bool = Field(..., description="Whether extraction succeeded")
    positions: list[ExtractedPosition] = Field(
        default_factory=list,
        description="Extracted positions from the image",
    )
    
    # Metadata
    image_quality: str | None = Field(
        default=None,
        description="Assessment of image quality (good, fair, poor)",
    )
    detected_broker: str | None = Field(
        default=None,
        description="Detected broker/app if identifiable",
    )
    currency_hint: str | None = Field(
        default=None,
        description="Detected base currency from the image",
    )
    
    # Processing info
    processing_time_ms: int | None = Field(
        default=None,
        description="Time taken to process the image",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if extraction failed",
    )
    
    # Warnings for the user
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings about the extraction",
    )


class BulkImportPosition(BaseModel):
    """A position to import (user-edited version of ExtractedPosition)."""
    
    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock ticker symbol (required)",
    )
    quantity: float = Field(
        ...,
        gt=0,
        description="Number of shares (required, must be positive)",
    )
    avg_cost: float | None = Field(
        default=None,
        ge=0,
        description="Average purchase price per share",
    )
    currency: str = Field(
        default="USD",
        max_length=3,
    )
    
    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: str) -> str:
        return v.upper().strip()


class BulkImportRequest(BaseModel):
    """Request to bulk import positions."""
    
    positions: list[BulkImportPosition] = Field(
        ...,
        min_length=1,
        description="Positions to import (at least one required)",
    )
    skip_duplicates: bool = Field(
        default=True,
        description="Skip positions that already exist (vs. update them)",
    )


class ImportResultStatus(str, Enum):
    """Status of a single position import."""
    
    CREATED = "created"
    UPDATED = "updated"
    SKIPPED = "skipped"
    FAILED = "failed"


class ImportPositionResult(BaseModel):
    """Result of importing a single position."""
    
    symbol: str
    status: ImportResultStatus
    message: str | None = None
    holding_id: int | None = None


class BulkImportResponse(BaseModel):
    """Response from bulk import endpoint."""
    
    success: bool
    total: int = Field(..., description="Total positions submitted")
    created: int = Field(..., description="Positions created")
    updated: int = Field(..., description="Positions updated")
    skipped: int = Field(..., description="Positions skipped (duplicates)")
    failed: int = Field(..., description="Positions that failed to import")
    results: list[ImportPositionResult] = Field(
        default_factory=list,
        description="Detailed result for each position",
    )
