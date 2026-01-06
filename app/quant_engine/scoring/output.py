"""
Output models for the unified scoring system.

These models are used by the ScoringOrchestrator and returned via the API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from app.quant_engine.core.technical_service import TechnicalSnapshot
from app.quant_engine.core.regime_service import RegimeState


@dataclass
class ScoreComponents:
    """Individual score components (all 0-100)."""
    
    technical: float = 50.0
    fundamental: float = 50.0
    regime: float = 50.0
    entry_timing: float = 50.0
    risk: float = 50.0  # Higher = lower risk
    
    # Final composite
    composite: float = 50.0
    
    def to_dict(self) -> dict[str, float]:
        return {
            "technical": round(self.technical, 1),
            "fundamental": round(self.fundamental, 1),
            "regime": round(self.regime, 1),
            "entry_timing": round(self.entry_timing, 1),
            "risk": round(self.risk, 1),
            "composite": round(self.composite, 1),
        }


@dataclass
class EntryAnalysis:
    """Entry timing analysis."""
    
    current_drawdown_pct: float = 0.0
    is_dip_entry: bool = False
    optimal_entry_price: float | None = None
    days_since_high: int = 0
    rsi_oversold: bool = False
    volume_capitulation: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "current_drawdown_pct": round(self.current_drawdown_pct, 2),
            "is_dip_entry": self.is_dip_entry,
            "optimal_entry_price": round(self.optimal_entry_price, 2) if self.optimal_entry_price else None,
            "days_since_high": self.days_since_high,
            "rsi_oversold": self.rsi_oversold,
            "volume_capitulation": self.volume_capitulation,
        }


@dataclass
class RiskAssessment:
    """Risk assessment for the position."""
    
    risk_score: float = 50.0  # 0-100, higher = safer
    risk_factors: list[str] = field(default_factory=list)
    max_position_pct: float = 5.0  # Suggested max portfolio allocation
    suggested_stop_loss_pct: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": round(self.risk_score, 1),
            "risk_factors": self.risk_factors,
            "max_position_pct": round(self.max_position_pct, 1),
            "suggested_stop_loss_pct": self.suggested_stop_loss_pct,
        }


@dataclass
class StockAnalysisDashboard:
    """
    Unified output model for all stock analysis.
    
    This is the single format returned by the ScoringOrchestrator,
    consumed by the API and frontend.
    """
    
    # Identity
    symbol: str
    name: str | None = None
    sector: str | None = None
    domain: str | None = None
    
    # Overall Recommendation
    recommendation: Literal["STRONG_BUY", "BUY", "ACCUMULATE", "HOLD", "AVOID", "SELL"] = "HOLD"
    confidence: float = 50.0  # 0-100
    summary: str = ""
    
    # Component Scores
    scores: ScoreComponents = field(default_factory=ScoreComponents)
    
    # Technical Details
    technicals: TechnicalSnapshot | None = None
    
    # Regime Context
    regime: RegimeState | None = None
    
    # Domain Quality (simplified)
    fundamental_score: float = 50.0
    fundamental_notes: list[str] = field(default_factory=list)
    
    # Entry Analysis
    entry: EntryAnalysis = field(default_factory=EntryAnalysis)
    
    # Risk Assessment
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    
    # Metadata
    analysis_date: date = field(default_factory=date.today)
    data_quality: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    scoring_version: str = "3.0.0"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector,
            "domain": self.domain,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 1),
            "summary": self.summary,
            "scores": self.scores.to_dict(),
            "technicals": self.technicals.to_dict() if self.technicals else None,
            "regime": self.regime.to_dict() if self.regime else None,
            "fundamental_score": round(self.fundamental_score, 1),
            "fundamental_notes": self.fundamental_notes,
            "entry": self.entry.to_dict(),
            "risk": self.risk.to_dict(),
            "metadata": {
                "analysis_date": str(self.analysis_date),
                "data_quality": self.data_quality,
                "scoring_version": self.scoring_version,
            },
        }
    
    def to_compact_dict(self) -> dict[str, Any]:
        """Compact version for list views."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 1),
            "composite_score": round(self.scores.composite, 1),
            "current_drawdown_pct": round(self.entry.current_drawdown_pct, 2),
            "is_dip_entry": self.entry.is_dip_entry,
            "regime": self.regime.regime.value if self.regime else None,
        }
