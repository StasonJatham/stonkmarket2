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
class ChartMarker:
    """Marker for frontend chart overlay."""
    price: float
    marker_type: Literal["buy", "sell", "support", "resistance", "entry_zone_low", "entry_zone_high", "optimal_entry", "current_price"]
    label: str
    color: str = "blue"
    timestamp: str | None = None  # ISO format, optional
    
    def to_dict(self) -> dict[str, Any]:
        result = {
            "price": round(self.price, 2),
            "type": self.marker_type,
            "label": self.label,
            "color": self.color,
        }
        if self.timestamp:
            result["timestamp"] = self.timestamp
        return result


@dataclass
class BadgeInfo:
    """UI badge for quick visual scanning."""
    text: str
    color: Literal["green", "yellow", "red", "blue", "gray", "orange", "purple"]
    tooltip: str
    icon: str  # Lucide icon name
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "color": self.color,
            "tooltip": self.tooltip,
            "icon": self.icon,
        }


@dataclass
class StockAnalysisDashboard:
    """
    Unified output model for all stock analysis.
    
    This is the single format returned by the ScoringOrchestrator,
    consumed by the API and frontend.
    
    V3 additions:
    - sector_regime: Sector-specific trend analysis
    - entry_trigger: Discrete BUY/WAIT signals
    - event_risk: Earnings/dividend event awareness
    - liquidity: Volume adequacy check
    - badges: UI-ready quick-glance info
    - chart_markers: Visualization data
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
    
    # Regime Context (Market-wide)
    regime: RegimeState | None = None
    
    # V3: Sector Regime
    sector_regime: dict[str, Any] | None = None
    
    # V3: Entry Trigger
    entry_trigger: dict[str, Any] | None = None
    
    # V3: Event Risk (Earnings/Dividends)
    event_risk: dict[str, Any] | None = None
    
    # V3: Liquidity
    liquidity: dict[str, Any] | None = None
    
    # Domain Quality (simplified)
    fundamental_score: float = 50.0
    fundamental_notes: list[str] = field(default_factory=list)
    
    # Entry Analysis (legacy, still used)
    entry: EntryAnalysis = field(default_factory=EntryAnalysis)
    
    # Risk Assessment
    risk: RiskAssessment = field(default_factory=RiskAssessment)
    
    # V3: UI Badges
    badges: list[BadgeInfo] = field(default_factory=list)
    
    # V3: Chart Markers
    chart_markers: list[ChartMarker] = field(default_factory=list)
    
    # Metadata
    analysis_date: date = field(default_factory=date.today)
    data_quality: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"
    scoring_version: str = "3.1.0"
    
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
            # V3 additions
            "sector_regime": self.sector_regime,
            "entry_trigger": self.entry_trigger,
            "event_risk": self.event_risk,
            "liquidity": self.liquidity,
            "badges": [b.to_dict() for b in self.badges],
            "chart_markers": [m.to_dict() for m in self.chart_markers],
            # Legacy
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
            # V3: Include key flags
            "entry_signal": self.entry_trigger.get("signal") if self.entry_trigger else None,
            "event_risk_level": self.event_risk.get("risk_level") if self.event_risk else None,
            "sector_regime": self.sector_regime.get("regime") if self.sector_regime else None,
        }
