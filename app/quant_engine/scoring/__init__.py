"""
Unified Scoring Module for the Quant Engine.

This module provides a single entry point for all stock scoring,
replacing the fragmented scoring systems (scoring.py, scoring_v2.py, trade_engine.py).

Components:
- ScoringOrchestrator: Main entry point that combines all scoring sources
- StockAnalysisDashboard: Unified output model

All scoring uses:
- TechnicalService for indicators
- RegimeService for market regime
- DomainScoringAdapter for fundamental quality
"""

from app.quant_engine.scoring.orchestrator import (
    ScoringOrchestrator,
    OrchestratorConfig,
    get_scoring_orchestrator,
)
from app.quant_engine.scoring.output import (
    StockAnalysisDashboard,
    ScoreComponents,
    EntryAnalysis,
    RiskAssessment,
)

__all__ = [
    "ScoringOrchestrator",
    "OrchestratorConfig",
    "get_scoring_orchestrator",
    "StockAnalysisDashboard",
    "ScoreComponents",
    "EntryAnalysis",
    "RiskAssessment",
]
