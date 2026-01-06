"""
Signals Module - Technical signal scanning and support/resistance detection.

This module provides:
- Signal scanner for identifying trading opportunities
- Support/resistance level detection
"""

from app.quant_engine.signals.scanner import (
    OptimizedSignal,
    scan_all_stocks,
    ScanResult,
    StockOpportunity,
)
from app.quant_engine.signals.support_resistance import (
    analyze_support_resistance,
    SupportResistanceAnalysis,
)

__all__ = [
    # Scanner
    "OptimizedSignal",
    "scan_all_stocks",
    "ScanResult",
    "StockOpportunity",
    # Support/Resistance
    "analyze_support_resistance",
    "SupportResistanceAnalysis",
]
