"""DipFinder signal engine for stock dip analysis.

This module provides:
- Dip severity calculations with O(n) peak-to-current algorithm
- Dip significance vs historical behavior
- Market context classification (market/stock-specific/mixed)
- Quality metrics from yfinance fundamentals
- Stability metrics (volatility, max drawdown, beta)
- Combined scoring and alerting
"""

from .config import DipFinderConfig, get_dipfinder_config
from .dip import (
    DipMetrics,
    compute_dip_percentile,
    compute_dip_series_windowed,
    compute_persistence,
)
from .fundamentals import QualityMetrics, compute_quality_score
from .service import DipFinderService, get_dipfinder_service
from .signal import AlertLevel, DipClass, DipSignal, MarketContext, compute_signal
from .stability import StabilityMetrics, compute_stability_score


__all__ = [
    "AlertLevel",
    "DipClass",
    "DipFinderConfig",
    "DipFinderService",
    "DipMetrics",
    "DipSignal",
    "MarketContext",
    "QualityMetrics",
    "StabilityMetrics",
    "compute_dip_percentile",
    "compute_dip_series_windowed",
    "compute_persistence",
    "compute_quality_score",
    "compute_signal",
    "compute_stability_score",
    "get_dipfinder_config",
    "get_dipfinder_service",
]
