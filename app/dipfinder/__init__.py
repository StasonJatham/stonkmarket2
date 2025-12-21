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
from .dip import compute_dip_series, compute_dip_percentile, DipMetrics
from .fundamentals import compute_quality_score, QualityMetrics
from .stability import compute_stability_score, StabilityMetrics
from .signal import compute_signal, DipSignal
from .service import DipFinderService

__all__ = [
    "DipFinderConfig",
    "get_dipfinder_config",
    "compute_dip_series",
    "compute_dip_percentile",
    "DipMetrics",
    "compute_quality_score",
    "QualityMetrics",
    "compute_stability_score",
    "StabilityMetrics",
    "compute_signal",
    "DipSignal",
    "DipFinderService",
]
