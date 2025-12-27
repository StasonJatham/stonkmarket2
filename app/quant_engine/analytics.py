"""
Portfolio Analytics Engine - Risk-Based Diagnostics.

This module provides deep risk analysis without trying to predict returns:
- Risk decomposition (where does portfolio risk come from?)
- Tail risk analysis (what happens in crashes?)
- Diversification metrics (how well-diversified are we?)
- Regime detection (what's the current market state?)
- Correlation analysis (which assets move together?)

All outputs are designed to translate into simple, actionable insights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RiskDecomposition:
    """Risk contribution analysis for a portfolio."""
    
    portfolio_volatility: float  # Annualized
    portfolio_var_95: float  # Daily VaR at 95%
    portfolio_cvar_95: float  # Daily CVaR at 95%
    
    # Per-asset contributions
    asset_volatilities: dict[str, float]
    marginal_risk: dict[str, float]  # ∂σ/∂w
    component_risk: dict[str, float]  # w * marginal_risk
    risk_contribution_pct: dict[str, float]  # Percentage of total risk
    
    # Risk concentration
    largest_risk_contributor: str
    top_3_risk_pct: float  # % of risk from top 3 assets


@dataclass
class TailRiskAnalysis:
    """Tail risk metrics for worst-case scenarios."""
    
    # Value at Risk
    var_95_daily: float  # 5% worst-case daily loss
    var_99_daily: float  # 1% worst-case daily loss
    
    # Conditional VaR (Expected Shortfall)
    cvar_95_daily: float  # Average loss in worst 5%
    cvar_99_daily: float  # Average loss in worst 1%
    
    # Cornish-Fisher adjusted (for non-normal distributions)
    cf_var_95: float
    cf_var_99: float
    
    # Drawdown analysis
    max_drawdown: float  # Largest peak-to-trough decline
    max_drawdown_duration_days: int  # Days to recover
    current_drawdown: float  # Current drawdown from peak
    
    # Distribution metrics
    skewness: float  # Negative = left tail fatter
    kurtosis: float  # > 3 = fat tails
    is_fat_tailed: bool
    
    # User-friendly summary
    risk_level: str  # LOW, MEDIUM, HIGH, VERY_HIGH
    risk_explanation: str


@dataclass
class DiversificationMetrics:
    """Diversification quality assessment."""
    
    # Concentration metrics
    hhi: float  # Herfindahl-Hirschman Index (0-1)
    effective_n: float  # Inverse HHI (equivalent number of equal positions)
    concentration_level: str  # LOW, MEDIUM, HIGH
    
    # Diversification benefit
    diversification_ratio: float  # Weighted avg vol / portfolio vol
    max_diversification_ratio: float  # Upper bound (equal weight)
    diversification_efficiency: float  # Actual / max (0-1)
    
    # Correlation-based
    avg_correlation: float  # Average pairwise correlation
    max_correlation: float  # Highest pairwise correlation
    correlation_cluster_count: int  # Number of correlated groups
    
    # Warnings
    is_well_diversified: bool
    diversification_warnings: list[str]


@dataclass
class RegimeState:
    """Current market regime detection."""
    
    regime: str  # bull_low, bull_high, bear_low, bear_high
    regime_description: str
    
    # Trend metrics
    market_return_3m: float  # 3-month market return
    is_bull: bool
    
    # Volatility metrics
    current_volatility: float  # Current annualized vol
    long_term_volatility: float  # 5-year average
    vol_ratio: float  # Current / long-term
    is_high_vol: bool
    
    # Correlation regime
    current_avg_correlation: float
    historical_avg_correlation: float
    correlation_elevated: bool
    
    # Recommendation
    risk_budget_recommendation: str  # "Normal", "Reduce", "Increase"
    explanation: str


@dataclass
class CorrelationAnalysis:
    """Correlation structure analysis."""
    
    correlation_matrix: pd.DataFrame
    
    # Summary stats
    avg_correlation: float
    min_correlation: float
    max_correlation: float
    
    # Highly correlated pairs
    high_correlation_pairs: list[tuple[str, str, float]]  # (asset1, asset2, corr)
    
    # Clustering
    clusters: list[list[str]]  # Groups of correlated assets
    n_clusters: int
    
    # Correlation stability
    correlation_vs_history: float  # Current avg / historical avg
    is_elevated: bool
    
    # Stress correlations (during worst 10% of market days)
    stress_avg_correlation: float
    correlation_breakdown_warning: bool


@dataclass
class PortfolioAnalytics:
    """Complete portfolio analytics report."""
    
    # Holdings info
    holdings: dict[str, float]  # symbol -> weight
    n_positions: int
    total_value: float | None
    
    # Core analytics
    risk_decomposition: RiskDecomposition
    tail_risk: TailRiskAnalysis
    diversification: DiversificationMetrics
    regime: RegimeState
    correlations: CorrelationAnalysis
    
    # User-friendly summary
    overall_risk_score: int  # 1-10
    key_insights: list[str]
    action_items: list[str]
    
    # Metadata
    analysis_date: str
    data_quality: str  # GOOD, FAIR, POOR


# =============================================================================
# Core Computation Functions
# =============================================================================


def compute_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "shrinkage",
    shrinkage_target: str = "constant_correlation",
) -> pd.DataFrame:
    """
    Compute robust covariance matrix with optional shrinkage.
    
    Methods:
    - sample: Raw sample covariance (unstable with limited data)
    - shrinkage: Ledoit-Wolf shrinkage (recommended)
    - ewma: Exponentially weighted (recent data weighted more)
    """
    if method == "sample":
        return returns.cov()
    
    elif method == "shrinkage":
        # Ledoit-Wolf shrinkage
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(returns.dropna())
        cov_shrunk = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns,
        )
        return cov_shrunk
    
    elif method == "ewma":
        # Exponentially weighted
        halflife = 63  # ~3 months
        ewma_cov = returns.ewm(halflife=halflife).cov()
        # Get last observation
        last_date = returns.index[-1]
        return ewma_cov.loc[last_date]
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_risk_decomposition(
    weights: np.ndarray,
    cov: np.ndarray,
    symbols: list[str],
    returns: pd.DataFrame | None = None,
) -> RiskDecomposition:
    """
    Decompose portfolio risk by asset contribution.
    
    Answers: "Where does my portfolio risk come from?"
    """
    # Portfolio volatility
    portfolio_var = weights @ cov @ weights
    portfolio_vol = np.sqrt(portfolio_var) * np.sqrt(252)  # Annualized
    
    # Marginal risk contribution: ∂σ/∂w
    marginal = (cov @ weights) / np.sqrt(portfolio_var)
    
    # Component risk contribution: w * marginal
    component = weights * marginal
    
    # Percentage contribution
    risk_pct = component / np.sqrt(portfolio_var)
    
    # VaR/CVaR if returns provided
    var_95, cvar_95 = 0.0, 0.0
    if returns is not None:
        port_returns = (returns @ weights).dropna()
        if len(port_returns) > 0:
            var_95 = float(np.percentile(port_returns, 5))
            cvar_95 = float(port_returns[port_returns <= var_95].mean())
    
    # Build dicts
    asset_vols = {s: float(np.sqrt(cov[i, i]) * np.sqrt(252)) for i, s in enumerate(symbols)}
    marginal_dict = {s: float(marginal[i]) for i, s in enumerate(symbols)}
    component_dict = {s: float(component[i]) for i, s in enumerate(symbols)}
    risk_pct_dict = {s: float(risk_pct[i]) for i, s in enumerate(symbols)}
    
    # Find largest contributor
    largest = max(risk_pct_dict, key=risk_pct_dict.get)
    top_3_values = sorted(risk_pct_dict.values(), reverse=True)[:3]
    top_3_pct = sum(top_3_values)
    
    return RiskDecomposition(
        portfolio_volatility=portfolio_vol,
        portfolio_var_95=var_95,
        portfolio_cvar_95=cvar_95,
        asset_volatilities=asset_vols,
        marginal_risk=marginal_dict,
        component_risk=component_dict,
        risk_contribution_pct=risk_pct_dict,
        largest_risk_contributor=largest,
        top_3_risk_pct=top_3_pct,
    )


def compute_tail_risk(
    returns: pd.Series,
    alpha_95: float = 0.05,
    alpha_99: float = 0.01,
) -> TailRiskAnalysis:
    """
    Compute tail risk metrics.
    
    Answers: "What happens to my portfolio in a crash?"
    """
    returns = returns.dropna()
    
    if len(returns) < 30:
        # Insufficient data
        return TailRiskAnalysis(
            var_95_daily=0,
            var_99_daily=0,
            cvar_95_daily=0,
            cvar_99_daily=0,
            cf_var_95=0,
            cf_var_99=0,
            max_drawdown=0,
            max_drawdown_duration_days=0,
            current_drawdown=0,
            skewness=0,
            kurtosis=3,
            is_fat_tailed=False,
            risk_level="UNKNOWN",
            risk_explanation="Insufficient data for tail risk analysis",
        )
    
    # Historical VaR
    var_95 = float(np.percentile(returns, alpha_95 * 100))
    var_99 = float(np.percentile(returns, alpha_99 * 100))
    
    # Historical CVaR (Expected Shortfall)
    cvar_95 = float(returns[returns <= var_95].mean())
    cvar_99 = float(returns[returns <= var_99].mean())
    
    # Distribution metrics
    skewness = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns))  # Excess kurtosis
    is_fat_tailed = kurtosis > 1  # Excess kurtosis > 1 means fat tails
    
    # Cornish-Fisher adjustment
    z_95 = stats.norm.ppf(alpha_95)
    z_99 = stats.norm.ppf(alpha_99)
    
    def cf_adjustment(z: float) -> float:
        return z + (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * kurtosis / 24 - (2*z**3 - 5*z) * skewness**2 / 36
    
    mu = returns.mean()
    sigma = returns.std()
    cf_var_95 = float(mu + sigma * cf_adjustment(z_95))
    cf_var_99 = float(mu + sigma * cf_adjustment(z_99))
    
    # Drawdown analysis
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    
    max_dd = float(drawdown.min())
    current_dd = float(drawdown.iloc[-1])
    
    # Max drawdown duration
    in_drawdown = drawdown < 0
    if in_drawdown.any():
        # Find longest consecutive drawdown period
        dd_groups = (~in_drawdown).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        max_dd_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    else:
        max_dd_duration = 0
    
    # Risk level classification
    ann_vol = sigma * np.sqrt(252)
    if ann_vol < 0.10:
        risk_level = "LOW"
        explanation = "Portfolio has low volatility, suitable for conservative investors"
    elif ann_vol < 0.20:
        risk_level = "MEDIUM"
        explanation = "Portfolio has moderate volatility, typical for balanced portfolios"
    elif ann_vol < 0.30:
        risk_level = "HIGH"
        explanation = "Portfolio has high volatility, expect significant swings"
    else:
        risk_level = "VERY_HIGH"
        explanation = "Portfolio has very high volatility, only for aggressive risk tolerance"
    
    if is_fat_tailed:
        explanation += ". Fat tails detected - extreme moves more likely than normal distribution suggests."
    
    return TailRiskAnalysis(
        var_95_daily=var_95,
        var_99_daily=var_99,
        cvar_95_daily=cvar_95,
        cvar_99_daily=cvar_99,
        cf_var_95=cf_var_95,
        cf_var_99=cf_var_99,
        max_drawdown=max_dd,
        max_drawdown_duration_days=max_dd_duration,
        current_drawdown=current_dd,
        skewness=skewness,
        kurtosis=kurtosis,
        is_fat_tailed=is_fat_tailed,
        risk_level=risk_level,
        risk_explanation=explanation,
    )


def compute_diversification_metrics(
    weights: np.ndarray,
    cov: np.ndarray,
    corr: np.ndarray,
    symbols: list[str],
) -> DiversificationMetrics:
    """
    Compute diversification quality metrics.
    
    Answers: "How well-diversified is my portfolio?"
    """
    n = len(weights)
    
    # Concentration: Herfindahl-Hirschman Index
    hhi = float(np.sum(weights ** 2))
    effective_n = 1 / hhi if hhi > 0 else n
    
    if effective_n >= n * 0.7:
        concentration_level = "LOW"
    elif effective_n >= n * 0.4:
        concentration_level = "MEDIUM"
    else:
        concentration_level = "HIGH"
    
    # Diversification ratio
    asset_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = weights @ asset_vols
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
    
    # Max diversification (equal weight)
    equal_weights = np.ones(n) / n
    max_weighted_avg_vol = equal_weights @ asset_vols
    max_portfolio_vol = np.sqrt(equal_weights @ cov @ equal_weights)
    max_div_ratio = max_weighted_avg_vol / max_portfolio_vol if max_portfolio_vol > 0 else 1.0
    
    div_efficiency = div_ratio / max_div_ratio if max_div_ratio > 0 else 0
    
    # Correlation metrics
    upper_tri = np.triu_indices(n, k=1)
    if len(upper_tri[0]) > 0:
        corr_values = corr[upper_tri]
        avg_corr = float(np.mean(corr_values))
        max_corr = float(np.max(corr_values))
    else:
        avg_corr, max_corr = 0.0, 0.0
    
    # Find high correlation pairs
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr[i, j] > 0.7:
                high_corr_pairs.append((symbols[i], symbols[j], float(corr[i, j])))
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Correlation clustering
    if n > 2:
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0)
        try:
            dist_condensed = squareform(dist, checks=False)
            link = linkage(dist_condensed, method="ward")
            # Cut at distance threshold to get clusters
            from scipy.cluster.hierarchy import fcluster
            cluster_labels = fcluster(link, t=0.5, criterion="distance")
            n_clusters = len(set(cluster_labels))
        except Exception:
            n_clusters = n
    else:
        n_clusters = n
    
    # Warnings
    warnings = []
    if effective_n < n * 0.5:
        warnings.append(f"Portfolio is concentrated - effectively only {effective_n:.1f} positions")
    if avg_corr > 0.5:
        warnings.append(f"Assets are highly correlated (avg {avg_corr:.2f}) - diversification limited")
    if len(high_corr_pairs) > 0:
        top_pair = high_corr_pairs[0]
        warnings.append(f"{top_pair[0]} and {top_pair[1]} are very similar (correlation {top_pair[2]:.2f})")
    
    is_well_diversified = div_efficiency > 0.7 and avg_corr < 0.5
    
    return DiversificationMetrics(
        hhi=hhi,
        effective_n=effective_n,
        concentration_level=concentration_level,
        diversification_ratio=float(div_ratio),
        max_diversification_ratio=float(max_div_ratio),
        diversification_efficiency=float(div_efficiency),
        avg_correlation=avg_corr,
        max_correlation=max_corr,
        correlation_cluster_count=n_clusters,
        is_well_diversified=is_well_diversified,
        diversification_warnings=warnings,
    )


def detect_regime(
    market_returns: pd.Series,
    asset_returns: pd.DataFrame,
    lookback_trend: int = 63,
    lookback_vol: int = 21,
) -> RegimeState:
    """
    Detect current market regime.
    
    Answers: "What's the current market environment?"
    """
    market_returns = market_returns.dropna()
    
    if len(market_returns) < lookback_trend:
        return RegimeState(
            regime="unknown",
            regime_description="Insufficient data",
            market_return_3m=0,
            is_bull=True,
            current_volatility=0,
            long_term_volatility=0,
            vol_ratio=1.0,
            is_high_vol=False,
            current_avg_correlation=0,
            historical_avg_correlation=0,
            correlation_elevated=False,
            risk_budget_recommendation="Normal",
            explanation="Not enough historical data to determine regime",
        )
    
    # Trend analysis
    return_3m = float(market_returns.iloc[-lookback_trend:].sum())
    is_bull = return_3m > 0
    
    # Volatility analysis
    current_vol = float(market_returns.iloc[-lookback_vol:].std() * np.sqrt(252))
    long_term_vol = float(market_returns.std() * np.sqrt(252))
    vol_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0
    is_high_vol = vol_ratio > 1.2
    
    # Correlation analysis
    if asset_returns is not None and len(asset_returns) >= lookback_trend:
        recent_corr = asset_returns.iloc[-lookback_trend:].corr()
        historical_corr = asset_returns.corr()
        
        n = recent_corr.shape[0]
        upper_tri = np.triu_indices(n, k=1)
        
        current_avg_corr = float(np.mean(recent_corr.values[upper_tri]))
        historical_avg_corr = float(np.mean(historical_corr.values[upper_tri]))
        correlation_elevated = current_avg_corr > historical_avg_corr * 1.2
    else:
        current_avg_corr = 0.0
        historical_avg_corr = 0.0
        correlation_elevated = False
    
    # Determine regime
    if is_bull and not is_high_vol:
        regime = "bull_low"
        description = "Rising market with low volatility - favorable conditions"
    elif is_bull and is_high_vol:
        regime = "bull_high"
        description = "Rising market but volatile - proceed with caution"
    elif not is_bull and not is_high_vol:
        regime = "bear_low"
        description = "Declining market with low volatility - slow correction"
    else:
        regime = "bear_high"
        description = "Declining market with high volatility - defensive posture recommended"
    
    # Risk budget recommendation
    if regime == "bull_low":
        recommendation = "Normal"
        explanation = "Market conditions are favorable. Normal risk allocation appropriate."
    elif regime == "bull_high":
        recommendation = "Slightly Reduce"
        explanation = "Market rising but choppy. Consider slightly reducing risk exposure."
    elif regime == "bear_low":
        recommendation = "Reduce"
        explanation = "Market declining. Consider reducing risk and raising cash."
    else:  # bear_high
        recommendation = "Significantly Reduce"
        explanation = "High volatility selloff. Defensive positioning recommended."
    
    if correlation_elevated:
        explanation += " Correlations are elevated - diversification benefits reduced."
    
    return RegimeState(
        regime=regime,
        regime_description=description,
        market_return_3m=return_3m,
        is_bull=is_bull,
        current_volatility=current_vol,
        long_term_volatility=long_term_vol,
        vol_ratio=float(vol_ratio),
        is_high_vol=is_high_vol,
        current_avg_correlation=current_avg_corr,
        historical_avg_correlation=historical_avg_corr,
        correlation_elevated=correlation_elevated,
        risk_budget_recommendation=recommendation,
        explanation=explanation,
    )


def compute_correlation_analysis(
    returns: pd.DataFrame,
    lookback_current: int = 63,
    stress_threshold: float = 0.10,
) -> CorrelationAnalysis:
    """
    Analyze correlation structure.
    
    Answers: "Which assets move together? How do correlations change in stress?"
    """
    symbols = list(returns.columns)
    n = len(symbols)
    
    # Current correlation matrix
    corr = returns.corr()
    
    # Summary stats
    upper_tri = np.triu_indices(n, k=1)
    if len(upper_tri[0]) > 0:
        corr_values = corr.values[upper_tri]
        avg_corr = float(np.mean(corr_values))
        min_corr = float(np.min(corr_values))
        max_corr = float(np.max(corr_values))
    else:
        avg_corr, min_corr, max_corr = 0.0, 0.0, 0.0
    
    # High correlation pairs
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr.iloc[i, j] > 0.7:
                high_pairs.append((symbols[i], symbols[j], float(corr.iloc[i, j])))
    high_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Clustering
    clusters = []
    if n > 2:
        try:
            dist = np.sqrt((1 - corr.values) / 2)
            np.fill_diagonal(dist, 0)
            dist_condensed = squareform(dist, checks=False)
            link = linkage(dist_condensed, method="ward")
            from scipy.cluster.hierarchy import fcluster
            cluster_labels = fcluster(link, t=0.7, criterion="distance")
            
            # Group symbols by cluster
            for cluster_id in set(cluster_labels):
                cluster_symbols = [symbols[i] for i, c in enumerate(cluster_labels) if c == cluster_id]
                if len(cluster_symbols) > 1:
                    clusters.append(cluster_symbols)
        except Exception:
            clusters = []
    
    # Correlation vs recent history
    if len(returns) >= lookback_current * 2:
        recent_corr = returns.iloc[-lookback_current:].corr()
        recent_avg = float(np.mean(recent_corr.values[upper_tri])) if len(upper_tri[0]) > 0 else 0
        historical_avg = avg_corr
        corr_vs_history = recent_avg / historical_avg if historical_avg > 0 else 1.0
        is_elevated = corr_vs_history > 1.2
    else:
        corr_vs_history = 1.0
        is_elevated = False
    
    # Stress correlations (during worst market days)
    if "SPY" in returns.columns or len(returns.columns) > 0:
        # Use first column as market proxy if SPY not available
        market_col = "SPY" if "SPY" in returns.columns else returns.columns[0]
        market_ret = returns[market_col]
        stress_threshold_value = np.percentile(market_ret.dropna(), stress_threshold * 100)
        stress_mask = market_ret <= stress_threshold_value
        
        if stress_mask.sum() >= 10:
            stress_returns = returns[stress_mask]
            stress_corr = stress_returns.corr()
            stress_avg = float(np.mean(stress_corr.values[upper_tri])) if len(upper_tri[0]) > 0 else 0
        else:
            stress_avg = avg_corr
    else:
        stress_avg = avg_corr
    
    breakdown_warning = stress_avg > avg_corr * 1.3  # Correlations spike in stress
    
    return CorrelationAnalysis(
        correlation_matrix=corr,
        avg_correlation=avg_corr,
        min_correlation=min_corr,
        max_correlation=max_corr,
        high_correlation_pairs=high_pairs[:5],  # Top 5
        clusters=clusters,
        n_clusters=len(clusters),
        correlation_vs_history=float(corr_vs_history),
        is_elevated=is_elevated,
        stress_avg_correlation=stress_avg,
        correlation_breakdown_warning=breakdown_warning,
    )


# =============================================================================
# Main Analysis Function
# =============================================================================


def analyze_portfolio(
    holdings: dict[str, float],
    returns: pd.DataFrame,
    market_returns: pd.Series | None = None,
    total_value: float | None = None,
) -> PortfolioAnalytics:
    """
    Complete portfolio analysis.
    
    Parameters
    ----------
    holdings : dict[str, float]
        Symbol -> weight mapping
    returns : pd.DataFrame
        Daily returns for each asset
    market_returns : pd.Series, optional
        Market benchmark returns (e.g., SPY). For accurate regime detection,
        provide actual SPY returns. If None, falls back to SPY in returns
        DataFrame or portfolio returns as proxy.
    total_value : float, optional
        Total portfolio value in EUR
    
    Returns
    -------
    PortfolioAnalytics
        Complete analysis report with user-friendly insights
    """
    from datetime import datetime
    
    symbols = list(holdings.keys())
    weights = np.array([holdings[s] for s in symbols])
    
    # Normalize weights
    if weights.sum() != 1.0:
        weights = weights / weights.sum()
    
    # Filter returns to only include held assets
    available_symbols = [s for s in symbols if s in returns.columns]
    if len(available_symbols) == 0:
        raise ValueError("No price data available for held assets")
    
    returns_filtered = returns[available_symbols].dropna()
    weights_filtered = np.array([holdings[s] for s in available_symbols])
    weights_filtered = weights_filtered / weights_filtered.sum()
    
    # Compute covariance and correlation
    cov = compute_covariance_matrix(returns_filtered, method="shrinkage")
    corr = returns_filtered.corr()
    
    # Portfolio returns
    port_returns = (returns_filtered * weights_filtered).sum(axis=1)
    
    # Use market returns or first asset as proxy
    if market_returns is None:
        if "SPY" in returns.columns:
            market_returns = returns["SPY"]
        else:
            market_returns = port_returns
    
    # Run all analytics
    risk_decomp = compute_risk_decomposition(
        weights_filtered, cov.values, available_symbols, returns_filtered
    )
    
    tail_risk = compute_tail_risk(port_returns)
    
    diversification = compute_diversification_metrics(
        weights_filtered, cov.values, corr.values, available_symbols
    )
    
    regime = detect_regime(market_returns, returns_filtered)
    
    correlations = compute_correlation_analysis(returns_filtered)
    
    # Generate insights
    insights = []
    action_items = []
    
    # Risk insights
    if risk_decomp.top_3_risk_pct > 0.8:
        insights.append(f"Top 3 positions account for {risk_decomp.top_3_risk_pct*100:.0f}% of portfolio risk")
        action_items.append("Consider rebalancing to spread risk more evenly")
    
    insights.append(f"Largest risk contributor: {risk_decomp.largest_risk_contributor}")
    
    # Tail risk insights
    if tail_risk.is_fat_tailed:
        insights.append("Portfolio returns have fat tails - extreme moves more likely than normal")
    
    if tail_risk.current_drawdown < -0.10:
        insights.append(f"Currently in {abs(tail_risk.current_drawdown)*100:.0f}% drawdown from peak")
    
    # Diversification insights
    if not diversification.is_well_diversified:
        insights.extend(diversification.diversification_warnings)
        action_items.append("Improve diversification by adding uncorrelated assets")
    
    # Regime insights
    insights.append(f"Market regime: {regime.regime_description}")
    if regime.risk_budget_recommendation != "Normal":
        action_items.append(regime.explanation)
    
    # Correlation insights
    if correlations.correlation_breakdown_warning:
        insights.append("Warning: Correlations spike during market stress - diversification may fail when needed most")
    
    # Overall risk score (1-10)
    risk_score = 5  # Start at medium
    
    # Adjust based on volatility
    if risk_decomp.portfolio_volatility < 0.10:
        risk_score -= 2
    elif risk_decomp.portfolio_volatility > 0.25:
        risk_score += 2
    elif risk_decomp.portfolio_volatility > 0.20:
        risk_score += 1
    
    # Adjust based on diversification
    if not diversification.is_well_diversified:
        risk_score += 1
    
    # Adjust based on regime
    if regime.regime == "bear_high":
        risk_score += 2
    elif regime.regime == "bear_low":
        risk_score += 1
    
    risk_score = max(1, min(10, risk_score))
    
    # Data quality assessment
    if len(returns_filtered) >= 252 * 3:
        data_quality = "GOOD"
    elif len(returns_filtered) >= 252:
        data_quality = "FAIR"
    else:
        data_quality = "POOR"
    
    return PortfolioAnalytics(
        holdings=holdings,
        n_positions=len(available_symbols),
        total_value=total_value,
        risk_decomposition=risk_decomp,
        tail_risk=tail_risk,
        diversification=diversification,
        regime=regime,
        correlations=correlations,
        overall_risk_score=risk_score,
        key_insights=insights,
        action_items=action_items,
        analysis_date=datetime.now().isoformat(),
        data_quality=data_quality,
    )


# =============================================================================
# User-Friendly Translation
# =============================================================================


def translate_for_user(analytics: PortfolioAnalytics, portfolio_value: float = 10000) -> dict:
    """
    Translate technical analytics into user-friendly language.
    
    Takes complex quant metrics and makes them accessible to retail investors.
    """
    result = {
        "summary": {
            "risk_score": analytics.overall_risk_score,
            "risk_label": _risk_score_label(analytics.overall_risk_score),
            "headline": _generate_headline(analytics),
        },
        "risk": {
            "volatility_explanation": f"Your portfolio typically swings ±{analytics.risk_decomposition.portfolio_volatility*100:.0f}% per year",
            "bad_day_loss": f"On a bad day (1 in 20), you could lose up to €{abs(analytics.tail_risk.var_95_daily * portfolio_value):.0f}",
            "crash_loss": f"In a market crash, expect to lose around €{abs(analytics.tail_risk.cvar_95_daily * portfolio_value):.0f}",
            "worst_ever": f"Historically, the worst drop was {abs(analytics.tail_risk.max_drawdown)*100:.0f}%",
        },
        "diversification": {
            "effective_positions": f"You effectively have {analytics.diversification.effective_n:.1f} independent investments",
            "quality": "Well diversified" if analytics.diversification.is_well_diversified else "Needs improvement",
            "warnings": analytics.diversification.diversification_warnings,
        },
        "market": {
            "current_regime": analytics.regime.regime_description,
            "recommendation": analytics.regime.risk_budget_recommendation,
            "explanation": analytics.regime.explanation,
        },
        "action_items": analytics.action_items[:3],  # Top 3 actions
        "insights": analytics.key_insights[:5],  # Top 5 insights
    }
    
    return result


def _risk_score_label(score: int) -> str:
    if score <= 2:
        return "Very Low Risk"
    elif score <= 4:
        return "Low Risk"
    elif score <= 6:
        return "Moderate Risk"
    elif score <= 8:
        return "High Risk"
    else:
        return "Very High Risk"


def _generate_headline(analytics: PortfolioAnalytics) -> str:
    """Generate a one-line summary of portfolio state."""
    risk_level = _risk_score_label(analytics.overall_risk_score)
    
    if analytics.regime.regime == "bear_high":
        return f"{risk_level} portfolio in volatile market - consider defensive adjustments"
    elif not analytics.diversification.is_well_diversified:
        return f"{risk_level} portfolio with concentration risk - diversification recommended"
    elif analytics.tail_risk.current_drawdown < -0.15:
        return f"{risk_level} portfolio in significant drawdown - review positions"
    else:
        return f"{risk_level} portfolio in {analytics.regime.regime_description.lower()}"
