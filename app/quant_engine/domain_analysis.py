"""
Domain-Specific Stock Analysis.

Different sectors require different analysis frameworks:
- Banks: Interest rate sensitivity, NIM, credit quality, capital ratios
- Tech: Growth metrics, R&D efficiency, TAM, revenue quality
- Healthcare: Pipeline, FDA approvals, patent cliffs, reimbursement risk
- Consumer: Brand strength, pricing power, cyclicality
- Energy: Commodity exposure, reserve life, breakeven costs
- REITs: FFO, occupancy, cap rates, debt/equity

This module provides sector-aware analysis that interprets dips differently
based on what matters for each domain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Domain/Sector Definitions
# =============================================================================


class Sector(str, Enum):
    """Stock sectors with domain-specific analysis."""
    
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"  # Banks, insurance
    HEALTHCARE = "healthcare"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    ENERGY = "energy"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    UTILITIES = "utilities"
    COMMUNICATION = "communication"
    UNKNOWN = "unknown"


# Map common sector names to our enum
SECTOR_MAPPING = {
    # Technology
    "technology": Sector.TECHNOLOGY,
    "information technology": Sector.TECHNOLOGY,
    "tech": Sector.TECHNOLOGY,
    "software": Sector.TECHNOLOGY,
    "semiconductors": Sector.TECHNOLOGY,
    
    # Financials
    "financials": Sector.FINANCIALS,
    "financial services": Sector.FINANCIALS,
    "banks": Sector.FINANCIALS,
    "banking": Sector.FINANCIALS,
    "insurance": Sector.FINANCIALS,
    
    # Healthcare
    "healthcare": Sector.HEALTHCARE,
    "health care": Sector.HEALTHCARE,
    "biotechnology": Sector.HEALTHCARE,
    "pharmaceuticals": Sector.HEALTHCARE,
    "biotech": Sector.HEALTHCARE,
    
    # Consumer
    "consumer discretionary": Sector.CONSUMER_DISCRETIONARY,
    "consumer cyclical": Sector.CONSUMER_DISCRETIONARY,
    "retail": Sector.CONSUMER_DISCRETIONARY,
    "consumer staples": Sector.CONSUMER_STAPLES,
    "consumer defensive": Sector.CONSUMER_STAPLES,
    
    # Energy
    "energy": Sector.ENERGY,
    "oil & gas": Sector.ENERGY,
    
    # Others
    "industrials": Sector.INDUSTRIALS,
    "materials": Sector.MATERIALS,
    "basic materials": Sector.MATERIALS,
    "real estate": Sector.REAL_ESTATE,
    "utilities": Sector.UTILITIES,
    "communication services": Sector.COMMUNICATION,
    "communication": Sector.COMMUNICATION,
    "telecommunications": Sector.COMMUNICATION,
}


def normalize_sector(sector_name: str | None) -> Sector:
    """Normalize sector name to enum."""
    if not sector_name:
        return Sector.UNKNOWN
    
    normalized = sector_name.lower().strip()
    return SECTOR_MAPPING.get(normalized, Sector.UNKNOWN)


# =============================================================================
# Domain-Specific Metrics
# =============================================================================


@dataclass
class DomainMetrics:
    """Base class for domain-specific metrics."""
    
    sector: Sector
    sector_name: str
    
    # Universal metrics
    volatility_regime: str  # "low", "normal", "high", "extreme"
    volatility_percentile: float  # 0-100
    correlation_to_sector: float  # -1 to 1
    
    # Dip context
    dip_interpretation: str  # How to interpret the dip for this sector
    typical_recovery_days: int  # Sector-typical recovery time
    
    # Risk factors specific to sector
    primary_risk_factors: list[str]
    current_risk_level: str  # "low", "medium", "high"
    
    # Recommendation adjustment
    sector_adjustment: float  # -1 to +1, adjusts overall score
    adjustment_reason: str


@dataclass
class BankMetrics(DomainMetrics):
    """Bank/Financial sector specific metrics."""
    
    # Interest rate sensitivity
    rate_sensitivity: str  # "asset_sensitive", "liability_sensitive", "neutral"
    rate_outlook_impact: str  # How rate changes affect the stock
    
    # Yield curve
    yield_curve_state: str  # "normal", "flat", "inverted"
    yield_curve_impact: str
    
    # Credit cycle
    credit_cycle_phase: str  # "expansion", "peak", "contraction", "trough"
    
    # Bank-specific dip factors
    dip_likely_causes: list[str]
    is_systemic_risk: bool


@dataclass
class TechMetrics(DomainMetrics):
    """Technology sector specific metrics."""
    
    # Growth vs Value
    is_growth_stock: bool
    growth_premium_at_risk: bool  # Is growth premium deflating?
    
    # Tech-specific factors
    rate_sensitivity_for_growth: str  # Growth stocks hate rising rates
    
    # Competitive moat
    has_network_effects: bool
    has_switching_costs: bool
    
    # Tech-specific dip factors
    dip_likely_causes: list[str]
    is_sector_rotation: bool  # Is it just rotation out of tech?


@dataclass
class HealthcareMetrics(DomainMetrics):
    """Healthcare sector specific metrics."""
    
    # Sub-sector
    subsector: str  # "pharma", "biotech", "devices", "services", "managed_care"
    
    # Pipeline/binary risk
    has_binary_catalyst: bool  # FDA decision, trial results pending
    binary_event_description: str
    
    # Defensive characteristics
    is_defensive: bool  # Stable demand regardless of economy
    
    # Healthcare-specific dip factors
    dip_likely_causes: list[str]
    is_drug_specific_risk: bool


@dataclass
class EnergyMetrics(DomainMetrics):
    """Energy sector specific metrics."""
    
    # Commodity exposure
    primary_commodity: str  # "oil", "gas", "renewables"
    commodity_trend: str  # "bullish", "bearish", "neutral"
    
    # Cost structure
    breakeven_context: str  # High/low cost producer
    
    # Energy-specific dip factors
    dip_likely_causes: list[str]
    is_commodity_driven: bool


@dataclass
class DomainAnalysis:
    """Complete domain-specific analysis result."""
    
    symbol: str
    sector: Sector
    sector_display_name: str
    
    # Domain metrics (one of the specific types)
    metrics: DomainMetrics
    
    # How domain context changes the dip interpretation
    dip_context: str  # Plain English explanation
    dip_adjustment: float  # -1 to +1 adjustment to opportunity score
    
    # Domain-specific signals
    sector_signals: list[dict]  # Signals that matter for this sector
    
    # Comparison to sector peers
    vs_sector_performance: float  # Relative to sector ETF
    sector_rank_percentile: float  # Where does this stock rank in sector
    
    # Final recommendation adjustment
    final_adjustment: float
    adjustment_explanation: str
    
    # Warnings specific to domain
    domain_warnings: list[str]


# =============================================================================
# Volatility Regime Detection
# =============================================================================


def compute_volatility_regime(
    returns: pd.Series,
    current_vol_window: int = 20,
    historical_vol_window: int = 252,
) -> tuple[str, float]:
    """
    Determine volatility regime (low/normal/high/extreme).
    
    Compares current volatility to historical distribution.
    """
    if len(returns) < historical_vol_window:
        return "normal", 50.0
    
    # Current volatility (20-day annualized)
    current_vol = returns.iloc[-current_vol_window:].std() * np.sqrt(252)
    
    # Historical volatility distribution
    rolling_vol = returns.rolling(current_vol_window).std() * np.sqrt(252)
    historical_vols = rolling_vol.dropna()
    
    if len(historical_vols) == 0:
        return "normal", 50.0
    
    # Percentile of current vol
    percentile = (historical_vols < current_vol).mean() * 100
    
    # Regime classification
    if percentile < 20:
        regime = "low"
    elif percentile < 60:
        regime = "normal"
    elif percentile < 85:
        regime = "high"
    else:
        regime = "extreme"
    
    return regime, float(percentile)


def compute_general_health(prices: pd.Series) -> tuple[str, str]:
    """
    Compute general price health metrics regardless of sector.
    
    Returns:
        (health_status, health_summary): Brief assessment of general health
    """
    if len(prices) < 50:
        return "unknown", ""
    
    # Calculate key metrics
    returns = prices.pct_change().dropna()
    
    # Momentum: 20-day vs 50-day return
    if len(prices) >= 50:
        ret_20d = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100
        ret_50d = (prices.iloc[-1] / prices.iloc[-50] - 1) * 100
    else:
        ret_20d = ret_50d = 0
    
    # RSI proxy (simplified)
    if len(returns) >= 14:
        gains = returns.where(returns > 0, 0).rolling(14).mean()
        losses = (-returns.where(returns < 0, 0)).rolling(14).mean()
        rs = gains / losses.replace(0, np.nan)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        if pd.isna(rsi):
            rsi = 50
    else:
        rsi = 50
    
    # Drawdown from peak
    peak = prices.rolling(min(252, len(prices)), min_periods=1).max()
    drawdown = ((prices.iloc[-1] - peak.iloc[-1]) / peak.iloc[-1]) * 100
    
    # Build health assessment
    health_parts = []
    
    # Momentum assessment
    if ret_20d > 5:
        health_parts.append("strong momentum")
        momentum_health = "good"
    elif ret_20d > 0:
        health_parts.append("positive trend")
        momentum_health = "okay"
    elif ret_20d > -5:
        health_parts.append("weak momentum")
        momentum_health = "weak"
    else:
        health_parts.append("downtrend")
        momentum_health = "poor"
    
    # RSI assessment  
    if rsi < 30:
        health_parts.append("oversold (RSI {:.0f})".format(rsi))
    elif rsi > 70:
        health_parts.append("overbought (RSI {:.0f})".format(rsi))
    
    # Drawdown severity
    if drawdown < -30:
        health_parts.append("severe drawdown ({:.0f}%)".format(drawdown))
        dd_health = "critical"
    elif drawdown < -20:
        health_parts.append("significant dip ({:.0f}%)".format(drawdown))
        dd_health = "concerning"
    elif drawdown < -10:
        health_parts.append("moderate pullback ({:.0f}%)".format(drawdown))
        dd_health = "okay"
    else:
        dd_health = "healthy"
    
    # Overall health
    if dd_health == "critical" or momentum_health == "poor":
        overall = "weak"
    elif momentum_health == "good" and dd_health in ("okay", "healthy"):
        overall = "strong"
    else:
        overall = "mixed"
    
    summary = "; ".join(health_parts) if health_parts else "neutral conditions"
    
    return overall, summary


# =============================================================================
# Sector-Specific Analysis Functions
# =============================================================================


def analyze_bank(
    symbol: str,
    prices: pd.Series,
    sector_etf_prices: pd.Series | None = None,
    rate_10y: pd.Series | None = None,
    rate_2y: pd.Series | None = None,
) -> BankMetrics:
    """
    Analyze a bank/financial stock with domain-specific factors.
    
    Banks are unique:
    - Asset-sensitive: benefit from rising rates
    - Yield curve matters: steeper = better for NIM
    - Credit cycle: early cycle = loan growth, late = credit losses
    - Systemic risk: can have contagion (SVB, 2008)
    """
    returns = prices.pct_change().dropna()
    vol_regime, vol_pct = compute_volatility_regime(returns)
    
    # Correlation to sector (use XLF proxy if no data)
    if sector_etf_prices is not None and len(sector_etf_prices) >= 60:
        sector_returns = sector_etf_prices.pct_change().dropna()
        aligned = pd.concat([returns, sector_returns], axis=1, join="inner")
        if len(aligned) >= 20:
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        else:
            correlation = 0.7  # Default for banks
    else:
        correlation = 0.7
    
    # Yield curve analysis
    yield_curve_state = "normal"
    yield_curve_impact = "Neutral yield curve environment."
    
    if rate_10y is not None and rate_2y is not None:
        if len(rate_10y) > 0 and len(rate_2y) > 0:
            spread = float(rate_10y.iloc[-1] - rate_2y.iloc[-1])
            if spread < 0:
                yield_curve_state = "inverted"
                yield_curve_impact = "⚠️ Inverted yield curve pressures bank NIM. Dips may be justified."
            elif spread < 0.5:
                yield_curve_state = "flat"
                yield_curve_impact = "Flat yield curve limits NIM expansion."
            else:
                yield_curve_state = "normal"
                yield_curve_impact = "Normal yield curve supports bank profitability."
    
    # Dip interpretation for banks
    dip_causes = []
    if vol_regime in ["high", "extreme"]:
        dip_causes.append("Market stress/risk-off sentiment")
    if yield_curve_state == "inverted":
        dip_causes.append("Yield curve inversion pressuring margins")
    if not dip_causes:
        dip_causes.append("Idiosyncratic bank-specific issue or sector rotation")
    
    # Systemic risk check (high vol + high correlation = systemic)
    is_systemic = vol_regime == "extreme" and correlation > 0.8
    if is_systemic:
        dip_causes.insert(0, "⚠️ SYSTEMIC RISK - sector-wide contagion possible")
    
    # Risk level
    if is_systemic:
        risk_level = "high"
    elif yield_curve_state == "inverted" and vol_regime == "high":
        risk_level = "high"
    elif vol_regime == "high":
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Sector adjustment
    # Banks: dips during inverted curves are often justified (avoid)
    # Banks: dips during normal curves with low vol are opportunities
    if is_systemic:
        adjustment = -0.5
        adj_reason = "Systemic risk detected - avoid catching falling knife"
    elif yield_curve_state == "inverted":
        adjustment = -0.3
        adj_reason = "Inverted yield curve headwind - reduced opportunity score"
    elif yield_curve_state == "normal" and vol_regime == "low":
        adjustment = 0.2
        adj_reason = "Favorable rate environment for banks"
    else:
        adjustment = 0.0
        adj_reason = "Neutral sector environment"
    
    # Add general health to dip interpretation
    general_health, health_summary = compute_general_health(prices)
    domain_context = f"Bank dip during {yield_curve_state} yield curve. {yield_curve_impact}"
    if health_summary:
        dip_interpretation = f"General: {health_summary}. Sector: {domain_context}"
    else:
        dip_interpretation = domain_context
    
    return BankMetrics(
        sector=Sector.FINANCIALS,
        sector_name="Financials / Banks",
        volatility_regime=vol_regime,
        volatility_percentile=vol_pct,
        correlation_to_sector=correlation,
        dip_interpretation=dip_interpretation,
        typical_recovery_days=40,  # Banks recover slower
        primary_risk_factors=["Interest rates", "Credit quality", "Regulatory", "Systemic"],
        current_risk_level=risk_level,
        sector_adjustment=adjustment,
        adjustment_reason=adj_reason,
        rate_sensitivity="asset_sensitive",  # Most banks
        rate_outlook_impact="Rising rates generally positive for bank earnings",
        yield_curve_state=yield_curve_state,
        yield_curve_impact=yield_curve_impact,
        credit_cycle_phase="expansion",  # Would need macro data to determine
        dip_likely_causes=dip_causes,
        is_systemic_risk=is_systemic,
    )


def analyze_tech(
    symbol: str,
    prices: pd.Series,
    sector_etf_prices: pd.Series | None = None,
    rate_10y: pd.Series | None = None,
) -> TechMetrics:
    """
    Analyze a technology stock with domain-specific factors.
    
    Tech stocks are unique:
    - High rate sensitivity (long duration assets)
    - Growth premium can compress in risk-off
    - Sector rotation can cause dips unrelated to fundamentals
    - Winner-take-all dynamics
    """
    returns = prices.pct_change().dropna()
    vol_regime, vol_pct = compute_volatility_regime(returns)
    
    # Correlation to sector (use XLK proxy if no data)
    if sector_etf_prices is not None and len(sector_etf_prices) >= 60:
        sector_returns = sector_etf_prices.pct_change().dropna()
        aligned = pd.concat([returns, sector_returns], axis=1, join="inner")
        if len(aligned) >= 20:
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        else:
            correlation = 0.8
    else:
        correlation = 0.8  # Tech stocks highly correlated
    
    # Growth stock analysis (high beta = likely growth)
    if len(returns) >= 60:
        beta = returns.std() / (returns.mean() + 0.0001)  # Rough proxy
        is_growth = beta > 0.8 or vol_pct > 60
    else:
        is_growth = True  # Assume growth for tech
    
    # Rate sensitivity for growth stocks
    rate_sensitivity = "high"
    rate_impact = "Rising rates compress growth valuations."
    
    if rate_10y is not None and len(rate_10y) >= 60:
        rate_change_3m = float(rate_10y.iloc[-1] - rate_10y.iloc[-60])
        if rate_change_3m > 0.5:
            rate_sensitivity = "high"
            rate_impact = "⚠️ Rising rate environment pressures growth valuations."
        elif rate_change_3m < -0.3:
            rate_sensitivity = "low"
            rate_impact = "Falling rates support growth valuations."
        else:
            rate_sensitivity = "moderate"
            rate_impact = "Stable rate environment."
    
    # Sector rotation detection (tech underperforming while market up)
    is_rotation = correlation < 0.5 and vol_regime in ["normal", "low"]
    
    # Dip causes
    dip_causes = []
    if rate_sensitivity == "high":
        dip_causes.append("Growth multiple compression from rising rates")
    if is_rotation:
        dip_causes.append("Sector rotation out of tech")
    if vol_regime == "high":
        dip_causes.append("Risk-off selling of high-beta names")
    if not dip_causes:
        dip_causes.append("Stock-specific issue or earnings miss")
    
    # Growth premium at risk
    growth_premium_at_risk = rate_sensitivity == "high" and is_growth
    
    # Risk level
    if growth_premium_at_risk and vol_regime == "high":
        risk_level = "high"
    elif is_rotation:
        risk_level = "medium"  # Rotation dips often recover
    elif vol_regime == "high":
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Sector adjustment
    if growth_premium_at_risk:
        adjustment = -0.2
        adj_reason = "Growth premium compression risk in rising rate environment"
    elif is_rotation and vol_regime == "low":
        adjustment = 0.3
        adj_reason = "Sector rotation dip - often a buying opportunity"
    elif vol_regime == "low" and rate_sensitivity == "low":
        adjustment = 0.2
        adj_reason = "Favorable environment for tech"
    else:
        adjustment = 0.0
        adj_reason = "Neutral sector conditions"
    
    # Add general health to dip interpretation
    general_health, health_summary = compute_general_health(prices)
    domain_context = f"Tech dip with {rate_sensitivity} rate sensitivity. {rate_impact}"
    if health_summary:
        dip_interpretation = f"General: {health_summary}. Sector: {domain_context}"
    else:
        dip_interpretation = domain_context
    
    return TechMetrics(
        sector=Sector.TECHNOLOGY,
        sector_name="Technology",
        volatility_regime=vol_regime,
        volatility_percentile=vol_pct,
        correlation_to_sector=correlation,
        dip_interpretation=dip_interpretation,
        typical_recovery_days=25,  # Tech recovers faster
        primary_risk_factors=["Interest rates", "Growth expectations", "Competition", "Regulation"],
        current_risk_level=risk_level,
        sector_adjustment=adjustment,
        adjustment_reason=adj_reason,
        is_growth_stock=is_growth,
        growth_premium_at_risk=growth_premium_at_risk,
        rate_sensitivity_for_growth=rate_sensitivity,
        has_network_effects=False,  # Would need fundamental data
        has_switching_costs=False,
        dip_likely_causes=dip_causes,
        is_sector_rotation=is_rotation,
    )


def analyze_healthcare(
    symbol: str,
    prices: pd.Series,
    sector_etf_prices: pd.Series | None = None,
    subsector: str | None = None,
) -> HealthcareMetrics:
    """
    Analyze a healthcare stock with domain-specific factors.
    
    Healthcare is unique:
    - Defensive in downturns (stable demand)
    - Binary risk for biotech (FDA decisions)
    - Patent cliffs for pharma
    - Reimbursement/political risk
    """
    returns = prices.pct_change().dropna()
    vol_regime, vol_pct = compute_volatility_regime(returns)
    
    # Determine subsector
    if subsector is None:
        # Infer from volatility (biotech = high vol, pharma = lower)
        if vol_pct > 70:
            subsector = "biotech"
        elif vol_pct > 50:
            subsector = "pharma"
        else:
            subsector = "managed_care"
    
    # Correlation to sector
    if sector_etf_prices is not None and len(sector_etf_prices) >= 60:
        sector_returns = sector_etf_prices.pct_change().dropna()
        aligned = pd.concat([returns, sector_returns], axis=1, join="inner")
        if len(aligned) >= 20:
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        else:
            correlation = 0.6
    else:
        correlation = 0.6
    
    # Binary risk (biotech primarily)
    has_binary_catalyst = subsector == "biotech" and vol_regime in ["high", "extreme"]
    binary_desc = "Potential FDA decision or trial results" if has_binary_catalyst else ""
    
    # Defensive characteristics
    is_defensive = subsector in ["managed_care", "pharma"] and vol_regime == "low"
    
    # Dip causes
    dip_causes = []
    if has_binary_catalyst:
        dip_causes.append("Binary catalyst risk (trial/FDA)")
    if subsector == "pharma":
        dip_causes.append("Patent cliff or pipeline concerns")
    if vol_regime == "high" and not has_binary_catalyst:
        dip_causes.append("Political/regulatory headline")
    if not dip_causes:
        dip_causes.append("Company-specific issue or sector rotation")
    
    # Risk level
    if has_binary_catalyst:
        risk_level = "high"
    elif subsector == "biotech" and vol_regime == "high":
        risk_level = "high"
    elif is_defensive:
        risk_level = "low"
    else:
        risk_level = "medium"
    
    # Sector adjustment
    if has_binary_catalyst:
        adjustment = -0.3
        adj_reason = "Binary catalyst pending - high uncertainty"
    elif is_defensive and vol_regime == "low":
        adjustment = 0.2
        adj_reason = "Defensive healthcare with stable demand"
    elif subsector == "biotech":
        adjustment = -0.1
        adj_reason = "Biotech carries elevated binary risk"
    else:
        adjustment = 0.0
        adj_reason = "Standard healthcare risk profile"
    
    # Recovery time varies by subsector
    if subsector == "biotech":
        recovery_days = 60  # Biotech can take longer
    elif subsector == "pharma":
        recovery_days = 35
    else:
        recovery_days = 25
    
    # Add general health to dip interpretation
    general_health, health_summary = compute_general_health(prices)
    domain_context = f"{subsector.title()} dip. {'High binary risk!' if has_binary_catalyst else 'Standard sector dynamics.'}"
    if health_summary:
        dip_interpretation = f"General: {health_summary}. Sector: {domain_context}"
    else:
        dip_interpretation = domain_context
    
    return HealthcareMetrics(
        sector=Sector.HEALTHCARE,
        sector_name=f"Healthcare / {subsector.title()}",
        volatility_regime=vol_regime,
        volatility_percentile=vol_pct,
        correlation_to_sector=correlation,
        dip_interpretation=dip_interpretation,
        typical_recovery_days=recovery_days,
        primary_risk_factors=["FDA/Regulatory", "Pipeline", "Reimbursement", "Patent cliffs"],
        current_risk_level=risk_level,
        sector_adjustment=adjustment,
        adjustment_reason=adj_reason,
        subsector=subsector,
        has_binary_catalyst=has_binary_catalyst,
        binary_event_description=binary_desc,
        is_defensive=is_defensive,
        dip_likely_causes=dip_causes,
        is_drug_specific_risk=subsector in ["biotech", "pharma"],
    )


def analyze_energy(
    symbol: str,
    prices: pd.Series,
    sector_etf_prices: pd.Series | None = None,
    oil_prices: pd.Series | None = None,
) -> EnergyMetrics:
    """
    Analyze an energy stock with domain-specific factors.
    
    Energy is unique:
    - Directly tied to commodity prices
    - Cyclical with global demand
    - Geopolitical risk
    - Transition risk (renewables)
    """
    returns = prices.pct_change().dropna()
    vol_regime, vol_pct = compute_volatility_regime(returns)
    
    # Correlation to sector
    if sector_etf_prices is not None and len(sector_etf_prices) >= 60:
        sector_returns = sector_etf_prices.pct_change().dropna()
        aligned = pd.concat([returns, sector_returns], axis=1, join="inner")
        if len(aligned) >= 20:
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        else:
            correlation = 0.75
    else:
        correlation = 0.75
    
    # Commodity analysis
    commodity_trend = "neutral"
    is_commodity_driven = False
    
    if oil_prices is not None and len(oil_prices) >= 60:
        oil_change_3m = (oil_prices.iloc[-1] / oil_prices.iloc[-60] - 1) * 100
        if oil_change_3m > 10:
            commodity_trend = "bullish"
        elif oil_change_3m < -10:
            commodity_trend = "bearish"
            is_commodity_driven = True
        
        # Check correlation to oil
        oil_returns = oil_prices.pct_change().dropna()
        aligned = pd.concat([returns, oil_returns], axis=1, join="inner")
        if len(aligned) >= 20:
            oil_corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            is_commodity_driven = is_commodity_driven or oil_corr > 0.6
    
    # Dip causes
    dip_causes = []
    if commodity_trend == "bearish":
        dip_causes.append("Oil/gas price decline")
    if vol_regime == "high":
        dip_causes.append("Macro/geopolitical uncertainty")
    if correlation > 0.8 and vol_regime == "high":
        dip_causes.append("Sector-wide selloff")
    if not dip_causes:
        dip_causes.append("Company-specific issue")
    
    # Risk level
    if commodity_trend == "bearish" and vol_regime == "high":
        risk_level = "high"
    elif is_commodity_driven:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Sector adjustment
    if commodity_trend == "bearish":
        adjustment = -0.3
        adj_reason = "Commodity headwinds - dip may be fundamental"
    elif commodity_trend == "bullish" and vol_regime != "high":
        adjustment = 0.3
        adj_reason = "Commodity tailwind - dip likely opportunity"
    else:
        adjustment = 0.0
        adj_reason = "Neutral commodity environment"
    
    # Add general health to dip interpretation
    general_health, health_summary = compute_general_health(prices)
    domain_context = f"Energy dip with {commodity_trend} commodity trend."
    if health_summary:
        dip_interpretation = f"General: {health_summary}. Sector: {domain_context}"
    else:
        dip_interpretation = domain_context
    
    return EnergyMetrics(
        sector=Sector.ENERGY,
        sector_name="Energy",
        volatility_regime=vol_regime,
        volatility_percentile=vol_pct,
        correlation_to_sector=correlation,
        dip_interpretation=dip_interpretation,
        typical_recovery_days=35,
        primary_risk_factors=["Commodity prices", "Geopolitical", "Demand/supply", "Transition risk"],
        current_risk_level=risk_level,
        sector_adjustment=adjustment,
        adjustment_reason=adj_reason,
        primary_commodity="oil",
        commodity_trend=commodity_trend,
        breakeven_context="Unknown",  # Would need fundamental data
        dip_likely_causes=dip_causes,
        is_commodity_driven=is_commodity_driven,
    )


def analyze_generic(
    symbol: str,
    prices: pd.Series,
    sector: Sector,
    sector_etf_prices: pd.Series | None = None,
) -> DomainMetrics:
    """Generic analysis for sectors without specific logic."""
    returns = prices.pct_change().dropna()
    vol_regime, vol_pct = compute_volatility_regime(returns)
    
    # Correlation to sector
    if sector_etf_prices is not None and len(sector_etf_prices) >= 60:
        sector_returns = sector_etf_prices.pct_change().dropna()
        aligned = pd.concat([returns, sector_returns], axis=1, join="inner")
        if len(aligned) >= 20:
            correlation = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        else:
            correlation = 0.6
    else:
        correlation = 0.6
    
    # Generic risk assessment
    if vol_regime == "extreme":
        risk_level = "high"
    elif vol_regime == "high":
        risk_level = "medium"
    else:
        risk_level = "low"
    
    sector_names = {
        Sector.CONSUMER_DISCRETIONARY: "Consumer Discretionary",
        Sector.CONSUMER_STAPLES: "Consumer Staples",
        Sector.INDUSTRIALS: "Industrials",
        Sector.MATERIALS: "Materials",
        Sector.REAL_ESTATE: "Real Estate",
        Sector.UTILITIES: "Utilities",
        Sector.COMMUNICATION: "Communication Services",
        Sector.UNKNOWN: "Unknown Sector",
    }
    
    # Add general health to dip interpretation
    general_health, health_summary = compute_general_health(prices)
    sector_display = sector_names.get(sector, 'this sector')
    if health_summary:
        dip_interpretation = f"General: {health_summary}. Standard analysis for {sector_display}."
    else:
        dip_interpretation = f"Standard dip analysis for {sector_display}."
    
    return DomainMetrics(
        sector=sector,
        sector_name=sector_names.get(sector, "Unknown"),
        volatility_regime=vol_regime,
        volatility_percentile=vol_pct,
        correlation_to_sector=correlation,
        dip_interpretation=dip_interpretation,
        typical_recovery_days=30,
        primary_risk_factors=["Market risk", "Sector risk", "Company-specific"],
        current_risk_level=risk_level,
        sector_adjustment=0.0,
        adjustment_reason="No sector-specific adjustment applied",
    )


# =============================================================================
# Main Analysis Interface
# =============================================================================


def perform_domain_analysis(
    symbol: str,
    prices: pd.Series,
    sector_name: str | None,
    sector_etf_prices: pd.Series | None = None,
    rate_10y: pd.Series | None = None,
    rate_2y: pd.Series | None = None,
    oil_prices: pd.Series | None = None,
) -> DomainAnalysis:
    """
    Perform domain-specific analysis for a stock.
    
    Routes to appropriate sector-specific analyzer and returns
    comprehensive domain analysis including dip interpretation
    and score adjustments.
    """
    sector = normalize_sector(sector_name)
    
    # Route to sector-specific analyzer
    if sector == Sector.FINANCIALS:
        metrics = analyze_bank(symbol, prices, sector_etf_prices, rate_10y, rate_2y)
    elif sector == Sector.TECHNOLOGY:
        metrics = analyze_tech(symbol, prices, sector_etf_prices, rate_10y)
    elif sector == Sector.HEALTHCARE:
        metrics = analyze_healthcare(symbol, prices, sector_etf_prices)
    elif sector == Sector.ENERGY:
        metrics = analyze_energy(symbol, prices, sector_etf_prices, oil_prices)
    else:
        metrics = analyze_generic(symbol, prices, sector, sector_etf_prices)
    
    # Build sector signals list
    sector_signals = []
    if hasattr(metrics, "dip_likely_causes"):
        for cause in metrics.dip_likely_causes:  # type: ignore
            sector_signals.append({
                "name": cause,
                "type": "risk_factor",
                "severity": "warning" if "⚠️" in cause else "info",
            })
    
    # Calculate performance vs sector
    vs_sector = 0.0
    sector_rank = 50.0
    if sector_etf_prices is not None and len(sector_etf_prices) >= 20 and len(prices) >= 20:
        stock_ret = (prices.iloc[-1] / prices.iloc[-20] - 1) * 100
        sector_ret = (sector_etf_prices.iloc[-1] / sector_etf_prices.iloc[-20] - 1) * 100
        vs_sector = stock_ret - sector_ret
    
    # Compile warnings
    warnings = []
    if metrics.current_risk_level == "high":
        warnings.append(f"High risk level detected for {metrics.sector_name}")
    if metrics.volatility_regime == "extreme":
        warnings.append("Extreme volatility regime - exercise caution")
    if hasattr(metrics, "is_systemic_risk") and metrics.is_systemic_risk:  # type: ignore
        warnings.append("⚠️ SYSTEMIC RISK: Sector-wide contagion possible")
    if hasattr(metrics, "has_binary_catalyst") and metrics.has_binary_catalyst:  # type: ignore
        warnings.append("Binary catalyst pending - high uncertainty")
    
    return DomainAnalysis(
        symbol=symbol,
        sector=sector,
        sector_display_name=metrics.sector_name,
        metrics=metrics,
        dip_context=metrics.dip_interpretation,
        dip_adjustment=metrics.sector_adjustment,
        sector_signals=sector_signals,
        vs_sector_performance=vs_sector,
        sector_rank_percentile=sector_rank,
        final_adjustment=metrics.sector_adjustment,
        adjustment_explanation=metrics.adjustment_reason,
        domain_warnings=warnings,
    )


def domain_analysis_to_dict(analysis: DomainAnalysis) -> dict[str, Any]:
    """Convert DomainAnalysis to dictionary for API response."""
    return {
        "symbol": analysis.symbol,
        "sector": analysis.sector.value,
        "sector_display_name": analysis.sector_display_name,
        "dip_context": analysis.dip_context,
        "dip_adjustment": analysis.dip_adjustment,
        "adjustment_explanation": analysis.adjustment_explanation,
        "sector_signals": analysis.sector_signals,
        "vs_sector_performance": round(analysis.vs_sector_performance, 2),
        "sector_rank_percentile": round(analysis.sector_rank_percentile, 1),
        "volatility_regime": analysis.metrics.volatility_regime,
        "volatility_percentile": round(analysis.metrics.volatility_percentile, 1),
        "correlation_to_sector": round(analysis.metrics.correlation_to_sector, 2),
        "risk_level": analysis.metrics.current_risk_level,
        "typical_recovery_days": analysis.metrics.typical_recovery_days,
        "primary_risk_factors": analysis.metrics.primary_risk_factors,
        "domain_warnings": analysis.domain_warnings,
    }
