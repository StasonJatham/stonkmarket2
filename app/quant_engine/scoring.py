"""
Dual-Mode Scoring Pipeline (APUS + DOUS)

Implements the complete scoring specification:
- Mode A (APUS): Certified Buy - statistically proven edge over benchmarks
- Mode B (DOUS): Dip Entry - fundamental + technical opportunity scoring

Key Features:
- Stationary bootstrap (Politis & Romano) for P(edge > 0) and CI
- Deflated Sharpe ratio (Lopez de Prado) with effective N for multiple testing
- Walk-forward OOS with embargo (no look-ahead)
- Compounded equity curves for all benchmarks
- Adjusted Close everywhere

Author: Quant Engine Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SCORING_VERSION = "1.0.0"

@dataclass
class ScoringConfig:
    """Configuration for the scoring pipeline."""
    
    # Walk-forward settings
    n_folds: int = 5  # K folds for walk-forward
    embargo_days: int = 5  # Gap between train and test
    min_train_days: int = 252  # Minimum training period
    
    # Bootstrap settings
    n_bootstrap: int = 2000  # Number of bootstrap samples
    block_length: int = 20  # Average block length for stationary bootstrap
    confidence_level: float = 0.95  # For CI calculation
    
    # Mode A gate thresholds
    min_p_outperf: float = 0.75  # P(edge > 0) threshold
    min_ci_low: float = 0.0  # CI lower bound must be > 0
    min_dsr: float = 0.50  # Deflated Sharpe threshold
    
    # Trading costs
    transaction_cost_bps: float = 10.0  # 10 bps per trade
    slippage_bps: float = 5.0  # 5 bps slippage
    
    # Dip entry settings
    dip_holding_days: int = 60  # Days to measure recovery
    event_window_days: int = 7  # Earnings/dividend warning window
    
    def config_hash(self) -> str:
        """Generate hash of configuration for versioning."""
        config_str = f"{self.n_folds}:{self.embargo_days}:{self.n_bootstrap}:{self.block_length}"
        config_str += f":{self.min_p_outperf}:{self.min_dsr}:{self.transaction_cost_bps}"
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


DEFAULT_CONFIG = ScoringConfig()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvidenceBlock:
    """Complete evidence block for transparency and auditability."""
    
    # Statistical validation
    p_outperf: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    dsr: float = 0.0
    psr: float = 0.0  # Probabilistic Sharpe Ratio
    
    # Edge metrics
    median_edge: float = 0.0
    mean_edge: float = 0.0
    edge_vs_stock: float = 0.0
    edge_vs_spy: float = 0.0
    worst_regime_edge: float = 0.0
    cvar_5: float = 0.0  # Conditional VaR at 5%
    
    # Sharpe metrics
    observed_sharpe: float = 0.0
    sr_max: float = 0.0  # Expected max Sharpe under null
    n_effective: float = 1.0  # Effective number of strategies tested
    
    # Regime edges
    edge_bull: float = 0.0
    edge_bear: float = 0.0
    edge_high_vol: float = 0.0
    
    # Stability
    sharpe_degradation: float = 0.0
    n_trades: int = 0
    
    # Fundamental metrics
    fund_mom: float = 0.0
    val_z: float = 0.0
    event_risk: bool = False
    
    # Dip metrics
    p_recovery: float = 0.0
    expected_value: float = 0.0
    sector_relative: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "p_outperf": round(self.p_outperf, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "dsr": round(self.dsr, 4),
            "psr": round(self.psr, 4),
            "median_edge": round(self.median_edge, 4),
            "mean_edge": round(self.mean_edge, 4),
            "edge_vs_stock": round(self.edge_vs_stock, 4),
            "edge_vs_spy": round(self.edge_vs_spy, 4),
            "worst_regime_edge": round(self.worst_regime_edge, 4),
            "cvar_5": round(self.cvar_5, 4),
            "observed_sharpe": round(self.observed_sharpe, 4),
            "sr_max": round(self.sr_max, 4),
            "n_effective": round(self.n_effective, 2),
            "edge_bull": round(self.edge_bull, 4),
            "edge_bear": round(self.edge_bear, 4),
            "edge_high_vol": round(self.edge_high_vol, 4),
            "sharpe_degradation": round(self.sharpe_degradation, 4),
            "n_trades": self.n_trades,
            "fund_mom": round(self.fund_mom, 4),
            "val_z": round(self.val_z, 4),
            "event_risk": self.event_risk,
            "p_recovery": round(self.p_recovery, 4),
            "expected_value": round(self.expected_value, 4),
            "sector_relative": round(self.sector_relative, 4),
        }


@dataclass
class ScoringResult:
    """Result of the dual-mode scoring pipeline."""
    
    symbol: str
    best_score: float  # 0-100
    mode: str  # "CERTIFIED_BUY" or "DIP_ENTRY"
    score_a: float  # Mode A score
    score_b: float  # Mode B score
    gate_pass: bool  # Did Mode A gate pass?
    
    evidence: EvidenceBlock = field(default_factory=EvidenceBlock)
    
    # Metadata
    config_hash: str = ""
    scoring_version: str = SCORING_VERSION
    data_start: date | None = None
    data_end: date | None = None
    computed_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# STATIONARY BOOTSTRAP (Politis & Romano, 1994)
# =============================================================================

def stationary_bootstrap(
    data: np.ndarray,
    n_samples: int = 2000,
    block_length: int = 20,
    seed: int | None = None,
) -> np.ndarray:
    """
    Stationary bootstrap for dependent time series data.
    
    Implements Politis & Romano (1994) stationary bootstrap where
    block lengths follow a geometric distribution.
    
    Args:
        data: 1D array of observations (e.g., edge values)
        n_samples: Number of bootstrap samples (B)
        block_length: Average block length (L), p = 1/L
        seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples,) containing bootstrap statistics (means)
    """
    rng = np.random.default_rng(seed)
    T = len(data)
    p = 1.0 / block_length  # Probability of starting new block
    
    bootstrap_means = np.zeros(n_samples)
    
    for b in range(n_samples):
        # Generate bootstrap sample
        sample = np.zeros(T)
        i = rng.integers(0, T)  # Random start
        
        for t in range(T):
            sample[t] = data[i]
            # With probability p, start new block
            if rng.random() < p:
                i = rng.integers(0, T)
            else:
                i = (i + 1) % T  # Wrap around
        
        bootstrap_means[b] = np.mean(sample)
    
    return bootstrap_means


def compute_bootstrap_stats(
    edge_values: np.ndarray,
    n_samples: int = 2000,
    block_length: int = 20,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """
    Compute bootstrap statistics for edge values.
    
    Returns:
        Tuple of (p_outperf, ci_low, ci_high, cvar_5)
    """
    if len(edge_values) < 10:
        return 0.0, 0.0, 0.0, 0.0
    
    bootstrap_means = stationary_bootstrap(
        edge_values, n_samples, block_length, seed
    )
    
    # P(edge > 0): proportion of bootstrap means > 0
    p_outperf = float(np.mean(bootstrap_means > 0))
    
    # Confidence interval
    alpha = 1 - confidence
    ci_low = float(np.percentile(bootstrap_means, alpha / 2 * 100))
    ci_high = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
    
    # CVaR at 5% (Conditional Value at Risk)
    var_5 = np.percentile(bootstrap_means, 5)
    cvar_5 = float(np.mean(bootstrap_means[bootstrap_means <= var_5]))
    
    return p_outperf, ci_low, ci_high, cvar_5


# =============================================================================
# DEFLATED SHARPE RATIO (Lopez de Prado)
# =============================================================================

def compute_effective_n(returns_matrix: np.ndarray) -> float:
    """
    Compute effective number of independent strategies.
    
    N_eff = (sum λ_i)² / sum(λ_i²)
    where λ_i are eigenvalues of return correlation matrix.
    
    Args:
        returns_matrix: (T, N) matrix of strategy returns
        
    Returns:
        Effective N (1 <= N_eff <= N)
    """
    if returns_matrix.ndim == 1:
        return 1.0
    
    n_strategies = returns_matrix.shape[1]
    if n_strategies <= 1:
        return 1.0
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(returns_matrix.T)
    
    # Handle NaN/Inf
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)  # Numerical stability
    
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues ** 2)
    
    if sum_lambda_sq == 0:
        return 1.0
    
    n_eff = (sum_lambda ** 2) / sum_lambda_sq
    return float(max(1.0, min(n_eff, n_strategies)))


def compute_probabilistic_sharpe(
    observed_sharpe: float,
    benchmark_sharpe: float,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """
    Compute Probabilistic Sharpe Ratio (PSR).
    
    PSR = Φ((SR_hat - SR0) * sqrt(n-1) / 
            sqrt(1 - γ3*SR_hat + (γ4-1)/4 * SR_hat²))
    
    Args:
        observed_sharpe: Annualized Sharpe ratio
        benchmark_sharpe: Benchmark Sharpe (SR0)
        n_obs: Number of observations
        skewness: γ3 (sample skewness)
        kurtosis: γ4 (sample kurtosis)
        
    Returns:
        PSR in [0, 1]
    """
    if n_obs <= 1:
        return 0.0
    
    sr = observed_sharpe
    sr0 = benchmark_sharpe
    
    # Denominator (standard error adjustment)
    denom_sq = 1 - skewness * sr + (kurtosis - 1) / 4 * (sr ** 2)
    if denom_sq <= 0:
        denom_sq = 1.0
    
    z = (sr - sr0) * np.sqrt(n_obs - 1) / np.sqrt(denom_sq)
    
    # PSR = Φ(z)
    psr = float(stats.norm.cdf(z))
    return psr


def compute_expected_max_sharpe(
    mean_sharpe: float,
    std_sharpe: float,
    n_effective: float,
) -> float:
    """
    Compute expected maximum Sharpe ratio under null hypothesis.
    
    SR_max = μ_SR + σ_SR * Φ^{-1}((N_eff - 0.375)/(N_eff + 0.25))
    
    Args:
        mean_sharpe: Mean Sharpe across strategies
        std_sharpe: Std of Sharpe across strategies
        n_effective: Effective number of strategies
        
    Returns:
        Expected max Sharpe
    """
    if n_effective <= 1:
        return mean_sharpe
    
    # Approximation for expected max of N_eff standard normals
    quantile_arg = (n_effective - 0.375) / (n_effective + 0.25)
    quantile_arg = min(max(quantile_arg, 0.001), 0.999)  # Bound for stability
    
    z_max = float(stats.norm.ppf(quantile_arg))
    sr_max = mean_sharpe + std_sharpe * z_max
    
    return sr_max


def compute_deflated_sharpe(
    returns: np.ndarray,
    n_strategies_tested: int = 1,
    mean_sharpe_null: float = 0.0,
    std_sharpe_null: float = 0.5,
) -> tuple[float, float, float, float]:
    """
    Compute Deflated Sharpe Ratio (DSR).
    
    DSR = PSR with SR0 = SR_max (expected max under null)
    
    Args:
        returns: Array of strategy returns
        n_strategies_tested: Total strategies tested
        mean_sharpe_null: Mean Sharpe under null
        std_sharpe_null: Std of Sharpe under null
        
    Returns:
        Tuple of (dsr, observed_sharpe, sr_max, n_effective)
    """
    if len(returns) < 20:
        return 0.0, 0.0, 0.0, 1.0
    
    # Observed Sharpe (annualized)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret == 0:
        return 0.0, 0.0, 0.0, 1.0
    
    observed_sharpe = float(mean_ret / std_ret * np.sqrt(252))
    
    # Higher moments
    skewness = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns, fisher=False))  # Excess kurtosis + 3
    
    # Effective N (for single strategy, use n_strategies_tested)
    n_effective = float(n_strategies_tested)
    
    # Expected max Sharpe under null
    sr_max = compute_expected_max_sharpe(mean_sharpe_null, std_sharpe_null, n_effective)
    
    # DSR = PSR with SR0 = SR_max
    dsr = compute_probabilistic_sharpe(
        observed_sharpe, sr_max, len(returns), skewness, kurtosis
    )
    
    return dsr, observed_sharpe, sr_max, n_effective


# =============================================================================
# WALK-FORWARD OOS WITH EMBARGO
# =============================================================================

def walk_forward_split(
    n_obs: int,
    n_folds: int = 5,
    embargo_days: int = 5,
    min_train: int = 252,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward splits with purging and embargo.
    
    Args:
        n_obs: Total observations
        n_folds: Number of folds
        embargo_days: Gap between train and test
        min_train: Minimum training observations
        
    Returns:
        List of (train_idx, test_idx) tuples
    """
    splits = []
    
    # Expanding window approach
    test_size = (n_obs - min_train) // n_folds
    if test_size < 20:
        # Not enough data for walk-forward
        return []
    
    for k in range(n_folds):
        train_end = min_train + k * test_size
        test_start = train_end + embargo_days
        test_end = test_start + test_size
        
        if test_end > n_obs:
            test_end = n_obs
        
        if test_start >= test_end:
            continue
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        
        splits.append((train_idx, test_idx))
    
    return splits


# =============================================================================
# EQUITY CURVE CALCULATION
# =============================================================================

def compute_equity_curve(
    returns: np.ndarray,
    weights: np.ndarray,
    cost_per_trade: float = 0.0015,
) -> np.ndarray:
    """
    Compute compounded equity curve.
    
    E_t = E_{t-1} * (1 + w_{t-1} * r_t - cost_t)
    
    Args:
        returns: Asset returns (T,)
        weights: Position weights (T,), 1 = long, 0 = flat, -1 = short
        cost_per_trade: Transaction cost per trade (as decimal)
        
    Returns:
        Equity curve (T+1,) starting at 1.0
    """
    T = len(returns)
    equity = np.ones(T + 1)
    
    prev_weight = 0.0
    for t in range(T):
        # Strategy return
        strat_return = weights[t] * returns[t]
        
        # Transaction cost (on weight change)
        weight_change = abs(weights[t] - prev_weight)
        cost = weight_change * cost_per_trade
        
        equity[t + 1] = equity[t] * (1 + strat_return - cost)
        prev_weight = weights[t]
    
    return equity


def compute_benchmark_return(prices: pd.Series) -> float:
    """Compute compounded buy-and-hold return."""
    if len(prices) < 2:
        return 0.0
    return float((prices.iloc[-1] / prices.iloc[0]) - 1)


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_regimes(
    returns: pd.Series,
    prices: pd.Series,
) -> dict[str, np.ndarray]:
    """
    Detect market regimes: bull, bear, high-vol.
    
    Returns:
        Dict mapping regime name to boolean mask (aligned with returns length)
    """
    n_returns = len(returns)
    
    if n_returns < 60:
        return {
            "bull": np.ones(n_returns, dtype=bool),
            "bear": np.zeros(n_returns, dtype=bool),
            "high_vol": np.zeros(n_returns, dtype=bool),
        }
    
    # Rolling metrics on returns
    rolling_vol = returns.rolling(60).std() * np.sqrt(252)
    median_vol = rolling_vol.median()
    
    # 200-day SMA for trend - align prices with returns
    # Prices has one more element than returns, so we use prices[1:] to align
    # This gives us the price at each return's date (returns[i] = prices[i+1]/prices[i] - 1)
    aligned_prices = prices.iloc[1:] if len(prices) > n_returns else prices
    aligned_prices = aligned_prices.iloc[-n_returns:].reset_index(drop=True)
    
    sma_200 = aligned_prices.rolling(200).mean()
    
    # Regimes - all masks now have length n_returns
    bull = (aligned_prices > sma_200).fillna(True).values
    bear = (aligned_prices < sma_200).fillna(False).values
    high_vol = (rolling_vol > median_vol * 1.5).fillna(False).values
    
    return {
        "bull": bull,
        "bear": bear,
        "high_vol": high_vol,
    }


def compute_regime_edges(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    regimes: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute edge in each regime."""
    edges = {}
    
    for regime_name, mask in regimes.items():
        if mask.sum() < 20:
            edges[regime_name] = 0.0
            continue
        
        strat_ret = strategy_returns[mask]
        bench_ret = benchmark_returns[mask]
        
        strat_total = float(np.prod(1 + strat_ret) - 1)
        bench_total = float(np.prod(1 + bench_ret) - 1)
        
        edges[regime_name] = strat_total - bench_total
    
    return edges


# =============================================================================
# MODE A: CERTIFIED BUY (APUS) SCORING
# =============================================================================

def compute_mode_a_score(
    p_outperf: float,
    median_edge: float,
    dsr: float,
    cvar_5: float,
    worst_regime_edge: float,
    sharpe_degradation: float,
) -> float:
    """
    Compute Mode A (APUS) score.
    
    Normalized scores:
    - S_prob = clip((P_outperf - 0.50)/0.35, 0, 1)
    - S_med = clip(median(edge)/0.15, 0, 1)
    - S_tail = clip(1 - |CVaR_5%|/0.15, 0, 1)
    - S_dsr = clip(DSR/0.60, 0, 1)
    - S_reg = clip(R/0.10, 0, 1)
    - S_stab = clip((1 - sharpe_degradation)/0.50, 0, 1)
    
    Score_A = 100 * (0.35*S_prob + 0.25*S_med + 0.15*S_dsr +
                     0.10*S_tail + 0.10*S_reg + 0.05*S_stab)
    """
    # Normalized components
    s_prob = np.clip((p_outperf - 0.50) / 0.35, 0, 1)
    s_med = np.clip(median_edge / 0.15, 0, 1)
    s_tail = np.clip(1 - abs(cvar_5) / 0.15, 0, 1)
    s_dsr = np.clip(dsr / 0.60, 0, 1)
    s_reg = np.clip(worst_regime_edge / 0.10, 0, 1)
    s_stab = np.clip((1 - sharpe_degradation) / 0.50, 0, 1)
    
    # Weighted combination
    score_a = 100 * (
        0.35 * s_prob +
        0.25 * s_med +
        0.15 * s_dsr +
        0.10 * s_tail +
        0.10 * s_reg +
        0.05 * s_stab
    )
    
    return float(score_a)


def check_mode_a_gate(
    p_outperf: float,
    ci_low: float,
    dsr: float,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> bool:
    """
    Check if Mode A gate passes.
    
    Gate requirements:
    - P_outperf >= 0.75
    - CI_low > 0
    - DSR >= 0.50
    """
    return (
        p_outperf >= config.min_p_outperf and
        ci_low > config.min_ci_low and
        dsr >= config.min_dsr
    )


# =============================================================================
# MODE B: DIP ENTRY (DOUS) SCORING
# =============================================================================

def compute_fundamental_momentum(
    revenue_z: float,
    earnings_z: float,
    margin_z: float,
) -> float:
    """
    Compute fundamental momentum score.
    
    F_mom = sigmoid(0.4*z_revenue + 0.4*z_earnings + 0.2*z_margin)
    """
    z_combined = 0.4 * revenue_z + 0.4 * earnings_z + 0.2 * margin_z
    return float(1 / (1 + np.exp(-z_combined)))


def compute_valuation_z(
    pe_z: float,
    ev_ebitda_z: float,
    ps_z: float,
) -> float:
    """
    Compute valuation z-score.
    
    Val_z = sigmoid(-(z_PE + z_EV_EBITDA + z_P_Sales)/3)
    """
    z_combined = -(pe_z + ev_ebitda_z + ps_z) / 3
    return float(1 / (1 + np.exp(-z_combined)))


def compute_mode_b_score(
    p_recovery: float,
    expected_value: float,
    fund_mom: float,
    val_z: float,
    sector_relative: float,
    event_risk: bool,
) -> float:
    """
    Compute Mode B (DOUS) score.
    
    Normalized:
    - S_rec = clip((P_rec - 0.50)/0.30, 0, 1)
    - S_ev = clip(EV/0.08, 0, 1)
    - S_fund = F_mom
    - S_val = Val_z
    - S_sec = Sec_rel
    
    Score_B = 100 * (0.30*S_rec + 0.25*S_ev + 0.15*S_fund +
                     0.10*S_val + 0.10*S_sec + 0.10*(1-Event))
    """
    s_rec = np.clip((p_recovery - 0.50) / 0.30, 0, 1)
    s_ev = np.clip(expected_value / 0.08, 0, 1)
    s_fund = fund_mom
    s_val = val_z
    s_sec = sector_relative
    s_event = 0.0 if event_risk else 1.0
    
    score_b = 100 * (
        0.30 * s_rec +
        0.25 * s_ev +
        0.15 * s_fund +
        0.10 * s_val +
        0.10 * s_sec +
        0.10 * s_event
    )
    
    return float(score_b)


# =============================================================================
# DIP ANALYSIS FOR MODE B
# =============================================================================

def analyze_dip_recovery(
    prices: pd.Series,
    current_drawdown: float,
    holding_days: int = 60,
) -> tuple[float, float]:
    """
    Analyze historical dip recovery statistics.
    
    Returns:
        Tuple of (p_recovery, expected_value)
    """
    if len(prices) < 252:
        return 0.5, 0.0
    
    # Find historical dips of similar magnitude
    returns = prices.pct_change().dropna()
    rolling_max = prices.rolling(60).max()
    drawdowns = (prices - rolling_max) / rolling_max
    
    # Dips within 20% of current drawdown magnitude
    dip_threshold = current_drawdown * 0.8
    similar_dips = drawdowns[drawdowns <= dip_threshold]
    
    if len(similar_dips) < 5:
        return 0.5, 0.0
    
    # Analyze recovery
    recoveries = []
    non_recoveries = []
    
    for idx in similar_dips.index:
        loc = prices.index.get_loc(idx)
        future_end = min(loc + holding_days, len(prices) - 1)
        
        if future_end <= loc:
            continue
        
        future_return = float((prices.iloc[future_end] / prices.iloc[loc]) - 1)
        
        if future_return > 0:
            recoveries.append(future_return)
        else:
            non_recoveries.append(future_return)
    
    n_total = len(recoveries) + len(non_recoveries)
    if n_total < 5:
        return 0.5, 0.0
    
    p_recovery = len(recoveries) / n_total
    avg_up = np.mean(recoveries) if recoveries else 0.0
    avg_dn = np.mean(non_recoveries) if non_recoveries else 0.0
    
    # Expected value: EV = P_rec * avg_up - (1-P_rec) * avg_dn
    ev = p_recovery * avg_up - (1 - p_recovery) * abs(avg_dn)
    
    return float(p_recovery), float(ev)


def compute_sector_relative(
    stock_drawdown: float,
    sector_drawdown: float,
) -> float:
    """
    Compute sector-relative drawdown percentile.
    
    Returns value in [0, 1] where higher = stock oversold vs sector.
    """
    if sector_drawdown == 0:
        return 0.5
    
    # Ratio of stock drawdown to sector drawdown
    ratio = stock_drawdown / sector_drawdown if sector_drawdown != 0 else 1.0
    
    # Convert to percentile (sigmoid transformation)
    # If stock drawdown >> sector drawdown, returns high value
    return float(1 / (1 + np.exp(-(ratio - 1) * 3)))


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def compute_symbol_score(
    symbol: str,
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    strategy_weights: pd.Series,
    fundamentals: dict | None = None,
    sector_prices: pd.Series | None = None,
    earnings_date: date | None = None,
    dividend_date: date | None = None,
    n_strategies_tested: int = 1,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> ScoringResult:
    """
    Compute complete dual-mode score for a symbol.
    
    Args:
        symbol: Stock ticker
        prices: DataFrame with 'Adj Close' column
        spy_prices: SPY Adjusted Close series
        strategy_weights: Position weights from optimized strategy
        fundamentals: Dict with fundamental metrics
        sector_prices: Sector ETF prices for relative comparison
        earnings_date: Next earnings date
        dividend_date: Next dividend date
        n_strategies_tested: Number of strategies tested (for deflated Sharpe)
        config: Scoring configuration
        
    Returns:
        ScoringResult with complete evidence block
    """
    logger.info(f"Computing score for {symbol}")
    
    evidence = EvidenceBlock()
    
    # Prepare data
    if "Adj Close" in prices.columns:
        stock_prices = prices["Adj Close"]
    elif "close" in prices.columns:
        stock_prices = prices["close"]
    else:
        stock_prices = prices.iloc[:, 0]
    
    stock_prices = stock_prices.dropna()
    spy_prices = spy_prices.reindex(stock_prices.index).dropna()
    
    # Align all series
    common_idx = stock_prices.index.intersection(spy_prices.index)
    if len(common_idx) < 252:
        logger.warning(f"{symbol}: Insufficient data ({len(common_idx)} days)")
        return ScoringResult(
            symbol=symbol,
            best_score=0.0,
            mode="DIP_ENTRY",
            score_a=0.0,
            score_b=0.0,
            gate_pass=False,
            evidence=evidence,
            config_hash=config.config_hash(),
            data_start=None,
            data_end=None,
        )
    
    stock_prices = stock_prices.loc[common_idx]
    spy_prices = spy_prices.loc[common_idx]
    strategy_weights = strategy_weights.reindex(common_idx).fillna(0)
    
    # Returns
    stock_returns = stock_prices.pct_change().dropna()
    spy_returns = spy_prices.pct_change().dropna()
    
    # Align after returns computation
    common_ret_idx = stock_returns.index.intersection(spy_returns.index)
    stock_returns = stock_returns.loc[common_ret_idx]
    spy_returns = spy_returns.loc[common_ret_idx]
    strategy_weights = strategy_weights.loc[common_ret_idx]
    
    # Transaction cost (in decimal)
    cost_per_trade = (config.transaction_cost_bps + config.slippage_bps) / 10000
    
    # Strategy returns (costed)
    strategy_returns = strategy_weights.shift(1).fillna(0) * stock_returns
    
    # Apply transaction costs on weight changes
    weight_changes = strategy_weights.diff().abs().fillna(0)
    costs = weight_changes * cost_per_trade
    strategy_returns = strategy_returns - costs
    
    strategy_returns_arr = strategy_returns.values
    stock_returns_arr = stock_returns.values
    
    # Equity curves (compounded)
    strategy_equity = compute_equity_curve(
        stock_returns_arr,
        strategy_weights.values,
        cost_per_trade,
    )
    
    # Benchmark returns
    R_strat = float(strategy_equity[-1] / strategy_equity[0] - 1)
    R_stock = compute_benchmark_return(stock_prices)
    R_spy = compute_benchmark_return(spy_prices)
    
    # Edge vs benchmarks
    edge_vs_stock = R_strat - R_stock
    edge_vs_spy = R_strat - R_spy
    edge = R_strat - max(R_stock, R_spy)
    
    evidence.edge_vs_stock = edge_vs_stock
    evidence.edge_vs_spy = edge_vs_spy
    
    # Walk-forward OOS edges
    splits = walk_forward_split(
        len(stock_returns_arr),
        config.n_folds,
        config.embargo_days,
        config.min_train_days,
    )
    
    oos_edges = []
    is_sharpes = []
    oos_sharpes = []
    
    for train_idx, test_idx in splits:
        # OOS strategy return
        oos_strat_ret = strategy_returns_arr[test_idx]
        oos_stock_ret = stock_returns_arr[test_idx]
        
        # Compounded returns
        oos_strat_total = float(np.prod(1 + oos_strat_ret) - 1)
        oos_stock_total = float(np.prod(1 + oos_stock_ret) - 1)
        oos_spy_total = float(np.prod(1 + spy_returns.values[test_idx]) - 1)
        
        oos_edge = oos_strat_total - max(oos_stock_total, oos_spy_total)
        oos_edges.append(oos_edge)
        
        # Sharpe for stability check
        if len(oos_strat_ret) > 10 and np.std(oos_strat_ret) > 0:
            oos_sharpe = np.mean(oos_strat_ret) / np.std(oos_strat_ret) * np.sqrt(252)
            oos_sharpes.append(float(oos_sharpe))
        
        is_strat_ret = strategy_returns_arr[train_idx]
        if len(is_strat_ret) > 10 and np.std(is_strat_ret) > 0:
            is_sharpe = np.mean(is_strat_ret) / np.std(is_strat_ret) * np.sqrt(252)
            is_sharpes.append(float(is_sharpe))
    
    oos_edges_arr = np.array(oos_edges) if oos_edges else np.array([edge])
    
    # Stationary bootstrap on OOS edges
    p_outperf, ci_low, ci_high, cvar_5 = compute_bootstrap_stats(
        oos_edges_arr,
        config.n_bootstrap,
        config.block_length,
        config.confidence_level,
    )
    
    evidence.p_outperf = p_outperf
    evidence.ci_low = ci_low
    evidence.ci_high = ci_high
    evidence.cvar_5 = cvar_5
    evidence.median_edge = float(np.median(oos_edges_arr))
    evidence.mean_edge = float(np.mean(oos_edges_arr))
    
    # Deflated Sharpe Ratio
    dsr, observed_sharpe, sr_max, n_effective = compute_deflated_sharpe(
        strategy_returns_arr,
        n_strategies_tested,
    )
    
    evidence.dsr = dsr
    evidence.observed_sharpe = observed_sharpe
    evidence.sr_max = sr_max
    evidence.n_effective = n_effective
    
    # Sharpe degradation
    if is_sharpes and oos_sharpes:
        avg_is_sharpe = np.mean(is_sharpes)
        avg_oos_sharpe = np.mean(oos_sharpes)
        if avg_is_sharpe > 0:
            sharpe_degradation = (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe
        else:
            sharpe_degradation = 0.0
    else:
        sharpe_degradation = 0.0
    
    evidence.sharpe_degradation = float(sharpe_degradation)
    evidence.n_trades = int(weight_changes[weight_changes > 0].sum())
    
    # Regime analysis
    regimes = detect_regimes(stock_returns, stock_prices)
    regime_edges = compute_regime_edges(
        strategy_returns_arr,
        np.maximum(stock_returns_arr, spy_returns.values),
        regimes,
    )
    
    evidence.edge_bull = regime_edges.get("bull", 0.0)
    evidence.edge_bear = regime_edges.get("bear", 0.0)
    evidence.edge_high_vol = regime_edges.get("high_vol", 0.0)
    
    worst_regime_edge = min(
        regime_edges.get("bull", 0.0),
        regime_edges.get("bear", 0.0),
        regime_edges.get("high_vol", 0.0),
    )
    evidence.worst_regime_edge = worst_regime_edge
    
    # Mode A gate check
    gate_pass = check_mode_a_gate(p_outperf, ci_low, dsr, config)
    
    # Mode A score
    score_a = compute_mode_a_score(
        p_outperf,
        evidence.median_edge,
        dsr,
        cvar_5,
        worst_regime_edge,
        sharpe_degradation,
    )
    
    # Mode B scoring (always computed as fallback)
    # Current drawdown
    rolling_max = stock_prices.rolling(60).max()
    current_drawdown = float((stock_prices.iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1])
    
    # Dip analysis
    p_recovery, expected_value = analyze_dip_recovery(
        stock_prices,
        current_drawdown,
        config.dip_holding_days,
    )
    
    evidence.p_recovery = p_recovery
    evidence.expected_value = expected_value
    
    # Fundamental momentum (default to 0.5 if no data)
    fund_mom = 0.5
    val_z = 0.5
    if fundamentals:
        # Z-scores for fundamentals (simplified)
        revenue_z = fundamentals.get("revenue_z", 0.0)
        earnings_z = fundamentals.get("earnings_z", 0.0)
        margin_z = fundamentals.get("margin_z", 0.0)
        fund_mom = compute_fundamental_momentum(revenue_z, earnings_z, margin_z)
        
        pe_z = fundamentals.get("pe_z", 0.0)
        ev_ebitda_z = fundamentals.get("ev_ebitda_z", 0.0)
        ps_z = fundamentals.get("ps_z", 0.0)
        val_z = compute_valuation_z(pe_z, ev_ebitda_z, ps_z)
    
    evidence.fund_mom = fund_mom
    evidence.val_z = val_z
    
    # Sector relative
    sector_relative = 0.5
    if sector_prices is not None and len(sector_prices) > 60:
        sector_max = sector_prices.rolling(60).max().iloc[-1]
        sector_drawdown = (sector_prices.iloc[-1] - sector_max) / sector_max
        sector_relative = compute_sector_relative(current_drawdown, sector_drawdown)
    
    evidence.sector_relative = sector_relative
    
    # Event risk
    event_risk = False
    today = date.today()
    if earnings_date and (earnings_date - today).days <= config.event_window_days:
        event_risk = True
    if dividend_date and (dividend_date - today).days <= config.event_window_days:
        event_risk = True
    
    evidence.event_risk = event_risk
    
    # Mode B score
    score_b = compute_mode_b_score(
        p_recovery,
        expected_value,
        fund_mom,
        val_z,
        sector_relative,
        event_risk,
    )
    
    # Final best score
    if gate_pass:
        best_score = score_a
        mode = "CERTIFIED_BUY"
    else:
        best_score = score_b
        mode = "DIP_ENTRY"
    
    return ScoringResult(
        symbol=symbol,
        best_score=best_score,
        mode=mode,
        score_a=score_a,
        score_b=score_b,
        gate_pass=gate_pass,
        evidence=evidence,
        config_hash=config.config_hash(),
        data_start=stock_prices.index[0].date() if hasattr(stock_prices.index[0], 'date') else None,
        data_end=stock_prices.index[-1].date() if hasattr(stock_prices.index[-1], 'date') else None,
    )


# =============================================================================
# BATCH SCORING
# =============================================================================

async def compute_all_scores(
    symbols: list[str],
    get_prices_func,
    get_spy_func,
    get_strategy_func,
    get_fundamentals_func=None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> list[ScoringResult]:
    """
    Compute scores for all symbols.
    
    Args:
        symbols: List of stock symbols
        get_prices_func: Async function(symbol) -> DataFrame
        get_spy_func: Async function() -> Series
        get_strategy_func: Async function(symbol) -> (weights, n_strategies)
        get_fundamentals_func: Optional async function(symbol) -> dict
        config: Scoring configuration
        
    Returns:
        List of ScoringResult for each symbol
    """
    import asyncio
    
    # Get SPY prices once
    spy_prices = await get_spy_func()
    
    results = []
    
    for symbol in symbols:
        try:
            prices = await get_prices_func(symbol)
            weights, n_strategies = await get_strategy_func(symbol)
            
            fundamentals = None
            if get_fundamentals_func:
                fundamentals = await get_fundamentals_func(symbol)
            
            result = compute_symbol_score(
                symbol=symbol,
                prices=prices,
                spy_prices=spy_prices,
                strategy_weights=weights,
                fundamentals=fundamentals,
                n_strategies_tested=n_strategies,
                config=config,
            )
            
            results.append(result)
            
        except Exception as e:
            logger.exception(f"Failed to compute score for {symbol}: {e}")
            # Add empty result
            results.append(ScoringResult(
                symbol=symbol,
                best_score=0.0,
                mode="DIP_ENTRY",
                score_a=0.0,
                score_b=0.0,
                gate_pass=False,
                config_hash=config.config_hash(),
            ))
    
    return results
