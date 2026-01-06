# Scoring System V3 Documentation

This document describes the unified scoring system used to rank stocks for investment opportunities.

## Overview

The Scoring V3 system uses a unified architecture with single sources of truth:

1. **TechnicalService** - All technical indicator computation
2. **RegimeService** - Market regime detection
3. **ScoringOrchestrator** - Unified scoring entry point

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   ScoringOrchestrator                   │
│    Unified entry point for stock analysis & scoring    │
└─────────────────┬───────────────────┬──────────────────┘
                  │                   │
     ┌────────────▼─────────┐ ┌──────▼────────────┐
     │   TechnicalService   │ │   RegimeService   │
     │  (Singleton Cache)   │ │  (Market Context) │
     └──────────────────────┘ └───────────────────┘
```

## Core Services

### TechnicalService
Centralized indicator computation with caching:
```python
from app.quant_engine.core import get_technical_service

tech = get_technical_service()
snapshot = tech.get_snapshot(price_df)

print(snapshot.rsi_14)          # RSI
print(snapshot.macd_histogram)  # MACD
print(snapshot.momentum_score)  # Composite momentum
print(snapshot.volatility_regime)  # LOW/NORMAL/HIGH/EXTREME
```

### RegimeService
Market regime detection:
```python
from app.quant_engine.core import get_regime_service, MarketRegime

regime = get_regime_service()
state = regime.get_current_regime(spy_df)

print(state.regime)       # BULL, BEAR, CRASH, RECOVERY, CORRECTION
print(state.strategy_mode)  # TECHNICAL, CAUTIOUS_TREND, ACCUMULATE, DEFENSIVE, HOLD
```

### ScoringOrchestrator
Unified stock analysis:
```python
from app.quant_engine.scoring import get_scoring_orchestrator

orchestrator = get_scoring_orchestrator()
dashboard = orchestrator.analyze_stock(
    symbol="NFLX",
    price_df=price_df,
    fundamentals=fundamentals_dict,
    spy_df=spy_df,
)

print(dashboard.overall_score)
print(dashboard.recommendation)
print(dashboard.score_components)
```

## Data Sources

### 1. Backtest V2 (`BacktestV2Input`)

Comprehensive strategy performance metrics from the `backtest_v2` engine:

| Metric | Description | Source |
|--------|-------------|--------|
| `sharpe_ratio` | Risk-adjusted return (excess return / std dev) | Strategy Analyzer |
| `sortino_ratio` | Downside risk-adjusted return | Strategy Analyzer |
| `calmar_ratio` | Return / max drawdown | Strategy Analyzer |
| `kelly_fraction` | Optimal position sizing (Kelly Criterion) | Strategy Analyzer |
| `sqn` | System Quality Number (expectancy × √N) | Strategy Analyzer |
| `profit_factor` | Gross profit / gross loss | Trade Statistics |
| `vs_buyhold_pct` | Alpha vs buy-and-hold benchmark | Baseline Comparison |
| `vs_spy_pct` | Alpha vs SPY benchmark | Baseline Comparison |
| `max_drawdown_pct` | Worst peak-to-trough decline | Risk Metrics |
| `cvar_5` | Conditional VaR at 5% (tail risk) | Risk Metrics |
| `wfo_passed` | Walk-forward validation passed? | WFO Engine |
| `wfo_oos_sharpe` | Out-of-sample Sharpe ratio | WFO Engine |
| `crash_outperformed_count` | # of crashes where strategy beat market | Crash Tester |
| `avg_crash_alpha` | Average alpha during market crashes | Crash Tester |

### 2. Dip Entry Optimizer (`DipEntryInput`)

Metrics from the dip entry analysis system:

| Metric | Description | Source |
|--------|-------------|--------|
| `current_drawdown_pct` | Current price vs 52-week high | DipState table |
| `optimal_threshold_pct` | Best risk-adjusted entry threshold | Optimizer |
| `recovery_rate` | % of similar dips that recovered to entry | Threshold Stats |
| `full_recovery_rate` | % that fully recovered to previous high | Threshold Stats |
| `avg_recovery_days` | Mean days to recovery | Threshold Stats |
| `avg_recovery_velocity` | Recovery % per day (speed of bounce) | Threshold Stats |
| `win_rate_optimal_hold` | Win rate at optimal holding period | Threshold Stats |
| `avg_return_optimal_hold` | Avg return at optimal holding period | Threshold Stats |
| `sharpe_optimal_hold` | Sharpe ratio at optimal holding period | Threshold Stats |
| `max_further_drawdown` | Maximum Adverse Excursion (MAE) | Risk Metrics |
| `prob_further_drop` | Probability of further decline | Risk Metrics |
| `continuation_risk` | "low" / "medium" / "high" | Risk Assessment |
| `entry_score` | Pre-computed risk-adjusted entry score | Optimizer |
| `signal_strength` | 0-100 signal quality | Optimizer |
| `is_buy_now` | Active buy signal? | Optimizer |

### 3. Fundamentals (`FundamentalsInput`)

Company financial metrics from Yahoo Finance:

| Metric | Description | Better |
|--------|-------------|--------|
| `pe_ratio` | Price to Earnings | Lower |
| `peg_ratio` | PE to Growth | Lower |
| `profit_margin` | Net income / revenue | Higher |
| `roe` | Return on Equity | Higher |
| `revenue_growth` | YoY revenue growth | Higher |
| `earnings_growth` | YoY earnings growth | Higher |
| `debt_to_equity` | Total debt / equity | Lower |
| `current_ratio` | Current assets / liabilities | Higher |
| `free_cash_flow` | Operating cash - capex | Higher |
| `target_upside_pct` | Analyst price target upside | Higher |

**Domain-Specific Metrics:**

- **Banks**: NIM, efficiency ratio, NPL ratio
- **REITs**: FFO yield, NAV discount
- **Insurance**: Combined ratio, loss ratio

## Score Components

The final score is composed of six weighted components:

### 1. Backtest Quality (20% base weight)
How good is the trading strategy's historical performance?

```
backtest_quality = weighted_avg(
    sharpe_percentile × 0.15,
    sortino_percentile × 0.10,
    calmar_percentile × 0.05,
    edge_vs_buyhold_percentile × 0.10,
    edge_vs_spy_percentile × 0.10,
    sqn_percentile × 0.10,
    kelly_percentile × 0.10,
    profit_factor_percentile × 0.05,
    crash_win_rate × 0.10,
    crash_alpha_percentile × 0.05,
    wfo_passed × 0.05,
    wfo_oos_sharpe_percentile × 0.05
)
```

### 2. Entry Timing (25% base weight)
Is NOW a good time to buy?

```
entry_timing = weighted_avg(
    distance_to_optimal × 0.25,     # How close to optimal threshold
    signal_strength × 0.25,         # Signal quality
    is_buy_now_score × 0.25,        # Active signal
    continuation_risk_inv × 0.25    # Low = better
)
```

### 3. Recovery Probability (25% base weight)
Will this stock recover from the current dip?

```
recovery_score = weighted_avg(
    recovery_rate_percentile × 0.30,
    recovery_velocity_percentile × 0.25,
    full_recovery_rate_percentile × 0.15,
    avg_return_percentile × 0.20,
    win_rate_percentile × 0.10
)
```

### 4. Fundamental Score (15% base weight)
Is this a good company?

```
fundamental_score = weighted_avg(
    pe_inv_percentile × 0.10,       # Lower PE = better
    peg_inv_percentile × 0.10,
    profit_margin_percentile × 0.15,
    roe_percentile × 0.15,
    revenue_growth_percentile × 0.10,
    earnings_growth_percentile × 0.10,
    debt_equity_inv_percentile × 0.10,
    current_ratio_percentile × 0.10,
    fcf_percentile × 0.05,
    + domain_specific_metrics
)
```

### 5. Risk Score (10% base weight)
How risky is this entry?

```
risk_score = weighted_avg(
    mae_inv_percentile × 0.30,         # Lower MAE = better
    max_drawdown_inv_percentile × 0.25,
    cvar_percentile × 0.20,            # Higher CVaR = better
    continuation_risk_inv × 0.25
)
```

### 6. Momentum Score (5% base weight)
What's the upside potential?

```
momentum_score = weighted_avg(
    dip_depth_inv_percentile × 0.40,   # Deeper dip = more oversold
    target_upside_percentile × 0.30,
    revenue_growth_percentile × 0.15,
    earnings_growth_percentile × 0.15
)
```

## Percentile Normalization

**The key innovation: NO HARDCODED THRESHOLDS.**

Every metric is normalized against the universe of all tracked symbols:

```python
def percentile_normalize(value, universe_values):
    """
    Returns the percentile rank [0, 1] of value in the universe.
    
    Example:
    - If NFLX has Sharpe = 1.5 and that's 85th percentile in universe
    - Returns 0.85
    """
    return percentileofscore(universe_values, value) / 100
```

For metrics where lower is better (PE, debt, drawdown):
```python
def inverse_normalize(value, universe_values):
    return 1.0 - percentile_normalize(value, universe_values)
```

## Dynamic Weight Adjustment

Weights are adjusted based on data confidence:

```python
final_weight[component] = base_weight[component] × confidence[component]
```

Confidence is calculated from:
- **Sample size**: More historical dip events = higher confidence
- **Data years**: More years of data = higher confidence
- **WFO validation**: Passed walk-forward = confidence boost

## Modes

The system assigns one of four modes:

| Mode | Criteria |
|------|----------|
| `CERTIFIED_BUY` | WFO passed + Sharpe > 1.0 + beats buy-and-hold |
| `DIP_ENTRY` | Buy signal active (at optimal threshold) |
| `HOLD` | No significant dip or below entry threshold |
| `DOWNTREND` | Extended decline (>1 year in dip) |

## Actions

Based on score and mode:

| Action | Criteria |
|--------|----------|
| `STRONG_BUY` | CERTIFIED_BUY with score ≥ 75, or DIP_ENTRY with is_buy_now + score ≥ 70 |
| `BUY` | CERTIFIED_BUY with score ≥ 60, or DIP_ENTRY with is_buy_now + score ≥ 50 |
| `HOLD` | Everything else |
| `AVOID` | DOWNTREND mode |

## Database Schema

Scores are stored in `quant_scores` table:

```sql
best_score       -- Final composite score (0-100)
mode             -- CERTIFIED_BUY, DIP_ENTRY, HOLD, DOWNTREND
score_a          -- Backtest quality component
score_b          -- Entry timing component
gate_pass        -- Is CERTIFIED_BUY?
p_recovery       -- Recovery probability (0-1)
expected_value   -- Entry score from dip optimizer
evidence         -- Full JSON with all components
```

## Job Execution

The `quant_scoring_daily` job runs nightly:

1. Load all symbols
2. Bulk-load precomputed data from `quant_precomputed`
3. Build universe statistics (arrays for percentile ranking)
4. Score each symbol using `compute_score_v2()`
5. Upsert results to `quant_scores`
6. Clear caches

## API Usage

```python
from app.quant_engine import (
    ScoringOrchestrator,
    get_scoring_orchestrator,
    get_technical_service,
    get_regime_service,
)

# Get unified scoring
orchestrator = get_scoring_orchestrator()
dashboard = orchestrator.analyze_stock(
    symbol="NFLX",
    price_df=price_df,
    fundamentals=fundamentals,
    spy_df=spy_df,
)

print(f"Score: {dashboard.overall_score}")
print(f"Recommendation: {dashboard.recommendation}")
print(f"Risk Level: {dashboard.risk_assessment.level}")

# Technical analysis
tech = get_technical_service()
snapshot = tech.get_snapshot(price_df)
print(f"RSI: {snapshot.rsi_14}")
print(f"Trend: {snapshot.trend_direction}")

# Regime detection
regime = get_regime_service()
state = regime.get_current_regime(spy_df)
print(f"Regime: {state.regime.value}")
print(f"Strategy: {state.strategy_mode.value}")
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Original | Dual-mode APUS/DOUS with hardcoded thresholds |
| 2.0.0 | Previous | Integrated scoring with percentile ranking |
| 3.0.0 | Current | Unified architecture with TechnicalService, RegimeService, ScoringOrchestrator |

## Key Differences from V2

| Aspect | V2 | V3 |
|--------|----|----|
| Entry point | `compute_score_v2()` | `ScoringOrchestrator.analyze_stock()` |
| Technical indicators | Computed inline | `TechnicalService` singleton |
| Regime detection | `RegimeDetector` class | `RegimeService` singleton |
| Caching | Per-function | Service-level with invalidation |
| Architecture | Multiple scattered modules | Unified core services |
