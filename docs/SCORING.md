# Scoring System V2 Documentation

This document describes the integrated scoring system used to rank stocks for dip buying opportunities.

## Overview

The Scoring V2 system replaces the old dual-mode (APUS/DOUS) scoring with a modern, statistically-driven approach that:

1. **Uses ALL available data sources** - backtest_v2, dip_entry_optimizer, fundamentals
2. **NO hardcoded thresholds** - All optimal values discovered through statistics
3. **Percentile ranking** - Scores relative to the universe, not absolute thresholds
4. **Confidence-weighted** - Higher data confidence = higher weight contribution

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
from app.quant_engine.scoring_v2 import compute_score_v2
from app.quant_engine.scoring_v2_adapters import load_all_inputs_for_symbol

async with get_session() as session:
    backtest, dip_entry, fundamentals = await load_all_inputs_for_symbol(
        session, "NFLX"
    )
    
    result = compute_score_v2(
        symbol="NFLX",
        backtest=backtest,
        dip_entry=dip_entry,
        fundamentals=fundamentals,
        universe_stats=universe_stats,
    )
    
    print(f"Score: {result.score}")
    print(f"Mode: {result.mode}")
    print(f"Action: {result.action}")
    print(f"Components: {result.components.to_dict()}")
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Original | Dual-mode APUS/DOUS with hardcoded thresholds |
| 2.0.0 | Current | Integrated scoring with percentile ranking, no hardcoded thresholds |

## Key Differences from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Recovery calculation | Fixed 60-day window, 20% dip matching | Uses dip_entry_optimizer's recovery_velocity |
| Thresholds | Hardcoded (min_p_outperf=0.75, etc.) | Percentile ranking against universe |
| Data sources | Computed inline | Reads from precomputed tables |
| Weights | Fixed | Confidence-adjusted |
| Fundamental scoring | Simple z-scores | Domain-specific with percentile ranking |
