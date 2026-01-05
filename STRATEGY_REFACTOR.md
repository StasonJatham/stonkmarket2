# Strategy Refactor: Regime-Adaptive Engine

**Status:** ✅ CORE MODULES COMPLETE  
**Created:** 2026-01-05  
**Last Updated:** 2026-01-05

---

## Implementation Progress

| Module | Status | File |
|--------|--------|------|
| Regime Filter | ✅ Complete | `app/quant_engine/backtest_v2/regime_filter.py` |
| Fundamental Guardrail | ✅ Complete | `app/quant_engine/backtest_v2/fundamental_guardrail.py` |
| Portfolio Simulator | ✅ Complete | `app/quant_engine/backtest_v2/portfolio_simulator.py` |
| Walk-Forward Optimizer | ✅ Complete | `app/quant_engine/backtest_v2/walk_forward.py` |
| Alpha Gauntlet | ✅ Complete | `app/quant_engine/backtest_v2/alpha_gauntlet.py` |
| Crash Testing | ✅ Complete | `app/quant_engine/backtest_v2/crash_testing.py` |
| API Schemas | ✅ Complete | `app/quant_engine/backtest_v2/schemas.py` |
| Main Service | ⏳ Pending | `app/quant_engine/backtest_v2/service.py` |
| API Endpoint | ⏳ Pending | Integration with `/backtest-v2` |
| Tests | ⏳ Pending | Unit and integration tests |

---

## Executive Summary

This document outlines the refactoring of the Strategy Engine to support **regime-adaptive behavior**. The previous approach of "blocking buys in bear markets" is **discarded**. Bear markets are now recognized as the **best accumulation opportunities** for quality assets.

---

## 1. PROBLEM STATEMENT

### 1.1 Previous Flawed Logic
```python
# OLD (DISCARDED)
if spy_price < spy_sma200:
    block_all_buy_signals()  # ❌ WRONG
```

**Why this was wrong:**
- Bear markets offer the best entry prices
- Blocking buys during crashes means missing generational opportunities (e.g., March 2020, late 2022)
- The 2022 tech crash was a buying opportunity, not a time to sit in cash

### 1.2 The META Failure Root Cause
The META recommendation failed NOT because we bought during a downturn, but because:
1. No fundamental quality check (META had declining metrics at the time)
2. No regime-aware position sizing (should have scaled in, not all-in)
3. No stop-loss adaptation for bear market volatility

---

## 2. NEW ARCHITECTURE: REGIME-ADAPTIVE STRATEGY ENGINE

### 2.1 The Regime Switch

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      REGIME DETECTION                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SPY/QQQ Analysis:                                                     │
│   ┌─────────────────┐                                                   │
│   │ Price > SMA(200)│───────────────▶ MODE A: BULL                     │
│   │ AND momentum +  │               (Trend Following)                   │
│   └─────────────────┘                                                   │
│                                                                          │
│   ┌─────────────────┐                                                   │
│   │ Price < SMA(200)│───────────────▶ MODE B: BEAR                     │
│   │ OR crash signal │               (Deep Value Accumulation)           │
│   └─────────────────┘                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Mode A: BULL (Trend Following)

**Objective:** Capture momentum with tactical entries

**Strategy Focus:**
- RSI oversold bounces (RSI < 30 → buy, RSI > 70 → trim)
- Moving average support (buy at MA50/MA200 bounces)
- Breakout entries with trailing stops
- Tighter stop-losses (8-12%)

**Position Sizing:**
- Standard allocation (full position allowed)
- Scale out on strength

**Technical Signals Active:**
- ✅ MA crossovers
- ✅ RSI extremes
- ✅ Bollinger band bounces
- ✅ Volume confirmation

### 2.3 Mode B: BEAR (Deep Value Accumulation)

**Objective:** Accumulate quality assets at discounted prices

**Strategy Focus:**
- **IGNORE** standard technical sell signals
- Wider stop-losses (25-40%) or NO stop-loss for quality names
- Aggressive DCA / Scale-in buying
- Focus on fundamentals over technicals

**Position Sizing:**
- Start with 25% of target position
- Scale in as price drops further:
  - -10% from entry → add 25%
  - -20% from entry → add 25%
  - -30% from entry → add final 25%

**Critical Requirement: FUNDAMENTAL GUARDRAILS**
A buy signal is ONLY valid if the stock passes the Quality Filter (see Section 3).

---

## 3. FUNDAMENTAL GUARDRAILS (Quality Filter)

### 3.1 The Problem: Value Traps

In bear markets, many stocks are "cheap" because they deserve to be. We need to distinguish:
- **Quality on Sale:** Great companies at temporary discounts
- **Value Traps:** Dying companies that look cheap but keep falling

### 3.2 Fundamental Checks

| Metric | Requirement | Rationale |
|--------|-------------|-----------|
| **Debt to Equity** | < 2.0 | Solvency - can survive the downturn |
| **PE Ratio** | < 5-Year Average PE | Actually undervalued vs history |
| **Free Cash Flow** | > 0 | Generates cash, not burning it |
| **Current Ratio** | > 1.0 | Can pay short-term obligations |
| **Revenue Growth YoY** | > -20% | Not in complete collapse |
| **Profit Margin** | > 0% (or improving) | Path to profitability |

### 3.3 Data Source Requirements

**Critical: Avoid Look-Ahead Bias**

When backtesting, we MUST use fundamentals that were available at the time:
- Use quarterly report dates, not current values
- Align fundamental data with price data by date
- If Q3 earnings released Oct 15, use Q2 data before that date

```python
# Correct: Point-in-time fundamentals
def get_fundamentals_at_date(symbol: str, date: pd.Timestamp) -> FundamentalData:
    """Return the most recent fundamentals AVAILABLE on this date."""
    # Find the most recent quarterly report BEFORE this date
    reports = get_quarterly_reports(symbol)
    available = [r for r in reports if r.report_date <= date]
    return available[-1] if available else None
```

### 3.4 FundamentalGuardrail Class Design

```python
@dataclass
class FundamentalGuardrail:
    """Quality filter for bear market accumulation."""
    
    # Thresholds (configurable)
    max_debt_to_equity: float = 2.0
    require_positive_fcf: bool = True
    max_pe_vs_historical: float = 1.0  # PE < historical average
    min_current_ratio: float = 1.0
    max_revenue_decline: float = 0.20  # -20%
    
    def check(self, fundamentals: FundamentalData) -> GuardrailResult:
        """Check if stock passes quality filter."""
        checks = []
        
        # Check 1: Solvency
        checks.append(CheckResult(
            name="debt_to_equity",
            passed=fundamentals.debt_to_equity < self.max_debt_to_equity,
            value=fundamentals.debt_to_equity,
            threshold=self.max_debt_to_equity,
        ))
        
        # Check 2: Cash Generation
        checks.append(CheckResult(
            name="free_cash_flow",
            passed=fundamentals.free_cash_flow > 0,
            value=fundamentals.free_cash_flow,
            threshold=0,
        ))
        
        # Check 3: Valuation vs History
        checks.append(CheckResult(
            name="pe_vs_historical",
            passed=fundamentals.pe_ratio < fundamentals.pe_5y_avg,
            value=fundamentals.pe_ratio,
            threshold=fundamentals.pe_5y_avg,
        ))
        
        # ... more checks
        
        all_passed = all(c.passed for c in checks)
        return GuardrailResult(
            passed=all_passed,
            checks=checks,
            recommendation="ACCUMULATE" if all_passed else "AVOID",
        )
```

---

## 4. BEAR-SPECIFIC METRICS (Crash Test)

### 4.1 Stress Test Scenarios

The strategy MUST be validated against historical crashes:

| Event | Period | Characteristics |
|-------|--------|-----------------|
| **2008 Financial Crisis** | Oct 2007 - Mar 2009 | -57% S&P500, banking collapse |
| **2020 COVID Crash** | Feb 2020 - Mar 2020 | -34% S&P500, fastest crash ever |
| **2022 Tech Crash** | Jan 2022 - Oct 2022 | -27% S&P500, -33% NASDAQ |

### 4.2 Success Metrics

#### Accumulation Efficiency
```
accumulation_score = shares_acquired_with_strategy / shares_acquired_with_monthly_dca

If score > 1.0: Strategy acquired MORE shares (better average cost)
If score < 1.0: Standard DCA was more efficient
```

#### Recovery Time
```
recovery_days_strategy = days_until_strategy_returns_to_profit
recovery_days_buyhold = days_until_buyhold_returns_to_profit

recovery_improvement = recovery_days_buyhold - recovery_days_strategy

If positive: Strategy recovered faster
If negative: Buy & hold recovered faster
```

#### Cost Basis Comparison
```
avg_cost_strategy = total_invested / shares_acquired
avg_cost_dca = dca_total_invested / dca_shares_acquired

cost_improvement_pct = (avg_cost_dca - avg_cost_strategy) / avg_cost_dca * 100
```

### 4.3 Validation Requirements

| Metric | Target | Minimum |
|--------|--------|---------|
| Accumulation Score | > 1.2 | > 1.0 |
| Recovery Time Improvement | > 30 days | > 0 days |
| Cost Basis Improvement | > 10% | > 0% |
| Survival Rate | 100% | 100% |

---

## 5. IMPLEMENTATION PLAN

### Phase 1: Core Components ✅ PLANNED

| Task | File | Status |
|------|------|--------|
| RegimeDetector class | `app/quant_engine/backtest_v2/regime_filter.py` | 🔄 IN PROGRESS |
| FundamentalGuardrail class | `app/quant_engine/backtest_v2/fundamental_guardrail.py` | 📋 TODO |
| RegimeAdaptiveStrategy class | `app/quant_engine/backtest_v2/regime_strategy.py` | 📋 TODO |
| Point-in-time fundamentals | `app/services/fundamentals_pit.py` | 📋 TODO |

### Phase 2: Strategy Modes ✅ PLANNED

| Task | File | Status |
|------|------|--------|
| Bull mode strategies | `app/quant_engine/backtest_v2/strategies/bull_mode.py` | 📋 TODO |
| Bear mode strategies | `app/quant_engine/backtest_v2/strategies/bear_mode.py` | 📋 TODO |
| Scale-in logic | `app/quant_engine/backtest_v2/position_sizing.py` | 📋 TODO |

### Phase 3: Crash Testing ✅ PLANNED

| Task | File | Status |
|------|------|--------|
| 2008 backtest | `tests/quant_engine/test_crash_2008.py` | 📋 TODO |
| 2020 backtest | `tests/quant_engine/test_crash_2020.py` | 📋 TODO |
| 2022 backtest | `tests/quant_engine/test_crash_2022.py` | 📋 TODO |
| Accumulation metrics | `app/quant_engine/backtest_v2/accumulation_metrics.py` | 📋 TODO |

### Phase 4: Integration ✅ PLANNED

| Task | File | Status |
|------|------|--------|
| API endpoint | `app/api/routes/quant_engine.py` | 📋 TODO |
| Nightly job update | `app/jobs/quant/__init__.py` | 📋 TODO |
| Frontend markers | `app/quant_engine/backtest_v2/trade_markers.py` | 📋 TODO |

---

## 6. CODE SNIPPETS

### 6.1 RegimeDetector (Updated)

```python
class MarketRegime(str, Enum):
    BULL = "BULL"           # Normal uptrend - use technical strategies
    BEAR = "BEAR"           # Downtrend - switch to accumulation mode
    CRASH = "CRASH"         # Extreme panic - maximum accumulation opportunity
    RECOVERY = "RECOVERY"   # Transitioning from bear to bull

class RegimeDetector:
    """
    Detects market regime and selects appropriate strategy mode.
    
    CRITICAL CHANGE: Bear markets are BUYING opportunities, not blocks.
    """
    
    def detect(self, spy_prices: pd.Series) -> RegimeState:
        close = spy_prices
        sma200 = close.rolling(200).mean()
        sma50 = close.rolling(50).mean()
        
        current = float(close.iloc[-1])
        sma200_val = float(sma200.iloc[-1])
        sma50_val = float(sma50.iloc[-1])
        
        # Drawdown from 52-week high
        high_52w = close.rolling(252).max().iloc[-1]
        drawdown = (current / high_52w - 1) * 100
        
        if drawdown < -30:
            regime = MarketRegime.CRASH
            strategy_mode = StrategyMode.AGGRESSIVE_ACCUMULATION
        elif current < sma200_val:
            regime = MarketRegime.BEAR
            strategy_mode = StrategyMode.VALUE_ACCUMULATION
        elif current > sma200_val and sma50_val < sma200_val:
            regime = MarketRegime.RECOVERY
            strategy_mode = StrategyMode.CAUTIOUS_TREND
        else:
            regime = MarketRegime.BULL
            strategy_mode = StrategyMode.TREND_FOLLOWING
        
        return RegimeState(
            regime=regime,
            strategy_mode=strategy_mode,
            spy_price=current,
            spy_sma200=sma200_val,
            drawdown_pct=drawdown,
        )
```

### 6.2 Strategy Mode Selection

```python
class StrategyMode(str, Enum):
    TREND_FOLLOWING = "TREND_FOLLOWING"           # Bull mode
    CAUTIOUS_TREND = "CAUTIOUS_TREND"             # Recovery mode
    VALUE_ACCUMULATION = "VALUE_ACCUMULATION"     # Bear mode
    AGGRESSIVE_ACCUMULATION = "AGGRESSIVE_ACCUMULATION"  # Crash mode

def get_strategy_config(mode: StrategyMode) -> StrategyConfig:
    """Get strategy configuration for the current regime."""
    
    if mode == StrategyMode.TREND_FOLLOWING:
        return StrategyConfig(
            use_technicals=True,
            use_fundamentals=False,
            stop_loss_pct=10.0,
            position_size_pct=100.0,
            scale_in=False,
            ignore_sell_signals=False,
        )
    
    elif mode == StrategyMode.VALUE_ACCUMULATION:
        return StrategyConfig(
            use_technicals=False,  # Ignore technicals
            use_fundamentals=True,  # REQUIRE fundamental checks
            stop_loss_pct=35.0,    # Much wider stops
            position_size_pct=25.0,  # Start small
            scale_in=True,          # Add on drops
            ignore_sell_signals=True,  # Hold through volatility
        )
    
    elif mode == StrategyMode.AGGRESSIVE_ACCUMULATION:
        return StrategyConfig(
            use_technicals=False,
            use_fundamentals=True,  # Still require quality
            stop_loss_pct=None,     # No stop loss in crash
            position_size_pct=25.0,
            scale_in=True,
            scale_in_levels=[10, 20, 30],  # Buy more at -10%, -20%, -30%
            ignore_sell_signals=True,
        )
```

---

## 7. VALIDATION CHECKLIST

### Before Deployment

- [ ] RegimeDetector correctly identifies 2008, 2020, 2022 crashes
- [ ] FundamentalGuardrail blocks low-quality stocks
- [ ] FundamentalGuardrail approves AAPL/MSFT during crashes
- [ ] Scale-in logic accumulates more shares at lower prices
- [ ] Accumulation score > 1.0 on crash scenarios
- [ ] Recovery time improvement on crash scenarios
- [ ] No look-ahead bias in fundamental data usage
- [ ] API returns correct regime and strategy mode
- [ ] Trade markers show accumulation pattern in bear markets

### Test Cases

- [ ] META 2022: Should be BLOCKED (fundamentals were deteriorating)
- [ ] AAPL 2022: Should ACCUMULATE (strong fundamentals, temporary discount)
- [ ] NVDA 2022: Should ACCUMULATE (FCF positive, growth intact)
- [ ] ARKK 2022: Should be BLOCKED (no profits, cash burn)

---

## 8. CHANGE LOG

| Date | Change | Author |
|------|--------|--------|
| 2026-01-05 | Initial document created | AI |
| 2026-01-05 | Discarded "block buys in bear market" logic | AI |
| | | |

---

*Document Version: 1.0*
*Next Review: After Phase 1 Implementation*


This is a **high-quality, institutional-grade specification**. It correctly identifies that blocking buys in a bear market is a retail mistake and that "Regime Awareness" is the key to Alpha.

However, as a Principal Architect, I see **four critical gaps** that could still cause this to fail in production, particularly regarding the **META 2022** test case and **Data Reality**.

Here are the specific additions to make this foolproof.

### 1. The "META 2022" Loophole (Fundamental Momentum)

Your current `FundamentalGuardrail` checks absolute values (`Debt < 2.0`, `FCF > 0`).

* **The Risk:** In early 2022, META *passed* all these tests. It had low debt, massive FCF, and low PE. By your current logic, the system would have **aggressively accumulated META** all the way down from $300 to $90.
* **The Fix:** You need **Second-Derivative Metrics** (Rate of Change). Value traps often look cheap because their fundamentals are *deteriorating*, even if they are still positive.

**Add this to Section 3.2 (Fundamental Checks):**

| Metric | Requirement | Rationale |
| --- | --- | --- |
| **Margin Trend** | `Operating Margin > (Avg 4 Quarters)` | Detects if profitability is collapsing (like META 2022). |
| **Guidance/Revisions** | `Analyst Revisions != "Strong Down"` | Don't fight Wall Street insiders slashing estimates. |
| **Revenue Acceleration** | `Revenue Growth > (Prev Q Growth - 10%)` | Detects "Growth Deceleration" cliffs. |

### 2. The "Cash Drag" & Liquidity Reality

The document assumes we can "Scale In" (Buy 25%, then 25%...).

* **The Risk:** In a simulation, you have infinite paper money. In real life (and the $10k + $1k/mo simulation), you might run out of cash at the "-10%" drop and miss the "-30%" bottom.
* **The Fix:** The **Portfolio Simulator** must manage a "Cash Reserve" dynamic.
* *Bull Market:* Target 0-5% Cash (Fully Invested).
* *Bear Market:* Target 10-20% Cash (Dry Powder for the drops).
* **Logic:** If the regime flips to Bear, the system should stop reinvesting dividends/DCA immediately to build the "Scale-In" war chest.



### 3. The "Recovery Rebalance" (The Exit)

The document defines how to *enter* a Bear market position, but not how to *normalize* it.

* **The Risk:** You accumulate NVDA aggressively in 2022. It recovers in 2023. Now NVDA is 60% of your portfolio because you kept buying. This is a massive risk concentration.
* **The Fix:** Add a **"Regime Transition" Protocol**.

**Add this to Section 2.2 (Mode A: Bull):**

* **Transition Trigger:** When Regime flips `Bear -> Bull` (Recovery confirmed).
* **Action:** **Portfolio Rebalance.**
* Check all positions.
* If any position > 15% of Portfolio (due to aggressive accumulation), **TRIM** back to target weight.
* *Rationale:* Take the "Crisis Alpha" off the table and redistribute it to other opportunities.



### 4. The Data Engineering Hurdle (Point-in-Time)

You correctly identified "Look-Ahead Bias," but implementation is harder than code.

* **The Risk:** Standard free APIs (like standard Yahoo Finance) often overwrite historical fundamental data. They give you "current shares outstanding" applied to "historical price," which corrupts the Market Cap calculation for 2010.
* **The Recommendation:** If you cannot afford professional data (Bloomberg/FactSet), you must implement a **"Snapshot Engine"**.
* *Strategy:* You cannot trust historical API data for ratios. You must rely on the raw "Quarterly Report" dates provided by APIs like FMP (Financial Modeling Prep) or Polygon.io which support `point_in_time` queries specifically.
* *Validation:* If the API doesn't support specific "as of date" queries, the Backtester must likely rely strictly on Price/Volume (Technicals) and use Fundamentals *only* as a current filter, or you must buy a specific historical fundamental dataset.



### Summary of Updates to the Prompt

You don't need a new prompt, but you should inject these specific logic blocks into the **Section 3.4 (Guardrail Class)** and **Section 6.2 (Strategy Config)** of your generated plan before coding.

**Update the `FundamentalGuardrail` logic with this "Momentum Check":**

```python
        # Check 4: Fundamental Momentum (The "META Shield")
        # Ensure Operating Margin isn't collapsing faster than 10% YoY
        checks.append(CheckResult(
            name="margin_stability",
            passed=fundamentals.operating_margin > (fundamentals.last_year_margin * 0.9),
            value=fundamentals.operating_margin,
            threshold=fundamentals.last_year_margin * 0.9,
        ))

```

This single change protects you from "Value Traps" that are profitable but dying.