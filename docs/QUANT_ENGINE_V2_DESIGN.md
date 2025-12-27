# Quant Engine V2: Risk-Based Portfolio Optimization

## Executive Summary

This document outlines the design for a **non-predictive, risk-focused portfolio optimization engine** that answers:

> "Given my current portfolio and the current market state, where should my next €1,000 go to maximize risk-adjusted quality?"

### Core Philosophy

| ❌ What We DON'T Do | ✅ What We DO |
|---------------------|---------------|
| Predict future returns | Measure current risk exposures |
| Forecast price movements | Optimize diversification |
| Generate "alpha" signals | Minimize tail risk (CVaR/ES) |
| Claim to beat the market | Build robust, stable allocations |

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUANT ENGINE V2                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────┐ │
│  │  ANALYTICS ENGINE   │    │  OPTIMIZATION ENGINE │    │  SIGNAL SCANNER  │ │
│  │  (Diagnostic Focus) │    │  (Risk-Based)        │    │  (Already Built) │ │
│  ├─────────────────────┤    ├─────────────────────┤    ├──────────────────┤ │
│  │ • Risk decomposition│    │ • Risk Parity        │    │ • Mean reversion │ │
│  │ • Factor exposures  │    │ • Min Variance       │    │ • Technical dips │ │
│  │ • Tail risk (CVaR)  │    │ • Max Diversification│    │ • Backtest stats │ │
│  │ • Regime detection  │    │ • CVaR minimization  │    │ • Optimal holding│ │
│  │ • Correlation shifts│    │ • Black-Litterman    │    │                  │ │
│  │ • Drawdown analysis │    │   (views from dips)  │    │                  │ │
│  └─────────────────────┘    └─────────────────────┘    └──────────────────┘ │
│            │                          │                          │           │
│            └──────────────────────────┼──────────────────────────┘           │
│                                       ▼                                      │
│                    ┌─────────────────────────────────┐                       │
│                    │     HPO & CALIBRATION ENGINE    │                       │
│                    │     (Scheduled Background Job)  │                       │
│                    ├─────────────────────────────────┤                       │
│                    │ • Per-stock lookback windows    │                       │
│                    │ • Covariance shrinkage params   │                       │
│                    │ • Regime thresholds             │                       │
│                    │ • Risk model selection          │                       │
│                    │ • Walk-forward validation       │                       │
│                    └─────────────────────────────────┘                       │
│                                       │                                      │
│                                       ▼                                      │
│                    ┌─────────────────────────────────┐                       │
│                    │     RECOMMENDATION TRANSLATOR   │                       │
│                    │     (Simple Actionable Output)  │                       │
│                    ├─────────────────────────────────┤                       │
│                    │ "Put €500 in AAPL, €300 in VOO" │                       │
│                    │ "Your portfolio is 40% too      │                       │
│                    │  concentrated in tech"          │                       │
│                    │ "Risk: 8/10 (high volatility)"  │                       │
│                    └─────────────────────────────────┘                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Analytics Engine (Diagnostic Focus)

### 2.1 Risk Decomposition

**Purpose**: Understand where portfolio risk comes from.

| Metric | Description | User-Facing Translation |
|--------|-------------|------------------------|
| Volatility (σ) | Annualized standard deviation | "Your portfolio swings ±X% in a typical year" |
| Marginal VaR | How much each position contributes to total VaR | "AAPL accounts for 30% of your risk" |
| Component VaR | VaR contribution per position | Risk contribution pie chart |
| Beta to market | Sensitivity to SPY | "When the market drops 10%, you drop ~12%" |

**Implementation**:
```python
def compute_risk_decomposition(weights: np.ndarray, cov: np.ndarray) -> dict:
    """
    Compute portfolio risk metrics.
    
    Returns:
        portfolio_vol: Total portfolio volatility
        marginal_risk: ∂σ/∂w for each asset
        component_risk: w_i * marginal_risk_i (sums to portfolio_vol)
        risk_contribution_pct: Percentage contribution per asset
    """
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    marginal_risk = (cov @ weights) / portfolio_vol
    component_risk = weights * marginal_risk
    risk_contribution_pct = component_risk / portfolio_vol
    return {...}
```

### 2.2 Tail Risk Analysis (CVaR/Expected Shortfall)

**Purpose**: Measure worst-case scenarios, not just average volatility.

| Metric | Description | User-Facing Translation |
|--------|-------------|------------------------|
| VaR 95% | 5% worst-case daily loss | "On a bad day (1 in 20), you could lose €X" |
| CVaR 95% | Average loss in worst 5% of days | "In a crash, expect to lose €Y on average" |
| Max Drawdown | Largest peak-to-trough decline | "Historically, the worst drop was Z%" |
| Drawdown Duration | Time to recover from drawdown | "Recovery took N months" |

**Implementation**:
```python
def compute_tail_risk(returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Compute tail risk metrics using historical simulation.
    
    Uses:
    - Historical VaR (empirical quantile)
    - Historical CVaR (mean of tail)
    - Cornish-Fisher adjustment for non-normality
    """
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, alpha * 100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    # Cornish-Fisher adjustment for skewness/kurtosis
    skew = scipy.stats.skew(portfolio_returns)
    kurt = scipy.stats.kurtosis(portfolio_returns)
    z = scipy.stats.norm.ppf(alpha)
    cf_var = var + (1/6)*(z**2 - 1)*skew + (1/24)*(z**3 - 3*z)*kurt
    
    return {"var_95": var, "cvar_95": cvar, "cf_var_95": cf_var, ...}
```

### 2.3 Diversification Metrics

**Purpose**: Quantify how well-diversified the portfolio is.

| Metric | Description | User-Facing Translation |
|--------|-------------|------------------------|
| Effective N | Number of independent bets | "You effectively have 4 stocks, not 10" |
| Diversification Ratio | σ(equal-weighted) / σ(portfolio) | "Your diversification is X% of optimal" |
| Concentration (HHI) | Herfindahl-Hirschman Index | "Your portfolio is 60% concentrated" |
| Correlation clustering | Hierarchical clustering of assets | "These 5 stocks move together" |

**Implementation**:
```python
def compute_diversification_metrics(weights: np.ndarray, cov: np.ndarray) -> dict:
    """
    Compute diversification quality metrics.
    """
    # Effective N (inverse HHI)
    hhi = np.sum(weights ** 2)
    effective_n = 1 / hhi
    
    # Diversification Ratio
    asset_vols = np.sqrt(np.diag(cov))
    weighted_avg_vol = weights @ asset_vols
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    div_ratio = weighted_avg_vol / portfolio_vol
    
    # Max diversification potential
    equal_weights = np.ones(len(weights)) / len(weights)
    max_div_ratio = (equal_weights @ asset_vols) / np.sqrt(equal_weights @ cov @ equal_weights)
    
    return {
        "effective_n": effective_n,
        "diversification_ratio": div_ratio,
        "max_diversification_ratio": max_div_ratio,
        "diversification_efficiency": div_ratio / max_div_ratio,
    }
```

### 2.4 Regime Detection

**Purpose**: Identify current market regime (bull/bear, high/low volatility).

| Regime | Characteristics | Portfolio Implication |
|--------|-----------------|----------------------|
| Bull + Low Vol | Rising prices, low uncertainty | Can take more risk |
| Bull + High Vol | Rising but choppy | Moderate risk |
| Bear + Low Vol | Slow decline | Defensive posture |
| Bear + High Vol | Crash/panic | Maximum defense |

**Implementation**:
```python
def detect_regime(market_returns: pd.Series, lookback: int = 63) -> str:
    """
    Simple regime detection using trend + volatility.
    
    Returns: 'bull_low', 'bull_high', 'bear_low', 'bear_high'
    """
    # Trend: 3-month return
    trend = market_returns.iloc[-lookback:].sum()
    is_bull = trend > 0
    
    # Volatility: compare current to long-term average
    current_vol = market_returns.iloc[-lookback:].std() * np.sqrt(252)
    long_term_vol = market_returns.std() * np.sqrt(252)
    is_high_vol = current_vol > long_term_vol * 1.2
    
    if is_bull and not is_high_vol:
        return "bull_low"
    elif is_bull and is_high_vol:
        return "bull_high"
    elif not is_bull and not is_high_vol:
        return "bear_low"
    else:
        return "bear_high"
```

### 2.5 Correlation Analysis

**Purpose**: Understand how assets move together and detect correlation breakdowns.

| Analysis | Description | User-Facing Translation |
|----------|-------------|------------------------|
| Current correlation matrix | Rolling 60-day correlations | Heatmap visualization |
| Correlation vs history | Are correlations elevated? | "Correlations are 20% higher than normal" |
| Correlation stress test | Correlations during past crashes | "In 2020 crash, all stocks correlated 0.9" |
| Correlation clustering | Group similar assets | "These 3 stocks are basically the same bet" |

---

## 3. Optimization Engine (Risk-Based)

### 3.1 Optimization Objectives

We use **risk-based objectives** that don't require return forecasts:

| Objective | Formula | When to Use |
|-----------|---------|-------------|
| **Risk Parity** | Equal risk contribution per asset | Default, most robust |
| **Minimum Variance** | min w'Σw | Conservative, low-risk focus |
| **Maximum Diversification** | max (w'σ) / √(w'Σw) | Maximize diversification benefit |
| **CVaR Minimization** | min E[loss \| loss > VaR] | Tail-risk focused |
| **Hierarchical Risk Parity** | Cluster-based allocation | Handles correlated assets better |

### 3.2 Risk Parity (Recommended Default)

**Why**: Doesn't require return forecasts. Allocates risk equally across assets.

```python
def risk_parity_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Objective: minimize deviation from equal risk contribution.
    
    Target: each asset contributes 1/N of total portfolio risk.
    """
    n = len(weights)
    target_risk = 1 / n
    
    portfolio_vol = np.sqrt(weights @ cov @ weights)
    marginal_risk = (cov @ weights) / portfolio_vol
    risk_contrib = weights * marginal_risk / portfolio_vol
    
    # Sum of squared deviations from target
    return np.sum((risk_contrib - target_risk) ** 2)


def optimize_risk_parity(cov: np.ndarray, constraints: dict = None) -> np.ndarray:
    """
    Find risk parity portfolio.
    
    Uses:
    - Sequential Least Squares Programming (SLSQP)
    - Constraints: long-only, sum to 1, position limits
    """
    n = cov.shape[0]
    x0 = np.ones(n) / n  # Start with equal weights
    
    bounds = [(0.01, 0.40)] * n  # 1-40% per position
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Sum to 1
    ]
    
    result = scipy.optimize.minimize(
        risk_parity_objective,
        x0,
        args=(cov,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    
    return result.x
```

### 3.3 CVaR Minimization

**Why**: Focuses on tail risk, not just volatility. Better for crash protection.

```python
def optimize_cvar(returns: pd.DataFrame, alpha: float = 0.05) -> np.ndarray:
    """
    Minimize Conditional Value at Risk using linear programming.
    
    Uses Rockafellar-Uryasev formulation:
    - CVaR can be computed via linear programming
    - Much faster than simulation-based approaches
    """
    import cvxpy as cp
    
    T, n = returns.shape
    w = cp.Variable(n)
    z = cp.Variable(T)  # Auxiliary variables for CVaR
    gamma = cp.Variable()  # VaR threshold
    
    portfolio_returns = returns.values @ w
    
    # CVaR constraints
    constraints = [
        z >= 0,
        z >= -portfolio_returns - gamma,
        cp.sum(w) == 1,
        w >= 0.01,  # Min 1%
        w <= 0.40,  # Max 40%
    ]
    
    # Objective: minimize CVaR
    cvar = gamma + (1 / (T * alpha)) * cp.sum(z)
    
    problem = cp.Problem(cp.Minimize(cvar), constraints)
    problem.solve()
    
    return w.value
```

### 3.4 Hierarchical Risk Parity (HRP)

**Why**: Handles correlated assets better than traditional optimization.

```python
def hierarchical_risk_parity(returns: pd.DataFrame) -> np.ndarray:
    """
    HRP algorithm (López de Prado, 2016):
    1. Tree clustering based on correlation distance
    2. Quasi-diagonalization of covariance matrix
    3. Recursive bisection to allocate weights
    
    Benefits:
    - No matrix inversion (more stable)
    - Handles correlated assets naturally
    - Less sensitive to estimation error
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    
    cov = returns.cov()
    corr = returns.corr()
    
    # 1. Correlation distance
    dist = np.sqrt((1 - corr) / 2)
    dist_condensed = squareform(dist.values, checks=False)
    
    # 2. Hierarchical clustering
    link = linkage(dist_condensed, method="ward")
    sort_ix = leaves_list(link)
    
    # 3. Recursive bisection
    weights = _recursive_bisection(cov.values, sort_ix)
    
    return weights
```

### 3.5 Black-Litterman with Dip Signals as Views

**Why**: Combines market equilibrium with our dip signals as "views".

```python
def black_litterman_with_dips(
    cov: np.ndarray,
    market_caps: np.ndarray,
    dip_signals: dict[str, float],  # symbol -> dip strength
    tau: float = 0.05,
    view_confidence: float = 0.5,
) -> np.ndarray:
    """
    Black-Litterman with dip signals as views.
    
    - Prior: Market cap weighted equilibrium
    - Views: "Stock X is oversold, expect mean reversion"
    - Posterior: Blend of prior + views weighted by confidence
    
    This is a principled way to incorporate our signal scanner
    into portfolio optimization.
    """
    # Equilibrium returns (implied by market cap weights)
    market_weights = market_caps / market_caps.sum()
    risk_aversion = 2.5  # Standard assumption
    pi = risk_aversion * cov @ market_weights  # Equilibrium excess returns
    
    # Convert dip signals to views
    # P = pick matrix, Q = expected excess return
    views = []
    for symbol, strength in dip_signals.items():
        if strength > 0:
            # View: this stock will outperform by strength * base_return
            views.append((symbol, strength * 0.05))  # 5% base return for strong dip
    
    # ... Black-Litterman math to compute posterior ...
    
    return posterior_weights
```

---

## 4. Hyperparameter Optimization (HPO)

### 4.1 Parameters to Optimize

#### Per-Stock Parameters

| Parameter | Range | What It Controls |
|-----------|-------|------------------|
| `vol_lookback` | [21, 63, 126, 252] | Days for volatility estimation |
| `corr_lookback` | [63, 126, 252] | Days for correlation estimation |
| `ewma_halflife` | [10, 21, 42, 63] | Exponential decay for recent data |
| `outlier_threshold` | [2.5, 3.0, 3.5, 4.0] | Z-score for outlier removal |
| `regime_sensitivity` | [0.5, 1.0, 1.5] | How quickly to detect regime change |

#### Portfolio-Level Parameters

| Parameter | Range | What It Controls |
|-----------|-------|------------------|
| `cov_shrinkage` | [0, 0.25, 0.5, 0.75, 1.0] | Ledoit-Wolf shrinkage intensity |
| `min_weight` | [0.01, 0.02, 0.05] | Minimum position size |
| `max_weight` | [0.20, 0.30, 0.40] | Maximum position size |
| `turnover_penalty` | [0, 0.001, 0.005] | Transaction cost penalty |
| `risk_target` | [0.10, 0.15, 0.20] | Target portfolio volatility |

### 4.2 Optimization Objective

**What we optimize**: Out-of-sample portfolio quality metrics.

```python
def hpo_objective(params: dict, train_returns: pd.DataFrame, test_returns: pd.DataFrame) -> float:
    """
    Objective for HPO: maximize out-of-sample risk-adjusted quality.
    
    Metrics (all computed out-of-sample):
    - Sharpe ratio (even without return forecasts, we can measure realized Sharpe)
    - Sortino ratio (downside-focused)
    - Calmar ratio (return / max drawdown)
    - Stability of weights (low turnover is good)
    
    We use a composite score to avoid overfitting to one metric.
    """
    # Build portfolio with train data using these params
    weights = build_portfolio(train_returns, params)
    
    # Evaluate on test data
    test_portfolio_returns = test_returns @ weights
    
    sharpe = test_portfolio_returns.mean() / test_portfolio_returns.std() * np.sqrt(252)
    sortino = compute_sortino(test_portfolio_returns)
    max_dd = compute_max_drawdown(test_portfolio_returns)
    calmar = test_portfolio_returns.mean() * 252 / abs(max_dd) if max_dd != 0 else 0
    
    # Composite score (weighted average)
    score = 0.4 * sharpe + 0.3 * sortino + 0.3 * calmar
    
    return score
```

### 4.3 Validation Protocol (Overfitting Prevention)

#### Walk-Forward Validation

```
Data: |-------- 5 years --------|

Fold 1: [Train: Year 1-2] [Gap] [Test: Year 3]
Fold 2: [Train: Year 1-3] [Gap] [Test: Year 4]
Fold 3: [Train: Year 1-4] [Gap] [Test: Year 5]

Gap = 21 trading days (avoid lookahead bias)
```

```python
def walk_forward_validation(
    returns: pd.DataFrame,
    param_grid: dict,
    n_folds: int = 3,
    gap_days: int = 21,
) -> dict:
    """
    Walk-forward validation for HPO.
    
    Rules:
    - Minimum 2 years training data
    - 21-day gap between train/test (avoid lookahead)
    - Test period = 1 year
    - Report mean ± std across folds
    """
    fold_scores = []
    
    for fold in range(n_folds):
        train_end = ...
        test_start = train_end + gap_days
        test_end = test_start + 252
        
        train_data = returns.iloc[:train_end]
        test_data = returns.iloc[test_start:test_end]
        
        # Grid search on this fold
        best_params, best_score = grid_search(param_grid, train_data, test_data)
        fold_scores.append(best_score)
    
    return {
        "mean_score": np.mean(fold_scores),
        "std_score": np.std(fold_scores),
        "is_stable": np.std(fold_scores) < 0.3 * np.mean(fold_scores),
    }
```

#### Stability Tests

```python
def stability_test(params: dict, returns: pd.DataFrame, n_bootstrap: int = 100) -> dict:
    """
    Test stability of optimal parameters.
    
    Procedure:
    1. Bootstrap resample the returns data
    2. Re-run HPO on each bootstrap sample
    3. Check if optimal params are consistent
    
    If params change wildly across bootstraps, they're overfit.
    """
    param_samples = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        boot_returns = returns.sample(frac=1.0, replace=True)
        
        # Re-optimize
        best_params = optimize_params(boot_returns)
        param_samples.append(best_params)
    
    # Check consistency
    stability = {}
    for param_name in params.keys():
        values = [p[param_name] for p in param_samples]
        stability[param_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "cv": np.std(values) / np.mean(values),  # Coefficient of variation
            "is_stable": np.std(values) / np.mean(values) < 0.2,
        }
    
    return stability
```

### 4.4 HPO Implementation

```python
class HPOEngine:
    """
    Hyperparameter Optimization Engine.
    
    Runs as scheduled background job (nightly/weekly).
    Results are cached and used by real-time recommendations.
    """
    
    def __init__(self, db_session, cache_client):
        self.db = db_session
        self.cache = cache_client
    
    async def run_full_optimization(self) -> HPOResult:
        """
        Full HPO run. Expensive - runs nightly.
        """
        # 1. Fetch all price data
        returns = await self.fetch_returns()
        
        # 2. Per-stock parameter optimization
        stock_params = {}
        for symbol in returns.columns:
            stock_params[symbol] = self.optimize_stock_params(returns[symbol])
        
        # 3. Portfolio-level parameter optimization
        portfolio_params = self.optimize_portfolio_params(returns, stock_params)
        
        # 4. Validation
        validation = self.validate_params(returns, stock_params, portfolio_params)
        
        # 5. Cache results
        result = HPOResult(
            stock_params=stock_params,
            portfolio_params=portfolio_params,
            validation=validation,
            timestamp=datetime.now(),
        )
        await self.cache.set("hpo:latest", result.to_json(), ttl=86400 * 7)
        
        return result
    
    def optimize_stock_params(self, stock_returns: pd.Series) -> dict:
        """
        Optimize per-stock parameters.
        
        Objective: Stable volatility estimates with good out-of-sample fit.
        """
        param_grid = {
            "vol_lookback": [21, 42, 63, 126],
            "ewma_halflife": [10, 21, 42],
            "outlier_zscore": [2.5, 3.0, 3.5],
        }
        
        best_score = -np.inf
        best_params = None
        
        for params in itertools.product(*param_grid.values()):
            params_dict = dict(zip(param_grid.keys(), params))
            
            # Walk-forward score for volatility forecasting
            score = self.eval_vol_forecast(stock_returns, params_dict)
            
            if score > best_score:
                best_score = score
                best_params = params_dict
        
        return best_params
```

---

## 5. Scheduled Jobs

### 5.1 Job Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    WEEKLY (Sunday Night)                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  HPO Full Optimization                                  │ │
│  │  - Per-stock parameter sweep                            │ │
│  │  - Portfolio parameter sweep                            │ │
│  │  - Walk-forward validation                              │ │
│  │  - Stability testing                                    │ │
│  │  Duration: 1-4 hours                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    DAILY (After Market Close)                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Risk Model Update                                      │ │
│  │  - Update covariance matrix                             │ │
│  │  - Refresh volatility estimates                         │ │
│  │  - Regime detection                                     │ │
│  │  Duration: 5-15 minutes                                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Signal Scanner Update                                  │ │
│  │  - Compute all technical signals                        │ │
│  │  - Update buy scores                                    │ │
│  │  Duration: 2-5 minutes                                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ON-DEMAND (Real-Time API)                 │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Portfolio Recommendation                               │ │
│  │  - Use cached HPO params                                │ │
│  │  - Use cached risk model                                │ │
│  │  - Generate allocation recommendation                   │ │
│  │  Duration: < 1 second                                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Celery Job Definitions

```python
# app/jobs/definitions.py

@celery.task(name="jobs.hpo_full_optimization")
def hpo_full_optimization():
    """
    Weekly HPO job. Runs Sunday night.
    
    - Optimizes per-stock parameters
    - Optimizes portfolio parameters
    - Runs validation
    - Caches results for 7 days
    """
    ...

@celery.task(name="jobs.risk_model_update")
def risk_model_update():
    """
    Daily risk model update. Runs after market close.
    
    - Updates covariance matrix
    - Computes current volatilities
    - Detects regime
    - Caches for 24 hours
    """
    ...

@celery.task(name="jobs.signal_scanner_update")
def signal_scanner_update():
    """
    Daily signal scanner. Runs after market close.
    
    - Computes all technical signals
    - Updates buy scores
    - Caches for 24 hours
    """
    ...
```

---

## 6. API Endpoints

### 6.1 Endpoint Design

```python
# Real-time recommendations (uses cached data)
GET /api/recommendations/allocation
    ?portfolio_value=10000
    &inflow=1000
    &current_holdings={"AAPL": 0.3, "GOOG": 0.2}
    
Response:
{
    "recommendations": [
        {
            "symbol": "VOO",
            "action": "BUY",
            "amount_eur": 500,
            "reason": "Improves diversification, reduces concentration risk",
            "confidence": "HIGH"
        },
        ...
    ],
    "portfolio_metrics": {
        "current": {
            "volatility": 0.22,
            "var_95": -0.032,
            "diversification_score": 0.45,
            "effective_n": 2.3
        },
        "after_trades": {
            "volatility": 0.18,
            "var_95": -0.025,
            "diversification_score": 0.72,
            "effective_n": 4.1
        }
    },
    "risk_warnings": [
        "Portfolio is 60% concentrated in tech sector",
        "Correlation with SPY is very high (0.95)"
    ]
}

# Analytics endpoint
GET /api/analytics/portfolio
    ?holdings={"AAPL": 0.3, "GOOG": 0.2, "VOO": 0.5}

Response:
{
    "risk_decomposition": {
        "total_volatility": 0.18,
        "risk_contributions": {
            "AAPL": 0.35,
            "GOOG": 0.28,
            "VOO": 0.37
        }
    },
    "tail_risk": {
        "var_95_daily": -0.025,
        "cvar_95_daily": -0.038,
        "max_drawdown_1y": -0.15
    },
    "diversification": {
        "effective_n": 2.8,
        "diversification_ratio": 1.12,
        "concentration_warning": true
    },
    "regime": {
        "current": "bull_low",
        "market_vol_vs_avg": 0.85,
        "recommendation": "Normal risk budget appropriate"
    }
}

# Signal scanner (already built)
GET /api/recommendations/signals

# HPO status
GET /api/admin/hpo/status
Response:
{
    "last_run": "2025-12-22T02:00:00Z",
    "next_scheduled": "2025-12-29T02:00:00Z",
    "validation_score": 0.85,
    "is_stable": true,
    "top_params": {
        "cov_shrinkage": 0.5,
        "vol_lookback_median": 63,
        ...
    }
}
```

### 6.2 User-Facing Translation Layer

```python
def translate_to_user_friendly(technical_result: dict) -> dict:
    """
    Translate quant jargon to user-friendly language.
    """
    translations = {
        "volatility": lambda v: f"Your portfolio typically swings ±{v*100:.0f}% per year",
        "var_95": lambda v: f"On a bad day (1 in 20), you could lose up to €{abs(v*10000):.0f}",
        "cvar_95": lambda v: f"In a crash, expect to lose around €{abs(v*10000):.0f}",
        "effective_n": lambda n: f"You effectively have {n:.1f} independent investments",
        "diversification_ratio": lambda d: (
            "Well diversified" if d > 1.3 else 
            "Moderately diversified" if d > 1.1 else 
            "Poorly diversified - your stocks move together"
        ),
    }
    
    user_friendly = {}
    for key, value in technical_result.items():
        if key in translations:
            user_friendly[key] = {
                "value": value,
                "explanation": translations[key](value)
            }
        else:
            user_friendly[key] = value
    
    return user_friendly
```

---

## 7. Data Requirements

### 7.1 Price Data

| Data | Source | Frequency | Lookback |
|------|--------|-----------|----------|
| Daily OHLCV | Yahoo Finance | Daily | 5 years |
| Adjusted Close | Yahoo Finance | Daily | 5 years |
| Dividends | Yahoo Finance | As available | 5 years |

### 7.2 Derived Data (Computed)

| Data | Computation | Storage |
|------|-------------|---------|
| Returns | log(P_t / P_{t-1}) | Cache 24h |
| Covariance | Shrunk covariance matrix | Cache 24h |
| Volatility | Rolling/EWMA volatility | Cache 24h |
| Correlations | Rolling correlations | Cache 24h |

### 7.3 Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| No intraday data | Can't capture intraday vol | Daily data sufficient for retail |
| No fundamental data | Can't do factor analysis | Use sector proxies |
| No options data | Can't compute implied vol | Historical vol is fine |
| Yahoo Finance reliability | Occasional gaps | Fallback to cached data |

---

## 8. Implementation Roadmap

### Phase 1: Core Analytics (Week 1)
- [ ] Risk decomposition module
- [ ] Tail risk computation (VaR/CVaR)
- [ ] Diversification metrics
- [ ] Basic regime detection
- [ ] API endpoint: `/api/analytics/portfolio`

### Phase 2: Optimization Engine (Week 2)
- [ ] Risk Parity optimizer
- [ ] Minimum Variance optimizer
- [ ] CVaR optimizer
- [ ] Hierarchical Risk Parity
- [ ] API endpoint: `/api/recommendations/allocation`

### Phase 3: HPO Framework (Week 3)
- [ ] Per-stock parameter grid
- [ ] Portfolio parameter grid
- [ ] Walk-forward validation
- [ ] Stability testing
- [ ] Celery job: weekly HPO

### Phase 4: Integration & Polish (Week 4)
- [ ] User-friendly translation layer
- [ ] Risk warnings and alerts
- [ ] Frontend integration
- [ ] Performance optimization
- [ ] Documentation

---

## 9. Success Metrics

| Metric | Target | How Measured |
|--------|--------|--------------|
| Recommendation latency | < 500ms | API response time |
| HPO job duration | < 4 hours | Celery task timing |
| Out-of-sample Sharpe | > 0.5 | Walk-forward backtest |
| Parameter stability | CV < 20% | Bootstrap stability test |
| User understanding | > 80% clarity | User surveys |

---

## 10. Appendix: Mathematical Details

### A. Ledoit-Wolf Shrinkage

The sample covariance matrix is often poorly estimated with limited data. Shrinkage improves stability:

```
Σ_shrunk = α * Σ_sample + (1-α) * F

Where:
- α = shrinkage intensity (0 to 1)
- F = shrinkage target (typically scaled identity or single-factor model)
```

### B. Cornish-Fisher VaR Adjustment

Standard VaR assumes normality. Cornish-Fisher adjusts for skewness/kurtosis:

```
CF-VaR = μ + σ * [z + (z² - 1)*S/6 + (z³ - 3z)*K/24 - (2z³ - 5z)*S²/36]

Where:
- z = standard normal quantile
- S = skewness
- K = excess kurtosis
```

### C. Risk Parity Optimization

Find weights such that each asset contributes equally to portfolio risk:

```
Target: RC_i = w_i * (Σw)_i / σ_p = 1/N for all i

Optimization: min Σ(RC_i - 1/N)²
Subject to: Σw_i = 1, w_i ≥ 0
```
