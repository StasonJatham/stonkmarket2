"""
Alpha Gauntlet - Hierarchy of Truth for Strategy Validation.

This module implements the "Gauntlet of Trials" concept:
Strategy must prove itself against increasingly difficult benchmarks.

Hierarchy:
1. Strategy vs Buy & Hold (same asset)
2. Strategy vs SPY B&H (market benchmark)
3. Consistency across market regimes
4. Risk-adjusted performance (Sharpe, Calmar)

Only strategies that pass ALL trials are considered valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from app.quant_engine.backtest_v2.walk_forward import WFOResult, WFOSummary
from app.quant_engine.core import MarketRegime, RegimeService, get_regime_service

logger = logging.getLogger(__name__)


class GauntletTrial(str, Enum):
    """Individual trials in the gauntlet."""

    POSITIVE_RETURNS = "POSITIVE_RETURNS"
    BEAT_ASSET_BH = "BEAT_ASSET_BH"
    BEAT_SPY_BH = "BEAT_SPY_BH"
    RISK_ADJUSTED = "RISK_ADJUSTED"
    REGIME_CONSISTENCY = "REGIME_CONSISTENCY"
    WFO_VALIDATION = "WFO_VALIDATION"


class GauntletVerdict(str, Enum):
    """Final verdict from the gauntlet."""

    CERTIFIED_ALPHA = "CERTIFIED_ALPHA"  # Passed all trials
    CONDITIONAL_ALPHA = "CONDITIONAL_ALPHA"  # Passed most, minor concerns
    USE_BH_ASSET = "USE_BH_ASSET"  # Strategy worse than B&H, use B&H for asset
    USE_SPY_BH = "USE_SPY_BH"  # Both fail, default to SPY
    REJECTED = "REJECTED"  # Strategy is harmful


@dataclass
class TrialResult:
    """Result of a single trial."""

    trial: GauntletTrial
    passed: bool
    score: float  # 0-100 score
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegimePerformance:
    """Strategy performance in a specific regime."""

    regime: MarketRegime
    n_days: int
    return_pct: float
    trades: int
    win_rate: float
    max_drawdown_pct: float


@dataclass
class GauntletResult:
    """Complete gauntlet evaluation result."""

    verdict: GauntletVerdict
    message: str
    overall_score: float  # 0-100 composite score

    # Trial results
    trials: list[TrialResult] = field(default_factory=list)

    # Detailed comparisons
    strategy_return: float = 0.0
    asset_bh_return: float = 0.0
    spy_bh_return: float = 0.0

    # Risk metrics
    strategy_sharpe: float = 0.0
    strategy_calmar: float = 0.0
    strategy_max_dd: float = 0.0

    # Regime breakdown
    regime_performance: list[RegimePerformance] = field(default_factory=list)

    # Recommendation
    recommended_action: str = ""
    fallback_strategy: str | None = None

    def passed_all(self) -> bool:
        """Check if all trials passed."""
        return all(t.passed for t in self.trials)

    def get_failed_trials(self) -> list[GauntletTrial]:
        """Get list of failed trials."""
        return [t.trial for t in self.trials if not t.passed]


@dataclass
class GauntletConfig:
    """Alpha Gauntlet configuration."""

    # Return thresholds
    min_absolute_return: float = 0.0
    min_alpha_vs_bh: float = 0.0  # Must beat B&H by this much
    min_alpha_vs_spy: float = 0.0

    # Risk thresholds
    min_sharpe_ratio: float = 0.3
    min_calmar_ratio: float = 0.3
    max_drawdown: float = 50.0

    # Consistency thresholds
    min_regime_win_rate: float = 45.0  # Must be > 45% in each regime
    max_regime_drawdown: float = 60.0  # No regime can have > 60% drawdown

    # Scoring weights
    return_weight: float = 0.3
    risk_weight: float = 0.3
    consistency_weight: float = 0.2
    alpha_weight: float = 0.2


class AlphaGauntlet:
    """
    The Alpha Gauntlet - Rigorous Strategy Validation.

    A strategy must prove it generates true alpha, not just luck.
    The gauntlet tests:
    1. Absolute returns (positive)
    2. Relative returns vs asset B&H
    3. Relative returns vs market (SPY)
    4. Risk-adjusted returns
    5. Consistency across market regimes

    Only strategies passing all trials earn "Certified Alpha" status.
    """

    def __init__(
        self,
        config: GauntletConfig | None = None,
        regime_service: RegimeService | None = None,
    ):
        self.config = config or GauntletConfig()
        self.regime_service = regime_service or get_regime_service()

    def run_gauntlet(
        self,
        strategy_return: float,
        strategy_sharpe: float,
        strategy_calmar: float,
        strategy_max_dd: float,
        strategy_win_rate: float,
        asset_bh_return: float,
        spy_bh_return: float,
        wfo_summary: WFOSummary | None = None,
        regime_breakdown: list[RegimePerformance] | None = None,
    ) -> GauntletResult:
        """
        Run the complete Alpha Gauntlet.

        Args:
            strategy_return: Strategy total return %
            strategy_sharpe: Strategy Sharpe ratio
            strategy_calmar: Strategy Calmar ratio
            strategy_max_dd: Strategy max drawdown %
            strategy_win_rate: Strategy win rate %
            asset_bh_return: Buy & Hold return for same asset %
            spy_bh_return: SPY Buy & Hold return %
            wfo_summary: Walk-forward optimization results
            regime_breakdown: Optional breakdown by market regime

        Returns:
            GauntletResult with verdict and detailed analysis
        """
        trials: list[TrialResult] = []

        # Trial 1: Positive Returns
        trials.append(self._trial_positive_returns(strategy_return))

        # Trial 2: Beat Asset B&H
        trials.append(self._trial_beat_asset_bh(strategy_return, asset_bh_return))

        # Trial 3: Beat SPY B&H
        trials.append(self._trial_beat_spy_bh(strategy_return, spy_bh_return))

        # Trial 4: Risk-Adjusted Performance
        trials.append(self._trial_risk_adjusted(
            strategy_sharpe, strategy_calmar, strategy_max_dd
        ))

        # Trial 5: Regime Consistency (if data available)
        if regime_breakdown:
            trials.append(self._trial_regime_consistency(regime_breakdown))

        # Trial 6: WFO Validation (if available)
        if wfo_summary:
            trials.append(self._trial_wfo(wfo_summary))

        # Calculate overall score
        overall_score = self._calculate_overall_score(trials)

        # Determine verdict
        verdict, message, recommended_action, fallback = self._determine_verdict(
            trials=trials,
            strategy_return=strategy_return,
            asset_bh_return=asset_bh_return,
            spy_bh_return=spy_bh_return,
            overall_score=overall_score,
        )

        return GauntletResult(
            verdict=verdict,
            message=message,
            overall_score=overall_score,
            trials=trials,
            strategy_return=strategy_return,
            asset_bh_return=asset_bh_return,
            spy_bh_return=spy_bh_return,
            strategy_sharpe=strategy_sharpe,
            strategy_calmar=strategy_calmar,
            strategy_max_dd=strategy_max_dd,
            regime_performance=regime_breakdown or [],
            recommended_action=recommended_action,
            fallback_strategy=fallback,
        )

    def _trial_positive_returns(self, strategy_return: float) -> TrialResult:
        """Trial 1: Strategy must have positive returns."""
        passed = strategy_return > self.config.min_absolute_return
        score = min(100, max(0, strategy_return))  # Cap at 100

        return TrialResult(
            trial=GauntletTrial.POSITIVE_RETURNS,
            passed=passed,
            score=score,
            message=f"Return: {strategy_return:.1f}% ({'PASS' if passed else 'FAIL'})",
            details={"strategy_return": strategy_return},
        )

    def _trial_beat_asset_bh(
        self,
        strategy_return: float,
        asset_bh_return: float,
    ) -> TrialResult:
        """Trial 2: Strategy must beat buy & hold of same asset."""
        alpha = strategy_return - asset_bh_return
        passed = alpha >= self.config.min_alpha_vs_bh

        # Score based on alpha (capped at ±50%)
        score = min(100, max(0, 50 + alpha))

        return TrialResult(
            trial=GauntletTrial.BEAT_ASSET_BH,
            passed=passed,
            score=score,
            message=f"Alpha vs B&H: {alpha:+.1f}% ({'PASS' if passed else 'FAIL'})",
            details={
                "strategy_return": strategy_return,
                "asset_bh_return": asset_bh_return,
                "alpha": alpha,
            },
        )

    def _trial_beat_spy_bh(
        self,
        strategy_return: float,
        spy_bh_return: float,
    ) -> TrialResult:
        """Trial 3: Strategy must beat SPY buy & hold."""
        alpha = strategy_return - spy_bh_return
        passed = alpha >= self.config.min_alpha_vs_spy

        score = min(100, max(0, 50 + alpha))

        return TrialResult(
            trial=GauntletTrial.BEAT_SPY_BH,
            passed=passed,
            score=score,
            message=f"Alpha vs SPY: {alpha:+.1f}% ({'PASS' if passed else 'FAIL'})",
            details={
                "strategy_return": strategy_return,
                "spy_bh_return": spy_bh_return,
                "alpha": alpha,
            },
        )

    def _trial_risk_adjusted(
        self,
        sharpe: float,
        calmar: float,
        max_dd: float,
    ) -> TrialResult:
        """Trial 4: Risk-adjusted performance check."""
        sharpe_pass = sharpe >= self.config.min_sharpe_ratio
        calmar_pass = calmar >= self.config.min_calmar_ratio
        dd_pass = max_dd <= self.config.max_drawdown

        passed = sharpe_pass and calmar_pass and dd_pass

        # Score based on all three metrics
        sharpe_score = min(100, max(0, sharpe * 50))
        calmar_score = min(100, max(0, calmar * 50))
        dd_score = max(0, 100 - max_dd)

        score = (sharpe_score + calmar_score + dd_score) / 3

        issues = []
        if not sharpe_pass:
            issues.append(f"Sharpe {sharpe:.2f} < {self.config.min_sharpe_ratio}")
        if not calmar_pass:
            issues.append(f"Calmar {calmar:.2f} < {self.config.min_calmar_ratio}")
        if not dd_pass:
            issues.append(f"MaxDD {max_dd:.1f}% > {self.config.max_drawdown}%")

        message = "Risk metrics OK" if passed else f"Risk issues: {', '.join(issues)}"

        return TrialResult(
            trial=GauntletTrial.RISK_ADJUSTED,
            passed=passed,
            score=score,
            message=message,
            details={
                "sharpe_ratio": sharpe,
                "calmar_ratio": calmar,
                "max_drawdown": max_dd,
            },
        )

    def _trial_regime_consistency(
        self,
        regime_breakdown: list[RegimePerformance],
    ) -> TrialResult:
        """Trial 5: Performance must be consistent across regimes."""
        issues = []

        for perf in regime_breakdown:
            if perf.win_rate < self.config.min_regime_win_rate:
                issues.append(f"{perf.regime.value}: win rate {perf.win_rate:.0f}%")
            if perf.max_drawdown_pct > self.config.max_regime_drawdown:
                issues.append(f"{perf.regime.value}: drawdown {perf.max_drawdown_pct:.0f}%")

        passed = len(issues) == 0

        # Score based on worst regime
        if regime_breakdown:
            min_win_rate = min(p.win_rate for p in regime_breakdown)
            max_dd = max(p.max_drawdown_pct for p in regime_breakdown)
            score = (min_win_rate + (100 - max_dd)) / 2
        else:
            score = 50

        message = "Consistent across regimes" if passed else f"Regime issues: {'; '.join(issues[:2])}"

        return TrialResult(
            trial=GauntletTrial.REGIME_CONSISTENCY,
            passed=passed,
            score=score,
            message=message,
            details={"regimes": [p.regime.value for p in regime_breakdown]},
        )

    def _trial_wfo(self, wfo_summary: WFOSummary) -> TrialResult:
        """Trial 6: Walk-forward optimization must pass."""
        passed = wfo_summary.result == WFOResult.PASSED

        # Score based on WFO metrics
        if wfo_summary.avg_oos_return_pct > 0:
            score = min(100, 50 + wfo_summary.avg_oos_return_pct)
        else:
            score = max(0, 50 + wfo_summary.avg_oos_return_pct)

        return TrialResult(
            trial=GauntletTrial.WFO_VALIDATION,
            passed=passed,
            score=score,
            message=f"WFO: {wfo_summary.result.value}",
            details={
                "wfo_result": wfo_summary.result.value,
                "avg_oos_return": wfo_summary.avg_oos_return_pct,
                "pct_folds_profitable": wfo_summary.pct_folds_profitable,
            },
        )

    def _calculate_overall_score(self, trials: list[TrialResult]) -> float:
        """Calculate weighted overall score."""
        if not trials:
            return 0.0

        # Weight each trial type
        weights = {
            GauntletTrial.POSITIVE_RETURNS: self.config.return_weight,
            GauntletTrial.BEAT_ASSET_BH: self.config.alpha_weight,
            GauntletTrial.BEAT_SPY_BH: self.config.alpha_weight / 2,
            GauntletTrial.RISK_ADJUSTED: self.config.risk_weight,
            GauntletTrial.REGIME_CONSISTENCY: self.config.consistency_weight,
            GauntletTrial.WFO_VALIDATION: self.config.consistency_weight,
        }

        total_weight = 0.0
        weighted_score = 0.0

        for trial in trials:
            weight = weights.get(trial.trial, 0.1)
            weighted_score += trial.score * weight
            total_weight += weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def _determine_verdict(
        self,
        trials: list[TrialResult],
        strategy_return: float,
        asset_bh_return: float,
        spy_bh_return: float,
        overall_score: float,
    ) -> tuple[GauntletVerdict, str, str, str | None]:
        """
        Determine final verdict based on trial results.

        Returns: (verdict, message, recommended_action, fallback_strategy)
        """
        failed_trials = [t.trial for t in trials if not t.passed]
        passed_trials = [t.trial for t in trials if t.passed]

        # All passed -> Certified Alpha
        if not failed_trials:
            return (
                GauntletVerdict.CERTIFIED_ALPHA,
                f"Strategy passed all {len(trials)} trials with score {overall_score:.0f}/100",
                "Use this strategy with confidence",
                None,
            )

        # Minor failures only -> Conditional Alpha
        minor_failures = {GauntletTrial.BEAT_SPY_BH, GauntletTrial.REGIME_CONSISTENCY}
        if all(f in minor_failures for f in failed_trials) and overall_score >= 60:
            return (
                GauntletVerdict.CONDITIONAL_ALPHA,
                f"Strategy passed core trials but has minor concerns: {[f.value for f in failed_trials]}",
                "Use strategy with monitoring",
                None,
            )

        # Failed to beat asset B&H -> Use B&H for this asset
        if GauntletTrial.BEAT_ASSET_BH in failed_trials and asset_bh_return > 0:
            if asset_bh_return > spy_bh_return:
                return (
                    GauntletVerdict.USE_BH_ASSET,
                    f"Strategy ({strategy_return:.1f}%) underperforms B&H ({asset_bh_return:.1f}%)",
                    "Use Buy & Hold for this asset",
                    "BUY_AND_HOLD",
                )

        # Both strategy and asset B&H fail -> Default to SPY
        if spy_bh_return > strategy_return and spy_bh_return > asset_bh_return:
            return (
                GauntletVerdict.USE_SPY_BH,
                f"Both strategy and asset B&H underperform SPY ({spy_bh_return:.1f}%)",
                "Consider SPY or broad market index instead",
                "SPY_BUY_AND_HOLD",
            )

        # Risk failures or negative returns -> Rejected
        if (
            GauntletTrial.POSITIVE_RETURNS in failed_trials
            or GauntletTrial.RISK_ADJUSTED in failed_trials
        ):
            return (
                GauntletVerdict.REJECTED,
                f"Strategy fails critical tests: {[f.value for f in failed_trials]}",
                "Do not use this strategy",
                None,
            )

        # Default: conditional based on score
        if overall_score >= 50:
            return (
                GauntletVerdict.CONDITIONAL_ALPHA,
                f"Mixed results (score: {overall_score:.0f}). Failed: {[f.value for f in failed_trials]}",
                "Review and improve strategy before use",
                "BUY_AND_HOLD" if asset_bh_return > 0 else None,
            )

        return (
            GauntletVerdict.REJECTED,
            f"Strategy fails too many trials (score: {overall_score:.0f})",
            "Redesign strategy or use fallback",
            "SPY_BUY_AND_HOLD",
        )


def quick_gauntlet(
    strategy_return: float,
    strategy_sharpe: float,
    strategy_max_dd: float,
    asset_bh_return: float,
) -> GauntletVerdict:
    """
    Quick gauntlet check without full analysis.

    Useful for filtering before detailed analysis.
    """
    gauntlet = AlphaGauntlet()
    calmar = strategy_return / strategy_max_dd if strategy_max_dd > 0 else 0

    result = gauntlet.run_gauntlet(
        strategy_return=strategy_return,
        strategy_sharpe=strategy_sharpe,
        strategy_calmar=calmar,
        strategy_max_dd=strategy_max_dd,
        strategy_win_rate=50,  # Assume average
        asset_bh_return=asset_bh_return,
        spy_bh_return=0,  # Not available in quick check
    )

    return result.verdict
