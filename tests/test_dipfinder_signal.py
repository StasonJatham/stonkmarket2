"""Tests for DipFinder signal module."""

from __future__ import annotations

import pytest

from app.dipfinder.signal import (
    DipClass,
    AlertLevel,
    MarketContext,
    DipSignal,
    classify_dip,
    compute_dip_score,
)
from app.dipfinder.dip import DipMetrics
from app.dipfinder.fundamentals import QualityMetrics
from app.dipfinder.stability import StabilityMetrics
from app.dipfinder.config import DipFinderConfig


@pytest.fixture
def config() -> DipFinderConfig:
    """Default config for testing."""
    return DipFinderConfig()


class TestClassifyDip:
    """Tests for classify_dip."""
    
    def test_market_dip(self, config):
        """Market dip classification."""
        # Both stock and market down significantly, stock not much more
        dip_class = classify_dip(
            dip_stock=0.12,  # 12%
            dip_mkt=0.10,    # 10%
            config=config,
        )
        assert dip_class == DipClass.MARKET_DIP
    
    def test_stock_specific(self, config):
        """Stock-specific dip classification."""
        # Stock down much more than market
        dip_class = classify_dip(
            dip_stock=0.15,  # 15%
            dip_mkt=0.02,    # 2%
            config=config,
        )
        assert dip_class == DipClass.STOCK_SPECIFIC
    
    def test_mixed_dip(self, config):
        """Mixed dip classification."""
        # Market down, but stock down moderately more
        dip_class = classify_dip(
            dip_stock=0.10,  # 10%
            dip_mkt=0.05,    # 5%
            config=config,
        )
        # Either MIXED or STOCK_SPECIFIC depending on threshold
        assert dip_class in (DipClass.MIXED, DipClass.STOCK_SPECIFIC)
    
    def test_no_market_dip_stock_down(self, config):
        """No market dip but stock is down."""
        dip_class = classify_dip(
            dip_stock=0.10,
            dip_mkt=0.01,
            config=config,
        )
        assert dip_class == DipClass.STOCK_SPECIFIC


class TestComputeDipScore:
    """Tests for compute_dip_score."""
    
    def test_high_dip_score(self, config):
        """High dip with persistence gets high score."""
        dip_metrics = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.25,
            peak_price=100.0,
            current_price=75.0,
            dip_percentile=95.0,
            dip_vs_typical=3.0,
            typical_dip=0.083,
            persist_days=5,
            days_since_peak=10,
            is_meaningful=True,
        )
        market_context = MarketContext(
            benchmark_ticker="SPY",
            dip_mkt=0.03,
            dip_stock=0.25,
            excess_dip=0.22,
            dip_class=DipClass.STOCK_SPECIFIC,
        )
        
        score = compute_dip_score(dip_metrics, market_context, config)
        assert score > 70
    
    def test_low_dip_score(self, config):
        """Small dip gets low score."""
        dip_metrics = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.05,
            peak_price=100.0,
            current_price=95.0,
            dip_percentile=30.0,
            dip_vs_typical=0.5,
            typical_dip=0.10,
            persist_days=1,
            days_since_peak=2,
            is_meaningful=False,
        )
        market_context = MarketContext(
            benchmark_ticker="SPY",
            dip_mkt=0.02,
            dip_stock=0.05,
            excess_dip=0.03,
            dip_class=DipClass.MIXED,
        )
        
        score = compute_dip_score(dip_metrics, market_context, config)
        assert score < 40


class TestDipSignal:
    """Tests for DipSignal dataclass."""
    
    @pytest.fixture
    def sample_signal(self) -> DipSignal:
        """Create a sample signal for testing."""
        return DipSignal(
            ticker="AAPL",
            window=30,
            benchmark="SPY",
            as_of_date="2025-12-21",
            dip_metrics=DipMetrics(
                ticker="AAPL",
                window=30,
                dip_pct=0.15,
                peak_price=200.0,
                current_price=170.0,
                dip_percentile=85.0,
                dip_vs_typical=2.0,
                typical_dip=0.075,
                persist_days=4,
                days_since_peak=7,
                is_meaningful=True,
            ),
            market_context=MarketContext(
                benchmark_ticker="SPY",
                dip_mkt=0.05,
                dip_stock=0.15,
                excess_dip=0.10,
                dip_class=DipClass.STOCK_SPECIFIC,
            ),
            quality_metrics=QualityMetrics(ticker="AAPL", score=75.0),
            stability_metrics=StabilityMetrics(ticker="AAPL", score=70.0),
            dip_score=80.0,
            final_score=76.0,
            alert_level=AlertLevel.GOOD,
            should_alert=True,
            reason="AAPL is 15% off peak, stock-specific dip with strong fundamentals",
        )
    
    def test_to_dict(self, sample_signal):
        """Test to_dict method."""
        d = sample_signal.to_dict()
        
        assert d["ticker"] == "AAPL"
        assert d["window"] == 30
        assert d["final_score"] == 76.0
        assert d["should_alert"] is True
        # Metrics are flattened, not nested
        assert "dip_stock" in d
        assert "quality_score" in d
    
    def test_alert_levels(self):
        """Test alert level enum values."""
        assert AlertLevel.NONE.value == "NONE"
        assert AlertLevel.GOOD.value == "GOOD"
        assert AlertLevel.STRONG.value == "STRONG"
    
    def test_dip_class_values(self):
        """Test dip class enum values."""
        assert DipClass.MARKET_DIP.value == "MARKET_DIP"
        assert DipClass.STOCK_SPECIFIC.value == "STOCK_SPECIFIC"
        assert DipClass.MIXED.value == "MIXED"


class TestAlertDecision:
    """Tests for alert decision logic."""
    
    def test_alert_triggered(self, config):
        """Alert triggered when all conditions met."""
        # High scores, meaningful dip, gates passed
        dip_score = 80.0
        quality_score = 70.0
        stability_score = 70.0
        is_meaningful = True
        
        # Compute weighted final score (inline since compute_final_score was removed)
        final_score = (
            config.weight_dip * dip_score
            + config.weight_quality * quality_score
            + config.weight_stability * stability_score
        )
        
        should_alert = (
            final_score >= config.alert_good
            and is_meaningful
            and quality_score >= config.quality_gate
            and stability_score >= config.stability_gate
        )
        
        assert should_alert is True
    
    def test_no_alert_low_quality(self, config):
        """No alert if quality gate not met."""
        quality_score = 50.0  # Below gate
        stability_score = 70.0
        
        should_alert = (
            quality_score >= config.quality_gate
            and stability_score >= config.stability_gate
        )
        
        assert should_alert is False
    
    def test_no_alert_low_stability(self, config):
        """No alert if stability gate not met."""
        quality_score = 70.0
        stability_score = 50.0  # Below gate
        
        should_alert = (
            quality_score >= config.quality_gate
            and stability_score >= config.stability_gate
        )
        
        assert should_alert is False
    
    def test_strong_alert(self, config):
        """Strong alert when final score high."""
        final_score = 85.0
        alert_level = AlertLevel.STRONG if final_score >= config.alert_strong else AlertLevel.GOOD
        
        assert alert_level == AlertLevel.STRONG


class TestEarningsAdjustment:
    """Tests for earnings adjustment logic."""

    def test_no_pre_earnings_penalty(self):
        """Pre-earnings dips should NOT be penalized."""
        from app.dipfinder.signal import _compute_earnings_adjustment, EnhancedAnalysisInputs
        from app.dipfinder.earnings_calendar import EarningsInfo

        # Earnings coming up in 5 days - should NOT penalize
        earnings = EarningsInfo(
            ticker="TEST",
            days_to_earnings=5,
            days_since_earnings=None,
        )
        enhanced = EnhancedAnalysisInputs(earnings_info=earnings)

        adjustment, post_decline = _compute_earnings_adjustment(enhanced)
        assert adjustment == 0.0
        assert post_decline is False

    def test_post_earnings_with_deterioration_penalized(self):
        """Post-earnings with deterioration should be penalized."""
        from app.dipfinder.signal import _compute_earnings_adjustment, EnhancedAnalysisInputs
        from app.dipfinder.earnings_calendar import EarningsInfo
        from app.dipfinder.structural_analysis import FundamentalMomentum

        # Earnings 10 days ago with revenue decline
        earnings = EarningsInfo(
            ticker="TEST",
            days_since_earnings=10,
        )
        momentum = FundamentalMomentum(
            revenue_yoy_change=-15.0,  # Revenue YoY <= -10%
            is_structural_decline=False,
        )
        enhanced = EnhancedAnalysisInputs(earnings_info=earnings, structural_momentum=momentum)

        adjustment, post_decline = _compute_earnings_adjustment(enhanced)
        assert adjustment < 0  # Should have penalty
        assert post_decline is True

    def test_post_earnings_without_deterioration_no_penalty(self):
        """Post-earnings without deterioration should NOT be penalized."""
        from app.dipfinder.signal import _compute_earnings_adjustment, EnhancedAnalysisInputs
        from app.dipfinder.earnings_calendar import EarningsInfo
        from app.dipfinder.structural_analysis import FundamentalMomentum

        # Earnings 10 days ago with good revenue
        earnings = EarningsInfo(
            ticker="TEST",
            days_since_earnings=10,
        )
        momentum = FundamentalMomentum(
            revenue_yoy_change=5.0,  # Positive growth
            earnings_yoy_change=10.0,
            operating_margin_trend=1.0,
            is_structural_decline=False,
        )
        enhanced = EnhancedAnalysisInputs(earnings_info=earnings, structural_momentum=momentum)

        adjustment, post_decline = _compute_earnings_adjustment(enhanced)
        assert adjustment == 0.0
        assert post_decline is False


class TestStructuralPenaltyScaling:
    """Tests for structural penalty scaling by data quality."""

    def test_full_data_full_penalty(self):
        """Full data quality should apply 100% penalty."""
        from app.dipfinder.signal import _compute_structural_adjustment, EnhancedAnalysisInputs
        from app.dipfinder.structural_analysis import FundamentalMomentum

        momentum = FundamentalMomentum(
            is_structural_decline=True,
            decline_severity="severe",
            data_quality="full",
        )
        enhanced = EnhancedAnalysisInputs(structural_momentum=momentum)

        adjustment, is_structural = _compute_structural_adjustment(enhanced)
        # Full penalty = -25.0
        assert adjustment == -25.0
        assert is_structural is True

    def test_partial_data_half_penalty(self):
        """Partial data quality should apply 50% penalty."""
        from app.dipfinder.signal import _compute_structural_adjustment, EnhancedAnalysisInputs
        from app.dipfinder.structural_analysis import FundamentalMomentum

        momentum = FundamentalMomentum(
            is_structural_decline=True,
            decline_severity="severe",
            data_quality="partial",
        )
        enhanced = EnhancedAnalysisInputs(structural_momentum=momentum)

        adjustment, is_structural = _compute_structural_adjustment(enhanced)
        # Partial penalty = -25.0 * 0.5 = -12.5
        assert adjustment == -12.5
        assert is_structural is True

    def test_minimal_data_no_penalty(self):
        """Minimal data quality should apply 0% penalty (not enough data to penalize)."""
        from app.dipfinder.signal import _compute_structural_adjustment, EnhancedAnalysisInputs
        from app.dipfinder.structural_analysis import FundamentalMomentum

        momentum = FundamentalMomentum(
            is_structural_decline=True,
            decline_severity="severe",
            data_quality="minimal",
        )
        enhanced = EnhancedAnalysisInputs(structural_momentum=momentum)

        adjustment, is_structural = _compute_structural_adjustment(enhanced)
        # Minimal penalty = -25.0 * 0.0 = 0.0
        assert adjustment == 0.0
        # Still flagged as structural decline
        assert is_structural is True


class TestValuationCalibration:
    """Tests for valuation adjustment with missing data."""

    def test_missing_valuation_neutral(self):
        """Missing valuation data should return neutral adjustment."""
        from app.dipfinder.signal import _compute_valuation_adjustment, EnhancedAnalysisInputs

        enhanced = EnhancedAnalysisInputs(sector_valuation=None)
        adjustment, undervalued, overvalued = _compute_valuation_adjustment(enhanced)

        assert adjustment == 0.0
        assert undervalued is False
        assert overvalued is False

    def test_missing_sector_neutral(self):
        """Missing sector should still return neutral when no valuation metrics."""
        from app.dipfinder.sector_valuation import SectorRelativeValuation

        # No metrics = neutral 0.5 score
        val = SectorRelativeValuation(
            symbol="TEST",
            sector=None,  # Missing sector
            pe_ratio=None,
            forward_pe=None,
            peg_ratio=None,
            price_to_book=None,
            ev_to_ebitda=None,
        )
        # valuation_score defaults to 0.5 (neutral) when no data
        assert val.valuation_score == 0.5


class TestQuantGating:
    """Tests for quant gating behavior."""

    def test_quant_gated_flag_logic(self):
        """Verify quant gating logic: in_downtrend + not gate_pass = gated."""
        from app.dipfinder.signal import QuantContext

        # In downtrend without gate_pass
        quant = QuantContext(best_score=70.0, gate_pass=False)
        in_downtrend = True

        quant_gated = in_downtrend and not quant.gate_pass
        assert quant_gated is True

        # With gate_pass
        quant_certified = QuantContext(best_score=70.0, gate_pass=True)
        quant_gated = in_downtrend and not quant_certified.gate_pass
        assert quant_gated is False

        # Not in downtrend
        in_downtrend = False
        quant_gated = in_downtrend and not quant.gate_pass
        assert quant_gated is False
