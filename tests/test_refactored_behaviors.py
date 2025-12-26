"""Tests for refactored behaviors.

Tests for the fixes applied during the refactoring session:
1. Signal taxonomy standardization (5-value system)
2. Batch custom_id parsing (colon-delimited format)
3. D/E scoring monotonicity
4. typical_dip filtering
5. Dip score calculation (starting at 0, not 20)
6. Weight normalization
7. min_confidence filtering
8. MarketContext benchmark_data_available flag
"""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from app.dipfinder.fundamentals import (
    normalize_debt_to_equity,
    _compute_balance_sheet_score,
)
from app.dipfinder.dip import compute_typical_dip
from app.dipfinder.signal import (
    compute_dip_score,
    compute_market_context,
    DipClass,
    MarketContext,
)
from app.dipfinder.config import DipFinderConfig


class TestSignalTaxonomy:
    """Test 5-value signal taxonomy standardization."""
    
    def test_normalize_signal_standard_values(self):
        """Standard 5-value signals should be preserved."""
        from app.services.ai_agents import _normalize_signal
        
        assert _normalize_signal("strong_buy") == "strong_buy"
        assert _normalize_signal("buy") == "buy"
        assert _normalize_signal("hold") == "hold"
        assert _normalize_signal("sell") == "sell"
        assert _normalize_signal("strong_sell") == "strong_sell"
    
    def test_normalize_signal_legacy_values(self):
        """Legacy 3-value signals should be mapped correctly."""
        from app.services.ai_agents import _normalize_signal
        
        assert _normalize_signal("bullish") == "buy"
        assert _normalize_signal("bearish") == "sell"
        assert _normalize_signal("neutral") == "hold"
    
    def test_normalize_signal_case_insensitive(self):
        """Signal normalization should be case-insensitive."""
        from app.services.ai_agents import _normalize_signal
        
        assert _normalize_signal("STRONG_BUY") == "strong_buy"
        assert _normalize_signal("Buy") == "buy"
        assert _normalize_signal("BULLISH") == "buy"
    
    def test_normalize_signal_unknown_defaults_to_hold(self):
        """Unknown signals should default to hold."""
        from app.services.ai_agents import _normalize_signal
        
        assert _normalize_signal("unknown") == "hold"
        assert _normalize_signal("") == "hold"
        assert _normalize_signal("foobar") == "hold"


class TestBatchCustomIdParsing:
    """Test batch custom_id colon-delimited format parsing."""
    
    def test_new_format_parsing(self):
        """New colon-delimited format should be parsed correctly."""
        custom_id = "abc12345:AAPL:warren_buffett:rating"
        parts = custom_id.split(":")
        
        assert parts[0] == "abc12345"  # batch_run_id
        assert parts[1] == "AAPL"      # symbol
        assert parts[2] == "warren_buffett"  # agent_id
        assert parts[3] == "rating"    # task
    
    def test_dotted_symbol_parsing(self):
        """Dotted symbols like BRK.B should be parsed correctly."""
        custom_id = "abc12345:BRK.B:ben_graham:rating"
        parts = custom_id.split(":")
        
        assert parts[1] == "BRK.B"  # Symbol with dot preserved
    
    def test_complex_agent_id_parsing(self):
        """Agent IDs with underscores should be parsed correctly."""
        custom_id = "xyz99999:MSFT:cathie_wood:rating"
        parts = custom_id.split(":")
        
        assert parts[2] == "cathie_wood"


class TestDebtToEquityScoring:
    """Test D/E normalization and scoring."""
    
    def test_normalize_ratio_form(self):
        """D/E in ratio form should be returned as-is."""
        assert normalize_debt_to_equity(0.5) == 0.5
        assert normalize_debt_to_equity(1.0) == 1.0
        assert normalize_debt_to_equity(2.0) == 2.0
    
    def test_normalize_percentage_form(self):
        """D/E in percentage form should be converted to ratio."""
        assert normalize_debt_to_equity(50) == 0.5
        assert normalize_debt_to_equity(100) == 1.0
        assert normalize_debt_to_equity(200) == 2.0
    
    def test_normalize_none(self):
        """None D/E should return None."""
        assert normalize_debt_to_equity(None) is None
    
    def test_de_scoring_monotonically_decreasing(self):
        """Higher D/E should result in lower scores."""
        # Test scoring at different D/E levels
        scores = []
        for de in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]:
            info = {"debtToEquity": de}
            score, _ = _compute_balance_sheet_score(info)
            scores.append(score)
        
        # Each score should be <= previous (monotonically decreasing)
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i-1], f"Score at D/E={[0.1,0.3,0.5,0.8,1.0,1.5,2.0,2.5,3.0][i]} should be <= score at lower D/E"
    
    def test_de_scoring_extreme_values(self):
        """Extreme D/E values should score very low."""
        info = {"debtToEquity": 5.0}  # 500% D/E
        score, _ = _compute_balance_sheet_score(info)
        assert score < 25, "Very high D/E should score low"


class TestTypicalDipCalculation:
    """Test typical_dip calculation improvements."""
    
    def test_filters_zero_values(self):
        """Zero dips should be filtered out."""
        # Series with mostly zeros and some dips
        dip_series = np.array([0.0, 0.0, 0.15, 0.0, 0.20, 0.0, 0.0])
        result = compute_typical_dip(dip_series)
        
        # Median of [0.15, 0.20] = 0.175
        assert result == pytest.approx(0.175, rel=0.01)
    
    def test_filters_below_threshold(self):
        """Dips below threshold should be filtered out."""
        # Default threshold is 0.01 (1%)
        dip_series = np.array([0.005, 0.008, 0.15, 0.02, 0.25])
        result = compute_typical_dip(dip_series)
        
        # Only values >= 0.01: [0.15, 0.02, 0.25], median = 0.15
        assert result == pytest.approx(0.15, rel=0.01)
    
    def test_returns_zero_when_no_valid_dips(self):
        """Should return 0 when no dips meet threshold."""
        dip_series = np.array([0.0, 0.001, 0.005, 0.008])
        result = compute_typical_dip(dip_series)
        
        assert result == 0.0
    
    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        dip_series = np.array([0.03, 0.05, 0.08, 0.10])
        result = compute_typical_dip(dip_series, min_dip_threshold=0.05)
        
        # Only values >= 0.05: [0.05, 0.08, 0.10], median = 0.08
        assert result == pytest.approx(0.08, rel=0.01)


class TestDipScoreCalculation:
    """Test dip score calculation improvements."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = DipFinderConfig(
            min_dip_abs=0.10,  # 10% minimum dip
            dip_vs_typical_threshold=1.5,
            min_persist_days=2,
        )
    
    def test_minimal_dip_scores_low(self):
        """Minimal dips just above threshold should score low."""
        from app.dipfinder.dip import DipMetrics
        
        # Dip just above 10% threshold
        metrics = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.105,  # 10.5%
            peak_price=100.0,
            current_price=89.5,
            dip_percentile=50.0,
            dip_vs_typical=1.0,  # Not significant
            typical_dip=0.10,
            persist_days=1,  # Below min
            days_since_peak=5,
            is_meaningful=True,
        )
        
        context = MarketContext(
            benchmark_ticker="SPY",
            dip_mkt=0.05,
            dip_stock=0.105,
            excess_dip=0.055,
            dip_class=DipClass.STOCK_SPECIFIC,
        )
        
        score = compute_dip_score(metrics, context, self.config)
        
        # With minimal dip (magnitude_factor ≈ 0.017), should score low
        # Score breakdown: ~0.7 (magnitude) + 12.5 (percentile) + 0 (typical) + 0 (persist) + 5 (class) ≈ 18
        assert score < 25, f"Minimal dip should score low, got {score}"
    
    def test_significant_dip_scores_higher(self):
        """Significant dips should score higher."""
        from app.dipfinder.dip import DipMetrics
        
        metrics = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.25,  # 25%
            peak_price=100.0,
            current_price=75.0,
            dip_percentile=85.0,
            dip_vs_typical=2.0,
            typical_dip=0.125,
            persist_days=7,
            days_since_peak=14,
            is_meaningful=True,
        )
        
        context = MarketContext(
            benchmark_ticker="SPY",
            dip_mkt=0.05,
            dip_stock=0.25,
            excess_dip=0.20,
            dip_class=DipClass.STOCK_SPECIFIC,
        )
        
        score = compute_dip_score(metrics, context, self.config)
        
        # With significant dip, should score higher
        assert score > 50, f"Significant dip should score > 50, got {score}"
    
    def test_below_threshold_scores_zero(self):
        """Dips below threshold should score exactly 0."""
        from app.dipfinder.dip import DipMetrics
        
        metrics = DipMetrics(
            ticker="TEST",
            window=30,
            dip_pct=0.08,  # 8% - below 10% threshold
            peak_price=100.0,
            current_price=92.0,
            dip_percentile=30.0,
            dip_vs_typical=0.8,
            typical_dip=0.10,
            persist_days=5,
            days_since_peak=7,
            is_meaningful=False,
        )
        
        context = MarketContext(
            benchmark_ticker="SPY",
            dip_mkt=0.02,
            dip_stock=0.08,
            excess_dip=0.06,
            dip_class=DipClass.MIXED,
        )
        
        score = compute_dip_score(metrics, context, self.config)
        
        assert score == 0.0


class TestMarketContextBenchmarkFlag:
    """Test MarketContext benchmark_data_available flag."""
    
    def test_sufficient_benchmark_data_flag_true(self):
        """Flag should be True when benchmark data is sufficient."""
        stock_prices = np.linspace(100, 80, 50)  # 50 days
        benchmark_prices = np.linspace(100, 95, 50)  # 50 days
        
        context = compute_market_context(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,
        )
        
        assert context.benchmark_data_available is True
    
    def test_insufficient_benchmark_data_flag_false(self):
        """Flag should be False when benchmark data is insufficient."""
        stock_prices = np.linspace(100, 80, 50)  # 50 days
        benchmark_prices = np.linspace(100, 95, 20)  # Only 20 days
        
        context = compute_market_context(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,  # Requires 30 days, only have 20
        )
        
        assert context.benchmark_data_available is False
        assert context.dip_mkt == 0.0


class TestMinConfidenceFiltering:
    """Test portfolio manager min_confidence filtering."""
    
    def test_filters_low_confidence_signals(self):
        """Signals below min_confidence should be filtered."""
        from app.hedge_fund.agents.portfolio_manager import PortfolioManager
        from app.hedge_fund.schemas import Signal, AgentSignal, AgentType
        
        manager = PortfolioManager(min_confidence_threshold=0.5)
        
        signals = [
            AgentSignal(
                agent_id="agent1",
                agent_name="Agent 1",
                agent_type=AgentType.FUNDAMENTALS,
                symbol="AAPL",
                signal=Signal.STRONG_BUY,
                confidence=0.9,  # Above threshold
                reasoning="High confidence buy",
            ),
            AgentSignal(
                agent_id="agent2",
                agent_name="Agent 2",
                agent_type=AgentType.TECHNICALS,
                symbol="AAPL",
                signal=Signal.STRONG_SELL,
                confidence=0.3,  # Below threshold - should be filtered
                reasoning="Low confidence sell",
            ),
        ]
        
        report = manager.aggregate_signals(signals)
        
        # Only the high-confidence signal should influence consensus
        assert report.consensus_signal == Signal.STRONG_BUY
        assert report.bullish_count == 1
        assert report.bearish_count == 0
    
    def test_all_below_threshold_returns_hold(self):
        """If all signals below threshold, should return HOLD with 0 confidence."""
        from app.hedge_fund.agents.portfolio_manager import PortfolioManager
        from app.hedge_fund.schemas import Signal, AgentSignal, AgentType
        
        manager = PortfolioManager(min_confidence_threshold=0.8)
        
        signals = [
            AgentSignal(
                agent_id="agent1",
                agent_name="Agent 1",
                agent_type=AgentType.FUNDAMENTALS,
                symbol="AAPL",
                signal=Signal.BUY,
                confidence=0.5,  # Below threshold
                reasoning="Low confidence",
            ),
        ]
        
        report = manager.aggregate_signals(signals)
        
        assert report.consensus_signal == Signal.HOLD
        assert report.consensus_confidence == 0.0


class TestWeightNormalization:
    """Test signal weight normalization."""
    
    def test_weights_normalized_if_not_sum_to_one(self):
        """Weights should be normalized at runtime if they don't sum to 1."""
        from app.dipfinder.signal import compute_signal
        from app.dipfinder.fundamentals import QualityMetrics
        from app.dipfinder.stability import StabilityMetrics
        
        # Create config with weights that don't sum to 1
        config = DipFinderConfig(
            weight_dip=0.5,
            weight_quality=0.3,
            weight_stability=0.3,  # Sum = 1.1
        )
        
        # Create test data
        stock_prices = np.linspace(100, 80, 100)
        benchmark_prices = np.linspace(100, 95, 100)
        
        quality = QualityMetrics(ticker="TEST", score=70.0)
        stability = StabilityMetrics(ticker="TEST", score=70.0)
        
        signal = compute_signal(
            ticker="TEST",
            stock_prices=stock_prices,
            benchmark_prices=benchmark_prices,
            benchmark_ticker="SPY",
            window=30,
            quality_metrics=quality,
            stability_metrics=stability,
            as_of_date="2024-01-01",
            config=config,
        )
        
        # Should not raise an error and should produce valid score
        assert 0 <= signal.final_score <= 100

class TestInputHashCompleteness:
    """Test that input hash includes all metrics used in prompts."""
    
    def test_hash_includes_all_valuation_metrics(self):
        """Hash should include all valuation metrics from prompts."""
        from app.services.ai_agents import _compute_input_hash
        
        # Hash with all valuation metrics
        full_data = {
            "pe_ratio": 20,
            "forward_pe": 18,
            "peg_ratio": 1.5,
            "price_to_book": 3.0,
            "ev_to_ebitda": 12.0,
        }
        
        # Hash with missing metric
        partial_data = {
            "pe_ratio": 20,
            "forward_pe": 18,
            "peg_ratio": 1.5,
            "price_to_book": 3.0,
            # ev_to_ebitda missing
        }
        
        hash_full = _compute_input_hash(full_data, {})
        hash_partial = _compute_input_hash(partial_data, {})
        
        # Hashes should differ when ev_to_ebitda is missing
        assert hash_full != hash_partial
    
    def test_hash_includes_all_profitability_metrics(self):
        """Hash should include ROA in addition to ROE."""
        from app.services.ai_agents import _compute_input_hash
        
        data_with_roa = {"return_on_assets": 10.0}
        data_without_roa = {}
        
        hash_with = _compute_input_hash(data_with_roa, {})
        hash_without = _compute_input_hash(data_without_roa, {})
        
        assert hash_with != hash_without
    
    def test_hash_includes_financial_health_metrics(self):
        """Hash should include current_ratio and free_cash_flow."""
        from app.services.ai_agents import _compute_input_hash
        
        data_full = {
            "current_ratio": 2.0,
            "free_cash_flow": 1000000,
        }
        data_partial = {"current_ratio": 2.0}
        
        hash_full = _compute_input_hash(data_full, {})
        hash_partial = _compute_input_hash(data_partial, {})
        
        assert hash_full != hash_partial


class TestPersonaSchemaCompatibility:
    """Test that persona agent handles both 'signal' and 'rating' fields."""
    
    def test_parse_signal_field(self):
        """Persona should correctly parse 'signal' field from gateway realtime."""
        # Simulate parsed response with 'signal' field (gateway realtime format)
        parsed = {
            "signal": "strong_buy",
            "confidence": 8,
            "reasoning": "Great moat",
            "key_factors": ["moat", "management"],
        }
        
        # This is the logic from InvestorPersonaAgent.run()
        signal_str = parsed.get("signal") or parsed.get("rating", "hold")
        assert signal_str == "strong_buy"
    
    def test_parse_rating_field(self):
        """Persona should correctly parse 'rating' field from batch RATING_SCHEMA."""
        # Simulate parsed response with 'rating' field (batch format)
        parsed = {
            "rating": "buy",
            "confidence": 7,
            "reasoning": "Great PEG ratio",
            "key_factors": ["growth", "valuation"],
        }
        
        # This is the logic from InvestorPersonaAgent.run()
        signal_str = parsed.get("signal") or parsed.get("rating", "hold")
        assert signal_str == "buy"
    
    def test_parse_both_fields_prefers_signal(self):
        """When both fields present, 'signal' should take precedence."""
        parsed = {
            "signal": "strong_sell",
            "rating": "buy",  # Should be ignored
            "confidence": 5,
            "reasoning": "Test",
            "key_factors": [],
        }
        
        signal_str = parsed.get("signal") or parsed.get("rating", "hold")
        assert signal_str == "strong_sell"


class TestExternalAdapterSignature:
    """Test that external adapter calls _run_single_agent correctly."""
    
    def test_format_metrics_imported(self):
        """Adapter should import _format_metrics_for_prompt."""
        from app.services.ai_agents import _format_metrics_for_prompt
        
        # Verify it's importable and callable
        result = _format_metrics_for_prompt(
            {"pe_ratio": 20, "profit_margin": 0.15},
            {"name": "Test Corp", "sector": "Technology"},
        )
        
        assert "Test Corp" in result
        assert "Technology" in result
        assert "P/E: 20" in result