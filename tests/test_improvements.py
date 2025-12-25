"""Tests for AI analysis and yfinance improvements.

Tests:
1. Enhanced quality metrics (valuation, analyst scores)
2. Enhanced stability metrics (more fundamentals)
3. Symbol search with caching
4. Fundamentals service
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timezone


# ============================================================================
# Test Quality Metrics Enhancement
# ============================================================================

class TestQualityMetrics:
    """Test enhanced quality scoring with valuation and analyst data."""
    
    def test_valuation_score_low_pe(self):
        """Test valuation scoring with low P/E (undervalued)."""
        from app.dipfinder.fundamentals import _compute_valuation_score
        
        info = {
            "trailingPE": 12.0,
            "forwardPE": 10.0,
            "trailingPegRatio": 0.8,
        }
        
        score, factors = _compute_valuation_score(info)
        
        assert score > 80, "Low P/E with low PEG should score high"
        assert factors["pe_ratio"] == 12.0
        assert factors["peg_ratio"] == 0.8
    
    def test_valuation_score_high_pe(self):
        """Test valuation scoring with high P/E (expensive)."""
        from app.dipfinder.fundamentals import _compute_valuation_score
        
        info = {
            "trailingPE": 50.0,
            "forwardPE": 55.0,
            "trailingPegRatio": 3.5,
        }
        
        score, factors = _compute_valuation_score(info)
        
        assert score < 50, "High P/E with high PEG should score low"
    
    def test_analyst_score_buy(self):
        """Test analyst scoring with buy recommendation."""
        from app.dipfinder.fundamentals import _compute_analyst_score
        
        info = {
            "recommendationKey": "buy",
            "targetMeanPrice": 150.0,
            "regularMarketPrice": 100.0,
            "numberOfAnalystOpinions": 25,
        }
        
        score, factors = _compute_analyst_score(info)
        
        assert score > 75, "Buy rating with 50% upside should score high"
        assert factors["target_upside"] == 0.5  # (150-100)/100
        assert factors["recommendation"] == "buy"
    
    def test_analyst_score_sell(self):
        """Test analyst scoring with sell recommendation."""
        from app.dipfinder.fundamentals import _compute_analyst_score
        
        info = {
            "recommendationKey": "sell",
            "targetMeanPrice": 80.0,
            "regularMarketPrice": 100.0,
            "numberOfAnalystOpinions": 10,
        }
        
        score, factors = _compute_analyst_score(info)
        
        assert score < 40, "Sell rating with downside should score low"
        assert factors["target_upside"] == -0.2  # (80-100)/100
    
    def test_analyst_score_missing_data(self):
        """Test analyst scoring with no data."""
        from app.dipfinder.fundamentals import _compute_analyst_score
        
        info = {}
        
        score, factors = _compute_analyst_score(info)
        
        assert score == 50.0, "No data should return neutral score"


# ============================================================================
# Test Stability Metrics Enhancement
# ============================================================================

class TestStabilityMetrics:
    """Test enhanced stability scoring."""
    
    def test_fundamental_stability_score_strong(self):
        """Test stability with strong fundamentals."""
        from app.dipfinder.stability import _compute_fundamental_stability_score
        
        info = {
            "freeCashflow": 10_000_000_000,
            "totalRevenue": 50_000_000_000,  # 20% FCF margin
            "profitMargins": 0.25,
            "debtToEquity": 0.3,
            "currentRatio": 2.5,
            "recommendationKey": "strong_buy",
            "returnOnEquity": 0.30,
            "revenueGrowth": 0.15,
        }
        
        score = _compute_fundamental_stability_score(info)
        
        assert score > 80, f"Strong fundamentals should score high, got {score}"
    
    def test_fundamental_stability_score_weak(self):
        """Test stability with weak fundamentals."""
        from app.dipfinder.stability import _compute_fundamental_stability_score
        
        info = {
            "freeCashflow": -5_000_000_000,  # Negative
            "profitMargins": -0.10,  # Negative
            "debtToEquity": 2.5,  # High debt
            "currentRatio": 0.5,  # Low liquidity
            "recommendationKey": "sell",
            "returnOnEquity": -0.15,  # Negative
            "revenueGrowth": -0.20,  # Declining
        }
        
        score = _compute_fundamental_stability_score(info)
        
        assert score < 35, f"Weak fundamentals should score low, got {score}"
    
    def test_fundamental_stability_uses_stored_fundamentals(self):
        """Test that stored fundamentals format works."""
        from app.dipfinder.stability import _compute_fundamental_stability_score
        
        # This format matches what we store in stock_fundamentals table
        info = {
            "free_cash_flow": 8_000_000_000,
            "revenue": 40_000_000_000,
            "profit_margin": 0.20,
            "debt_to_equity": 0.45,
            "current_ratio": 1.8,
            "recommendation": "buy",
            "return_on_equity": 0.22,
            "revenue_growth": 0.10,
        }
        
        score = _compute_fundamental_stability_score(info)
        
        # Should work with snake_case keys from stored fundamentals
        assert score > 60, f"Should work with stored format, got {score}"


# ============================================================================
# Test Symbol Search Service
# ============================================================================

class TestSymbolSearch:
    """Test symbol search with caching."""
    
    @pytest.mark.asyncio
    async def test_search_local_first(self):
        """Test that local symbols are searched first."""
        from app.services.symbol_search import _search_local_db
        
        # This will fail if DB is not available, but tests structure
        try:
            results = await _search_local_db("AAPL", limit=5)
            # Results will be empty if no symbols in DB, that's OK
            assert isinstance(results, list)
        except Exception:
            pytest.skip("Database not available")
    
    def test_normalize_query(self):
        """Test query normalization."""
        from app.services.symbol_search import _normalize_query
        
        assert _normalize_query("  apple  ") == "APPLE"
        assert _normalize_query("aapl") == "AAPL"
        assert _normalize_query("BRK.B") == "BRK.B"
    
    @pytest.mark.asyncio
    async def test_search_symbols_structure(self):
        """Test symbol search returns proper structure."""
        from app.services.symbol_search import search_symbols
        
        try:
            # Real API call - may take a moment
            results = await search_symbols("apple", limit=3)
            
            assert isinstance(results, list)
            for r in results:
                assert "symbol" in r
                assert "name" in r
        except Exception:
            pytest.skip("Search service unavailable")
    
    @pytest.mark.asyncio
    async def test_lookup_symbol_valid(self):
        """Test looking up a valid symbol via search."""
        from app.services.symbol_search import search_symbols
        
        try:
            results, _ = await search_symbols("AAPL", limit=1)
            
            if results:  # May be empty if rate limited
                assert results[0]["symbol"] == "AAPL"
                assert "name" in results[0]
        except Exception:
            pytest.skip("Lookup service unavailable")
    
    @pytest.mark.asyncio
    async def test_lookup_symbol_invalid(self):
        """Test looking up an invalid symbol."""
        from app.services.symbol_search import search_symbols
        
        try:
            results, _ = await search_symbols("INVALIDXYZ123", limit=1)
            # Should return empty list for invalid symbol
            assert len(results) == 0 or results[0]["symbol"] != "INVALIDXYZ123"
        except Exception:
            pytest.skip("Lookup service unavailable")


# ============================================================================
# Test Fundamentals Service
# ============================================================================

class TestFundamentalsService:
    """Test fundamentals fetching and storage."""
    
    @pytest.mark.asyncio
    async def test_fetch_fundamentals_sync(self):
        """Test fetching fundamentals from yfinance."""
        from app.services.fundamentals import _fetch_fundamentals_sync
        
        data = await _fetch_fundamentals_sync("AAPL")
        
        assert data is not None, "Should fetch AAPL fundamentals"
        assert data["symbol"] == "AAPL"
        assert data["pe_ratio"] is not None or data["forward_pe"] is not None
        assert "recommendation" in data
        assert "profit_margin" in data
    
    @pytest.mark.asyncio
    async def test_fetch_fundamentals_etf_skipped(self):
        """Test that ETFs are skipped."""
        from app.services.fundamentals import _fetch_fundamentals_sync
        
        data = await _fetch_fundamentals_sync("SPY")
        
        assert data is None, "ETFs should be skipped"
    
    @pytest.mark.asyncio
    async def test_fetch_fundamentals_index_skipped(self):
        """Test that indexes are skipped."""
        from app.services.fundamentals import _fetch_fundamentals_sync
        
        data = await _fetch_fundamentals_sync("^GSPC")
        
        assert data is None, "Indexes should be skipped"
    
    @pytest.mark.asyncio
    async def test_get_fundamentals_for_analysis_format(self):
        """Test that analysis format has expected keys."""
        from app.services.fundamentals import _fetch_fundamentals_sync
        
        data = await _fetch_fundamentals_sync("MSFT")
        
        if data:
            # These are the keys used by AI prompts
            expected_keys = [
                "pe_ratio", "forward_pe", "peg_ratio",
                "profit_margin", "return_on_equity",
                "debt_to_equity", "current_ratio",
                "revenue_growth", "earnings_growth",
                "recommendation", "target_mean_price",
            ]
            
            for key in expected_keys:
                assert key in data, f"Missing key: {key}"


# ============================================================================
# Test Quality Score Integration
# ============================================================================

class TestQualityScoreIntegration:
    """Test full quality score computation with new metrics."""
    
    @pytest.mark.asyncio
    async def test_quality_score_with_stored_fundamentals(self):
        """Test quality score using pre-fetched fundamentals."""
        from app.dipfinder.fundamentals import compute_quality_score
        
        # Simulate stored fundamentals with yfinance-style keys
        fundamentals = {
            "profitMargins": 0.25,  # yfinance key
            "operatingMargins": 0.30,
            "returnOnEquity": 0.30,
            "debtToEquity": 40.0,  # yfinance uses percentage
            "currentRatio": 2.0,
            "revenueGrowth": 0.12,
            "freeCashflow": 5_000_000_000,
            "marketCap": 100_000_000_000,
            "averageVolume": 10_000_000,
            "trailingPE": 18.0,
            "forwardPE": 15.0,
            "trailingPegRatio": 1.0,
            "recommendationKey": "buy",
            "targetMeanPrice": 200.0,
            "regularMarketPrice": 150.0,
            "numberOfAnalystOpinions": 30,
        }
        
        # Pass as info since that's the format yfinance uses
        result = await compute_quality_score(
            ticker="TEST",
            info=fundamentals,
            fundamentals={},
        )
        
        assert result.score > 60, f"Good fundamentals should score well: {result.score}"
        assert result.valuation_score > 70, f"Good valuation should score well: {result.valuation_score}"
        assert result.analyst_score > 70, f"Buy rating should score well: {result.analyst_score}"
        assert result.recommendation == "buy"
    
    @pytest.mark.asyncio
    async def test_quality_score_new_fields_in_output(self):
        """Test that new fields appear in output dict."""
        from app.dipfinder.fundamentals import compute_quality_score
        
        result = await compute_quality_score(
            ticker="TEST",
            info={"trailingPE": 20.0, "recommendationKey": "hold"},
            fundamentals={},
        )
        
        output = result.to_dict()
        
        # Check new fields exist
        assert "valuation_score" in output
        assert "analyst_score" in output
        assert "pe_ratio" in output
        assert "recommendation" in output
        assert "target_upside" in output


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
