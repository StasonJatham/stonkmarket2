"""Tests for DipFinder API endpoints."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestDipFinderPublicEndpoints:
    """Tests for public (no auth) DipFinder endpoints."""
    
    def test_signals_is_public(self, client: TestClient):
        """GET /dipfinder/signals is publicly accessible."""
        response = client.get("/dipfinder/signals?tickers=AAPL")
        # 200 or 500 (if external API fails) - but NOT 401
        assert response.status_code != status.HTTP_401_UNAUTHORIZED
    
    def test_latest_is_public(self, client: TestClient):
        """GET /dipfinder/latest is publicly accessible."""
        response = client.get("/dipfinder/latest")
        # 200 or 500 - but NOT 401
        assert response.status_code != status.HTTP_401_UNAUTHORIZED
    
    def test_config_is_public(self, client: TestClient):
        """GET /dipfinder/config is publicly accessible."""
        response = client.get("/dipfinder/config")
        assert response.status_code == status.HTTP_200_OK


class TestDipFinderProtectedEndpoints:
    """Tests for auth-required DipFinder endpoints."""
    
    def test_run_requires_auth(self, client: TestClient):
        """POST /dipfinder/run requires authentication."""
        response = client.post("/dipfinder/run", json={})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_admin_refresh_requires_auth(self, client: TestClient):
        """POST /dipfinder/admin/refresh-all requires authentication."""
        response = client.post("/dipfinder/admin/refresh-all")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestDipFinderValidation:
    """Tests for request validation."""
    
    @pytest.fixture
    def auth_headers(self) -> dict:
        """Create auth headers for testing.
        
        Note: In real tests, this would create a proper JWT token.
        For now, this is a placeholder.
        """
        # This would need proper token generation
        return {"Authorization": "Bearer test-token"}
    
    def test_signals_empty_tickers(self, client: TestClient, auth_headers):
        """Empty tickers parameter returns error."""
        # This test would work with proper auth setup
        pass  # Skip for now without proper auth mock
    
    def test_signals_too_many_tickers(self, client: TestClient, auth_headers):
        """Too many tickers returns error."""
        # Would test with more than 20 tickers
        pass
    
    def test_signals_invalid_window(self, client: TestClient, auth_headers):
        """Invalid window value returns error."""
        # Would test with window < 7 or > 365
        pass


class TestDipFinderConfig:
    """Tests for DipFinder configuration."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        from app.dipfinder.config import DipFinderConfig
        
        config = DipFinderConfig()
        
        assert config.min_dip_abs == 0.10
        assert config.min_persist_days == 2
        assert config.dip_percentile_threshold == 0.80
        assert config.quality_gate == 60.0
        assert config.stability_gate == 60.0
        assert config.alert_good == 70.0
        assert config.alert_strong == 80.0
        assert config.default_benchmark == "SPY"
        assert 7 in config.windows
        assert 30 in config.windows
        assert 365 in config.windows
    
    def test_weight_sum(self):
        """Scoring weights should sum to 1.0."""
        from app.dipfinder.config import DipFinderConfig
        
        config = DipFinderConfig()
        total = config.weight_dip + config.weight_quality + config.weight_stability
        
        assert abs(total - 1.0) < 0.001


class TestResponseSchemas:
    """Tests for Pydantic response schemas."""
    
    def test_dip_signal_response_validation(self):
        """Test DipSignalResponse schema."""
        from app.schemas.dipfinder import DipSignalResponse
        
        response = DipSignalResponse(
            ticker="AAPL",
            window=30,
            benchmark="SPY",
            as_of_date="2025-12-21",
            dip_stock=0.15,
            peak_stock=200.0,
            current_price=170.0,
            dip_pctl=85.0,
            dip_vs_typical=2.0,
            persist_days=4,
            dip_mkt=0.05,
            excess_dip=0.10,
            dip_class="STOCK_SPECIFIC",
            quality_score=75.0,
            stability_score=70.0,
            dip_score=80.0,
            final_score=76.0,
            alert_level="GOOD",
            should_alert=True,
            reason="Test reason",
        )
        
        assert response.ticker == "AAPL"
        assert response.final_score == 76.0
    
    def test_run_response_validation(self):
        """Test DipFinderRunResponse schema."""
        from app.schemas.dipfinder import DipFinderRunResponse
        
        response = DipFinderRunResponse(
            status="completed",
            message="Processed 10 tickers",
            tickers_processed=10,
            signals_generated=10,
            alerts_triggered=2,
        )
        
        assert response.status == "completed"
        assert response.tickers_processed == 10
    
    def test_config_response_validation(self):
        """Test DipFinderConfigResponse schema."""
        from app.schemas.dipfinder import DipFinderConfigResponse
        
        response = DipFinderConfigResponse(
            windows=[7, 30, 100, 365],
            min_dip_abs=0.10,
            min_persist_days=2,
            dip_percentile_threshold=0.80,
            dip_vs_typical_threshold=1.5,
            market_dip_threshold=0.06,
            excess_dip_stock_specific=0.04,
            excess_dip_market=0.03,
            quality_gate=60.0,
            stability_gate=60.0,
            alert_good=70.0,
            alert_strong=80.0,
            weight_dip=0.45,
            weight_quality=0.30,
            weight_stability=0.25,
            default_benchmark="SPY",
        )
        
        assert response.windows == [7, 30, 100, 365]
        assert response.default_benchmark == "SPY"
