"""Tests for dip analysis API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestBenchmarksEndpoint:
    """Tests for GET /dips/benchmarks (public endpoint)."""

    def test_benchmarks_returns_200(self, client: TestClient):
        """GET /benchmarks returns 200 OK."""
        response = client.get("/dips/benchmarks")
        assert response.status_code == status.HTTP_200_OK

    def test_benchmarks_returns_list(self, client: TestClient):
        """GET /benchmarks returns a list."""
        response = client.get("/dips/benchmarks")
        assert isinstance(response.json(), list)


class TestRankingEndpoint:
    """Tests for GET /dips/ranking."""

    def test_ranking_returns_200(self, client: TestClient):
        """GET /ranking returns 200 OK (no auth required)."""
        response = client.get("/dips/ranking")
        assert response.status_code == status.HTTP_200_OK

    def test_ranking_returns_list(self, client: TestClient):
        """GET /ranking returns a list."""
        response = client.get("/dips/ranking")
        assert isinstance(response.json(), list)

    def test_ranking_with_show_all_true(self, client: TestClient):
        """GET /ranking?show_all=true includes all stocks."""
        response = client.get("/dips/ranking?show_all=true")
        assert response.status_code == status.HTTP_200_OK
        assert isinstance(response.json(), list)

    def test_ranking_with_show_all_false(self, client: TestClient):
        """GET /ranking?show_all=false filters by dip threshold."""
        response = client.get("/dips/ranking?show_all=false")
        assert response.status_code == status.HTTP_200_OK


class TestRefreshRankingEndpoint:
    """Tests for POST /dips/ranking/refresh (admin only)."""

    def test_refresh_ranking_without_auth_returns_401(self, client: TestClient):
        """POST /ranking/refresh without auth returns 401."""
        response = client.post("/dips/ranking/refresh")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_ranking_with_user_token_returns_403(
        self, client: TestClient
    ):
        """POST /ranking/refresh with non-admin token returns 403."""
        from datetime import UTC, datetime

        from app.api.dependencies import require_user
        from app.core.security import TokenData

        async def override_require_user():
            return TokenData(
                sub="test_user",
                exp=datetime.now(UTC),
                iat=datetime.now(UTC),
                iss="stonkmarket",
                aud="stonkmarket-api",
                jti="test-jti",
                is_admin=False,
            )

        client.app.dependency_overrides[require_user] = override_require_user
        try:
            response = client.post("/dips/ranking/refresh")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            client.app.dependency_overrides.clear()


class TestStatesEndpoint:
    """Tests for GET /dips/states."""

    def test_states_returns_200(self, client: TestClient):
        """GET /states returns 200 OK."""
        response = client.get("/dips/states")
        assert response.status_code == status.HTTP_200_OK

    def test_states_returns_list(self, client: TestClient):
        """GET /states returns a list."""
        response = client.get("/dips/states")
        assert isinstance(response.json(), list)


class TestSymbolStateEndpoint:
    """Tests for GET /dips/{symbol}/state."""

    def test_symbol_state_without_auth_returns_401(self, client: TestClient):
        """GET /{symbol}/state without auth returns 401."""
        response = client.get("/dips/AAPL/state")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestChartEndpoint:
    """Tests for GET /dips/{symbol}/chart (public endpoint)."""

    def test_chart_with_valid_symbol(self, client: TestClient, monkeypatch):
        """GET /{symbol}/chart returns chart data."""
        from datetime import date

        import pandas as pd

        from app.api.routes import dips as dips_routes

        class DummyPriceProvider:
            async def get_prices(self, symbol, start_date, end_date):
                dates = pd.date_range(end=date.today(), periods=3, freq="D")
                return pd.DataFrame({"Close": [100.0, 105.0, 102.0]}, index=dates)

        class DummyService:
            price_provider = DummyPriceProvider()

        async def fake_get_symbol_min_dip_pct(symbol):
            return 0.10

        async def fake_cache_get(key):
            return None

        async def fake_cache_set(key, value, ttl=None):
            return True

        monkeypatch.setattr(dips_routes, "get_dipfinder_service", lambda: DummyService())
        monkeypatch.setattr(dips_routes.dips_repo, "get_symbol_min_dip_pct", fake_get_symbol_min_dip_pct)
        monkeypatch.setattr(dips_routes._chart_cache, "get", fake_cache_get)
        monkeypatch.setattr(dips_routes._chart_cache, "set", fake_cache_set)

        response = client.get("/dips/AAPL/chart?days=10")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        assert "close" in data[0]

    def test_chart_with_invalid_days_too_low(self, client: TestClient):
        """GET /chart with days < 7 returns validation error."""
        response = client.get("/dips/AAPL/chart?days=5")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_chart_with_invalid_days_too_high(self, client: TestClient):
        """GET /chart with days > 1825 returns validation error."""
        response = client.get("/dips/AAPL/chart?days=2000")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestBuildRankingHelper:
    """Tests for the _build_ranking helper function."""

    @pytest.mark.asyncio
    async def test_build_ranking_returns_tuple(self, monkeypatch):
        """_build_ranking returns tuple of (all, filtered) lists."""
        from app.api.routes import dips as dips_routes

        monkeypatch.setattr(
            dips_routes.dips_repo,
            "get_ranking_data",
            AsyncMock(return_value=[]),
        )

        all_entries, filtered_entries = await dips_routes._build_ranking()
        assert all_entries == []
        assert filtered_entries == []


class TestRankingCache:
    """Tests for ranking cache behavior."""

    def test_ranking_cache_key_format(self):
        """Verify ranking cache uses correct key format."""
        # Cache key format sanitizes colons to underscores
        from app.cache.cache import cache_key
        
        key = cache_key("all:True", prefix="ranking")
        assert "ranking" in key
        # Colons are sanitized to underscores in cache keys
        assert "all_True" in key
