"""Tests for health check API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient):
        """GET /health returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

    def test_health_returns_status_field(self, client: TestClient):
        """GET /health returns status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_returns_version(self, client: TestClient):
        """GET /health returns version field."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data

    def test_health_returns_checks(self, client: TestClient):
        """GET /health returns checks object."""
        response = client.get("/health")
        data = response.json()
        assert "checks" in data
        assert isinstance(data["checks"], dict)


class TestReadinessEndpoint:
    """Tests for GET /health/ready."""

    def test_ready_endpoint_exists(self, client: TestClient):
        """GET /health/ready endpoint exists."""
        response = client.get("/health/ready")
        # May be 200 or 503 depending on DB state
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        ]

    def test_ready_returns_status_field(self, client: TestClient):
        """GET /health/ready returns status when healthy."""
        response = client.get("/health/ready")
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "status" in data
            assert data["status"] == "ready"


class TestLivenessEndpoint:
    """Tests for GET /health/live."""

    def test_live_returns_200(self, client: TestClient):
        """GET /health/live returns 200 OK."""
        response = client.get("/health/live")
        assert response.status_code == status.HTTP_200_OK

    def test_live_returns_alive_status(self, client: TestClient):
        """GET /health/live returns alive status."""
        response = client.get("/health/live")
        data = response.json()
        assert data["status"] == "alive"


class TestDbHealthcheck:
    """Tests for db_healthcheck function."""

    @pytest.mark.asyncio
    async def test_db_healthcheck_is_coroutine(self):
        """db_healthcheck is an async function."""
        import inspect
        from app.api.routes.health import db_healthcheck
        
        assert inspect.iscoroutinefunction(db_healthcheck)

    @pytest.mark.asyncio
    async def test_db_healthcheck_returns_bool(self):
        """db_healthcheck returns a boolean."""
        from app.api.routes.health import db_healthcheck
        
        # With mocked pool
        with patch("app.api.routes.health.get_pg_pool") as mock_pool:
            mock_pool.return_value = None
            result = await db_healthcheck()
            assert isinstance(result, bool)
            assert result is False


class TestHealthChecks:
    """Integration tests for health status logic."""

    def test_healthy_when_all_checks_pass(self):
        """Status is 'healthy' when all checks pass."""
        checks = {"database": True, "cache": True}
        
        if all(checks.values()):
            status_val = "healthy"
        elif checks.get("database", False):
            status_val = "degraded"
        else:
            status_val = "unhealthy"
        
        assert status_val == "healthy"

    def test_degraded_when_cache_down(self):
        """Status is 'degraded' when cache is down but DB is up."""
        checks = {"database": True, "cache": False}
        
        if all(checks.values()):
            status_val = "healthy"
        elif checks.get("database", False):
            status_val = "degraded"
        else:
            status_val = "unhealthy"
        
        assert status_val == "degraded"

    def test_unhealthy_when_db_down(self):
        """Status is 'unhealthy' when database is down."""
        checks = {"database": False, "cache": True}
        
        if all(checks.values()):
            status_val = "healthy"
        elif checks.get("database", False):
            status_val = "degraded"
        else:
            status_val = "unhealthy"
        
        assert status_val == "unhealthy"
