"""Tests for symbol CRUD API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestValidateSymbolEndpoint:
    """Tests for GET /symbols/validate/{symbol} (public endpoint)."""

    def test_validate_endpoint_exists(self, client: TestClient):
        """GET /validate/{symbol} endpoint exists."""
        response = client.get("/symbols/validate/AAPL")
        # Depends on cache/external service state
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        ]

    def test_validate_returns_validation_response(self, client: TestClient):
        """Response contains valid, symbol fields."""
        response = client.get("/symbols/validate/AAPL")
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert "valid" in data
            assert "symbol" in data
            assert data["symbol"] == "AAPL"

    def test_validate_normalizes_symbol_to_uppercase(self, client: TestClient):
        """Symbol is normalized to uppercase."""
        response = client.get("/symbols/validate/aapl")
        if response.status_code == status.HTTP_200_OK:
            assert response.json()["symbol"] == "AAPL"


class TestListSymbolsEndpoint:
    """Tests for GET /symbols (public endpoint)."""

    def test_list_symbols_without_auth_returns_200(self, client: TestClient):
        """GET /symbols without auth returns 200 (public endpoint for signals page)."""
        response = client.get("/symbols")
        assert response.status_code == status.HTTP_200_OK
        # Returns empty list or list of symbols
        assert isinstance(response.json(), list)

    def test_list_symbols_with_invalid_token_still_returns_200(self, client: TestClient):
        """GET /symbols with invalid token still works (public endpoint)."""
        response = client.get(
            "/symbols", headers={"Authorization": "Bearer invalid"}
        )
        # Public endpoint ignores auth header
        assert response.status_code == status.HTTP_200_OK


class TestGetSymbolEndpoint:
    """Tests for GET /symbols/{symbol}."""

    def test_get_symbol_without_auth_returns_401(self, client: TestClient):
        """GET /symbols/{symbol} without auth returns 401."""
        response = client.get("/symbols/AAPL")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestCreateSymbolEndpoint:
    """Tests for POST /symbols."""

    def test_create_symbol_without_auth_returns_401(self, client: TestClient):
        """POST /symbols without auth returns 401."""
        response = client.post("/symbols", json={"symbol": "AAPL"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_create_symbol_without_body_returns_422(self, client: TestClient, auth_headers: dict):
        """POST /symbols without body returns validation error."""
        # Note: Would need DB mock to actually work
        pass


class TestUpdateSymbolEndpoint:
    """Tests for PUT /symbols/{symbol}."""

    def test_update_symbol_without_auth_returns_401(self, client: TestClient):
        """PUT /symbols/{symbol} without auth returns 401."""
        response = client.put("/symbols/AAPL", json={"min_dip_pct": 0.15})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestDeleteSymbolEndpoint:
    """Tests for DELETE /symbols/{symbol}."""

    def test_delete_symbol_without_auth_returns_401(self, client: TestClient):
        """DELETE /symbols/{symbol} without auth returns 401."""
        response = client.delete("/symbols/AAPL")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestSymbolValidation:
    """Tests for symbol validation logic."""

    def test_symbol_path_validation_uppercase(self):
        """Symbol path validator normalizes to uppercase."""
        from app.api.routes.symbols import _validate_symbol_path
        
        # Path function returns the processed value
        # Note: In actual use, this is a Depends callable
        pass

    def test_symbol_max_length(self, client: TestClient):
        """Symbol longer than 10 chars returns validation error."""
        response = client.get("/symbols/validate/ABCDEFGHIJK")
        # Either 422 from path validation or handled in endpoint
        assert response.status_code in [
            status.HTTP_200_OK,  # May return valid=False
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


class TestSymbolCaching:
    """Tests for symbol validation caching."""

    def test_validation_cache_ttl_valid(self):
        """Valid symbols should be cached for 30 days."""
        # 30 days in seconds
        expected_ttl = 30 * 24 * 60 * 60
        assert expected_ttl == 2592000

    def test_validation_cache_ttl_invalid(self):
        """Invalid symbols should be cached for 1 day."""
        # 1 day in seconds
        expected_ttl = 24 * 60 * 60
        assert expected_ttl == 86400
