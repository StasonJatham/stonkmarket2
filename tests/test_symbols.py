"""Tests for symbol CRUD API endpoints."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient


class TestListSymbolsEndpoint:
    """Tests for GET /symbols (public endpoint)."""

    def test_list_symbols_without_auth_returns_200(self, client: TestClient):
        """GET /symbols without auth returns 200 (public endpoint for signals page)."""
        response = client.get("/symbols")
        assert response.status_code == status.HTTP_200_OK
        # Returns empty list or list of symbols
        assert isinstance(response.json(), list)


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

    def test_create_symbol_without_body_returns_422(self, client: TestClient):
        """POST /symbols without body returns validation error."""
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
            response = client.post("/symbols", json={})
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        finally:
            client.app.dependency_overrides.clear()


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
        assert _validate_symbol_path(" aapl ") == "AAPL"

    def test_symbol_max_length(self, client: TestClient):
        """Symbol longer than 10 chars returns validation error."""
        response = client.get("/symbols/validate/ABCDEFGHIJK")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

