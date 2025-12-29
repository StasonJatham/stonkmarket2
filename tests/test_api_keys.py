"""Tests for API key management endpoints."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient


class TestListAPIKeysEndpoint:
    """Tests for GET /admin/user-keys (admin only)."""

    def test_list_api_keys_without_auth_returns_401(self, client: TestClient):
        """GET /admin/user-keys without auth returns 401."""
        response = client.get("/admin/user-keys")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_api_keys_with_invalid_token_returns_401(self, client: TestClient):
        """GET /admin/user-keys with invalid token returns 401."""
        response = client.get(
            "/admin/user-keys",
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestCreateAPIKeyEndpoint:
    """Tests for POST /admin/user-keys."""

    def test_create_api_key_without_auth_returns_401(self, client: TestClient):
        """POST /admin/user-keys without auth returns 401."""
        response = client.post("/admin/user-keys", json={"name": "test-key"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestRevokeAPIKeyEndpoint:
    """Tests for DELETE /admin/user-keys/{key_id}."""

    def test_revoke_api_key_without_auth_returns_401(self, client: TestClient):
        """DELETE /admin/user-keys/{key_id} without auth returns 401."""
        response = client.delete("/admin/user-keys/1")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAdminAPIKeysEndpoint:
    """Tests for admin API key management (GET /admin/api-keys)."""

    def test_admin_list_api_keys_without_auth_returns_401(self, client: TestClient):
        """GET /admin/api-keys without auth returns 401."""
        response = client.get("/admin/api-keys")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_admin_list_api_keys_with_user_token_requires_admin(
        self, client: TestClient
    ):
        """GET /admin/api-keys with non-admin token returns 403."""
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
            response = client.get("/admin/api-keys")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            client.app.dependency_overrides.clear()
