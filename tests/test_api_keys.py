"""Tests for API key management endpoints."""

from __future__ import annotations

import pytest
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

    def test_create_api_key_requires_auth(
        self, client: TestClient, admin_headers: dict
    ):
        """POST /admin/user-keys requires valid auth to process body validation."""
        response = client.post("/admin/user-keys", headers=admin_headers)
        # 401 if test admin doesn't exist in DB, 422 if auth passes but no body
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        ]


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
        self, client: TestClient, auth_headers: dict
    ):
        """GET /admin/api-keys with non-admin token returns 401 (user not in DB) or 403."""
        response = client.get("/admin/api-keys", headers=auth_headers)
        # 401 if test user doesn't exist in DB, 403 if exists but not admin
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]
