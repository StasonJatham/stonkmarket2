"""Tests for API key management endpoints."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestListAPIKeysEndpoint:
    """Tests for GET /api-keys."""

    def test_list_api_keys_without_auth_returns_401(self, client: TestClient):
        """GET /api-keys without auth returns 401."""
        response = client.get("/api-keys")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_api_keys_with_invalid_token_returns_401(self, client: TestClient):
        """GET /api-keys with invalid token returns 401."""
        response = client.get(
            "/api-keys",
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestCreateAPIKeyEndpoint:
    """Tests for POST /api-keys."""

    def test_create_api_key_without_auth_returns_401(self, client: TestClient):
        """POST /api-keys without auth returns 401."""
        response = client.post("/api-keys", json={"name": "test-key"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_create_api_key_without_body_returns_422(
        self, client: TestClient, auth_headers: dict
    ):
        """POST /api-keys without body returns validation error."""
        response = client.post("/api-keys", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestRevokeAPIKeyEndpoint:
    """Tests for DELETE /api-keys/{key_id}."""

    def test_revoke_api_key_without_auth_returns_401(self, client: TestClient):
        """DELETE /api-keys/{key_id} without auth returns 401."""
        response = client.delete("/api-keys/some-key-id")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAdminAPIKeysEndpoint:
    """Tests for admin API key management (GET /admin/api-keys)."""

    def test_admin_list_api_keys_without_auth_returns_401(self, client: TestClient):
        """GET /admin/api-keys without auth returns 401."""
        response = client.get("/admin/api-keys")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_admin_list_api_keys_with_user_token_returns_403(
        self, client: TestClient, auth_headers: dict
    ):
        """GET /admin/api-keys with non-admin token returns 403."""
        response = client.get("/admin/api-keys", headers=auth_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN
