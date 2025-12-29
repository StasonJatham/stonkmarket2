"""Tests for MFA (Multi-Factor Authentication) API endpoints."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient


class TestMFASetupEndpoint:
    """Tests for POST /auth/mfa/setup."""

    def test_mfa_setup_without_auth_returns_401(self, client: TestClient):
        """POST /auth/mfa/setup without auth returns 401."""
        response = client.post("/auth/mfa/setup")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_mfa_setup_with_invalid_token_returns_401(self, client: TestClient):
        """POST /auth/mfa/setup with invalid token returns 401."""
        response = client.post(
            "/auth/mfa/setup",
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestMFAVerifyEndpoint:
    """Tests for POST /auth/mfa/verify."""

    def test_mfa_verify_without_auth_returns_401(self, client: TestClient):
        """POST /auth/mfa/verify without auth returns 401."""
        response = client.post("/auth/mfa/verify", json={"code": "123456"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestMFADisableEndpoint:
    """Tests for POST /auth/mfa/disable."""

    def test_mfa_disable_without_auth_returns_401(self, client: TestClient):
        """POST /auth/mfa/disable without auth returns 401."""
        response = client.post("/auth/mfa/disable", json={"code": "123456"})
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestMFAStatusEndpoint:
    """Tests for GET /auth/mfa/status."""

    def test_mfa_status_without_auth_returns_401(self, client: TestClient):
        """GET /auth/mfa/status without auth returns 401."""
        response = client.get("/auth/mfa/status")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
