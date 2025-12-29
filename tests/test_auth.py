"""Tests for authentication API endpoints."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestLoginEndpoint:
    """Tests for POST /auth/login."""

    def test_login_without_body_returns_422(self, client: TestClient):
        """Login without request body returns validation error."""
        response = client.post("/auth/login")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_login_with_empty_username_returns_422(self, client: TestClient):
        """Login with empty username returns validation error."""
        response = client.post("/auth/login", json={"username": "", "password": "test"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_login_with_empty_password_returns_422(self, client: TestClient):
        """Login with empty password returns validation error."""
        response = client.post("/auth/login", json={"username": "test", "password": ""})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_login_with_nonexistent_user_returns_401(self, client: TestClient, monkeypatch):
        """Login with non-existent user returns authentication error."""
        async def fake_get_user(username: str):
            return None

        from app.api.routes import auth as auth_routes

        monkeypatch.setattr(auth_routes.auth_repo, "get_user", fake_get_user)
        response = client.post(
            "/auth/login", json={"username": "nonexistent_user_xyz", "password": "test"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestLogoutEndpoint:
    """Tests for POST /auth/logout."""

    def test_logout_without_auth_returns_401(self, client: TestClient):
        """Logout without authentication returns 401."""
        response = client.post("/auth/logout")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_logout_with_invalid_token_returns_401(self, client: TestClient):
        """Logout with invalid token returns 401."""
        response = client.post(
            "/auth/logout", headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestMeEndpoint:
    """Tests for GET /auth/me."""

    def test_me_without_auth_returns_401(self, client: TestClient):
        """GET /me without authentication returns 401."""
        response = client.get("/auth/me")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_me_with_invalid_token_returns_401(self, client: TestClient):
        """GET /me with invalid token returns 401."""
        response = client.get(
            "/auth/me", headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestCredentialsEndpoint:
    """Tests for PUT /auth/credentials."""

    def test_credentials_without_auth_returns_401(self, client: TestClient):
        """Update credentials without auth returns 401."""
        response = client.put(
            "/auth/credentials",
            json={"current_password": "old", "new_password": "new"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_credentials_without_body_returns_422(self, client: TestClient):
        """Update credentials without body returns validation error."""
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
            response = client.put("/auth/credentials", json={})
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        finally:
            client.app.dependency_overrides.clear()


class TestTokenGeneration:
    """Tests for JWT token generation and validation."""

    def test_create_access_token_returns_string(self):
        """create_access_token returns a JWT string."""
        from app.core.security import create_access_token

        token = create_access_token(username="test_user", is_admin=False)
        assert isinstance(token, str)
        assert len(token) > 0
        # JWT tokens have 3 parts separated by dots
        assert token.count(".") == 2

    def test_create_admin_token_has_admin_claim(self):
        """Admin token includes is_admin=True claim."""
        from app.core.security import create_access_token, decode_access_token

        token = create_access_token(username="admin_user", is_admin=True)
        decoded = decode_access_token(token)
        
        assert decoded.sub == "admin_user"
        assert decoded.is_admin is True

    def test_decode_invalid_token_raises_error(self):
        """Decoding invalid token raises AuthenticationError."""
        from app.core.security import decode_access_token
        from app.core.exceptions import AuthenticationError

        with pytest.raises(AuthenticationError):
            decode_access_token("invalid.token.here")

    def test_decode_expired_token_raises_error(self):
        """Decoding expired token raises AuthenticationError."""
        from datetime import timedelta

        from app.core.security import create_access_token, decode_access_token
        from app.core.exceptions import AuthenticationError

        # Create a token that's already expired
        token = create_access_token(
            username="test", expires_delta=timedelta(seconds=-10)
        )

        with pytest.raises(AuthenticationError) as exc_info:
            decode_access_token(token)
        
        assert exc_info.value.error_code == "TOKEN_EXPIRED"


class TestPasswordHashing:
    """Tests for password hashing utilities."""

    def test_hash_password_returns_hash(self):
        """hash_password returns a bcrypt hash."""
        from app.core.security import hash_password

        password = "test_password_123"
        hashed = hash_password(password)
        
        assert isinstance(hashed, str)
        assert hashed != password
        # bcrypt hashes start with $2b$
        assert hashed.startswith("$2b$")

    def test_verify_password_correct_password(self):
        """verify_password returns True for correct password."""
        from app.core.security import hash_password, verify_password

        password = "correct_password"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True

    def test_verify_password_wrong_password(self):
        """verify_password returns False for wrong password."""
        from app.core.security import hash_password, verify_password

        hashed = hash_password("correct_password")
        
        assert verify_password("wrong_password", hashed) is False

    def test_verify_password_invalid_hash(self):
        """verify_password returns False for invalid hash."""
        from app.core.security import verify_password

        assert verify_password("password", "not_a_valid_hash") is False
