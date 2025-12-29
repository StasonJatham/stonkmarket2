"""Tests for cronjob management API endpoints."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient


class TestListCronJobsEndpoint:
    """Tests for GET /cronjobs."""

    def test_list_cronjobs_without_auth_returns_401(self, client: TestClient):
        """GET /cronjobs without auth returns 401."""
        response = client.get("/cronjobs")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_cronjobs_with_invalid_token_returns_401(self, client: TestClient):
        """GET /cronjobs with invalid token returns 401."""
        response = client.get(
            "/cronjobs", headers={"Authorization": "Bearer invalid_token"}
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestUpdateCronJobEndpoint:
    """Tests for PUT /cronjobs/{name}."""

    def test_update_cronjob_without_auth_returns_401(self, client: TestClient):
        """PUT /cronjobs/{name} without auth returns 401."""
        response = client.put(
            "/cronjobs/data_grab",
            json={"cron": "0 23 * * 1-5"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_update_cronjob_with_user_token_requires_admin(
        self, client: TestClient
    ):
        """PUT /cronjobs/{name} with non-admin token returns 403."""
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
            response = client.put(
                "/cronjobs/data_grab",
                json={"cron": "0 23 * * 1-5"},
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            client.app.dependency_overrides.clear()


class TestRunCronJobEndpoint:
    """Tests for POST /cronjobs/{name}/run."""

    def test_run_cronjob_without_auth_returns_401(self, client: TestClient):
        """POST /cronjobs/{name}/run without auth returns 401."""
        response = client.post("/cronjobs/data_grab/run")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_run_cronjob_with_user_token_requires_admin(
        self, client: TestClient
    ):
        """POST /cronjobs/{name}/run with non-admin token returns 403."""
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
            response = client.post("/cronjobs/data_grab/run")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            client.app.dependency_overrides.clear()


class TestCronLogsEndpoint:
    """Tests for GET /cronjobs/logs."""

    def test_logs_without_auth_returns_401(self, client: TestClient):
        """GET /cronjobs/logs without auth returns 401."""
        response = client.get("/cronjobs/logs/all")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_logs_with_user_token_requires_admin(
        self, client: TestClient
    ):
        """GET /cronjobs/logs with non-admin token returns 403."""
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
            response = client.get("/cronjobs/logs/all")
            assert response.status_code == status.HTTP_403_FORBIDDEN
        finally:
            client.app.dependency_overrides.clear()
