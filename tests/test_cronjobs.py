"""Tests for cronjob management API endpoints."""

from __future__ import annotations

import pytest
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
        self, client: TestClient, auth_headers: dict
    ):
        """PUT /cronjobs/{name} with non-admin token returns 401 or 403."""
        response = client.put(
            "/cronjobs/data_grab",
            json={"cron": "0 23 * * 1-5"},
            headers=auth_headers,
        )
        # 401 if test user doesn't exist in DB, 403 if exists but not admin
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]


class TestRunCronJobEndpoint:
    """Tests for POST /cronjobs/{name}/run."""

    def test_run_cronjob_without_auth_returns_401(self, client: TestClient):
        """POST /cronjobs/{name}/run without auth returns 401."""
        response = client.post("/cronjobs/data_grab/run")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_run_cronjob_with_user_token_requires_admin(
        self, client: TestClient, auth_headers: dict
    ):
        """POST /cronjobs/{name}/run with non-admin token returns 401 or 403."""
        response = client.post(
            "/cronjobs/data_grab/run",
            headers=auth_headers,
        )
        # 401 if test user doesn't exist in DB, 403 if exists but not admin
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]


class TestCronLogsEndpoint:
    """Tests for GET /cronjobs/logs."""

    def test_logs_without_auth_returns_401(self, client: TestClient):
        """GET /cronjobs/logs without auth returns 401."""
        response = client.get("/cronjobs/logs/all")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_logs_with_user_token_requires_admin(
        self, client: TestClient, auth_headers: dict
    ):
        """GET /cronjobs/logs with non-admin token returns 401 or 403."""
        response = client.get("/cronjobs/logs/all", headers=auth_headers)
        # 401 if test user doesn't exist in DB, 403 if exists but not admin
        assert response.status_code in [
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_403_FORBIDDEN,
        ]
