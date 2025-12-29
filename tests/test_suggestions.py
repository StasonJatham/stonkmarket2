"""Tests for suggestions API endpoints."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient


class TestListSuggestionsEndpoint:
    """Tests for GET /suggestions (admin only)."""

    def test_list_suggestions_requires_auth(self, client: TestClient):
        """GET /suggestions requires admin auth."""
        response = client.get("/suggestions")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestTopSuggestionsEndpoint:
    """Tests for GET /suggestions/top (public)."""

    def test_top_suggestions_returns_200(self, client: TestClient):
        """GET /suggestions/top returns 200 OK (public endpoint)."""
        response = client.get("/suggestions/top")
        assert response.status_code == status.HTTP_200_OK

    def test_top_suggestions_returns_list(self, client: TestClient):
        """GET /suggestions/top returns a list."""
        response = client.get("/suggestions/top")
        data = response.json()
        assert isinstance(data, list)


class TestCreateSuggestionEndpoint:
    """Tests for POST /suggestions."""

    def test_create_suggestion_without_body_returns_422(self, client: TestClient):
        """POST /suggestions without body returns validation error."""
        response = client.post("/suggestions")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_create_suggestion_with_invalid_symbol(self, client: TestClient):
        """POST /suggestions with invalid symbol format returns error."""
        response = client.post("/suggestions?symbol=INVALID!!!SYMBOL")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestApproveSuggestionEndpoint:
    """Tests for POST /suggestions/{symbol}/approve (admin only)."""

    def test_approve_without_auth_returns_401(self, client: TestClient):
        """POST /approve without auth returns 401."""
        response = client.post("/suggestions/AAPL/approve")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestRejectSuggestionEndpoint:
    """Tests for POST /suggestions/{symbol}/reject (admin only)."""

    def test_reject_without_auth_returns_401(self, client: TestClient):
        """POST /reject without auth returns 401."""
        response = client.post("/suggestions/AAPL/reject")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
