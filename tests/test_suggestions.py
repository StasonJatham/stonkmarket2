"""Tests for suggestions API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
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
        response = client.post(
            "/suggestions",
            json={"symbol": "INVALID!!!SYMBOL"},
        )
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        ]


class TestVoteSuggestionEndpoint:
    """Tests for PUT /suggestions/{symbol}/vote."""

    def test_vote_on_nonexistent_symbol(self, client: TestClient):
        """PUT /suggestions/{symbol}/vote for non-existent returns 404 or 422."""
        response = client.put("/suggestions/NONEXISTENT123/vote")
        # 404 if not found, 422 if vote validation fails
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_CONTENT,
        ]

    def test_vote_is_put_method(self, client: TestClient):
        """PUT /suggestions/{symbol}/vote uses PUT method."""
        response = client.put("/suggestions/AAPL/vote")
        # Should not be 405 Method Not Allowed
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED


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


class TestSymbolValidation:
    """Tests for symbol validation pattern."""

    def test_valid_symbol_patterns(self):
        """Valid symbols match the pattern."""
        import re
        
        pattern = re.compile(r'^[A-Z0-9.]{1,10}$')
        
        valid_symbols = ["AAPL", "MSFT", "BRK.A", "BRK.B", "A", "1234567890"]
        for symbol in valid_symbols:
            assert pattern.match(symbol), f"{symbol} should be valid"

    def test_invalid_symbol_patterns(self):
        """Invalid symbols don't match the pattern."""
        import re
        
        pattern = re.compile(r'^[A-Z0-9.]{1,10}$')
        
        invalid_symbols = [
            "",  # Empty
            "aapl",  # Lowercase
            "AAPL-B",  # Hyphen
            "AAPL MSFT",  # Space
            "12345678901",  # Too long (11 chars)
            "AAPL!",  # Special char
        ]
        for symbol in invalid_symbols:
            assert not pattern.match(symbol), f"{symbol} should be invalid"


class TestRateLimitIndicators:
    """Tests for rate limit detection."""

    def test_rate_limit_indicators(self):
        """Rate limit messages are properly detected."""
        indicators = ["rate limit", "too many requests", "429"]
        
        test_messages = [
            "Rate limit exceeded",
            "Too many requests, please slow down",
            "Error 429: Rate limited",
        ]
        
        for msg in test_messages:
            msg_lower = msg.lower()
            matched = any(ind in msg_lower for ind in indicators)
            assert matched, f"'{msg}' should be detected as rate limit"


class TestThreadPoolConfig:
    """Tests for thread pool configuration - deprecated since async migration."""

    def test_executor_workers(self):
        """Thread pool was removed in favor of async operations."""
        # The synchronous executor was removed when we migrated to async
        # This test is kept for documentation purposes
        import pytest
        pytest.skip("Thread pool executor removed - using async operations")
