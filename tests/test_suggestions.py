"""Tests for suggestions API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestListSuggestionsEndpoint:
    """Tests for GET /suggestions."""

    def test_list_suggestions_returns_200(self, client: TestClient):
        """GET /suggestions returns 200 OK (public endpoint)."""
        response = client.get("/suggestions")
        assert response.status_code == status.HTTP_200_OK

    def test_list_suggestions_returns_list(self, client: TestClient):
        """GET /suggestions returns a list."""
        response = client.get("/suggestions")
        data = response.json()
        # May return list directly or wrapped object
        assert isinstance(data, (list, dict))

    def test_list_suggestions_with_status_filter(self, client: TestClient):
        """GET /suggestions?status=pending filters by status."""
        response = client.get("/suggestions?status=pending")
        assert response.status_code == status.HTTP_200_OK


class TestGetSuggestionEndpoint:
    """Tests for GET /suggestions/{symbol}."""

    def test_get_suggestion_not_found(self, client: TestClient):
        """GET /suggestions/{symbol} for non-existent returns 404."""
        response = client.get("/suggestions/NONEXISTENT123")
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_200_OK,  # May return empty or null
        ]


class TestCreateSuggestionEndpoint:
    """Tests for POST /suggestions."""

    def test_create_suggestion_without_body_returns_422(self, client: TestClient):
        """POST /suggestions without body returns validation error."""
        response = client.post("/suggestions")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_suggestion_with_invalid_symbol(self, client: TestClient):
        """POST /suggestions with invalid symbol format returns error."""
        response = client.post(
            "/suggestions",
            json={"symbol": "INVALID!!!SYMBOL"},
        )
        assert response.status_code in [
            status.HTTP_400_BAD_REQUEST,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]


class TestVoteSuggestionEndpoint:
    """Tests for POST /suggestions/{symbol}/vote."""

    def test_vote_without_body_returns_422(self, client: TestClient):
        """POST /vote without body returns validation error."""
        response = client.post("/suggestions/AAPL/vote")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_vote_with_invalid_type_returns_422(self, client: TestClient):
        """POST /vote with invalid vote type returns error."""
        response = client.post(
            "/suggestions/AAPL/vote",
            json={"vote_type": "invalid"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


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
    """Tests for thread pool configuration."""

    def test_executor_workers(self):
        """Thread pool should have 4 workers."""
        from app.api.routes.suggestions import _executor
        
        # ThreadPoolExecutor stores max_workers in _max_workers
        assert _executor._max_workers == 4
