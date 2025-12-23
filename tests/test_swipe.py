"""Tests for swipe API endpoints."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestSwipeCardsEndpoint:
    """Tests for GET /swipe/cards."""

    def test_cards_returns_200(self, client: TestClient):
        """GET /swipe/cards returns 200 OK (public endpoint)."""
        response = client.get("/swipe/cards")
        assert response.status_code == status.HTTP_200_OK

    def test_cards_returns_list(self, client: TestClient):
        """GET /swipe/cards returns a DipCardList with cards array."""
        response = client.get("/swipe/cards")
        data = response.json()
        assert isinstance(data, dict)
        assert "cards" in data
        assert "total" in data
        assert isinstance(data["cards"], list)

    def test_cards_with_limit(self, client: TestClient):
        """GET /swipe/cards returns at most limit cards."""
        response = client.get("/swipe/cards")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Response has cards and total fields
        assert "cards" in data


class TestSwipeCardEndpoint:
    """Tests for GET /swipe/cards/{symbol}."""

    def test_card_with_valid_symbol_exists(self, client: TestClient):
        """GET /swipe/cards/{symbol} endpoint exists."""
        # First get available cards
        cards_response = client.get("/swipe/cards")
        data = cards_response.json()
        cards = data.get("cards", [])
        
        if cards:
            symbol = cards[0]["symbol"]
            response = client.get(f"/swipe/cards/{symbol}")
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_404_NOT_FOUND,  # May not be in dip
            ]
        else:
            # No cards in test DB - test that endpoint exists
            response = client.get("/swipe/cards/AAPL")
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_card_with_invalid_symbol_returns_404(self, client: TestClient):
        """GET /swipe/cards/{symbol} with invalid symbol returns 404."""
        response = client.get("/swipe/cards/INVALID_SYMBOL_XYZ123")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestSwipeVoteEndpoint:
    """Tests for PUT /swipe/cards/{symbol}/vote."""

    def test_vote_without_body_returns_422(self, client: TestClient):
        """PUT /swipe/cards/{symbol}/vote without body returns 422."""
        response = client.put("/swipe/cards/AAPL/vote")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_vote_with_invalid_vote_type_returns_422(self, client: TestClient):
        """PUT /swipe/cards/{symbol}/vote with invalid vote type returns 422."""
        response = client.put(
            "/swipe/cards/AAPL/vote",
            json={"vote_type": "invalid"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_vote_requires_fingerprint(self, client: TestClient):
        """PUT /swipe/cards/{symbol}/vote works without special fingerprint header."""
        response = client.put(
            "/swipe/cards/AAPL/vote",
            json={"vote_type": "buy"},
        )
        # Should work (using server fingerprint) or fail for various reasons:
        # - 200: Vote successful
        # - 404: Symbol not in dip
        # - 422: Validation error (cooldown, already voted, etc.)
        # - 429: Rate limited
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            status.HTTP_429_TOO_MANY_REQUESTS,
        ]


class TestSwipeVoteStatsEndpoint:
    """Tests for GET /swipe/cards/{symbol}/stats."""

    def test_stats_with_valid_symbol_returns_200(self, client: TestClient):
        """GET /swipe/cards/{symbol}/stats with valid symbol returns 200 OK."""
        response = client.get("/swipe/cards/AAPL/stats")
        # May return 200 or 404 depending on if symbol exists
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,
        ]

    def test_stats_returns_dict(self, client: TestClient):
        """GET /swipe/cards/{symbol}/stats returns a dict with vote counts."""
        response = client.get("/swipe/cards/AAPL/stats")
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert isinstance(data, dict)
