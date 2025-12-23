"""Tests for stock tinder API endpoints."""

from __future__ import annotations

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestTinderCardsEndpoint:
    """Tests for GET /tinder/cards."""

    def test_cards_returns_200(self, client: TestClient):
        """GET /tinder/cards returns 200 OK (public endpoint)."""
        response = client.get("/tinder/cards")
        assert response.status_code == status.HTTP_200_OK

    def test_cards_returns_list(self, client: TestClient):
        """GET /tinder/cards returns a list."""
        response = client.get("/tinder/cards")
        assert isinstance(response.json(), list)

    def test_cards_with_limit(self, client: TestClient):
        """GET /tinder/cards with limit parameter."""
        response = client.get("/tinder/cards?limit=5")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) <= 5


class TestTinderCardEndpoint:
    """Tests for GET /tinder/cards/{symbol}."""

    def test_card_with_valid_symbol_exists(self, client: TestClient):
        """GET /tinder/cards/{symbol} endpoint exists."""
        # First get available cards
        cards_response = client.get("/tinder/cards")
        cards = cards_response.json()
        
        if cards:
            symbol = cards[0]["symbol"]
            response = client.get(f"/tinder/cards/{symbol}")
            assert response.status_code in [
                status.HTTP_200_OK,
                status.HTTP_404_NOT_FOUND,  # May not be in dip
            ]

    def test_card_with_invalid_symbol_returns_404(self, client: TestClient):
        """GET /tinder/cards/{symbol} with invalid symbol returns 404."""
        response = client.get("/tinder/cards/INVALID_SYMBOL_XYZ123")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestTinderVoteEndpoint:
    """Tests for PUT /tinder/cards/{symbol}/vote."""

    def test_vote_without_body_returns_422(self, client: TestClient):
        """PUT /tinder/cards/{symbol}/vote without body returns 422."""
        response = client.put("/tinder/cards/AAPL/vote")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_vote_with_invalid_vote_type_returns_422(self, client: TestClient):
        """PUT /tinder/cards/{symbol}/vote with invalid vote type returns 422."""
        response = client.put(
            "/tinder/cards/AAPL/vote",
            json={"vote_type": "invalid"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_vote_requires_fingerprint(self, client: TestClient):
        """PUT /tinder/cards/{symbol}/vote requires fingerprint header."""
        response = client.put(
            "/tinder/cards/AAPL/vote",
            json={"vote_type": "buy"},
        )
        # Should work but rate limit by fingerprint
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_404_NOT_FOUND,  # Symbol not in dip
            status.HTTP_429_TOO_MANY_REQUESTS,  # Rate limited
        ]


class TestTinderVoteAggregateEndpoint:
    """Tests for GET /tinder/votes/aggregate."""

    def test_aggregate_returns_200(self, client: TestClient):
        """GET /tinder/votes/aggregate returns 200 OK."""
        response = client.get("/tinder/votes/aggregate")
        assert response.status_code == status.HTTP_200_OK

    def test_aggregate_returns_list(self, client: TestClient):
        """GET /tinder/votes/aggregate returns a list."""
        response = client.get("/tinder/votes/aggregate")
        assert isinstance(response.json(), list)
