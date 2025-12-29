"""Tests for swipe API endpoints."""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient


class TestSwipeCardsEndpoint:
    """Tests for GET /swipe/cards."""

    def test_cards_returns_list(self, client: TestClient):
        """GET /swipe/cards returns a DipCardList with cards array."""
        response = client.get("/swipe/cards")
        data = response.json()
        assert isinstance(data, dict)
        assert "cards" in data
        assert "total" in data
        assert isinstance(data["cards"], list)


class TestSwipeCardEndpoint:
    """Tests for GET /swipe/cards/{symbol}."""

    def test_card_with_invalid_symbol_returns_404(self, client: TestClient):
        """GET /swipe/cards/{symbol} with invalid symbol returns 404."""
        response = client.get("/swipe/cards/INVALID_SYMBOL_XYZ123")
        assert response.status_code == status.HTTP_404_NOT_FOUND


class TestSwipeVoteEndpoint:
    """Tests for PUT /swipe/cards/{symbol}/vote."""

    def test_vote_without_body_returns_422(self, client: TestClient):
        """PUT /swipe/cards/{symbol}/vote without body returns 422."""
        response = client.put("/swipe/cards/AAPL/vote")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_vote_with_invalid_vote_type_returns_422(self, client: TestClient):
        """PUT /swipe/cards/{symbol}/vote with invalid vote type returns 422."""
        response = client.put(
            "/swipe/cards/AAPL/vote",
            json={"vote_type": "invalid"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
