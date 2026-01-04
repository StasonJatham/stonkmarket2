"""Tests for typed YFinance service methods.

Validates the typed wrapper methods return proper Pydantic models.
"""

from datetime import date
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.domain import PriceHistory, TickerInfo, TickerSearchResult
from app.services.data_providers import get_yfinance_service


class TestYFinanceTypedMethods:
    """Tests for YFinanceService typed wrapper methods."""

    @pytest.fixture
    def yf_service(self):
        """Get YFinance service instance."""
        return get_yfinance_service()

    @pytest.fixture
    def mock_ticker_info(self) -> dict:
        """Sample ticker info dict."""
        return {
            "symbol": "AAPL",
            "name": "Apple Inc.",
            "quote_type": "EQUITY",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "current_price": 175.50,
            "previous_close": 174.00,
            "market_cap": 2_800_000_000_000,
            "pe_ratio": 28.5,
            "profit_margin": 0.25,
            "return_on_equity": 1.47,
            "debt_to_equity": 1.5,
        }

    @pytest.fixture
    def mock_price_df(self) -> pd.DataFrame:
        """Sample price DataFrame."""
        return pd.DataFrame({
            "Open": [100.0, 102.0, 105.0],
            "High": [105.0, 108.0, 110.0],
            "Low": [99.0, 101.0, 103.0],
            "Close": [102.0, 107.0, 109.0],
            "Volume": [1000000, 1500000, 2000000],
        }, index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]))

    @pytest.fixture
    def mock_search_results(self) -> list[dict]:
        """Sample search results."""
        return [
            {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "quote_type": "EQUITY",
                "relevance_score": 0.95,
            },
            {
                "symbol": "APLE",
                "name": "Apple Hospitality REIT",
                "exchange": "NYSE",
                "quote_type": "EQUITY",
                "relevance_score": 0.75,
            },
        ]

    @pytest.mark.asyncio
    async def test_get_ticker_info_typed_returns_model(
        self,
        yf_service,
        mock_ticker_info,
    ):
        """get_ticker_info_typed should return TickerInfo model."""
        with patch.object(
            yf_service,
            "get_ticker_info",
            new_callable=AsyncMock,
            return_value=mock_ticker_info,
        ):
            result = await yf_service.get_ticker_info_typed("AAPL")

            assert result is not None
            assert isinstance(result, TickerInfo)
            assert result.symbol == "AAPL"
            assert result.name == "Apple Inc."
            assert result.is_etf is False
            assert result.current_price == 175.50
            assert result.pe_ratio == 28.5

    @pytest.mark.asyncio
    async def test_get_ticker_info_typed_returns_none_when_no_data(
        self,
        yf_service,
    ):
        """get_ticker_info_typed should return None if underlying method returns None."""
        with patch.object(
            yf_service,
            "get_ticker_info",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await yf_service.get_ticker_info_typed("INVALID")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_price_history_typed_returns_model(
        self,
        yf_service,
        mock_price_df,
    ):
        """get_price_history_typed should return PriceHistory model."""
        with patch.object(
            yf_service,
            "get_price_history",
            new_callable=AsyncMock,
            return_value=(mock_price_df, "abc123"),
        ):
            result = await yf_service.get_price_history_typed("AAPL", period="1y")

            assert result is not None
            assert isinstance(result, PriceHistory)
            assert result.symbol == "AAPL"
            assert len(result) == 3
            assert result.latest_close == 109.0
            assert result.version_hash == "abc123"

    @pytest.mark.asyncio
    async def test_get_price_history_typed_returns_none_when_empty(
        self,
        yf_service,
    ):
        """get_price_history_typed should return None if DataFrame is empty."""
        with patch.object(
            yf_service,
            "get_price_history",
            new_callable=AsyncMock,
            return_value=(pd.DataFrame(), None),
        ):
            result = await yf_service.get_price_history_typed("INVALID")
            assert result is None

    @pytest.mark.asyncio
    async def test_search_tickers_typed_returns_models(
        self,
        yf_service,
        mock_search_results,
    ):
        """search_tickers_typed should return list of TickerSearchResult models."""
        with patch.object(
            yf_service,
            "search_tickers",
            new_callable=AsyncMock,
            return_value=mock_search_results,
        ):
            results = await yf_service.search_tickers_typed("apple")

            assert len(results) == 2
            assert all(isinstance(r, TickerSearchResult) for r in results)
            assert results[0].symbol == "AAPL"
            assert results[0].score == 0.95
            assert results[1].symbol == "APLE"

    @pytest.mark.asyncio
    async def test_search_tickers_typed_returns_empty_list_when_no_results(
        self,
        yf_service,
    ):
        """search_tickers_typed should return empty list if no results."""
        with patch.object(
            yf_service,
            "search_tickers",
            new_callable=AsyncMock,
            return_value=[],
        ):
            results = await yf_service.search_tickers_typed("xyznonexistent")
            assert results == []

    @pytest.mark.asyncio
    async def test_price_history_model_utility_methods(
        self,
        yf_service,
        mock_price_df,
    ):
        """PriceHistory model should have working utility methods."""
        with patch.object(
            yf_service,
            "get_price_history",
            new_callable=AsyncMock,
            return_value=(mock_price_df, "hash123"),
        ):
            history = await yf_service.get_price_history_typed("AAPL")
            assert history is not None

            # Test slice
            sliced = history.slice(2)
            assert len(sliced) == 2
            assert sliced.symbol == "AAPL"

            # Test to_dataframe roundtrip
            df = history.to_dataframe()
            assert len(df) == 3
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

            # Test iteration
            bars = list(history)
            assert len(bars) == 3
            assert bars[0].close == 102.0

    @pytest.mark.asyncio
    async def test_ticker_info_computed_fields(
        self,
        yf_service,
        mock_ticker_info,
    ):
        """TickerInfo computed fields should work correctly."""
        # Add target price for upside calculation
        mock_ticker_info["target_mean_price"] = 200.0
        mock_ticker_info["free_cash_flow"] = 100_000_000_000

        with patch.object(
            yf_service,
            "get_ticker_info",
            new_callable=AsyncMock,
            return_value=mock_ticker_info,
        ):
            info = await yf_service.get_ticker_info_typed("AAPL")
            assert info is not None

            # target_upside = (200 - 175.5) / 175.5 ≈ 0.1396
            assert info.target_upside == pytest.approx(0.1396, rel=0.01)

            # fcf_to_market_cap = 100B / 2800B ≈ 0.0357
            assert info.fcf_to_market_cap == pytest.approx(0.0357, rel=0.01)
