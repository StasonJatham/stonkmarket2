"""Tests for FinanceDatabase integration service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from app.database.orm import FinancialUniverse


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_equities_df() -> pd.DataFrame:
    """Sample equities DataFrame matching FinanceDatabase format."""
    return pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "name": ["Apple Inc.", "Microsoft Corporation", "Alphabet Inc."],
        "currency": ["USD", "USD", "USD"],
        "sector": ["Technology", "Technology", "Communication Services"],
        "industry_group": ["Technology Hardware & Equipment", "Software & Services", "Media & Entertainment"],
        "industry": ["Technology Hardware, Storage & Peripherals", "Systems Software", "Interactive Media & Services"],
        "exchange": ["NMS", "NMS", "NMS"],
        "market": ["NASDAQ", "NASDAQ", "NASDAQ"],
        "country": ["United States", "United States", "United States"],
        "market_cap": ["Mega Cap", "Mega Cap", "Mega Cap"],
        "isin": ["US0378331005", "US5949181045", "US02079K3059"],
        "cusip": ["037833100", "594918104", "02079K305"],
        "figi": ["BBG000B9XRY4", "BBG000BPH459", "BBG009S39JX6"],
        "summary": ["Apple designs...", "Microsoft develops...", "Alphabet provides..."],
    }).set_index("symbol")


@pytest.fixture
def mock_etfs_df() -> pd.DataFrame:
    """Sample ETFs DataFrame matching FinanceDatabase format."""
    return pd.DataFrame({
        "symbol": ["SPY", "QQQ", "VTI"],
        "name": ["SPDR S&P 500 ETF Trust", "Invesco QQQ Trust", "Vanguard Total Stock Market ETF"],
        "currency": ["USD", "USD", "USD"],
        "category_group": ["Equity", "Equity", "Equity"],
        "category": ["Large Blend", "Large Growth", "Large Blend"],
        "family": ["State Street", "Invesco", "Vanguard"],
        "exchange": ["PCX", "NMS", "PCX"],
    }).set_index("symbol")


# =============================================================================
# Unit Tests (no database needed)
# =============================================================================


class TestFinanceDatabaseService:
    """Tests for financedatabase_service module - unit tests with mocking."""

    def test_normalize_dataframe_equities(self, mock_equities_df: pd.DataFrame):
        """Test normalizing equities DataFrame to our schema."""
        from app.services.financedatabase_service import _normalize_dataframe
        
        # Reset index to match what _load_financedatabase_class returns
        df = mock_equities_df.reset_index()
        df = df.rename(columns={"index": "symbol"})
        
        records = _normalize_dataframe(df, "equity")
        
        assert len(records) == 3
        
        # Check AAPL record
        aapl = next(r for r in records if r["symbol"] == "AAPL")
        assert aapl["name"] == "Apple Inc."
        assert aapl["asset_class"] == "equity"
        assert aapl["sector"] == "Technology"
        assert aapl["industry"] == "Technology Hardware, Storage & Peripherals"
        assert aapl["country"] == "United States"
        assert aapl["isin"] == "US0378331005"
        assert aapl["cusip"] == "037833100"
        assert aapl["market_cap_category"] == "Mega Cap"
        assert aapl["is_active"] is True

    def test_normalize_dataframe_etfs(self, mock_etfs_df: pd.DataFrame):
        """Test normalizing ETFs DataFrame to our schema."""
        from app.services.financedatabase_service import _normalize_dataframe
        
        df = mock_etfs_df.reset_index()
        df = df.rename(columns={"index": "symbol"})
        
        records = _normalize_dataframe(df, "etf")
        
        assert len(records) == 3
        
        # Check SPY record
        spy = next(r for r in records if r["symbol"] == "SPY")
        assert spy["name"] == "SPDR S&P 500 ETF Trust"
        assert spy["asset_class"] == "etf"
        assert spy["category_group"] == "Equity"
        assert spy["category"] == "Large Blend"
        assert spy["family"] == "State Street"
        # ETFs shouldn't have sector/industry
        assert spy.get("sector") is None

    def test_clean_str_handles_nulls(self):
        """Test _clean_str handles various null values."""
        from app.services.financedatabase_service import _clean_str
        
        assert _clean_str(None, 100) is None
        assert _clean_str("", 100) is None
        assert _clean_str("nan", 100) is None
        assert _clean_str("None", 100) is None
        assert _clean_str(float("nan"), 100) is None

    def test_clean_str_truncates(self):
        """Test _clean_str truncates long strings."""
        from app.services.financedatabase_service import _clean_str
        
        long_string = "A" * 200
        result = _clean_str(long_string, 50)
        assert len(result) == 50
        assert result == "A" * 50

    def test_clean_str_preserves_valid(self):
        """Test _clean_str preserves valid strings."""
        from app.services.financedatabase_service import _clean_str
        
        assert _clean_str("Apple Inc.", 100) == "Apple Inc."
        assert _clean_str("  Trimmed  ", 100) == "Trimmed"


class TestFinanceDatabaseSearchMocked:
    """Tests for universe search with mocked database."""

    @pytest.mark.asyncio
    async def test_search_universe_calls_database(self, mocker):
        """Test search_universe queries the database correctly."""
        from app.services.financedatabase_service import search_universe
        
        # Mock the database session and result
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [
            {
                "id": 1,
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "asset_class": "equity",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "country": "United States",
                "exchange": "NMS",
                "category": None,
                "market_cap_category": "Mega Cap",
                "name_similarity": 0.9,
            }
        ]
        
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mocker.patch("app.services.financedatabase_service.get_session", return_value=mock_session)
        
        results = await search_universe("AAPL", use_trigram=False)
        
        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"
        assert results[0]["source"] == "universe"

    @pytest.mark.asyncio
    async def test_get_by_symbol_found(self, mocker):
        """Test get_by_symbol returns record when found."""
        from app.services.financedatabase_service import get_by_symbol
        
        # Create a mock row object
        mock_row = MagicMock()
        mock_row.symbol = "MSFT"
        mock_row.name = "Microsoft Corporation"
        mock_row.asset_class = "equity"
        mock_row.sector = "Technology"
        mock_row.industry = "Systems Software"
        mock_row.industry_group = "Software & Services"
        mock_row.country = "United States"
        mock_row.exchange = "NMS"
        mock_row.market = "NASDAQ"
        mock_row.currency = "USD"
        mock_row.category = None
        mock_row.category_group = None
        mock_row.family = None
        mock_row.market_cap_category = "Mega Cap"
        mock_row.isin = "US5949181045"
        mock_row.cusip = "594918104"
        mock_row.figi = "BBG000BPH459"
        mock_row.summary = "Microsoft develops..."
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row
        
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mocker.patch("app.services.financedatabase_service.get_session", return_value=mock_session)
        
        result = await get_by_symbol("MSFT")
        
        assert result is not None
        assert result["symbol"] == "MSFT"
        assert result["name"] == "Microsoft Corporation"
        assert result["sector"] == "Technology"
        assert result["isin"] == "US5949181045"
        assert result["source"] == "universe"

    @pytest.mark.asyncio
    async def test_get_by_symbol_not_found(self, mocker):
        """Test get_by_symbol returns None when not found."""
        from app.services.financedatabase_service import get_by_symbol
        
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mocker.patch("app.services.financedatabase_service.get_session", return_value=mock_session)
        
        result = await get_by_symbol("NOTEXIST")
        
        assert result is None


class TestUniverseStats:
    """Tests for universe statistics."""

    @pytest.mark.asyncio
    async def test_get_universe_stats(self, mocker):
        """Test getting universe statistics."""
        from app.services.financedatabase_service import get_universe_stats
        
        # First call returns asset class counts
        mock_counts_result = MagicMock()
        mock_counts_result.all.return_value = [
            ("equity", 24000),
            ("etf", 3000),
            ("fund", 31000),
            ("index", 62000),
            ("crypto", 3000),
        ]
        
        # Second call returns last updated timestamp
        mock_updated_result = MagicMock()
        mock_updated_result.scalar.return_value = datetime(2025, 12, 30, 10, 0, 0, tzinfo=UTC)
        
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=[mock_counts_result, mock_updated_result])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mocker.patch("app.services.financedatabase_service.get_session", return_value=mock_session)
        
        stats = await get_universe_stats()
        
        assert stats["counts"]["equity"] == 24000
        assert stats["counts"]["etf"] == 3000
        assert stats["total"] == 123000
        assert stats["last_updated"] is not None


class TestSymbolSearchIntegration:
    """Tests for symbol_search integration with universe."""

    @pytest.mark.asyncio
    async def test_lookup_symbol_checks_universe_before_api(self, mocker):
        """Test lookup_symbol checks universe before calling yfinance."""
        from app.services.symbol_search import lookup_symbol
        
        # Mock Symbol query - returns None (not in tracked symbols)
        mock_symbol_result = MagicMock()
        mock_symbol_result.one_or_none.return_value = None
        
        # Mock SymbolSearchResult query - returns None (not in cache)
        mock_cache_result = MagicMock()
        mock_cache_result.scalar_one_or_none.return_value = None
        
        # Mock FinancialUniverse query - returns a record
        mock_universe_row = MagicMock()
        mock_universe_row.symbol = "AMZN"
        mock_universe_row.name = "Amazon.com, Inc."
        mock_universe_row.asset_class = "equity"
        mock_universe_row.sector = "Consumer Discretionary"
        mock_universe_row.industry = "Internet & Direct Marketing Retail"
        mock_universe_row.industry_group = "Retailing"
        mock_universe_row.country = "United States"
        mock_universe_row.exchange = "NMS"
        mock_universe_row.market = "NASDAQ"
        mock_universe_row.currency = "USD"
        mock_universe_row.category = None
        mock_universe_row.category_group = None
        mock_universe_row.family = None
        mock_universe_row.market_cap_category = "Mega Cap"
        mock_universe_row.isin = "US0231351067"
        mock_universe_row.cusip = "023135106"
        mock_universe_row.figi = "BBG000BVPV84"
        mock_universe_row.summary = "Amazon.com operates..."
        
        mock_universe_result = MagicMock()
        mock_universe_result.scalar_one_or_none.return_value = mock_universe_row
        
        # Create mock sessions that return the right results
        call_count = 0
        
        async def mock_execute(stmt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_symbol_result
            elif call_count == 2:
                return mock_cache_result
            else:
                return mock_universe_result
        
        mock_session = AsyncMock()
        mock_session.execute = mock_execute
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        
        mocker.patch("app.services.symbol_search.get_session", return_value=mock_session)
        
        # Mock yfinance - should NOT be called
        mock_yf = mocker.patch("app.services.symbol_search._yf_service.validate_symbol")
        
        result = await lookup_symbol("AMZN")
        
        # Should find in universe, not call yfinance
        mock_yf.assert_not_called()
        
        assert result is not None
        assert result["symbol"] == "AMZN"
        assert result["source"] == "universe"
        assert result["valid"] is True
        assert result["quote_type"] == "EQUITY"
