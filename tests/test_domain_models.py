"""Tests for domain models.

Validates Pydantic domain models: TickerInfo, PriceHistory, QualityMetrics, etc.
"""

from datetime import date, datetime

import pandas as pd
import pytest

from app.domain import (
    EarningsEvent,
    EconomicEvent,
    FundamentalsData,
    IpoEvent,
    PriceBar,
    PriceHistory,
    QualityMetrics,
    SplitEvent,
    TickerInfo,
    TickerSearchResult,
)


class TestTickerInfo:
    """Tests for TickerInfo domain model."""

    def test_minimal_ticker_info(self):
        """TickerInfo should work with only symbol."""
        info = TickerInfo(symbol="AAPL")
        assert info.symbol == "AAPL"
        assert info.name is None
        assert info.is_etf is False

    def test_full_ticker_info(self):
        """TickerInfo should accept all fields."""
        info = TickerInfo(
            symbol="AAPL",
            name="Apple Inc.",
            quote_type="EQUITY",
            sector="Technology",
            industry="Consumer Electronics",
            current_price=175.50,
            previous_close=174.00,
            market_cap=2_800_000_000_000,
            pe_ratio=28.5,
            profit_margin=0.25,
            return_on_equity=1.47,
        )
        assert info.symbol == "AAPL"
        assert info.name == "Apple Inc."
        assert info.is_etf is False
        assert info.current_price == 175.50

    def test_etf_detection(self):
        """ETFs should be detected via quote_type."""
        etf = TickerInfo(symbol="SPY", quote_type="ETF")
        assert etf.is_etf is True

        index = TickerInfo(symbol="^GSPC", quote_type="INDEX")
        assert index.is_etf is True

        stock = TickerInfo(symbol="AAPL", quote_type="EQUITY")
        assert stock.is_etf is False

    def test_target_upside_computed_field(self):
        """target_upside should be computed from prices."""
        info = TickerInfo(
            symbol="AAPL",
            current_price=150.0,
            target_mean_price=180.0,
        )
        assert info.target_upside == pytest.approx(0.2, rel=0.01)

    def test_fcf_to_market_cap_computed_field(self):
        """fcf_to_market_cap should be computed from financials."""
        info = TickerInfo(
            symbol="AAPL",
            free_cash_flow=100_000_000_000,
            market_cap=2_500_000_000_000,
        )
        assert info.fcf_to_market_cap == pytest.approx(0.04, rel=0.01)

    def test_from_dict_extra_fields_ignored(self):
        """Unknown fields from yfinance should be ignored."""
        data = {
            "symbol": "AAPL",
            "name": "Apple",
            "unknownField": "should be ignored",
            "anotherUnknown": 12345,
        }
        info = TickerInfo(**data)
        assert info.symbol == "AAPL"
        assert not hasattr(info, "unknownField")


class TestTickerSearchResult:
    """Tests for TickerSearchResult domain model."""

    def test_search_result(self):
        """TickerSearchResult should store search results."""
        result = TickerSearchResult(
            symbol="AAPL",
            name="Apple Inc.",
            exchange="NASDAQ",
            quote_type="EQUITY",
            score=0.95,
        )
        assert result.symbol == "AAPL"
        assert result.score == 0.95


class TestPriceBar:
    """Tests for PriceBar domain model."""

    def test_price_bar(self):
        """PriceBar should store OHLCV data."""
        bar = PriceBar(
            date=date(2025, 1, 3),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1_000_000,
        )
        assert bar.open == 100.0
        assert bar.close == 103.0

    def test_computed_fields(self):
        """PriceBar should compute change and range."""
        bar = PriceBar(
            date=date(2025, 1, 3),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=1_000_000,
        )
        assert bar.change == 5.0
        assert bar.change_pct == pytest.approx(0.05, rel=0.01)
        assert bar.range == 15.0


class TestPriceHistory:
    """Tests for PriceHistory domain model."""

    def test_empty_history(self):
        """PriceHistory should work when empty."""
        history = PriceHistory(symbol="AAPL", bars=[])
        assert len(history) == 0
        assert history.latest_close is None
        assert history.start_date is None

    def test_history_with_bars(self):
        """PriceHistory should contain bars."""
        bars = [
            PriceBar(date=date(2025, 1, 1), open=100, high=105, low=99, close=102, volume=1000),
            PriceBar(date=date(2025, 1, 2), open=102, high=108, low=101, close=107, volume=1500),
            PriceBar(date=date(2025, 1, 3), open=107, high=110, low=105, close=109, volume=2000),
        ]
        history = PriceHistory(symbol="AAPL", bars=bars)

        assert len(history) == 3
        assert history.latest_close == 109.0
        assert history.start_date == date(2025, 1, 1)
        assert history.end_date == date(2025, 1, 3)
        assert history.high_52w == 110.0
        assert history.low_52w == 99.0

    def test_iteration(self):
        """PriceHistory should be iterable."""
        bars = [
            PriceBar(date=date(2025, 1, 1), open=100, high=105, low=99, close=102, volume=1000),
            PriceBar(date=date(2025, 1, 2), open=102, high=108, low=101, close=107, volume=1500),
        ]
        history = PriceHistory(symbol="AAPL", bars=bars)

        count = 0
        for bar in history:
            count += 1
            assert isinstance(bar, PriceBar)
        assert count == 2

    def test_to_dataframe(self):
        """PriceHistory should convert to DataFrame."""
        bars = [
            PriceBar(date=date(2025, 1, 1), open=100, high=105, low=99, close=102, volume=1000),
            PriceBar(date=date(2025, 1, 2), open=102, high=108, low=101, close=107, volume=1500),
        ]
        history = PriceHistory(symbol="AAPL", bars=bars)

        df = history.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_from_dataframe(self):
        """PriceHistory should be created from DataFrame."""
        df = pd.DataFrame({
            "Open": [100.0, 102.0],
            "High": [105.0, 108.0],
            "Low": [99.0, 101.0],
            "Close": [102.0, 107.0],
            "Volume": [1000, 1500],
        }, index=pd.to_datetime(["2025-01-01", "2025-01-02"]))

        history = PriceHistory.from_dataframe("AAPL", df)
        assert len(history) == 2
        assert history.symbol == "AAPL"
        assert history.bars[0].close == 102.0

    def test_slice(self):
        """PriceHistory.slice should return recent bars."""
        bars = [
            PriceBar(date=date(2025, 1, i), open=100, high=105, low=99, close=102, volume=1000)
            for i in range(1, 11)
        ]
        history = PriceHistory(symbol="AAPL", bars=bars)

        sliced = history.slice(5)
        assert len(sliced) == 5
        assert sliced.bars[0].date == date(2025, 1, 6)


class TestFundamentalsData:
    """Tests for FundamentalsData domain model."""

    def test_fundamentals_from_ticker_info(self):
        """FundamentalsData should be created from ticker info dict."""
        info = {
            "symbol": "AAPL",
            "profit_margin": 0.25,
            "operating_margin": 0.30,
            "debt_to_equity": 1.5,
            "current_ratio": 1.2,
            "free_cash_flow": 100_000_000_000,
            "market_cap": 2_500_000_000_000,
            "current_price": 175.0,
            "target_mean_price": 200.0,
        }
        fundamentals = FundamentalsData.from_ticker_info(info)

        assert fundamentals.symbol == "AAPL"
        assert fundamentals.profit_margin == 0.25
        assert fundamentals.fcf_to_market_cap == pytest.approx(0.04, rel=0.01)
        assert fundamentals.target_upside == pytest.approx(0.143, rel=0.01)

    def test_fields_available(self):
        """fields_available should count non-None fields."""
        fundamentals = FundamentalsData(
            symbol="AAPL",
            profit_margin=0.25,
            operating_margin=0.30,
            debt_to_equity=1.5,
            # Other fields are None
        )
        assert fundamentals.fields_available == 3


class TestQualityMetrics:
    """Tests for QualityMetrics domain model."""

    def test_quality_metrics(self):
        """QualityMetrics should store scores."""
        metrics = QualityMetrics(
            symbol="AAPL",
            score=75.0,
            profitability_score=80.0,
            balance_sheet_score=70.0,
            cash_generation_score=75.0,
            growth_score=72.0,
            liquidity_score=78.0,
            valuation_score=68.0,
            analyst_score=82.0,
            risk_score=75.0,
            fields_available=14,
            fields_total=16,
        )

        assert metrics.score == 75.0
        assert metrics.data_completeness == pytest.approx(0.875, rel=0.01)

    def test_get_weakest_category(self):
        """get_weakest_category should find lowest sub-score."""
        metrics = QualityMetrics(
            symbol="AAPL",
            score=70.0,
            profitability_score=80.0,
            balance_sheet_score=70.0,
            cash_generation_score=65.0,
            growth_score=60.0,
            liquidity_score=55.0,
            valuation_score=45.0,  # Lowest
            analyst_score=75.0,
            risk_score=70.0,
        )
        category, score = metrics.get_weakest_category()
        assert category == "valuation"
        assert score == 45.0

    def test_get_strongest_category(self):
        """get_strongest_category should find highest sub-score."""
        metrics = QualityMetrics(
            symbol="AAPL",
            score=70.0,
            profitability_score=85.0,  # Highest
            balance_sheet_score=70.0,
            valuation_score=55.0,
        )
        category, score = metrics.get_strongest_category()
        assert category == "profitability"
        assert score == 85.0

    def test_from_dataclass_conversion(self):
        """QualityMetrics should be creatable from dipfinder dataclass."""
        from app.dipfinder.fundamentals import QualityMetrics as QMDataclass

        # Create dataclass instance
        dc = QMDataclass(
            ticker="AAPL",
            score=75.5,
            profit_margin=0.25,
            profitability_score=80.0,
            balance_sheet_score=70.0,
            fields_available=5,
            fields_total=16,
        )

        # Convert to Pydantic
        pydantic_qm = dc.to_pydantic()

        assert pydantic_qm.symbol == "AAPL"
        assert pydantic_qm.score == 75.5
        assert pydantic_qm.profit_margin == 0.25
        assert pydantic_qm.profitability_score == 80.0
        assert pydantic_qm.data_completeness == pytest.approx(5 / 16, rel=0.01)

    def test_ticker_alias(self):
        """QualityMetrics should accept both 'symbol' and 'ticker' field names."""
        # Using ticker (alias)
        metrics1 = QualityMetrics(ticker="AAPL", score=70.0)
        assert metrics1.symbol == "AAPL"

        # Using symbol (primary)
        metrics2 = QualityMetrics(symbol="GOOG", score=80.0)
        assert metrics2.symbol == "GOOG"


class TestEarningsEvent:
    """Tests for EarningsEvent domain model."""

    def test_earnings_event(self):
        """EarningsEvent should store earnings data."""
        event = EarningsEvent(
            symbol="AAPL",
            company_name="Apple Inc.",
            report_date=date(2025, 1, 30),
            eps_estimate=2.10,
            eps_actual=2.25,
            time_of_day="after_market",
        )
        assert event.symbol == "AAPL"
        assert event.has_reported is True
        assert event.eps_surprise == pytest.approx(0.15, rel=0.01)
        assert event.eps_surprise_pct == pytest.approx(0.0714, rel=0.01)

    def test_upcoming_earnings(self):
        """Upcoming earnings should have no actuals."""
        event = EarningsEvent(
            symbol="AAPL",
            report_date=date(2025, 4, 30),
            eps_estimate=2.20,
        )
        assert event.has_reported is False
        assert event.eps_surprise is None


class TestIpoEvent:
    """Tests for IpoEvent domain model."""

    def test_ipo_event(self):
        """IpoEvent should store IPO data."""
        event = IpoEvent(
            symbol="NEWCO",
            company_name="New Company Inc.",
            ipo_date=date(2025, 2, 15),
            price_low=18.0,
            price_high=22.0,
            shares_offered=10_000_000,
            exchange="NASDAQ",
        )
        assert event.price_range == (18.0, 22.0)
        assert event.price_midpoint == 20.0


class TestSplitEvent:
    """Tests for SplitEvent domain model."""

    def test_forward_split(self):
        """Forward split should have factor > 1."""
        event = SplitEvent(
            symbol="AAPL",
            split_date=date(2025, 3, 1),
            split_ratio="4:1",
            numerator=4,
            denominator=1,
        )
        assert event.is_reverse_split is False
        assert event.split_factor == 4.0

    def test_reverse_split(self):
        """Reverse split should have factor < 1."""
        event = SplitEvent(
            symbol="XYZ",
            split_date=date(2025, 3, 1),
            split_ratio="1:10",
            numerator=1,
            denominator=10,
        )
        assert event.is_reverse_split is True
        assert event.split_factor == 0.1


class TestEconomicEvent:
    """Tests for EconomicEvent domain model."""

    def test_economic_event(self):
        """EconomicEvent should store economic data."""
        event = EconomicEvent(
            event_name="Non-Farm Payrolls",
            event_date=date(2025, 2, 7),
            country="US",
            previous_value=200000,
            forecast_value=180000,
            actual_value=250000,
            importance="high",
        )
        assert event.has_released is True
        assert event.surprise == 70000

    def test_upcoming_economic_event(self):
        """Upcoming events should have no actual."""
        event = EconomicEvent(
            event_name="CPI",
            event_date=date(2025, 2, 14),
            forecast_value=3.2,
        )
        assert event.has_released is False
        assert event.surprise is None
