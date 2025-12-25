"""
Tests for the hedge fund analysis module.
"""

import pytest
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.hedge_fund.schemas import (
    AgentSignal,
    AgentType,
    AnalysisBundle,
    AnalysisRequest,
    Fundamentals,
    LLMMode,
    MarketData,
    PerTickerReport,
    PortfolioDecision,
    PricePoint,
    PriceSeries,
    Signal,
    TickerInput,
)


# =============================================================================
# Schema Tests
# =============================================================================


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_ticker_input_uppercase(self):
        """Symbol should be uppercased."""
        ticker = TickerInput(symbol="aapl")
        assert ticker.symbol == "AAPL"

    def test_ticker_input_strips_whitespace(self):
        """Symbol should strip whitespace."""
        ticker = TickerInput(symbol="  msft  ")
        assert ticker.symbol == "MSFT"

    def test_agent_signal_confidence_normalization(self):
        """Confidence should be normalized to 0-1."""
        # Already 0-1
        signal = AgentSignal(
            agent_id="test",
            agent_name="Test",
            agent_type=AgentType.FUNDAMENTALS,
            symbol="AAPL",
            signal=Signal.BUY,
            confidence=0.8,
            reasoning="Test",
        )
        assert signal.confidence == 0.8

        # 1-10 scale should be normalized
        signal2 = AgentSignal(
            agent_id="test",
            agent_name="Test",
            agent_type=AgentType.FUNDAMENTALS,
            symbol="AAPL",
            signal=Signal.BUY,
            confidence=8,  # Should become 0.8
            reasoning="Test",
        )
        assert signal2.confidence == 0.8

    def test_price_series_returns_calculation(self):
        """PriceSeries should calculate returns correctly."""
        prices = PriceSeries(
            symbol="AAPL",
            prices=[
                PricePoint(date=date(2024, 1, 1), open=100, high=105, low=99, close=100, volume=1000),
                PricePoint(date=date(2024, 1, 2), open=100, high=110, low=100, close=110, volume=1200),
                PricePoint(date=date(2024, 1, 3), open=110, high=115, low=108, close=105, volume=800),
            ]
        )
        returns = prices.returns
        assert len(returns) == 2
        assert returns[0] == pytest.approx(0.10, rel=1e-3)  # 10% gain
        assert returns[1] == pytest.approx(-0.0454545, rel=1e-3)  # ~4.5% loss

    def test_per_ticker_report_agreement(self):
        """PerTickerReport should calculate agent agreement."""
        # Create signals to populate the report properly
        signals = [
            AgentSignal(agent_id="a1", agent_name="A1", agent_type=AgentType.FUNDAMENTALS, symbol="AAPL", signal=Signal.BUY, confidence=0.8, reasoning="Bull"),
            AgentSignal(agent_id="a2", agent_name="A2", agent_type=AgentType.TECHNICALS, symbol="AAPL", signal=Signal.BUY, confidence=0.7, reasoning="Bull"),
            AgentSignal(agent_id="a3", agent_name="A3", agent_type=AgentType.VALUATION, symbol="AAPL", signal=Signal.BUY, confidence=0.75, reasoning="Bull"),
            AgentSignal(agent_id="a4", agent_name="A4", agent_type=AgentType.SENTIMENT, symbol="AAPL", signal=Signal.SELL, confidence=0.6, reasoning="Bear"),
            AgentSignal(agent_id="a5", agent_name="A5", agent_type=AgentType.RISK, symbol="AAPL", signal=Signal.HOLD, confidence=0.5, reasoning="Neutral"),
        ]
        report = PerTickerReport(
            symbol="AAPL",
            signals=signals,
            consensus_signal=Signal.BUY,
            consensus_confidence=0.7,
            bullish_count=3,
            bearish_count=1,
            neutral_count=1,
        )
        # Max count (3) / total signals (5) = 0.6
        assert report.agent_agreement == 0.6


# =============================================================================
# Agent Base Tests
# =============================================================================


class TestAgentBase:
    """Test agent base classes."""

    def test_custom_id_generation(self):
        """Test deterministic custom ID generation."""
        from app.hedge_fund.agents.base import AgentBase

        class DummyAgent(AgentBase):
            async def run(self, symbol, data, *, mode=LLMMode.REALTIME, run_id=None):
                pass

        agent = DummyAgent(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type=AgentType.FUNDAMENTALS,
        )
        
        custom_id = agent.create_custom_id("run123", "AAPL", "analysis")
        assert custom_id == "run123:AAPL:test_agent:analysis"


# =============================================================================
# Calculation Agent Tests
# =============================================================================


class TestFundamentalsAgent:
    """Test fundamentals agent."""

    @pytest.mark.asyncio
    async def test_fundamentals_agent_scoring(self):
        """Test fundamentals scoring logic."""
        from app.hedge_fund.agents.fundamentals import FundamentalsAgent

        agent = FundamentalsAgent()

        # Create test data with strong fundamentals
        fundamentals = Fundamentals(
            symbol="AAPL",
            name="Apple Inc.",
            roe=0.25,  # 25% - excellent
            profit_margin=0.20,  # 20% - excellent
            current_ratio=1.8,  # Good
            debt_to_equity=0.4,  # Low
            revenue_growth=0.15,  # 15% - strong
            gross_margin=0.40,
        )

        prices = PriceSeries(
            symbol="AAPL",
            prices=[
                PricePoint(date=date(2024, 1, 1), open=180, high=185, low=178, close=180, volume=50000000),
            ],
        )

        data = MarketData(
            symbol="AAPL",
            prices=prices,
            fundamentals=fundamentals,
        )

        signal = await agent.calculate("AAPL", data)

        assert signal.symbol == "AAPL"
        assert signal.agent_id == "fundamentals"
        assert signal.signal in [Signal.STRONG_BUY, Signal.BUY]
        assert signal.confidence > 0.5
        assert len(signal.key_factors) > 0

    @pytest.mark.asyncio
    async def test_fundamentals_agent_weak_company(self):
        """Test fundamentals agent with weak company."""
        from app.hedge_fund.agents.fundamentals import FundamentalsAgent

        agent = FundamentalsAgent()

        # Create test data with weak fundamentals
        fundamentals = Fundamentals(
            symbol="WEAK",
            name="Weak Co",
            roe=0.02,  # 2% - poor
            profit_margin=-0.05,  # Negative
            current_ratio=0.5,  # Poor liquidity
            debt_to_equity=3.0,  # High debt
            revenue_growth=-0.10,  # Declining
        )

        prices = PriceSeries(
            symbol="WEAK",
            prices=[
                PricePoint(date=date(2024, 1, 1), open=10, high=11, low=9, close=10, volume=1000000),
            ],
        )

        data = MarketData(
            symbol="WEAK",
            prices=prices,
            fundamentals=fundamentals,
        )

        signal = await agent.calculate("WEAK", data)

        assert signal.signal in [Signal.SELL, Signal.STRONG_SELL, Signal.HOLD]


class TestTechnicalsAgent:
    """Test technicals agent."""

    @pytest.mark.asyncio
    async def test_technicals_insufficient_data(self):
        """Test technicals agent with insufficient data."""
        from app.hedge_fund.agents.technicals import TechnicalsAgent

        agent = TechnicalsAgent()

        # Only 5 price points - not enough for indicators
        prices = PriceSeries(
            symbol="AAPL",
            prices=[
                PricePoint(date=date(2024, 1, i), open=100+i, high=102+i, low=99+i, close=101+i, volume=1000000)
                for i in range(1, 6)
            ],
        )

        data = MarketData(
            symbol="AAPL",
            prices=prices,
            fundamentals=Fundamentals(symbol="AAPL", name="Apple"),
        )

        signal = await agent.calculate("AAPL", data)

        assert signal.signal == Signal.HOLD
        assert signal.confidence < 0.5
        assert "insufficient" in signal.reasoning.lower() or "need" in signal.reasoning.lower()


class TestValuationAgent:
    """Test valuation agent."""

    @pytest.mark.asyncio
    async def test_valuation_undervalued(self):
        """Test valuation agent with undervalued stock."""
        from app.hedge_fund.agents.valuation import ValuationAgent

        agent = ValuationAgent()

        fundamentals = Fundamentals(
            symbol="VALUE",
            name="Value Co",
            pe_ratio=8.0,  # Low P/E
            peg_ratio=0.5,  # Very low PEG
            price_to_book=0.8,  # Below book value
            free_cash_flow=1000000000,
            shares_outstanding=100000000,
            market_cap=5000000000,
            earnings_growth=0.15,
        )

        prices = PriceSeries(
            symbol="VALUE",
            prices=[
                PricePoint(date=date(2024, 1, 1), open=50, high=52, low=49, close=50, volume=5000000),
            ],
        )

        data = MarketData(
            symbol="VALUE",
            prices=prices,
            fundamentals=fundamentals,
        )

        signal = await agent.calculate("VALUE", data)

        assert signal.signal in [Signal.BUY, Signal.STRONG_BUY]


# =============================================================================
# Portfolio Manager Tests
# =============================================================================


class TestPortfolioManager:
    """Test portfolio manager aggregation."""

    def test_aggregate_signals_consensus(self):
        """Test signal aggregation."""
        from app.hedge_fund.agents.portfolio_manager import PortfolioManager

        pm = PortfolioManager()

        signals = [
            AgentSignal(
                agent_id="a1",
                agent_name="Agent 1",
                agent_type=AgentType.FUNDAMENTALS,
                symbol="AAPL",
                signal=Signal.BUY,
                confidence=0.8,
                reasoning="Bullish",
            ),
            AgentSignal(
                agent_id="a2",
                agent_name="Agent 2",
                agent_type=AgentType.TECHNICALS,
                symbol="AAPL",
                signal=Signal.BUY,
                confidence=0.7,
                reasoning="Also bullish",
            ),
            AgentSignal(
                agent_id="a3",
                agent_name="Agent 3",
                agent_type=AgentType.SENTIMENT,
                symbol="AAPL",
                signal=Signal.HOLD,
                confidence=0.5,
                reasoning="Neutral",
            ),
        ]

        report = pm.aggregate_signals(signals)

        assert report.symbol == "AAPL"
        assert report.consensus_signal == Signal.BUY
        assert report.bullish_count == 2
        assert report.neutral_count == 1
        assert report.bearish_count == 0

    def test_portfolio_decision_creation(self):
        """Test portfolio decision creation."""
        from app.hedge_fund.agents.portfolio_manager import PortfolioManager

        pm = PortfolioManager(max_allocation_per_stock=0.10)

        report = PerTickerReport(
            symbol="AAPL",
            signals=[],
            consensus_signal=Signal.BUY,
            consensus_confidence=0.8,
            bullish_count=4,
            bearish_count=1,
            neutral_count=0,
        )

        decision = pm.create_portfolio_decision(report, risk_score=0.3)

        assert decision.symbol == "AAPL"
        assert decision.action == Signal.BUY
        assert 0 < decision.allocation_pct <= 0.10
        assert decision.stop_loss_pct > 0


# =============================================================================
# LLM Gateway Tests
# =============================================================================


class TestLLMGateway:
    """Test LLM gateway."""

    @pytest.mark.asyncio
    async def test_gateway_task_creation(self):
        """Test LLM task creation."""
        from app.hedge_fund.llm.gateway import OpenAIGateway
        from app.hedge_fund.schemas import LLMTask

        gateway = OpenAIGateway(model="gpt-5-mini")

        task = LLMTask(
            custom_id="run1:AAPL:buffett:analysis",
            agent_id="buffett",
            symbol="AAPL",
            prompt="Analyze this stock",
            context={"system_prompt": "You are Warren Buffett"},
            require_json=True,
        )

        assert task.custom_id == "run1:AAPL:buffett:analysis"
        assert task.require_json is True


# =============================================================================
# Orchestrator Tests
# =============================================================================


class TestOrchestrator:
    """Test orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_analysis_request(self):
        """Test analysis request creation."""
        request = AnalysisRequest(
            tickers=[
                TickerInput(symbol="AAPL"),
                TickerInput(symbol="MSFT"),
            ],
            run_id="test123",
            mode=LLMMode.REALTIME,
            personas=["warren_buffett", "peter_lynch"],
        )

        assert len(request.tickers) == 2
        assert request.run_id == "test123"
        assert request.mode == LLMMode.REALTIME
        assert "warren_buffett" in request.personas


# =============================================================================
# Integration Tests (with mocks)
# =============================================================================


class TestIntegration:
    """Integration tests with mocked external dependencies."""

    @pytest.mark.asyncio
    async def test_quick_signal(self):
        """Test quick signal function with mocked data."""
        from app.hedge_fund.orchestrator import get_quick_signal
        from app.hedge_fund import data as data_module

        # Mock market data with proper date generation
        base_date = date(2024, 1, 1)
        mock_data = MarketData(
            symbol="AAPL",
            prices=PriceSeries(
                symbol="AAPL",
                prices=[
                    PricePoint(
                        date=base_date + timedelta(days=i),
                        open=180 + i,
                        high=185 + i,
                        low=178 + i,
                        close=182 + i,
                        volume=50000000,
                    )
                    for i in range(99)  # 99 days of data
                ],
            ),
            fundamentals=Fundamentals(
                symbol="AAPL",
                name="Apple Inc.",
                roe=0.20,
                profit_margin=0.25,
                pe_ratio=28,
                peg_ratio=1.5,
                current_ratio=1.5,
                debt_to_equity=0.5,
                revenue_growth=0.08,
            ),
        )

        with patch.object(data_module, "get_market_data", return_value=mock_data):
            signal, confidence = await get_quick_signal("AAPL")

            assert signal in Signal
            assert 0 <= confidence <= 1


# =============================================================================
# Persona Agent Tests
# =============================================================================


class TestInvestorPersona:
    """Test investor persona agents."""

    def test_persona_definitions(self):
        """Test all personas are properly defined."""
        from app.hedge_fund.agents.investor_persona import PERSONAS

        assert "warren_buffett" in PERSONAS
        assert "peter_lynch" in PERSONAS
        assert "cathie_wood" in PERSONAS
        assert "michael_burry" in PERSONAS
        assert "ben_graham" in PERSONAS
        assert "charlie_munger" in PERSONAS

        for persona_id, persona in PERSONAS.items():
            assert persona.id == persona_id
            assert len(persona.name) > 0
            assert len(persona.philosophy) > 0
            assert len(persona.system_prompt) > 0
            assert len(persona.focus_areas) > 0

    def test_persona_agent_creation(self):
        """Test persona agent creation."""
        from app.hedge_fund.agents.investor_persona import get_persona_agent, PERSONAS

        for persona_id in PERSONAS:
            agent = get_persona_agent(persona_id)
            assert agent is not None
            assert agent.persona.id == persona_id
            assert agent.requires_llm is True

    def test_unknown_persona(self):
        """Test that unknown persona returns None."""
        from app.hedge_fund.agents.investor_persona import get_persona_agent

        agent = get_persona_agent("unknown_investor")
        assert agent is None
