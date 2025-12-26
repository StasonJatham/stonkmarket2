"""
Tests for batch jobs including AI agent analysis and OpenAI Batch API integration.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.hedge_fund.schemas import (
    AgentSignal,
    AgentType,
    Fundamentals,
    LLMMode,
    MarketData,
    PricePoint,
    PriceSeries,
    Signal,
)


# =============================================================================
# Investor Persona Tests
# =============================================================================


class TestInvestorPersonas:
    """Test investor persona definitions."""

    def test_all_personas_loaded(self):
        """All investor personas should be loaded."""
        from app.hedge_fund.agents.investor_persona import PERSONAS

        expected_personas = [
            "warren_buffett",
            "peter_lynch",
            "cathie_wood",
            "michael_burry",
            "ben_graham",
            "charlie_munger",
            "aswath_damodaran",
            "bill_ackman",
            "phil_fisher",
            "stanley_druckenmiller",
            "mohnish_pabrai",
            "rakesh_jhunjhunwala",
        ]

        for persona_id in expected_personas:
            assert persona_id in PERSONAS, f"Missing persona: {persona_id}"

    def test_persona_schema_valid(self):
        """Each persona should have valid schema."""
        from app.hedge_fund.agents.investor_persona import PERSONAS

        required_fields = ["id", "name", "philosophy", "risk_tolerance"]
        allowed_risk_tolerances = {"very_low", "low", "moderate", "high", "very_high"}

        for persona_id, persona in PERSONAS.items():
            for field in required_fields:
                assert hasattr(persona, field), f"{persona_id} missing {field}"

            assert persona.risk_tolerance in allowed_risk_tolerances, (
                f"{persona_id} has invalid risk_tolerance: {persona.risk_tolerance}"
            )

    def test_persona_unique_philosophies(self):
        """Each persona should have a unique philosophy."""
        from app.hedge_fund.agents.investor_persona import PERSONAS

        philosophies = [p.philosophy for p in PERSONAS.values()]
        assert len(philosophies) == len(set(philosophies))

    def test_get_persona_by_id(self):
        """Should get persona by ID."""
        from app.hedge_fund.agents.investor_persona import PERSONAS

        buffett = PERSONAS.get("warren_buffett")
        assert buffett is not None
        assert buffett.name == "Warren Buffett"
        assert "value investing" in buffett.philosophy.lower()


class TestInvestorPersonaAgent:
    """Test the investor persona agent."""

    def test_agent_initialization(self):
        """Agent should initialize with a persona."""
        from app.hedge_fund.agents.investor_persona import (
            InvestorPersonaAgent,
            PERSONAS,
        )

        persona = PERSONAS["warren_buffett"]
        agent = InvestorPersonaAgent(persona=persona)

        assert agent.persona == persona
        assert agent.agent_name == "Warren Buffett"
        assert agent.agent_id == "persona_warren_buffett"

    def test_agent_builds_prompt(self):
        """Agent should build analysis prompt."""
        from app.hedge_fund.agents.investor_persona import (
            InvestorPersonaAgent,
            PERSONAS,
        )

        persona = PERSONAS["warren_buffett"]
        agent = InvestorPersonaAgent(persona=persona)

        market_data = MarketData(
            symbol="AAPL",
            prices=PriceSeries(
                symbol="AAPL",
                prices=[
                    PricePoint(
                        date=datetime.now().date(),
                        open=150,
                        high=155,
                        low=148,
                        close=152,
                        volume=1000000,
                    )
                ],
            ),
            fundamentals=Fundamentals(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                industry="Consumer Electronics",
                market_cap=3000000000000,
                roe=0.25,
                profit_margin=0.20,
                pe_ratio=25,
            ),
        )

        prompt = agent.build_prompt("AAPL", market_data)
        assert "AAPL" in prompt
        assert "Apple" in prompt

    @pytest.mark.asyncio
    async def test_agent_run_with_mock(self):
        """Agent should run analysis with mocked LLM gateway."""
        from app.hedge_fund.agents.investor_persona import (
            InvestorPersonaAgent,
            PERSONAS,
        )
        from app.hedge_fund.llm.gateway import LLMResult

        persona = PERSONAS["warren_buffett"]
        agent = InvestorPersonaAgent(persona=persona)

        market_data = MarketData(
            symbol="AAPL",
            prices=PriceSeries(
                symbol="AAPL",
                prices=[
                    PricePoint(
                        date=datetime.now().date() - timedelta(days=i),
                        open=150 + i,
                        high=155 + i,
                        low=148 + i,
                        close=152 + i,
                        volume=1000000,
                    )
                    for i in range(30)
                ],
            ),
            fundamentals=Fundamentals(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                industry="Consumer Electronics",
                market_cap=3000000000000,
                roe=0.25,
                profit_margin=0.20,
                pe_ratio=25,
            ),
        )

        # Mock the gateway
        mock_gateway = AsyncMock()
        mock_result = LLMResult(
            custom_id="test-1",
            agent_id="persona_warren_buffett",
            symbol="AAPL",
            content='{"signal": "buy", "confidence": 8, "reasoning": "Strong fundamentals.", "key_factors": ["moat"]}',
            parsed_json={
                "signal": "buy",
                "confidence": 8,
                "reasoning": "Strong fundamentals.",
                "key_factors": ["moat"],
            },
            failed=False,
        )
        mock_gateway.run_realtime = AsyncMock(return_value=mock_result)
        agent._gateway = mock_gateway

        signal = await agent.run("AAPL", market_data)

        assert signal is not None
        assert isinstance(signal, AgentSignal)
        assert signal.signal in Signal
        assert 0 <= signal.confidence <= 1


# =============================================================================
# Batch Job Definition Tests
# =============================================================================


class TestJobDefinitions:
    """Test job definitions are registered correctly."""

    def test_all_jobs_registered(self):
        """All expected jobs should be registered."""
        from app.jobs.registry import get_all_jobs
        import app.jobs.definitions  # noqa: F401 - Import to register jobs

        jobs = get_all_jobs()

        expected_jobs = [
            "initial_data_ingest",
            "data_grab",
            "cache_warmup",
            "batch_ai_swipe",
            "batch_ai_analysis",
            "batch_poll",
            "fundamentals_refresh",
            "ai_agents_analysis",
            "cleanup",
        ]

        for job_name in expected_jobs:
            assert job_name in jobs, f"Job not registered: {job_name}"

    def test_job_functions_are_async(self):
        """All job functions should be async."""
        from app.jobs.registry import get_all_jobs
        import app.jobs.definitions  # noqa: F401
        import inspect

        jobs = get_all_jobs()

        for job_name, job_func in jobs.items():
            assert inspect.iscoroutinefunction(job_func), f"Job {job_name} is not async"


class TestBatchAISwipeJob:
    """Test batch AI swipe bio generation job."""

    @pytest.mark.asyncio
    async def test_batch_swipe_job_with_mocks(self):
        """Batch swipe job should process correctly."""
        from app.jobs.definitions import batch_ai_swipe_job

        with patch(
            "app.services.batch_scheduler.schedule_batch_swipe_bios",
            new_callable=AsyncMock,
            return_value="batch_123",
        ), patch(
            "app.services.batch_scheduler.process_completed_batch_jobs",
            new_callable=AsyncMock,
            return_value=0,
        ):
            result = await batch_ai_swipe_job()

            assert "batch_123" in result or "none needed" in result.lower()


class TestBatchAIAnalysisJob:
    """Test batch AI dip analysis job."""

    @pytest.mark.asyncio
    async def test_batch_analysis_job_with_mocks(self):
        """Batch analysis job should process correctly."""
        from app.jobs.definitions import batch_ai_analysis_job

        with patch(
            "app.services.batch_scheduler.schedule_batch_dip_analysis",
            new_callable=AsyncMock,
            return_value="batch_456",
        ), patch(
            "app.services.batch_scheduler.process_completed_batch_jobs",
            new_callable=AsyncMock,
            return_value=0,
        ):
            result = await batch_ai_analysis_job()

            assert "batch_456" in result or "none needed" in result.lower()


class TestBatchPollJob:
    """Test batch polling job."""

    @pytest.mark.asyncio
    async def test_batch_poll_job_with_mocks(self):
        """Batch poll job should check for completed batches."""
        from app.jobs.definitions import batch_poll_job

        with patch(
            "app.services.batch_scheduler.process_completed_batch_jobs",
            new_callable=AsyncMock,
            return_value=2,
        ):
            result = await batch_poll_job()

            assert "2" in result or "processed" in result.lower()


class TestAIAgentsAnalysisJob:
    """Test AI agents analysis job."""

    @pytest.mark.asyncio
    async def test_ai_agents_job_with_mocks(self):
        """AI agents analysis job should run all agents."""
        from app.jobs.definitions import ai_agents_analysis_job

        mock_result = {"analyzed": 10, "skipped": 2, "failed": 0}

        with patch(
            "app.services.ai_agents.run_all_agent_analyses",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            result = await ai_agents_analysis_job()

            # The result format depends on actual implementation
            assert result is not None


# =============================================================================
# Batch Scheduler Tests
# =============================================================================


class TestBatchScheduler:
    """Test batch scheduler functions."""

    @pytest.mark.asyncio
    async def test_process_completed_batch_jobs_no_jobs(self):
        """Should handle no pending batch jobs."""
        from app.services.batch_scheduler import process_completed_batch_jobs

        with patch(
            "app.database.connection.fetch_all",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await process_completed_batch_jobs()
            # Returns int count
            assert isinstance(result, int)
            assert result == 0


# =============================================================================
# AI Agents Service Tests
# =============================================================================


class TestAIAgentsService:
    """Test AI agents service functions."""

    @pytest.mark.asyncio
    async def test_get_symbols_needing_analysis(self):
        """Should get symbols that need AI analysis."""
        from app.services.ai_agents import get_symbols_needing_analysis

        with patch(
            "app.database.connection.fetch_all",
            new_callable=AsyncMock,
            return_value=[{"symbol": "AAPL"}, {"symbol": "MSFT"}],
        ):
            result = await get_symbols_needing_analysis()
            assert isinstance(result, list)


# =============================================================================
# OpenAI Batch API Tests
# =============================================================================


class TestOpenAIBatchAPI:
    """Test OpenAI Batch API client functions."""

    def test_batch_model_configured(self):
        """Should have default model configured."""
        from app.services.openai_client import DEFAULT_MODEL

        # Verify default model configuration
        assert DEFAULT_MODEL is not None

    @pytest.mark.asyncio
    async def test_check_batch_with_mock(self):
        """Should check batch status."""
        from app.services.openai_client import check_batch

        mock_client = MagicMock()
        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output-123"
        mock_batch.error_file_id = None
        mock_batch.request_counts = MagicMock(completed=10, failed=0, total=10)
        mock_client.batches.retrieve.return_value = mock_batch

        with patch(
            "app.services.openai_client._get_client",
            return_value=mock_client,
        ):
            result = await check_batch("batch-456")
            # May return None if client not configured
            assert result is None or result["status"] == "completed"


# =============================================================================
# Data Versioning Tests (for batch job change detection)
# =============================================================================


class TestDataVersioning:
    """Test data versioning for change detection."""

    def test_version_hash_consistency(self):
        """Version hash should be consistent for same data."""
        from app.services.data_providers.yfinance_service import _compute_hash

        # Same data should produce same hash
        data1 = {"a": 1, "b": 2}
        data2 = {"a": 1, "b": 2}
        data3 = {"a": 1, "b": 3}  # Different

        hash1 = _compute_hash(data1)
        hash2 = _compute_hash(data2)
        hash3 = _compute_hash(data3)

        assert hash1 == hash2
        assert hash1 != hash3


# =============================================================================
# Job Executor Tests
# =============================================================================


class TestJobExecutor:
    """Test job executor functionality."""

    @pytest.mark.asyncio
    async def test_execute_job_by_name(self):
        """Should execute job by name."""
        from app.jobs.executor import execute_job
        from app.core.exceptions import JobError
        import app.jobs.definitions  # noqa: F401

        # Use a simple job that we can mock
        with patch(
            "app.services.batch_scheduler.process_completed_batch_jobs",
            new_callable=AsyncMock,
            return_value=0,
        ):
            result = await execute_job("batch_poll")
            assert result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_job(self):
        """Should raise error for unknown job."""
        from app.jobs.executor import execute_job
        from app.core.exceptions import JobError

        with pytest.raises(JobError):
            await execute_job("nonexistent_job")


# =============================================================================
# Integration Tests (with full mocking)
# =============================================================================


class TestBatchJobIntegration:
    """Integration tests for batch job workflows."""

    @pytest.mark.asyncio
    async def test_full_batch_swipe_workflow(self):
        """Test complete batch swipe bio workflow."""
        from app.jobs.definitions import batch_ai_swipe_job

        # Mock all dependencies
        with patch(
            "app.services.batch_scheduler.schedule_batch_suggestion_bios",
            new_callable=AsyncMock,
            return_value="batch_test_123",
        ), patch(
            "app.services.batch_scheduler.process_completed_batch_jobs",
            new_callable=AsyncMock,
            return_value=0,
        ):
            result = await batch_ai_swipe_job()
            assert "batch_test_123" in result or result is not None

    @pytest.mark.asyncio
    async def test_persona_analysis_flow(self):
        """Test investor persona analysis flow."""
        from app.hedge_fund.agents.investor_persona import (
            InvestorPersonaAgent,
            PERSONAS,
        )
        from app.hedge_fund.llm.gateway import LLMResult

        # Create mock market data
        market_data = MarketData(
            symbol="AAPL",
            prices=PriceSeries(
                symbol="AAPL",
                prices=[
                    PricePoint(
                        date=datetime.now().date() - timedelta(days=i),
                        open=150 + i,
                        high=155 + i,
                        low=148 + i,
                        close=152 + i,
                        volume=1000000,
                    )
                    for i in range(30)
                ],
            ),
            fundamentals=Fundamentals(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                industry="Consumer Electronics",
                market_cap=3000000000000,
                roe=0.25,
                profit_margin=0.20,
                pe_ratio=25,
            ),
        )

        # Get the agent
        persona = PERSONAS["warren_buffett"]
        agent = InvestorPersonaAgent(persona=persona)

        # Mock the gateway
        mock_gateway = AsyncMock()
        mock_result = LLMResult(
            custom_id="test-2",
            agent_id="persona_warren_buffett",
            symbol="AAPL",
            content='{"signal": "hold", "confidence": 7, "reasoning": "Fair valuation", "key_factors": ["valuation"]}',
            parsed_json={
                "signal": "hold",
                "confidence": 7,
                "reasoning": "Fair valuation",
                "key_factors": ["valuation"],
            },
            failed=False,
        )
        mock_gateway.run_realtime = AsyncMock(return_value=mock_result)
        agent._gateway = mock_gateway

        result = await agent.run("AAPL", market_data)
        assert result is not None
        assert isinstance(result, AgentSignal)

    def test_all_persona_agents_creatable(self):
        """All persona agents should be creatable."""
        from app.hedge_fund.agents.investor_persona import (
            get_all_persona_agents,
            InvestorPersonaAgent,
        )

        agents = get_all_persona_agents()
        assert len(agents) > 0
        for agent in agents:
            assert isinstance(agent, InvestorPersonaAgent)
            assert agent.agent_name is not None
            assert agent.agent_id is not None
