"""
Base agent protocol and abstract class.

All agents in the hedge fund module implement this interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from app.hedge_fund.schemas import (
    AgentSignal,
    AgentType,
    LLMMode,
    LLMTask,
    MarketData,
)


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol defining the agent interface."""

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent."""
        ...

    @property
    def agent_name(self) -> str:
        """Human-readable name."""
        ...

    @property
    def agent_type(self) -> AgentType:
        """Type of agent."""
        ...

    @property
    def requires_llm(self) -> bool:
        """Whether this agent needs LLM for processing."""
        ...

    async def run(
        self,
        symbol: str,
        data: MarketData,
        *,
        mode: LLMMode = LLMMode.REALTIME,
        run_id: str | None = None,
    ) -> AgentSignal:
        """
        Run analysis and return a signal.
        
        Args:
            symbol: Ticker symbol
            data: Market data for the symbol
            mode: LLM execution mode (realtime or batch)
            run_id: Optional run ID for tracking
            
        Returns:
            AgentSignal with the analysis result
        """
        ...


class AgentBase(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality and enforces the agent interface.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: AgentType,
        requires_llm: bool = False,
    ):
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_type = agent_type
        self._requires_llm = requires_llm

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_name(self) -> str:
        return self._agent_name

    @property
    def agent_type(self) -> AgentType:
        return self._agent_type

    @property
    def requires_llm(self) -> bool:
        return self._requires_llm

    @abstractmethod
    async def run(
        self,
        symbol: str,
        data: MarketData,
        *,
        mode: LLMMode = LLMMode.REALTIME,
        run_id: str | None = None,
    ) -> AgentSignal:
        """Run analysis and return a signal."""
        pass

    def create_custom_id(
        self,
        run_id: str,
        symbol: str,
        task_name: str = "analysis",
    ) -> str:
        """
        Create deterministic custom ID for batch tracking.
        
        Format: {run_id}:{ticker}:{agent_name}:{task_name}
        """
        return f"{run_id}:{symbol}:{self.agent_id}:{task_name}"

    def _build_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        reasoning: str,
        key_factors: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> AgentSignal:
        """Helper to build a properly formatted AgentSignal."""
        from app.hedge_fund.schemas import Signal

        return AgentSignal(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            symbol=symbol,
            signal=Signal(signal),
            confidence=confidence,
            reasoning=reasoning,
            key_factors=key_factors or [],
            metrics=metrics,
        )


class LLMAgentBase(AgentBase):
    """
    Base class for agents that require LLM calls.
    
    Provides common LLM interaction patterns and prompt building.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: AgentType,
        system_prompt: str,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            requires_llm=True,
        )
        self._system_prompt = system_prompt
        self._llm_gateway: Any | None = None

    def set_llm_gateway(self, gateway: Any) -> None:
        """Set the LLM gateway for this agent."""
        self._llm_gateway = gateway

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @abstractmethod
    def build_prompt(self, symbol: str, data: MarketData) -> str:
        """Build the analysis prompt for the LLM."""
        pass

    def create_llm_task(
        self,
        symbol: str,
        data: MarketData,
        run_id: str,
        require_json: bool = True,
    ) -> LLMTask:
        """Create an LLM task for batch processing."""
        return LLMTask(
            custom_id=self.create_custom_id(run_id, symbol, "analysis"),
            agent_id=self.agent_id,
            symbol=symbol,
            prompt=self.build_prompt(symbol, data),
            context={
                "system_prompt": self.system_prompt,
                "agent_name": self.agent_name,
            },
            require_json=require_json,
        )


class CalculationAgentBase(AgentBase):
    """
    Base class for agents that perform pure calculations without LLM.
    
    These agents analyze data using deterministic algorithms.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        agent_type: AgentType,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            agent_type=agent_type,
            requires_llm=False,
        )

    async def run(
        self,
        symbol: str,
        data: MarketData,
        *,
        mode: LLMMode = LLMMode.REALTIME,
        run_id: str | None = None,
    ) -> AgentSignal:
        """Run calculation-based analysis."""
        # Calculation agents ignore mode - they always run synchronously
        return await self.calculate(symbol, data)

    @abstractmethod
    async def calculate(self, symbol: str, data: MarketData) -> AgentSignal:
        """Perform the calculation and return a signal."""
        pass
