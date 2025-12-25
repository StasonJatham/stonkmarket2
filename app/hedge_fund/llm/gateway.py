"""
LLM Gateway for unified access to OpenAI realtime and batch APIs.

Wraps the existing openai_client.py to provide a clean interface for agents.
"""

import asyncio
import json
import logging
from typing import Any, Optional, Protocol

from app.hedge_fund.schemas import (
    BatchStatus,
    LLMMode,
    LLMResult,
    LLMTask,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Gateway Protocol
# =============================================================================


class LLMGatewayProtocol(Protocol):
    """Protocol for LLM gateway implementations."""

    async def run_realtime(self, task: LLMTask) -> LLMResult:
        """Execute a single task in realtime."""
        ...

    async def run_batch(
        self,
        tasks: list[LLMTask],
    ) -> str:
        """Submit tasks for batch processing. Returns batch_id."""
        ...

    async def check_batch_status(self, batch_id: str) -> BatchStatus:
        """Check status of a batch job."""
        ...

    async def collect_batch_results(self, batch_id: str) -> list[LLMResult]:
        """Collect results from a completed batch."""
        ...

    async def run(
        self,
        tasks: list[LLMTask],
        mode: LLMMode = LLMMode.REALTIME,
    ) -> list[LLMResult]:
        """
        Run tasks in the specified mode.
        
        For realtime: runs all concurrently and returns results.
        For batch: submits batch and polls until complete.
        """
        ...


# =============================================================================
# OpenAI Gateway Implementation
# =============================================================================


# Investment analysis schema for structured output
INVESTMENT_SIGNAL_SCHEMA = {
    "type": "json_schema",
    "strict": True,
    "name": "investment_signal",
    "schema": {
        "type": "object",
        "properties": {
            "signal": {
                "type": "string",
                "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
            },
            "confidence": {
                "type": "integer",
                "description": "Confidence level from 1-10",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed explanation of the investment thesis",
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Top 3-5 factors driving the recommendation",
            },
        },
        "required": ["signal", "confidence", "reasoning", "key_factors"],
        "additionalProperties": False,
    },
}


class OpenAIGateway:
    """
    LLM Gateway using OpenAI APIs.
    
    Supports both realtime (Responses API) and batch modes.
    Uses direct OpenAI client for custom persona analysis.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        default_temperature: float = 0.7,
        default_max_tokens: int = 1000,
        batch_poll_interval: float = 30.0,
        batch_max_wait: float = 3600.0,
    ):
        self.model = model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.batch_poll_interval = batch_poll_interval
        self.batch_max_wait = batch_max_wait

    async def run_realtime(self, task: LLMTask) -> LLMResult:
        """Execute a single task using the OpenAI Responses API directly."""
        import time
        from app.services.openai_client import _get_client

        try:
            client = await _get_client()
            if not client:
                return LLMResult(
                    custom_id=task.custom_id,
                    agent_id=task.agent_id,
                    symbol=task.symbol,
                    content="",
                    error="OpenAI client not configured - check OPENAI_API_KEY",
                    failed=True,
                )

            # Build messages
            system_prompt = task.context.get("system_prompt", "You are an expert investment analyst.")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task.prompt},
            ]
            
            start = time.monotonic()
            
            # Build request params
            params: dict = {
                "model": self.model,
                "messages": messages,
                "max_tokens": task.max_tokens or self.default_max_tokens,
            }
            
            # Add structured output for JSON responses
            if task.require_json:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "investment_signal",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "signal": {
                                    "type": "string",
                                    "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
                                },
                                "confidence": {
                                    "type": "integer",
                                    "description": "Confidence level from 1-10",
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Detailed explanation of the investment thesis",
                                },
                                "key_factors": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Top 3-5 factors driving the recommendation",
                                },
                            },
                            "required": ["signal", "confidence", "reasoning", "key_factors"],
                            "additionalProperties": False,
                        },
                    },
                }
            
            # Make the API call
            response = await client.chat.completions.create(**params)
            
            latency = (time.monotonic() - start) * 1000
            
            if not response.choices:
                return LLMResult(
                    custom_id=task.custom_id,
                    agent_id=task.agent_id,
                    symbol=task.symbol,
                    content="",
                    error="No response choices from OpenAI",
                    failed=True,
                    latency_ms=latency,
                )
            
            content = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            # Parse JSON if required
            parsed_json = None
            if task.require_json and content:
                try:
                    parsed_json = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
            
            return LLMResult(
                custom_id=task.custom_id,
                agent_id=task.agent_id,
                symbol=task.symbol,
                content=content,
                parsed_json=parsed_json,
                tokens_used=tokens_used,
                latency_ms=latency,
            )

        except Exception as e:
            logger.error(f"Realtime LLM error for {task.symbol}: {e}")
            return LLMResult(
                custom_id=task.custom_id,
                agent_id=task.agent_id,
                symbol=task.symbol,
                content="",
                error=str(e),
                failed=True,
            )

    async def run_batch(self, tasks: list[LLMTask]) -> str:
        """Submit tasks for batch processing."""
        from app.services import openai_client

        # Convert tasks to openai_client format
        items = []
        for task in tasks:
            items.append({
                "symbol": task.symbol,
                "custom_id": task.custom_id,
                "prompt": task.prompt,
                "system": task.context.get("system_prompt", ""),
                "agent_name": task.context.get("agent_name", "analyst"),
            })

        # Submit batch
        batch_id = await openai_client.submit_batch(
            task="rating",  # Use rating task for structured output
            items=items,
            model=self.model,
        )

        if not batch_id:
            raise RuntimeError("Failed to submit batch")

        return batch_id

    async def check_batch_status(self, batch_id: str) -> BatchStatus:
        """Check status of a batch job."""
        from app.services import openai_client
        from datetime import datetime

        status = await openai_client.check_batch(batch_id)
        if not status:
            raise RuntimeError(f"Failed to check batch {batch_id}")

        return BatchStatus(
            batch_id=batch_id,
            status=status["status"],
            total_count=status["total_count"],
            completed_count=status["completed_count"],
            failed_count=status["failed_count"],
            created_at=datetime.fromtimestamp(status["created_at"]) if isinstance(status["created_at"], (int, float)) else status["created_at"],
            completed_at=datetime.fromtimestamp(status["completed_at"]) if status.get("completed_at") and isinstance(status["completed_at"], (int, float)) else status.get("completed_at"),
            output_file_id=status.get("output_file_id"),
        )

    async def collect_batch_results(self, batch_id: str) -> list[LLMResult]:
        """Collect results from a completed batch."""
        from app.services import openai_client

        results = await openai_client.collect_batch(batch_id)
        if results is None:
            return []

        llm_results = []
        for item in results:
            # Parse custom_id to extract agent_id
            custom_id = item.get("custom_id", "")
            parts = custom_id.split(":")
            agent_id = parts[2] if len(parts) >= 3 else ""
            symbol = parts[1] if len(parts) >= 2 else item.get("symbol", "")

            llm_results.append(LLMResult(
                custom_id=custom_id,
                agent_id=agent_id,
                symbol=symbol,
                content=json.dumps(item.get("result", {})) if isinstance(item.get("result"), dict) else str(item.get("result", "")),
                parsed_json=item.get("result") if isinstance(item.get("result"), dict) else None,
                error=item.get("error"),
                failed=item.get("failed", False),
            ))

        return llm_results

    async def run(
        self,
        tasks: list[LLMTask],
        mode: LLMMode = LLMMode.REALTIME,
    ) -> list[LLMResult]:
        """
        Run tasks in the specified mode.
        
        For realtime: runs all concurrently and returns results.
        For batch: submits batch and polls until complete.
        """
        if not tasks:
            return []

        if mode == LLMMode.REALTIME:
            # Run all tasks concurrently
            results = await asyncio.gather(
                *[self.run_realtime(task) for task in tasks],
                return_exceptions=True,
            )
            
            # Convert exceptions to error results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(LLMResult(
                        custom_id=tasks[i].custom_id,
                        agent_id=tasks[i].agent_id,
                        symbol=tasks[i].symbol,
                        content="",
                        error=str(result),
                        failed=True,
                    ))
                else:
                    final_results.append(result)
            
            return final_results

        else:
            # Batch mode
            batch_id = await self.run_batch(tasks)
            
            # Poll for completion
            elapsed = 0.0
            while elapsed < self.batch_max_wait:
                status = await self.check_batch_status(batch_id)
                
                if status.status == "completed":
                    return await self.collect_batch_results(batch_id)
                elif status.status in ("failed", "cancelled", "expired"):
                    raise RuntimeError(f"Batch {batch_id} {status.status}")
                
                await asyncio.sleep(self.batch_poll_interval)
                elapsed += self.batch_poll_interval
            
            raise TimeoutError(f"Batch {batch_id} did not complete within {self.batch_max_wait}s")


# =============================================================================
# Helper Functions
# =============================================================================


async def get_investment_analysis(
    symbol: str,
    data_context: str,
    agent_name: str,
    system_prompt: str,
    gateway: Optional[OpenAIGateway] = None,
) -> LLMResult:
    """
    Convenience function to get investment analysis for a single symbol.
    
    Uses realtime mode by default.
    """
    gw = gateway or OpenAIGateway()
    
    task = LLMTask(
        custom_id=f"realtime:{symbol}:{agent_name}:analysis",
        agent_id=agent_name,
        symbol=symbol,
        prompt=data_context,
        context={
            "system_prompt": system_prompt,
            "agent_name": agent_name,
        },
        require_json=True,
    )
    
    return await gw.run_realtime(task)


# Singleton gateway instance
_gateway: Optional[OpenAIGateway] = None


def get_gateway() -> OpenAIGateway:
    """Get the singleton gateway instance."""
    global _gateway
    if _gateway is None:
        _gateway = OpenAIGateway()
    return _gateway


def set_gateway(gateway: OpenAIGateway) -> None:
    """Set the gateway instance (for testing)."""
    global _gateway
    _gateway = gateway
