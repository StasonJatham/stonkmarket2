"""
Core AI generation functions.

Provides the main generate() function with:
- Type-safe structured outputs
- Automatic retry with exponential backoff
- Token budget calculation (using tiktoken)
- Telemetry and usage tracking
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, TypeVar, overload

import tiktoken
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.logging import get_logger
from app.services.openai.client import get_client, get_client_manager
from app.services.openai.config import (
    TaskType,
    get_model_limits,
    get_settings,
    get_task_config,
    is_reasoning_model,
)
from app.services.openai.contexts import TaskContext, context_to_dict
from app.services.openai.prompts import get_instructions
from app.services.openai.schemas import (
    AgentOutput,
    BioOutput,
    PortfolioOutput,
    RatingOutput,
    SummaryOutput,
    TASK_SCHEMAS,
    get_output_model,
)
from app.services.openai.validation import repair_output, validate_output

logger = get_logger("openai.generate")


# Type variable for output types
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# TOKEN COUNTING WITH TIKTOKEN
# =============================================================================


@lru_cache(maxsize=8)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """
    Get the tiktoken encoding for a model.
    
    Cached to avoid repeated initialization overhead.
    Falls back to cl100k_base for unknown models.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # GPT-5, o3, and newer models likely use cl100k_base or similar
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str | None = None) -> int:
    """
    Count exact tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for encoding (defaults to configured model)
    
    Returns:
        Exact token count
    """
    if not text:
        return 0
    
    model = model or get_settings().default_model
    encoding = _get_encoding(model)
    return len(encoding.encode(text))


def estimate_tokens(text: str, model: str | None = None) -> int:
    """
    Count tokens using tiktoken.
    
    This is the primary token counting function used throughout the module.
    Uses exact tiktoken counting for accuracy.
    """
    return count_tokens(text, model)


@dataclass
class UsageMetrics:
    """Metrics from an API call."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    duration_ms: int = 0
    model: str = ""
    task: str = ""
    
    @property
    def cost_usd(self) -> float:
        """Estimate cost in USD based on current pricing."""
        # Pricing per 1M tokens (approximate, varies by model)
        prices = {
            "gpt-5": {"input": 15.0, "output": 60.0},
            "gpt-5-mini": {"input": 0.15, "output": 0.60},
            "o3": {"input": 15.0, "output": 60.0},
            "o3-mini": {"input": 1.10, "output": 4.40},
            "gpt-4o": {"input": 2.50, "output": 10.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }
        
        # Find matching price tier
        model_prices = None
        for prefix, price in prices.items():
            if self.model.startswith(prefix):
                model_prices = price
                break
        
        if not model_prices:
            model_prices = {"input": 2.50, "output": 10.0}  # Default to GPT-4o pricing
        
        input_cost = (self.input_tokens / 1_000_000) * model_prices["input"]
        output_cost = (self.output_tokens / 1_000_000) * model_prices["output"]
        
        return input_cost + output_cost


def calculate_safe_output_tokens(
    model: str,
    instructions: str,
    prompt: str,
    desired_output: int,
    task: TaskType,
) -> tuple[int, bool]:
    """
    Calculate safe output token budget based on input size.
    
    Uses tiktoken for accurate token counting.
    
    Returns:
        Tuple of (safe_output_tokens, input_overflow)
        If input_overflow is True, the input is too large for the model.
    """
    limits = get_model_limits(model)
    config = get_task_config(task)
    
    # Count input tokens using tiktoken
    input_tokens = estimate_tokens(instructions + prompt, model)
    
    # Add reasoning overhead for GPT-5/o3 models
    if is_reasoning_model(model):
        input_tokens += config.reasoning_overhead
    
    # Calculate available tokens for output
    available = limits.context_window - input_tokens - limits.reserved_overhead
    
    if available <= 0:
        return 0, True  # Input overflow
    
    # Use minimum of desired and available, capped by model max
    safe_output = min(desired_output, available, limits.max_output)
    
    # Ensure we have at least some output budget
    if safe_output < 50:
        return 0, True  # Not enough room for meaningful output
    
    return safe_output, False


def build_prompt(task: TaskType, context: dict[str, Any]) -> str:
    """
    Build a task-specific prompt from context data.
    
    If context contains a "prompt" key, use that directly (for custom prompts).
    """
    # Support custom prompts (e.g., from AI agents)
    if "prompt" in context:
        return context["prompt"]
    
    parts: list[str] = []
    settings = get_settings()
    
    if task == TaskType.BIO:
        if symbol := context.get("symbol"):
            parts.append(f"Stock: {symbol}")
        if name := context.get("name"):
            parts.append(f"Company: {name}")
        if sector := context.get("sector"):
            parts.append(f"Sector: {sector}")
        if summary := context.get("summary"):
            parts.append(f"Business: {summary[:400]}")
        
        # Pass mood as boolean only, no specific numbers
        dip = context.get("dip_pct") or context.get("dip_percentage")
        if dip and dip > 5:
            parts.append("Mood: currently down")
    
    elif task == TaskType.SUMMARY:
        if symbol := context.get("symbol"):
            parts.append(f"Stock: {symbol}")
        if name := context.get("name"):
            parts.append(f"Company: {name}")
        if desc := context.get("description"):
            max_chars = settings.max_description_chars
            if len(desc) > max_chars:
                logger.warning(
                    f"Description for {context.get('symbol', 'unknown')} is {len(desc)} chars, "
                    f"trimming to {max_chars}"
                )
                desc = desc[:max_chars] + "..."
            parts.append(f"\nFull Description:\n{desc}")
    
    elif task == TaskType.RATING:
        if symbol := context.get("symbol"):
            parts.append(f"Stock: {symbol}")
        if name := context.get("name"):
            parts.append(f"Company: {name}")
        if sector := context.get("sector"):
            parts.append(f"Sector: {sector}")
        
        # Dip metrics
        if dip := context.get("dip_pct"):
            parts.append(f"Dip from High: {dip:.1f}%")
        if days := context.get("days_in_dip"):
            parts.append(f"Days in Dip: {days}")
        if dip_type := context.get("dip_type"):
            parts.append(f"Dip Type: {dip_type}")
        
        # Quality and stability
        if quality := context.get("quality_score"):
            parts.append(f"Quality Score: {quality:.0f}")
        if stability := context.get("stability_score"):
            parts.append(f"Stability Score: {stability:.0f}")
        
        # Valuation metrics
        valuation = []
        if pe := context.get("pe_ratio"):
            valuation.append(f"P/E: {pe:.1f}")
        if fpe := context.get("forward_pe"):
            valuation.append(f"Fwd P/E: {fpe:.1f}")
        if ev := context.get("ev_to_ebitda"):
            valuation.append(f"EV/EBITDA: {ev:.1f}")
        if valuation:
            parts.append(f"Valuation: {', '.join(valuation)}")
        
        if cap := context.get("market_cap"):
            if cap >= 1e12:
                parts.append(f"Market Cap: ${cap/1e12:.1f}T")
            elif cap >= 1e9:
                parts.append(f"Market Cap: ${cap/1e9:.1f}B")
            else:
                parts.append(f"Market Cap: ${cap/1e6:.0f}M")
    
    elif task == TaskType.PORTFOLIO:
        if name := context.get("portfolio_name"):
            parts.append(f"Portfolio: {name}")
        if value := context.get("total_value"):
            parts.append(f"Total Value: ${value:,.2f}")
        if gain := context.get("total_gain"):
            pct = context.get("total_gain_pct", 0)
            parts.append(f"Total Gain: ${gain:+,.2f} ({pct:+.1f}%)")
        
        # Performance metrics
        if perf := context.get("performance", {}):
            parts.append("\n## Performance Metrics")
            if cagr := perf.get("cagr"):
                parts.append(f"- CAGR: {cagr:.1%}")
            if sharpe := perf.get("sharpe"):
                parts.append(f"- Sharpe Ratio: {sharpe:.2f}")
            if sortino := perf.get("sortino"):
                parts.append(f"- Sortino Ratio: {sortino:.2f}")
            if vol := perf.get("volatility"):
                parts.append(f"- Volatility: {vol:.1%}")
            if mdd := perf.get("max_drawdown"):
                parts.append(f"- Max Drawdown: {mdd:.1%}")
            if beta := perf.get("beta"):
                parts.append(f"- Beta: {beta:.2f}")
        
        # Risk metrics
        if risk := context.get("risk", {}):
            parts.append("\n## Risk Analysis")
            if score := risk.get("risk_score"):
                parts.append(f"- Risk Score: {score}/10")
            if var95 := risk.get("var_95_daily"):
                parts.append(f"- Daily VaR (95%): {var95:.2%}")
            if cvar95 := risk.get("cvar_95_daily"):
                parts.append(f"- Daily CVaR (95%): {cvar95:.2%}")
            if eff_n := risk.get("effective_n"):
                parts.append(f"- Effective Diversification: {eff_n:.1f}")
            if top_risk := risk.get("top_risk_contributors"):
                parts.append("- Top Risk Contributors:")
                for symbol, contrib in top_risk.items():
                    parts.append(f"  - {symbol}: {contrib:.1%}")
        
        # Holdings
        if holdings := context.get("holdings", []):
            parts.append("\n## Holdings")
            for h in holdings:
                sym = h.get("symbol", "?")
                weight = h.get("weight", 0)
                gain_pct = h.get("gain_pct", 0)
                line = f"- {sym}: {weight:.1f}% weight, {gain_pct:+.1f}% gain"
                if sector := h.get("sector"):
                    line += f", {sector}"
                parts.append(line)
        
        # Sector allocation
        if sectors := context.get("sector_allocation", {}):
            parts.append("\n## Sector Allocation")
            for sector, weight in sorted(sectors.items(), key=lambda x: -x[1]):
                parts.append(f"- {sector}: {weight:.1f}%")
    
    return "\n".join(parts)


# Overloaded generate signatures for type safety
@overload
async def generate(
    task: TaskType,
    context: TaskContext | dict[str, Any],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    return_model: type[T],
) -> T | None: ...


@overload
async def generate(
    task: TaskType,
    context: TaskContext | dict[str, Any],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any] | None: ...


async def generate(
    task: TaskType,
    context: TaskContext | dict[str, Any],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    return_model: type[BaseModel] | None = None,
) -> dict[str, Any] | BaseModel | None:
    """
    Generate AI content using OpenAI's Responses API with structured outputs.
    
    All tasks use structured outputs (JSON Schema) for guaranteed schema compliance.
    
    Args:
        task: Type of generation task
        context: Task-specific context data (typed or dict)
        model: Override default model
        max_tokens: Override default max output tokens
        return_model: If provided, parse and return as Pydantic model
    
    Returns:
        Parsed JSON dict, or Pydantic model if return_model specified.
        None on failure.
    
    Example:
        >>> result = await generate(
        ...     task=TaskType.RATING,
        ...     context=RatingContext(symbol="AAPL", dip_pct=15.2),
        ...     return_model=RatingOutput,
        ... )
        >>> print(result.rating)
        RatingValue.BUY
    """
    settings = get_settings()
    model = model or settings.default_model
    
    # Normalize task if string
    if isinstance(task, str):
        task = TaskType(task)
    
    # Convert context to dict
    ctx_dict = context_to_dict(context)
    
    # Get instructions and build prompt
    instructions = get_instructions(task)
    prompt = build_prompt(task, ctx_dict)
    
    # Get task config for defaults
    config = get_task_config(task)
    if max_tokens is None:
        max_tokens = config.default_max_tokens
    
    # Calculate safe output tokens
    safe_output, overflow = calculate_safe_output_tokens(
        model=model,
        instructions=instructions,
        prompt=prompt,
        desired_output=max_tokens,
        task=task,
    )
    
    if overflow:
        logger.error(f"Input too large for {task.value}, cannot proceed")
        return None
    
    # Build request parameters
    params: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": safe_output,
        "store": False,
    }
    
    # Use minimal reasoning effort for simple tasks
    if is_reasoning_model(model):
        params["reasoning"] = {"effort": "minimal"}
    
    # All tasks use structured outputs
    params["text"] = {
        "verbosity": "low",
        "format": TASK_SCHEMAS[task],
    }
    
    # Execute with retry
    manager = await get_client_manager()
    client = await manager.get_client()
    
    if not client:
        return None
    
    start_time = datetime.now(UTC)
    
    try:
        # Retry with exponential backoff
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(settings.max_retries),
            wait=wait_exponential_jitter(
                initial=settings.retry_delay,
                max=settings.retry_max_delay,
                jitter=1.0,
            ),
            retry=retry_if_exception_type((
                Exception,  # Retry on any exception for now
            )),
            reraise=True,
        ):
            with attempt:
                response = await client.responses.create(**params)
                
                # Check response status
                if response.status not in ("completed", "incomplete"):
                    logger.warning(f"Unexpected response status: {response.status}")
                    raise RuntimeError(f"Bad response status: {response.status}")
                
                # Handle incomplete with token boost
                if response.status == "incomplete":
                    reason = getattr(response.incomplete_details, "reason", None) if response.incomplete_details else None
                    if reason == "max_output_tokens":
                        # Boost tokens and retry
                        current = params.get("max_output_tokens", safe_output)
                        params["max_output_tokens"] = min(current * 2, 2000)
                        logger.info(f"Boosting max_output_tokens to {params['max_output_tokens']}")
                        raise RuntimeError("Incomplete response, retrying with more tokens")
                
                # Get output
                output = response.output_text or ""
                if not output.strip():
                    raise RuntimeError("Empty output from API")
                
                # Parse JSON
                try:
                    result = json.loads(output)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {e}")
                    raise
                
                # Record success
                manager.record_success()
                
                # Record metrics
                duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)
                metrics = UsageMetrics(
                    input_tokens=getattr(response.usage, "input_tokens", 0) if response.usage else 0,
                    output_tokens=getattr(response.usage, "output_tokens", 0) if response.usage else 0,
                    total_tokens=getattr(response.usage, "total_tokens", 0) if response.usage else 0,
                    duration_ms=duration_ms,
                    model=model,
                    task=task.value,
                )
                
                # Log token usage for worker job visibility
                logger.info(
                    f"[{task.value.upper()}] {ctx_dict.get('symbol', 'unknown')} - "
                    f"{metrics.input_tokens} in / {metrics.output_tokens} out tokens, "
                    f"{metrics.duration_ms}ms, ${metrics.cost_usd:.4f}"
                )
                
                # Record usage if enabled
                if settings.record_usage:
                    await _record_usage(metrics, ctx_dict.get("symbol"))
                
                # Validate and optionally repair output
                is_valid, errors = validate_output(task, result)
                if not is_valid:
                    logger.warning(f"Validation errors for {task.value}: {errors}")
                    result = repair_output(task, result)
                
                # Return as model or dict
                if return_model:
                    return return_model.model_validate(result)
                return result
    
    except RetryError as e:
        logger.error(f"All retries exhausted for {task.value}: {e}")
        manager.record_failure()
        return None
    except Exception as e:
        logger.error(f"Generation failed for {task.value}: {e}")
        manager.record_failure()
        return None


async def _record_usage(metrics: UsageMetrics, symbol: str | None) -> None:
    """Record API usage to database."""
    try:
        from app.repositories import api_usage_orm
        
        await api_usage_orm.record_usage(
            endpoint="responses.create",
            method="POST",
            status_code=200,
            response_time_ms=metrics.duration_ms,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            model=metrics.model,
            cost_usd=metrics.cost_usd,
            symbol=symbol,
        )
    except Exception as e:
        logger.debug(f"Failed to record usage: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

async def generate_bio(
    symbol: str,
    name: str | None = None,
    sector: str | None = None,
    summary: str | None = None,
    dip_pct: float | None = None,
) -> str | None:
    """Generate a dating-app style bio for a stock."""
    result = await generate(
        task=TaskType.BIO,
        context={
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "summary": summary,
            "dip_pct": dip_pct,
        },
        max_tokens=150,
    )
    
    if isinstance(result, dict):
        return result.get("bio")
    return None


async def rate_dip(
    symbol: str,
    current_price: float | None = None,
    ref_high: float | None = None,
    dip_pct: float | None = None,
    **extra_context: Any,
) -> dict[str, Any] | None:
    """Rate a stock dip opportunity."""
    return await generate(
        task=TaskType.RATING,
        context={
            "symbol": symbol,
            "current_price": current_price,
            "ref_high": ref_high,
            "dip_pct": dip_pct,
            **extra_context,
        },
        max_tokens=250,
    )


async def summarize_company(
    symbol: str,
    name: str | None = None,
    description: str = "",
) -> str | None:
    """Summarize a company description to 300-400 characters."""
    if not description or len(description) < 50:
        return description
    
    result = await generate(
        task=TaskType.SUMMARY,
        context={
            "symbol": symbol,
            "name": name,
            "description": description,
        },
        max_tokens=250,
    )
    
    if isinstance(result, dict):
        summary = result.get("summary")
        if summary:
            # Apply truncation fallback if needed
            from app.services.openai.validation import truncate_at_sentence
            return truncate_at_sentence(summary, max_chars=500)
    return None
