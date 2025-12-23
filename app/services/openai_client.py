"""
Unified OpenAI client for all AI operations.

This module provides a clean abstraction over the OpenAI API:
- Centralized client configuration with retries and timeouts
- Responses API for real-time generation
- Batch API for bulk processing
- Cost tracking and telemetry

Usage Examples:

    # Real-time text generation
    from app.services.openai_client import ai
    
    result = await ai.generate(
        task="bio",
        context={"symbol": "AAPL", "name": "Apple Inc.", "dip_pct": 15.2}
    )
    
    # JSON structured output
    rating = await ai.generate(
        task="rating",
        context={"symbol": "AAPL", "dip_pct": 15.2},
        json_output=True
    )
    
    # Batch processing
    batch_id = await ai.submit_batch(
        task="bio",
        items=[{"symbol": "AAPL", ...}, {"symbol": "MSFT", ...}]
    )
    results = await ai.collect_batch(batch_id)
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

from openai import AsyncOpenAI, APITimeoutError, RateLimitError

from app.core.logging import get_logger
from app.repositories import api_keys as api_keys_repo
from app.repositories import api_usage as api_usage_repo
from app.services.text_cleaner import clean_ai_text

logger = get_logger("openai")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TIMEOUT = 60.0  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds, doubles on each retry

# GPT-5 models use reasoning tokens - need higher output limits
GPT5_PREFIXES = ("gpt-5",)


class TaskType(str, Enum):
    """Supported AI tasks."""
    BIO = "bio"              # Dating-app style stock bio
    DIP_BIO = "dip_bio"      # Bio emphasizing the dip
    RATING = "rating"        # Buy/hold/sell rating with reasoning
    SUMMARY = "summary"      # Company description summary


# =============================================================================
# Prompt Templates
# =============================================================================

INSTRUCTIONS: dict[TaskType, str] = {
    TaskType.BIO: """You write dating-app bios for stocks. The stock is the person looking for a match.

RULES:
- First person, BE THE STOCK's personality
- Match the company's vibe (tech = nerdy, retail = friendly, energy = rugged)
- Flirty, confident, maybe a little unhinged
- Make investors LAUGH then think "maybe I should buy this"
- Max 2-3 sentences, include 1-2 emojis
- NO investor jargon - this is a dating app not CNBC""",

    TaskType.DIP_BIO: """You write dating-app bios for stocks going through a dip.

RULES:
- First person, self-aware about being down
- Dramatic, funny, self-deprecating about the price drop
- "Looking for someone who sees my true value" energy
- Max 2-3 sentences, include 1-2 emojis
- Use dating/investing puns""",

    TaskType.RATING: """You are a decisive stock analyst rating dip buying opportunities.

Rate this dip opportunity with:
- rating: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
- reasoning: 2-3 sentence explanation with specific insight
- confidence: 1-10 (how sure you are)

RATING GUIDE:
- strong_buy: Dip >20% on quality company, rare opportunity
- buy: Dip 10-20% with solid fundamentals
- hold: Wait for more data or fair price
- sell: Red flags despite the dip
- strong_sell: Major problems, could get worse

BE DECISIVE - take a stance. Always respond with valid JSON.""",

    TaskType.SUMMARY: """You explain complex businesses in simple terms.

RULES:
- Maximum 400 characters
- Simple language anyone can understand
- Focus on what the company actually does
- No jargon or complex terms
- Two or three sentences max""",
}


def _build_prompt(task: TaskType, context: dict[str, Any]) -> str:
    """Build a prompt from task type and context data."""
    parts = []
    
    # Stock identification
    if symbol := context.get("symbol"):
        parts.append(f"Stock: {symbol}")
    if name := context.get("name"):
        parts.append(f"Company: {name}")
    if sector := context.get("sector"):
        parts.append(f"Sector: {sector}")
    
    # Business info
    if summary := context.get("summary"):
        parts.append(f"Business: {summary[:400]}")
    
    # Price data
    if price := context.get("current_price"):
        parts.append(f"Current Price: ${price:.2f}")
    if high := context.get("ref_high") or context.get("ath_price"):
        parts.append(f"Recent High: ${high:.2f}")
    if dip := context.get("dip_pct") or context.get("dip_percentage"):
        parts.append(f"Dip: -{dip:.1f}% from high")
    if days := context.get("days_below"):
        parts.append(f"Days in dip: {days}")
    
    # Fundamentals
    if pe := context.get("pe_ratio"):
        parts.append(f"P/E: {pe:.1f}")
    if cap := context.get("market_cap"):
        if cap > 1e12:
            parts.append(f"Market Cap: ${cap/1e12:.1f}T")
        elif cap > 1e9:
            parts.append(f"Market Cap: ${cap/1e9:.1f}B")
    
    # Full description for summary task
    if task == TaskType.SUMMARY and (desc := context.get("description")):
        parts.append(f"\nFull Description:\n{desc}")
    
    return "\n".join(parts)


# =============================================================================
# Client Management
# =============================================================================

_client: Optional[AsyncOpenAI] = None
_client_created_at: Optional[datetime] = None
CLIENT_TTL = timedelta(hours=1)  # Refresh client hourly


async def _get_client() -> Optional[AsyncOpenAI]:
    """Get or create OpenAI client with connection pooling."""
    global _client, _client_created_at
    
    # Reuse existing client if still valid
    if _client and _client_created_at:
        if datetime.utcnow() - _client_created_at < CLIENT_TTL:
            return _client
    
    # Get API key from env or database
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = await api_keys_repo.get_decrypted_key("OPENAI_API_KEY")
    
    if not api_key:
        logger.warning("OpenAI API key not configured")
        return None
    
    _client = AsyncOpenAI(
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT,
        max_retries=0,  # We handle retries ourselves
    )
    _client_created_at = datetime.utcnow()
    return _client


def _is_gpt5(model: str) -> bool:
    """Check if model is GPT-5 family (uses reasoning tokens)."""
    return model.startswith(GPT5_PREFIXES)


def _adjust_tokens(model: str, desired: int) -> int:
    """Adjust max_output_tokens for GPT-5 reasoning overhead.
    
    GPT-5 reasoning models use some output tokens for internal reasoning.
    Even with reasoning effort set to 'low', we need 4x multiplier to ensure
    enough headroom for the actual output after reasoning tokens are used.
    """
    return desired * 4 if _is_gpt5(model) else desired


# =============================================================================
# Telemetry
# =============================================================================

@dataclass
class UsageMetrics:
    """Token usage and cost metrics."""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0


async def _record_usage(
    task: TaskType,
    model: str,
    metrics: UsageMetrics,
    success: bool,
    is_batch: bool = False,
) -> None:
    """Record API usage for telemetry."""
    try:
        await api_usage_repo.record_usage(
            service="openai",
            operation=f"{task.value}{'_batch' if is_batch else ''}",
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            cost_usd=metrics.cost_usd,
            is_batch=is_batch,
            metadata={
                "model": model,
                "reasoning_tokens": metrics.reasoning_tokens,
                "duration_ms": metrics.duration_ms,
                "success": success,
            },
        )
    except Exception as e:
        logger.debug(f"Failed to record usage: {e}")


def _calculate_cost(input_tokens: int, output_tokens: int, model: str, is_batch: bool = False) -> float:
    """Calculate cost in USD (estimates, update as pricing changes)."""
    # Pricing per 1M tokens (batch is 50% off)
    pricing = {
        "gpt-5-mini": (0.10, 0.40),
        "gpt-5": (2.00, 8.00),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
    }
    
    input_rate, output_rate = pricing.get(model, (0.15, 0.60))
    if is_batch:
        input_rate *= 0.5
        output_rate *= 0.5
    
    return (input_tokens / 1_000_000 * input_rate) + (output_tokens / 1_000_000 * output_rate)


# =============================================================================
# Core Generation Function
# =============================================================================

async def generate(
    task: TaskType | str,
    context: dict[str, Any],
    *,
    model: Optional[str] = None,
    json_output: bool = False,
    max_tokens: int = 200,
) -> Optional[str | dict]:
    """
    Generate AI content using the Responses API.
    
    Args:
        task: Type of generation task (bio, rating, summary, etc.)
        context: Dict with task-specific data (symbol, name, dip_pct, etc.)
        model: Override default model
        json_output: If True, parse response as JSON
        max_tokens: Desired output tokens (auto-adjusted for GPT-5)
    
    Returns:
        Generated text, or parsed JSON dict if json_output=True.
        None on failure.
    
    Example:
        >>> result = await generate(
        ...     task=TaskType.RATING,
        ...     context={"symbol": "AAPL", "dip_pct": 15.2},
        ...     json_output=True
        ... )
        >>> print(result)
        {"rating": "buy", "reasoning": "...", "confidence": 7}
    """
    client = await _get_client()
    if not client:
        return None
    
    # Normalize task
    if isinstance(task, str):
        task = TaskType(task)
    
    model = model or DEFAULT_MODEL
    instructions = INSTRUCTIONS.get(task, "")
    prompt = _build_prompt(task, context)
    
    # OpenAI Responses API requires 'json' in input when using json_object format
    if json_output:
        prompt += "\n\nRespond with valid JSON."
    
    # Request parameters
    params: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": _adjust_tokens(model, max_tokens),
        "store": False,
    }
    
    # For GPT-5 reasoning models, use low effort for simple tasks
    # This minimizes internal "thinking" tokens for straightforward generations
    if _is_gpt5(model):
        params["reasoning"] = {"effort": "low"}
    
    if json_output:
        params["text"] = {"format": {"type": "json_object"}}
    
    # Retry loop with exponential backoff
    last_error: Optional[Exception] = None
    start_time = datetime.utcnow()
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.responses.create(**params)
            
            if response.status != "completed":
                logger.warning(f"Response incomplete: {response.incomplete_details}")
                continue
            
            output = response.output_text
            if not output:
                continue
            
            # Record metrics
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            metrics = UsageMetrics(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                reasoning_tokens=getattr(response.usage.output_tokens_details, 'reasoning_tokens', 0),
                total_tokens=response.usage.total_tokens,
                cost_usd=_calculate_cost(response.usage.input_tokens, response.usage.output_tokens, model),
                duration_ms=duration_ms,
            )
            await _record_usage(task, model, metrics, success=True)
            
            logger.info(f"{task.value} for {context.get('symbol', 'unknown')}: {metrics.total_tokens} tokens, ${metrics.cost_usd:.4f}")
            
            # Parse, clean, and return
            output = output.strip()
            if json_output:
                result = json.loads(output)
                # Clean string values in JSON response
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, str):
                            result[key] = clean_ai_text(value)
                return result
            # Clean text output and remove quotes
            return clean_ai_text(output.strip('"\''))
            
        except RateLimitError as e:
            last_error = e
            delay = RETRY_DELAY * (2 ** attempt)
            logger.warning(f"Rate limited, waiting {delay}s...")
            await asyncio.sleep(delay)
            
        except APITimeoutError as e:
            last_error = e
            logger.warning(f"Timeout on attempt {attempt + 1}")
            
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"Invalid JSON response: {e}")
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
    
    # All retries exhausted
    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    await _record_usage(task, model, UsageMetrics(duration_ms=duration_ms), success=False)
    logger.error(f"All retries failed for {task.value}: {last_error}")
    return None


# =============================================================================
# High-Level Task Functions (Convenience Wrappers)
# =============================================================================

async def generate_bio(
    symbol: str,
    name: Optional[str] = None,
    sector: Optional[str] = None,
    summary: Optional[str] = None,
    dip_pct: Optional[float] = None,
) -> Optional[str]:
    """Generate a dating-app style bio for a stock."""
    task = TaskType.DIP_BIO if dip_pct else TaskType.BIO
    return await generate(
        task=task,
        context={
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "summary": summary,
            "dip_pct": dip_pct,
        },
        max_tokens=150,
    )


async def rate_dip(
    symbol: str,
    current_price: Optional[float] = None,
    ref_high: Optional[float] = None,
    dip_pct: Optional[float] = None,
    **extra_context,
) -> Optional[dict]:
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
        json_output=True,
        max_tokens=250,
    )


async def summarize_company(
    symbol: str,
    name: Optional[str] = None,
    description: str = "",
) -> Optional[str]:
    """Summarize a company description to ~400 characters."""
    if not description or len(description) < 50:
        return description
    
    return await generate(
        task=TaskType.SUMMARY,
        context={
            "symbol": symbol,
            "name": name,
            "description": description,
        },
        max_tokens=140,
    )


# =============================================================================
# Batch API
# =============================================================================

async def submit_batch(
    task: TaskType | str,
    items: list[dict[str, Any]],
    *,
    model: Optional[str] = None,
) -> Optional[str]:
    """
    Submit a batch job for bulk processing.
    
    Args:
        task: Type of task to run
        items: List of context dicts, each with at least 'symbol'
        model: Override default model
    
    Returns:
        Batch job ID, or None on failure
    
    Example:
        >>> batch_id = await submit_batch(
        ...     task=TaskType.BIO,
        ...     items=[
        ...         {"symbol": "AAPL", "name": "Apple Inc."},
        ...         {"symbol": "MSFT", "name": "Microsoft Corp."},
        ...     ]
        ... )
    """
    client = await _get_client()
    if not client or not items:
        return None
    
    if isinstance(task, str):
        task = TaskType(task)
    
    model = model or DEFAULT_MODEL
    instructions = INSTRUCTIONS.get(task, "")
    json_output = task == TaskType.RATING
    
    # Build JSONL
    jsonl_lines = []
    for item in items:
        custom_id = f"{task.value}_{item.get('symbol', 'unknown')}"
        prompt = _build_prompt(task, item)
        
        # OpenAI Responses API requires 'json' in input when using json_object format
        if json_output:
            prompt += "\n\nRespond with valid JSON."
        
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": _adjust_tokens(model, 300),
            "store": False,
        }
        
        if json_output:
            body["text"] = {"format": {"type": "json_object"}}
        
        jsonl_lines.append(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }))
    
    try:
        # Upload batch file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(jsonl_lines))
            temp_path = f.name
        
        with open(temp_path, "rb") as f:
            batch_file = await client.files.create(file=f, purpose="batch")
        
        Path(temp_path).unlink(missing_ok=True)
        
        # Create batch job
        batch = await client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "task": task.value,
                "model": model,
                "count": str(len(items)),
            },
        )
        
        logger.info(f"Created batch {batch.id}: {len(items)} {task.value} items")
        return batch.id
        
    except Exception as e:
        logger.error(f"Failed to create batch: {e}")
        return None


async def check_batch(batch_id: str) -> Optional[dict]:
    """Check status of a batch job."""
    client = await _get_client()
    if not client:
        return None
    
    try:
        batch = await client.batches.retrieve(batch_id)
        completed = batch.request_counts.completed if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            # Backward compatible names
            "completed_count": completed,
            "failed_count": failed,
            "total_count": total,
            # Also include nested structure for new code
            "counts": {
                "total": total,
                "completed": completed,
                "failed": failed,
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
    except Exception as e:
        logger.error(f"Failed to check batch {batch_id}: {e}")
        return None


async def collect_batch(batch_id: str) -> Optional[list[dict]]:
    """
    Collect results from a completed batch job.
    
    Returns list of dicts with 'custom_id' and 'result' keys.
    """
    client = await _get_client()
    if not client:
        return None
    
    try:
        batch = await client.batches.retrieve(batch_id)
        
        if batch.status != "completed" or not batch.output_file_id:
            logger.warning(f"Batch {batch_id} not ready: {batch.status}")
            return None
        
        output = await client.files.content(batch.output_file_id)
        
        results = []
        for line in output.text.strip().split("\n"):
            if not line:
                continue
            
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            
            # Parse response body
            response_body = data.get("response", {}).get("body", {})
            output_text = ""
            
            # Extract text from Responses API format
            for item in response_body.get("output", []):
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            output_text = content.get("text", "")
                            break
            
            results.append({
                "custom_id": custom_id,
                "symbol": custom_id.split("_", 1)[-1] if "_" in custom_id else custom_id,
                "result": output_text,
            })
        
        logger.info(f"Collected {len(results)} results from batch {batch_id}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to collect batch {batch_id}: {e}")
        return None


# =============================================================================
# Utilities
# =============================================================================

async def check_api_key() -> tuple[bool, Optional[str]]:
    """Verify API key is valid."""
    client = await _get_client()
    if not client:
        return False, "API key not configured"
    
    try:
        await client.models.list()
        return True, None
    except Exception as e:
        return False, str(e)


async def get_available_models() -> list[str]:
    """List available GPT models."""
    client = await _get_client()
    if not client:
        return []
    
    try:
        models = await client.models.list()
        return sorted([
            m.id for m in models.data
            if m.id.startswith(("gpt-4", "gpt-5"))
        ])
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


# Module-level convenience instance
class _AI:
    """Convenience wrapper for module functions."""
    generate = staticmethod(generate)
    generate_bio = staticmethod(generate_bio)
    rate_dip = staticmethod(rate_dip)
    summarize_company = staticmethod(summarize_company)
    submit_batch = staticmethod(submit_batch)
    check_batch = staticmethod(check_batch)
    collect_batch = staticmethod(collect_batch)
    check_api_key = staticmethod(check_api_key)
    get_available_models = staticmethod(get_available_models)
    
    # Expose types
    TaskType = TaskType


ai = _AI()

