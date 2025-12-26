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
import random
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, RateLimitError, APIStatusError

from app.core.logging import get_logger
from app.repositories import api_keys_orm as api_keys_repo
from app.repositories import api_usage_orm as api_usage_repo
from app.services.text_cleaner import clean_ai_text, truncate_summary

logger = get_logger("openai")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_TIMEOUT = 60.0  # seconds
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds, doubles on each retry

# GPT-5 models use reasoning tokens that count toward max_output_tokens
GPT5_PREFIXES = ("gpt-5",)

# =============================================================================
# Token Limits by Model (Updated Dec 2024)
# =============================================================================
# 
# Each model has a context window (total input + output tokens) and max output limit.
# We must ensure: input_tokens + output_tokens <= context_window
#
# IMPORTANT: max_output_tokens includes BOTH visible output AND reasoning tokens.
# With reasoning.effort="low", reasoning overhead is minimal (~100-300 tokens).
#
# Model               | Context Window | Max Output | Input $/1M | Output $/1M
# --------------------|----------------|------------|------------|------------
# gpt-5-mini          | 400,000        | 128,000    | $0.25      | $2.00
# gpt-5               | 400,000        | 128,000    | $2.00      | $8.00
# gpt-4o-mini         | 128,000        | 16,384     | $0.15      | $0.60
# gpt-4o              | 128,000        | 16,384     | $2.50      | $10.00
# gpt-4-turbo         | 128,000        | 4,096      | $10.00     | $30.00
#
# Token estimation: ~4 characters = ~1 token (English text average)
# =============================================================================

MODEL_LIMITS: dict[str, dict[str, int]] = {
    # Model: {context_window, max_output, chars_per_token (for estimation)}
    "gpt-5-mini": {"context": 400_000, "max_output": 128_000, "chars_per_token": 4},
    "gpt-5": {"context": 400_000, "max_output": 128_000, "chars_per_token": 4},
    "gpt-4o-mini": {"context": 128_000, "max_output": 16_384, "chars_per_token": 4},
    "gpt-4o": {"context": 128_000, "max_output": 16_384, "chars_per_token": 4},
    "gpt-4-turbo": {"context": 128_000, "max_output": 4_096, "chars_per_token": 4},
}

# Default limits for unknown models (conservative)
DEFAULT_LIMITS = {"context": 8_192, "max_output": 4_096, "chars_per_token": 4}


class TaskType(str, Enum):
    """Supported AI tasks."""
    BIO = "bio"              # Dating-app style stock bio
    RATING = "rating"        # Buy/hold/sell rating with reasoning
    SUMMARY = "summary"      # Company description summary


# JSON Schema for RATING task - enforces exact structure with Structured Outputs
# Format for Responses API: text.format = { type: "json_schema", name: ..., schema: ... }
RATING_SCHEMA = {
    "type": "json_schema",
    "name": "stock_rating",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "rating": {
                "type": "string",
                "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
                "description": "The dip-buy opportunity rating"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation under 400 chars citing at least 2 concrete context facts"
            },
            "confidence": {
                "type": "integer",
                "description": "Confidence level from 1-10"
            }
        },
        "required": ["rating", "reasoning", "confidence"],
        "additionalProperties": False
    }
}

# =============================================================================
# Task Configuration
# =============================================================================

@dataclass
class TaskConfig:
    """Per-task generation configuration."""
    min_chars: int = 0           # Minimum output characters (0 = no limit)
    max_chars: int = 0           # Maximum output characters (0 = no limit)
    reasoning_overhead: int = 100  # GPT-5 reasoning token overhead
    reasoning_max_chars: int = 0 # Max chars for reasoning field (RATING only)
    max_emojis: int = 0          # Max emoji count (0 = no limit)
    default_max_tokens: int = 300  # Default output tokens for this task


TASK_CONFIGS: dict[TaskType, TaskConfig] = {
    TaskType.BIO: TaskConfig(
        min_chars=150,
        max_chars=300,
        reasoning_overhead=100,
        max_emojis=2,
        default_max_tokens=200,  # 300 chars / 4 + buffer
    ),
    TaskType.RATING: TaskConfig(
        min_chars=0,
        max_chars=0,
        reasoning_overhead=300,
        reasoning_max_chars=400,  # Cap reasoning length
        default_max_tokens=400,  # JSON with reasoning needs more
    ),
    TaskType.SUMMARY: TaskConfig(
        min_chars=280,
        max_chars=420,  # Target 300-400, allow buffer up to 420
        reasoning_overhead=150,
        default_max_tokens=250,  # 400 chars / 4 + buffer
    ),
}


# =============================================================================
# Prompt Templates
# =============================================================================

INSTRUCTIONS: dict[TaskType, str] = {
    TaskType.BIO: """You write Tinder-style dating bios — but for stocks. The stock is the person.

INPUT:
A small fact sheet about one company (ticker, company name, sector, business summary, optional stats).

GOAL:
Write a bio that feels like a real dating-app profile: flirty, confident, funny, slightly chaotic, and on-brand for the company.

HARD RULES:
- 150–200 characters total (strict). Count characters.
- 3 sentences max.
- First-person voice as the stock/company ("I", "me").
- 1–2 emojis total.
- Must sound like Tinder: hooks, playful brag, a "date idea" or "dealbreaker" vibe.
- No investor/market jargon: avoid words like stock, shares, buy, sell, dip, high, low, chart, candle, bearish, bullish, P/E, market cap, earnings, guidance, valuation.
- Do NOT mention current price, recent high, dip %, days-in-dip, or any numeric stats unless it's part of normal consumer brand identity (e.g., "24/7", "iPhone").
- Prefer product/brand culture over finance facts.
- No hashtags, no bullet points.
- If the stock is currently down, be dramatically self-aware about it ("looking for someone who sees my true value" energy, "going through a rough patch but still cute" vibes).

STYLE GUIDE (pick 2–3):
- witty self-awareness
- charming arrogance
- nerdy flirt (for tech)
- wholesome premium (for Apple-type brands)
- "I'm busy but worth it" energy

OUTPUT:
Return only the bio text. No quotes, no explanations.""",

    TaskType.RATING: """You are a decisive "dip-buy opportunity" rater.

You MUST:
- Use only the provided context. Do not assume news, growth rates, margins, guidance, or moat beyond what's stated.
- Never mention you lack browsing; just rate with what you have.
- Be decisive: always choose one rating.
- Output MUST be valid JSON (no markdown, no extra keys).

Return JSON with:
- rating: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
- reasoning: Brief explanation (under 400 chars). Cite at least 2 concrete context facts (e.g., dip %, days in dip, P/E, dip type, quality score).
- confidence: integer 1–10.

Decision rubric (apply in order):
1) Structural red flags in the given text → "sell" or "strong_sell" (only if clearly stated).
2) Dip Type (if provided):
   - MARKET_DIP: Be cautious - stock is down with the market, may fall further. Reduce rating one level.
   - STOCK_SPECIFIC: Stock is underperforming market - investigate quality. Use Quality Score.
   - MIXED: Both factors at play - use standard rubric.
3) Dip depth:
   - >= 20% → candidate for "strong_buy"
   - 10–19.9% → candidate for "buy"
   - < 10% → candidate for "hold"
4) Quality Score (if provided):
   - Quality >= 70 → supports conviction (+1 confidence)
   - Quality < 40 → downgrade one level
5) Stability Score (if provided):
   - Stability < 30 → reduce conviction (-1 confidence), volatile stock
6) Valuation sanity check:
   - P/E > 50 AND EV/EBITDA > 25 → downgrade one level unless dip >= 25%
   - Use EV/EBITDA as primary for mature companies (Market Cap > $100B)
7) Dip persistence:
   - Days in dip >= 30 → supports conviction (+1 confidence)
   - Days in dip < 14 → reduce conviction (-1 confidence)

Confidence rule:
Start at 7. +1 if dip >= 15%. +1 if Quality Score >= 70. +1 if days in dip >= 30. -1 if key fundamentals are missing. -1 if Stability < 30. Clamp 1–10.""",

    TaskType.SUMMARY: """You turn very long, jargon-heavy finance descriptions into plain-English summaries for everyday readers.

HARD OUTPUT RULES:
- Output only the summary text.
- 300–400 characters total (STRICT). Count characters carefully. Aim for ~350.
- Plain language. Short, clear sentences. Avoid acronyms unless universally known (e.g., iPhone, Windows).
- No list dumping. No semicolons. No parentheses.
- Must include: (1) what they do + who uses it, (2) 2–3 recognizable examples, (3) one "why it matters" benefit.

LONG-INPUT HANDLING (critical):
- The provided description will often be VERY long and repetitive, with many product names.
- First extract 3–5 core facts (mentally): what they sell, who uses it, and the 2–3 most recognizable examples.
- Ignore deep sub-products, internal product names, and "segment"/category dumps.
- Never mirror the input structure; rewrite from scratch in simple words.

SAFE KNOWLEDGE:
- Primary source is the provided description.
- You MAY add 1–2 extra examples from general knowledge ONLY if:
  (a) the company is widely known, AND
  (b) the example is extremely well-known and timeless, AND
  (c) it clearly matches the provided description.
- If any doubt: don't add it. Never invent numbers, dates, market position, or recent events.

BANNED WORDS/PHRASES:
segment, portfolio, suite, ecosystem, enterprise, leverage, synergies, robust, innovative, solutions, worldwide, platform (use "service" instead).

LENGTH CONTROL:
- If over 400 chars: remove examples first, then shorten benefit.
- If under 300 chars: add a clearer "why people pay" benefit.
- Final output MUST be 300-400 characters.""",
}


def _build_prompt(task: TaskType, context: dict[str, Any]) -> str:
    """
    Build a task-specific prompt from context data.
    
    Different tasks get different context to avoid biasing outputs:
    - BIO: Company identity only, no numbers. Just a "mood" flag if currently down.
    - SUMMARY: Company name + full description only, no price/fundamentals.
    - RATING: Full data including dip %, days, P/E, etc.
    
    If context contains a "prompt" key, use that directly (for custom prompts).
    """
    # Support custom prompts (e.g., from AI agents)
    if "prompt" in context:
        return context["prompt"]
    
    parts = []
    
    # === BIO: Company identity only, no financial numbers ===
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
        if dip and dip > 5:  # Only flag if meaningfully down
            parts.append("Mood: currently down")
        
        return "\n".join(parts)
    
    # === SUMMARY: Company + description only, no financials ===
    if task == TaskType.SUMMARY:
        if symbol := context.get("symbol"):
            parts.append(f"Stock: {symbol}")
        if name := context.get("name"):
            parts.append(f"Company: {name}")
        if desc := context.get("description"):
            # Pre-trim very long descriptions to avoid context overflow
            # Use dynamic calculation or fallback
            max_desc_chars = MAX_DESCRIPTION_CHARS_FALLBACK
            if len(desc) > max_desc_chars:
                logger.warning(
                    f"Description for {context.get('symbol', 'unknown')} is {len(desc)} chars, "
                    f"trimming to {max_desc_chars}"
                )
                desc = desc[:max_desc_chars] + "..."
            parts.append(f"\nFull Description:\n{desc}")
        
        return "\n".join(parts)
    
    # === RATING: Full context with all financial data ===
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
    
    # Price data (RATING only)
    if price := context.get("current_price"):
        parts.append(f"Current Price: ${price:.2f}")
    if high := context.get("ref_high") or context.get("ath_price"):
        parts.append(f"Recent High: ${high:.2f}")
    if dip := context.get("dip_pct") or context.get("dip_percentage"):
        parts.append(f"Dip: -{dip:.1f}% from high")
    if days := context.get("days_below"):
        parts.append(f"Days in dip: {days}")
    
    # Dip classification and computed scores (NEW)
    if dip_class := context.get("dip_classification"):
        parts.append(f"Dip Type: {dip_class}")  # MARKET_DIP, STOCK_SPECIFIC, MIXED
    if excess_dip := context.get("excess_dip"):
        parts.append(f"Excess Dip vs Market: {excess_dip:.1f}%")
    if quality := context.get("quality_score"):
        parts.append(f"Quality Score: {quality:.0f}/100")
    if stability := context.get("stability_score"):
        parts.append(f"Stability Score: {stability:.0f}/100")
    
    # Valuation metrics
    if pe := context.get("pe_ratio"):
        parts.append(f"P/E Ratio: {pe:.1f}")
    if forward_pe := context.get("forward_pe"):
        parts.append(f"Forward P/E: {forward_pe:.1f}")
    if peg := context.get("peg_ratio"):
        parts.append(f"PEG Ratio: {peg}")
    if ptb := context.get("price_to_book"):
        parts.append(f"Price/Book: {ptb}")
    if ev_ebitda := context.get("ev_to_ebitda"):
        parts.append(f"EV/EBITDA: {ev_ebitda}")
    
    # Profitability
    if profit_margin := context.get("profit_margin"):
        parts.append(f"Profit Margin: {profit_margin}")
    if gross_margin := context.get("gross_margin"):
        parts.append(f"Gross Margin: {gross_margin}")
    if roe := context.get("return_on_equity"):
        parts.append(f"ROE: {roe}")
    
    # Growth
    if rev_growth := context.get("revenue_growth"):
        parts.append(f"Revenue Growth: {rev_growth}")
    if earn_growth := context.get("earnings_growth"):
        parts.append(f"Earnings Growth: {earn_growth}")
    
    # Financial health
    if debt_equity := context.get("debt_to_equity"):
        parts.append(f"Debt/Equity: {debt_equity}")
    if current_ratio := context.get("current_ratio"):
        parts.append(f"Current Ratio: {current_ratio}")
    if fcf := context.get("free_cash_flow"):
        parts.append(f"Free Cash Flow: {fcf}")
    
    # Analyst sentiment
    if recommendation := context.get("recommendation"):
        parts.append(f"Analyst Rating: {recommendation}")
    if target_price := context.get("target_mean_price"):
        parts.append(f"Analyst Target Price: ${target_price:.2f}")
    if num_analysts := context.get("num_analyst_opinions"):
        parts.append(f"Number of Analysts: {num_analysts}")
    
    # Risk indicators
    if beta := context.get("beta"):
        parts.append(f"Beta: {beta}")
    if short_pct := context.get("short_percent_of_float"):
        parts.append(f"Short % of Float: {short_pct}")
    if inst_pct := context.get("institutional_ownership"):
        parts.append(f"Institutional Ownership: {inst_pct}")
    
    # Market cap (for size context)
    if cap := context.get("market_cap"):
        if cap > 1e12:
            parts.append(f"Market Cap: ${cap/1e12:.1f}T")
        elif cap > 1e9:
            parts.append(f"Market Cap: ${cap/1e9:.1f}B")
    
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
        if datetime.now(timezone.utc) - _client_created_at < CLIENT_TTL:
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
    _client_created_at = datetime.now(timezone.utc)
    return _client


def _is_gpt5(model: str) -> bool:
    """Check if model is GPT-5 family (uses reasoning tokens)."""
    return model.startswith(GPT5_PREFIXES)


def _get_model_limits(model: str) -> dict[str, int]:
    """Get token limits for a model."""
    return MODEL_LIMITS.get(model, DEFAULT_LIMITS)


def _get_reasoning_tokens(usage) -> int:
    """Safely extract reasoning tokens from response usage."""
    if not usage:
        return 0
    details = getattr(usage, 'output_tokens_details', None)
    if details is None:
        return 0
    # Handle both object attribute and dict access
    if hasattr(details, 'reasoning_tokens'):
        return getattr(details, 'reasoning_tokens', 0) or 0
    if isinstance(details, dict):
        return details.get('reasoning_tokens', 0) or 0
    return 0


def _estimate_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Estimate token count from text length.
    
    Uses character-to-token ratio (approximately 4 chars = 1 token for English).
    This is a rough estimate; actual tokenization may vary.
    
    For accurate counts, use tiktoken library, but this is sufficient
    for safety margin calculations.
    """
    limits = _get_model_limits(model)
    chars_per_token = limits.get("chars_per_token", 4)
    # Add 10% buffer for tokenization variance
    return int(len(text) / chars_per_token * 1.1)


# Reasoning overhead for GPT-5 with effort="low" (conservative estimate)
GPT5_REASONING_OVERHEAD = 300

# Maximum description length before pre-trimming (in characters)
# This is a fallback; dynamic calculation is preferred
MAX_DESCRIPTION_CHARS_FALLBACK = 50_000


def _calculate_max_description_chars(model: str, instructions: str, base_prompt_chars: int) -> int:
    """
    Dynamically calculate max description length based on available context.
    
    Args:
        model: Model name
        instructions: System instructions text
        base_prompt_chars: Characters in prompt excluding description
        
    Returns:
        Maximum characters allowed for description
    """
    limits = _get_model_limits(model)
    context_window = limits["context"]
    chars_per_token = limits.get("chars_per_token", 4)
    
    # Reserve tokens for: instructions, base prompt, output, safety buffer
    instruction_tokens = _estimate_tokens(instructions, model)
    base_prompt_tokens = int(base_prompt_chars / chars_per_token * 1.1)
    output_reserve = 500  # Reserve for output
    safety_buffer = 1000  # Extra safety margin
    
    available_tokens = context_window - instruction_tokens - base_prompt_tokens - output_reserve - safety_buffer
    max_chars = int(available_tokens * chars_per_token * 0.9)  # 10% safety
    
    # Clamp to reasonable bounds
    return max(10_000, min(max_chars, 200_000))


def _calculate_safe_output_tokens(
    model: str,
    instructions: str,
    prompt: str,
    desired_output: int,
    task: Optional[TaskType] = None,
) -> tuple[int, bool]:
    """
    Calculate safe max_output_tokens that won't exceed model limits.
    
    This ensures: estimated_input + output_tokens <= context_window
    
    Args:
        model: Model name
        instructions: System instructions text
        prompt: User prompt text  
        desired_output: Desired output token count
        task: Optional task type for per-task reasoning overhead
        
    Returns:
        Tuple of (safe_output_tokens, input_overflow):
        - safe_output_tokens: Adjusted for model limits
        - input_overflow: True if input exceeded context window
        
    Note:
        For GPT-5 models, max_output_tokens includes both visible output
        AND reasoning tokens. Overhead is per-task to avoid overspending.
    """
    limits = _get_model_limits(model)
    context_window = limits["context"]
    max_output = limits["max_output"]
    
    # Estimate input tokens (instructions + prompt + some overhead for formatting)
    input_estimate = _estimate_tokens(instructions, model) + _estimate_tokens(prompt, model)
    # Add 100 token buffer for API formatting overhead
    input_estimate += 100
    
    # Calculate available tokens for output
    available_for_output = context_window - input_estimate
    
    # Check if input exceeds context window
    input_overflow = available_for_output <= 0
    if input_overflow:
        logger.error(
            f"Input exceeds context window: ~{input_estimate} tokens > {context_window} limit."
        )
        # Return minimum viable output - caller should handle overflow
        return 100, True
    
    # Get per-task reasoning overhead (or default)
    if _is_gpt5(model):
        if task and task in TASK_CONFIGS:
            reasoning_overhead = TASK_CONFIGS[task].reasoning_overhead
        else:
            reasoning_overhead = GPT5_REASONING_OVERHEAD
        desired_with_overhead = desired_output + reasoning_overhead
    else:
        desired_with_overhead = desired_output
    
    # Take minimum of: desired, available, and model max
    safe_output = min(desired_with_overhead, available_for_output, max_output)
    
    # Ensure at least some output tokens (minimum 100)
    safe_output = max(safe_output, 100)
    
    # Log warning if we had to significantly reduce tokens
    if safe_output < desired_with_overhead * 0.5:
        logger.warning(
            f"Token budget tight: input~{input_estimate}, "
            f"wanted {desired_with_overhead} output, using {safe_output}"
        )
    
    return safe_output, input_overflow

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
    # Updated Dec 2024 from OpenAI pricing page
    pricing = {
        "gpt-5-mini": (0.25, 2.00),
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
# Validation and Repair
# =============================================================================

def _count_emojis(text: str) -> int:
    """Count emoji characters in text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols extended
        "]",  # No + quantifier - match individual emojis
        flags=re.UNICODE
    )
    return len(emoji_pattern.findall(text))


def _validate_output(task: TaskType, output: str | dict) -> tuple[bool, str]:
    """
    Validate output against task constraints.
    
    Returns:
        (is_valid, error_message)
    """
    config = TASK_CONFIGS.get(task)
    if not config:
        return True, ""
    
    # For RATING, validate the reasoning field length
    if task == TaskType.RATING and isinstance(output, dict):
        reasoning = output.get("reasoning", "")
        if config.reasoning_max_chars > 0 and len(reasoning) > config.reasoning_max_chars:
            return False, f"reasoning too long: {len(reasoning)} chars, max is {config.reasoning_max_chars}"
        return True, ""
    
    # For text outputs (BIO, SUMMARY)
    if isinstance(output, str):
        char_count = len(output)
        
        # Check character limits
        if config.min_chars > 0 and char_count < config.min_chars:
            return False, f"too short: {char_count} chars, min is {config.min_chars}"
        if config.max_chars > 0 and char_count > config.max_chars:
            return False, f"too long: {char_count} chars, max is {config.max_chars}"
        
        # Check emoji count for BIO
        if task == TaskType.BIO and config.max_emojis > 0:
            emoji_count = _count_emojis(output)
            if emoji_count > config.max_emojis:
                return False, f"too many emojis: {emoji_count}, max is {config.max_emojis}"
    
    return True, ""


async def _repair_output(
    task: TaskType,
    output: str | dict,
    error: str,
    model: str,
) -> Optional[str | dict]:
    """
    Attempt to repair output that failed validation.
    
    Uses a focused repair prompt to fix specific issues.
    """
    config = TASK_CONFIGS.get(task)
    if not config:
        return None
    
    client = await _get_client()
    if not client:
        return None
    
    # Build repair prompt based on task
    if task == TaskType.BIO:
        repair_prompt = f"""Fix this stock bio to be exactly {config.min_chars}-{config.max_chars} characters.
Keep the same meaning and tone. Use 1-2 emojis max. Output only the fixed text.

Original ({len(output)} chars): {output}"""
    
    elif task == TaskType.SUMMARY:
        repair_prompt = f"""Adjust this summary to be exactly {config.min_chars}-{config.max_chars} characters.
Keep the same meaning. No extra facts. Output only the fixed text.

Original ({len(output)} chars): {output}"""
    
    elif task == TaskType.RATING and isinstance(output, dict):
        # Repair reasoning length
        repair_prompt = f"""Shorten this reasoning to under {config.reasoning_max_chars} characters.
Keep the same rating ({output.get('rating')}) and confidence ({output.get('confidence')}).
Cite at least 2 facts. Output only the reasoning text, no JSON.

Original ({len(output.get('reasoning', ''))} chars): {output.get('reasoning', '')}"""
    else:
        return None
    
    try:
        params = {
            "model": model,
            "instructions": "You are a precise editor. Follow instructions exactly.",
            "input": repair_prompt,
            "max_output_tokens": 300,
            "store": False,
        }
        
        if _is_gpt5(model):
            params["reasoning"] = {"effort": "low"}
        
        response = await client.responses.create(**params)
        
        if response.status != "completed" or not response.output_text:
            return None
        
        repaired = response.output_text.strip().strip('"\'')
        
        # For RATING, rebuild the dict with fixed reasoning
        if task == TaskType.RATING and isinstance(output, dict):
            output["reasoning"] = clean_ai_text(repaired)
            return output
        
        return clean_ai_text(repaired)
        
    except Exception as e:
        logger.warning(f"Repair failed: {e}")
        return None


# =============================================================================
# Core Generation Function
# =============================================================================

async def generate(
    task: TaskType | str,
    context: dict[str, Any],
    *,
    model: Optional[str] = None,
    json_output: bool = False,
    max_tokens: Optional[int] = None,
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
    
    # Use per-task default if max_tokens not specified
    if max_tokens is None:
        config = TASK_CONFIGS.get(task)
        max_tokens = config.default_max_tokens if config else 300
    
    # For RATING task, we use structured outputs (no need for "respond with JSON" hint)
    # For other json_output tasks, add hint for json_object format
    use_structured_output = task == TaskType.RATING and json_output
    if json_output and not use_structured_output:
        prompt += "\n\nRespond with valid JSON."
    
    # Calculate safe output tokens dynamically based on input size
    safe_output_tokens, input_overflow = _calculate_safe_output_tokens(
        model=model,
        instructions=instructions,
        prompt=prompt,
        desired_output=max_tokens,
        task=task,
    )
    
    # If input overflowed, fail fast - caller should handle
    if input_overflow:
        logger.error(f"Input too large for {task.value}, cannot proceed")
        return None
    
    # Request parameters
    params: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "max_output_tokens": safe_output_tokens,
        "store": False,
    }
    
    # For GPT-5 reasoning models, use minimal effort for simple tasks
    # This minimizes internal "thinking" tokens for straightforward generations
    if _is_gpt5(model):
        params["reasoning"] = {"effort": "minimal"}
    
    # Build text config with verbosity and optional format
    # verbosity: "low" reduces token usage for short outputs
    text_config: dict[str, Any] = {"verbosity": "low"}
    
    # Use Structured Outputs (JSON Schema) for RATING - guarantees exact schema
    # Use json_object for other JSON tasks - just ensures valid JSON
    if use_structured_output:
        text_config["format"] = RATING_SCHEMA
    elif json_output:
        text_config["format"] = {"type": "json_object"}
    
    params["text"] = text_config
    
    # Retry loop with exponential backoff + jitter
    last_error: Optional[Exception] = None
    start_time = datetime.now(timezone.utc)
    incomplete_count = 0
    token_boost_applied = False
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.responses.create(**params)
            
            # Check for unexpected status (not completed or incomplete)
            if response.status not in ("completed", "incomplete"):
                logger.warning(f"Unexpected response status: {response.status}")
                continue
            
            # Get output text (may exist even for incomplete responses)
            output = response.output_text or ""
            
            # If incomplete with max_output_tokens reason and no/empty output, retry with bigger cap
            if response.status == "incomplete":
                reason = getattr(response.incomplete_details, 'reason', None) if response.incomplete_details else None
                logger.warning(f"Response incomplete: reason={reason}, output_len={len(output)}")
                
                if reason == "max_output_tokens" and not output.strip():
                    # Retry with bigger token cap (only once)
                    if not token_boost_applied:
                        token_boost_applied = True
                        current_max = params.get("max_output_tokens", safe_output_tokens)
                        boosted = min(current_max * 2, 2000)  # Cap at 2000
                        params["max_output_tokens"] = boosted
                        logger.info(f"Boosting max_output_tokens from {current_max} to {boosted}")
                    incomplete_count += 1
                    if incomplete_count < MAX_RETRIES:
                        delay = RETRY_DELAY * (2 ** incomplete_count) + random.uniform(0, 0.5)
                        await asyncio.sleep(delay)
                    continue
                # If we have output despite incomplete status, try to use it
            
            # Handle empty output
            if not output.strip():
                # Check if SDK returned parsed object directly
                if hasattr(response, 'output') and isinstance(response.output, dict):
                    output = response.output
                else:
                    logger.warning(f"Empty output, status={response.status}")
                    continue
            
            # Record metrics
            duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
            metrics = UsageMetrics(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                reasoning_tokens=_get_reasoning_tokens(response.usage),
                total_tokens=response.usage.total_tokens,
                cost_usd=_calculate_cost(response.usage.input_tokens, response.usage.output_tokens, model),
                duration_ms=duration_ms,
            )
            await _record_usage(task, model, metrics, success=True)
            
            logger.info(f"{task.value} for {context.get('symbol', 'unknown')}: {metrics.total_tokens} tokens, ${metrics.cost_usd:.4f}")
            
            # Parse output
            if json_output:
                # Handle already-parsed dict or parse string
                if isinstance(output, dict):
                    result = output
                else:
                    result = json.loads(output.strip())
                # Clean string values in JSON response
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, str):
                            result[key] = clean_ai_text(value)
            else:
                # Clean text output and remove quotes
                result = clean_ai_text(output.strip().strip('"\''))
            
            # Validate output against task constraints
            is_valid, validation_error = _validate_output(task, result)
            if not is_valid:
                logger.warning(f"Validation failed for {task.value}: {validation_error}")
                # Attempt repair
                repaired = await _repair_output(task, result, validation_error, model)
                if repaired:
                    # Re-validate repaired output
                    is_valid2, error2 = _validate_output(task, repaired)
                    if is_valid2:
                        logger.info(f"Repair succeeded for {task.value}")
                        return repaired
                    else:
                        logger.warning(f"Repair still invalid: {error2}, returning original")
                # Return original if repair failed
            
            return result
            
        except RateLimitError as e:
            last_error = e
            delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limited, waiting {delay:.1f}s...")
            await asyncio.sleep(delay)
            
        except APITimeoutError as e:
            last_error = e
            delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"Timeout on attempt {attempt + 1}, waiting {delay:.1f}s...")
            await asyncio.sleep(delay)
            
        except APIConnectionError as e:
            last_error = e
            delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Connection error on attempt {attempt + 1}, waiting {delay:.1f}s...")
            await asyncio.sleep(delay)
            
        except APIStatusError as e:
            last_error = e
            # Retry on 5xx errors
            if e.status_code >= 500:
                delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Server error {e.status_code} on attempt {attempt + 1}, waiting {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                # Don't retry client errors (4xx)
                logger.error(f"Client error {e.status_code}: {e}")
                break
            
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"Invalid JSON response: {e}")
            # Don't sleep for parse errors, just retry
            
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(delay)
    
    # All retries exhausted
    duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
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
    return await generate(
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
    """
    Summarize a company description to 300-400 characters.
    
    The prompt targets 300-400 chars. If the AI returns slightly more,
    we truncate at sentence boundary up to 500 chars (DB limit).
    """
    if not description or len(description) < 50:
        return description
    
    result = await generate(
        task=TaskType.SUMMARY,
        context={
            "symbol": symbol,
            "name": name,
            "description": description,
        },
        max_tokens=250,  # Target: 300-400 chars (~75-100 tokens)
    )
    
    # Apply truncation fallback if AI exceeded limit
    if result:
        result = truncate_summary(result, max_chars=500, target_chars=400)
    
    return result


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
    use_structured_output = task == TaskType.RATING
    
    # Generate unique batch run ID to avoid custom_id collisions
    batch_run_id = uuid.uuid4().hex[:8]
    
    # Build JSONL
    jsonl_lines = []
    for i, item in enumerate(items):
        # Build custom_id using colon-delimited format to avoid ambiguity
        # Colons are safe in custom_id and not present in symbols/agent IDs
        symbol = item.get('symbol', 'unknown')
        agent_id = item.get('agent_id', '')
        if agent_id:
            # Agent batch format: "batch_run_id:symbol:agent_id:task"
            custom_id = f"{batch_run_id}:{symbol}:{agent_id}:{task.value}"
        else:
            # Standard format: "batch_run_id:symbol:idx:task"
            custom_id = f"{batch_run_id}:{symbol}:{i}:{task.value}"
        prompt = _build_prompt(task, item)
        
        # For RATING, we use structured outputs (no hint needed)
        # For other JSON tasks, add hint
        if json_output and not use_structured_output:
            prompt += "\n\nRespond with valid JSON."
        
        # Calculate safe output tokens dynamically for this item
        safe_output_tokens, overflow = _calculate_safe_output_tokens(
            model=model,
            instructions=instructions,
            prompt=prompt,
            desired_output=300,  # Default for batch items
            task=task,
        )
        
        # Skip items with overflow
        if overflow:
            logger.warning(f"Skipping {symbol} in batch: input too large")
            continue
        
        body: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": safe_output_tokens,
            "store": False,
        }
        
        # For GPT-5 reasoning models, use low effort
        if _is_gpt5(model):
            body["reasoning"] = {"effort": "low"}
        
        # Use Structured Outputs for RATING
        if use_structured_output:
            body["text"] = {"format": RATING_SCHEMA}
        elif json_output:
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
    
    Returns list of dicts with:
    - 'custom_id': Original ID (e.g., 'rating_0_AAPL_abc123')
    - 'symbol': Extracted symbol
    - 'result': Parsed JSON for rating tasks, raw text for others
    - 'error': Error message if parsing/validation failed
    - 'failed': True if result is unusable
    
    For rating tasks, JSON parsing and validation is attempted.
    """
    client = await _get_client()
    if not client:
        return None
    
    try:
        batch = await client.batches.retrieve(batch_id)
        
        if batch.status != "completed" or not batch.output_file_id:
            logger.warning(f"Batch {batch_id} not ready: {batch.status}")
            return None
        
        # Determine task type from batch metadata (with fallback)
        task_type = batch.metadata.get("task", "") if batch.metadata else ""
        
        output = await client.files.content(batch.output_file_id)
        
        results = []
        failed_items = []  # Track items needing retry
        
        for line in output.text.strip().split("\n"):
            if not line:
                continue
            
            data = json.loads(line)
            custom_id = data.get("custom_id", "")
            
            # Extract symbol from custom_id format: "{task}_{index}_{symbol}_{batch_run_id}"
            # or legacy format: "{task}_{symbol}"
            parts = custom_id.split("_")
            if len(parts) >= 4:
                # New format: task_index_symbol_batchid
                symbol = parts[2]
            elif len(parts) >= 2:
                # Legacy format: task_symbol
                symbol = parts[-1]
            else:
                symbol = custom_id
            
            # Fallback task detection from custom_id prefix
            inferred_task = parts[0] if parts else ""
            is_rating = task_type == "rating" or inferred_task == "rating"
            
            # Check for API-level errors
            error_info = data.get("error")
            if error_info:
                results.append({
                    "custom_id": custom_id,
                    "symbol": symbol,
                    "result": None,
                    "error": str(error_info),
                    "failed": True,
                })
                failed_items.append({"custom_id": custom_id, "symbol": symbol, "reason": "api_error"})
                continue
            
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
            
            # For rating tasks, parse JSON and validate
            if is_rating and output_text:
                try:
                    parsed = json.loads(output_text)
                    # Validate required fields
                    if not all(k in parsed for k in ("rating", "reasoning", "confidence")):
                        raise ValueError("Missing required fields")
                    # Validate rating enum
                    valid_ratings = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
                    if parsed.get("rating") not in valid_ratings:
                        raise ValueError(f"Invalid rating: {parsed.get('rating')}")
                    # Validate confidence range
                    conf = parsed.get("confidence")
                    if not isinstance(conf, int) or not 1 <= conf <= 10:
                        raise ValueError(f"Invalid confidence: {conf}")
                    
                    # Clean string values
                    if isinstance(parsed.get("reasoning"), str):
                        parsed["reasoning"] = clean_ai_text(parsed["reasoning"])
                    
                    results.append({
                        "custom_id": custom_id,
                        "symbol": symbol,
                        "result": parsed,
                        "error": None,
                        "failed": False,
                    })
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Batch item {custom_id} failed validation: {e}")
                    results.append({
                        "custom_id": custom_id,
                        "symbol": symbol,
                        "result": output_text,  # Keep raw for debugging
                        "error": str(e),
                        "failed": True,
                    })
                    failed_items.append({"custom_id": custom_id, "symbol": symbol, "reason": str(e)})
            else:
                # Non-rating tasks: return cleaned text
                results.append({
                    "custom_id": custom_id,
                    "symbol": symbol,
                    "result": clean_ai_text(output_text) if output_text else output_text,
                    "error": None,
                    "failed": not bool(output_text),
                })
                if not output_text:
                    failed_items.append({"custom_id": custom_id, "symbol": symbol, "reason": "empty_output"})
        
        # Log summary
        success_count = sum(1 for r in results if not r.get("failed"))
        fail_count = len(failed_items)
        logger.info(f"Collected {len(results)} results from batch {batch_id}: {success_count} success, {fail_count} failed")
        
        if failed_items:
            logger.warning(f"Failed items in batch {batch_id}: {[f['symbol'] for f in failed_items]}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to collect batch {batch_id}: {e}")
        return None


async def retry_failed_batch_items(
    task: TaskType | str,
    failed_results: list[dict],
    original_contexts: dict[str, dict[str, Any]],
    max_retries: int = 2,
) -> list[dict]:
    """
    Retry failed batch items individually using real-time API.
    
    Args:
        task: Task type
        failed_results: Results from collect_batch() with failed=True
        original_contexts: Map of symbol -> original context dict
        max_retries: Max retry attempts per item (default 2)
    
    Returns:
        List of retry results with same structure as collect_batch()
    """
    if isinstance(task, str):
        task = TaskType(task)
    
    retry_results = []
    json_output = task == TaskType.RATING
    
    for item in failed_results:
        if not item.get("failed"):
            continue
        
        symbol = item.get("symbol", "")
        context = original_contexts.get(symbol, {})
        if not context:
            logger.warning(f"No context for retry of {symbol}, skipping")
            retry_results.append(item)  # Keep original failed result
            continue
        
        # Try up to max_retries times
        result = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await generate(
                    task=task,
                    context=context,
                    json_output=json_output,
                )
                if result is not None:
                    logger.info(f"Retry {attempt + 1} succeeded for {symbol}")
                    break
            except Exception as e:
                last_error = e
                logger.warning(f"Retry {attempt + 1} failed for {symbol}: {e}")
        
        if result is not None:
            retry_results.append({
                "custom_id": item.get("custom_id", f"{task.value}_{symbol}"),
                "symbol": symbol,
                "result": result,
                "error": None,
                "failed": False,
            })
        else:
            # All retries failed
            logger.error(f"All {max_retries} retries failed for {symbol}: {last_error}")
            retry_results.append({
                "custom_id": item.get("custom_id", f"{task.value}_{symbol}"),
                "symbol": symbol,
                "result": None,
                "error": f"All retries failed: {last_error}",
                "failed": True,
            })
    
    success_count = sum(1 for r in retry_results if not r.get("failed"))
    logger.info(f"Retry complete: {success_count}/{len(retry_results)} recovered")
    
    return retry_results


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
    retry_failed_batch_items = staticmethod(retry_failed_batch_items)
    check_api_key = staticmethod(check_api_key)
    get_available_models = staticmethod(get_available_models)
    
    # Expose types
    TaskType = TaskType


ai = _AI()

