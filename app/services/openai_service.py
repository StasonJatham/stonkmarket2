"""OpenAI service for generating stock tinder bios and dip ratings."""

from __future__ import annotations

import json
from typing import Optional, Any

from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.repositories import api_keys as api_keys_repo

logger = get_logger("openai_service")

# gpt-4o-mini for cost-effective generation
MODEL = "gpt-4o-mini"


async def _get_client() -> Optional[AsyncOpenAI]:
    """Get OpenAI client with API key from database."""
    api_key = await api_keys_repo.get_decrypted_key(api_keys_repo.OPENAI_API_KEY)

    if not api_key:
        logger.warning("OpenAI API key not configured")
        return None

    return AsyncOpenAI(api_key=api_key)


async def generate_stock_bio(
    symbol: str,
    name: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    summary: Optional[str] = None,
    last_price: Optional[float] = None,
    price_change_pct: Optional[float] = None,
) -> Optional[str]:
    """
    Generate a fun Tinder-style bio for a stock.

    Args:
        symbol: Stock ticker symbol
        name: Company name
        sector: Company sector
        industry: Company industry
        summary: Company business summary
        last_price: Current stock price
        price_change_pct: 90-day price change percentage

    Returns:
        Generated bio string, or None if generation fails
    """
    client = await _get_client()
    if not client:
        return None

    # Build context
    context_parts = [f"Stock: {symbol}"]
    if name:
        context_parts.append(f"Company: {name}")
    if sector:
        context_parts.append(f"Sector: {sector}")
    if industry:
        context_parts.append(f"Industry: {industry}")
    if summary:
        context_parts.append(f"Business: {summary[:500]}")  # Limit summary length
    if last_price:
        context_parts.append(f"Current Price: ${last_price:.2f}")
    if price_change_pct is not None:
        direction = "up" if price_change_pct > 0 else "down"
        context_parts.append(f"90-day change: {direction} {abs(price_change_pct):.1f}%")

    context = "\n".join(context_parts)

    prompt = f"""Generate a fun, witty Tinder-style dating profile bio for this stock. 
The bio should be written from the stock's perspective, as if it were on a dating app looking for investors.
Keep it playful, use investing/dating puns, and highlight the stock's personality based on its business.
Max 3-4 sentences. Be creative but informative.

{context}

Write the bio now (no intro, just the bio):"""

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative copywriter who writes fun, witty stock profiles in the style of Tinder bios. Keep responses short, punchy, and entertaining.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.8,
        )

        bio = response.choices[0].message.content
        logger.info(f"Generated bio for {symbol}")
        return bio.strip() if bio else None

    except Exception as e:
        logger.error(f"Failed to generate bio for {symbol}: {e}")
        return None


async def rate_dip(
    symbol: str,
    dip_data: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """
    Rate a stock dip using AI analysis.

    Args:
        symbol: Stock ticker symbol
        dip_data: Dictionary containing:
            - name: Company name
            - sector: Sector
            - industry: Industry
            - summary: Business summary
            - current_price: Current price
            - ref_high: Reference high price
            - dip_pct: Current dip percentage
            - days_below: Days below threshold
            - fundamentals: Optional dict of fundamental data (P/E, market cap, etc.)

    Returns:
        Dict with:
            - rating: 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
            - reasoning: AI explanation
            - confidence: 1-10 confidence score
        Or None if generation fails
    """
    client = await _get_client()
    if not client:
        return None

    # Build context
    context_parts = [f"Stock: {symbol}"]

    if dip_data.get("name"):
        context_parts.append(f"Company: {dip_data['name']}")
    if dip_data.get("sector"):
        context_parts.append(f"Sector: {dip_data['sector']}")
    if dip_data.get("industry"):
        context_parts.append(f"Industry: {dip_data['industry']}")
    if dip_data.get("summary"):
        context_parts.append(f"Business: {dip_data['summary'][:500]}")

    # Price data
    if dip_data.get("current_price"):
        context_parts.append(f"Current Price: ${dip_data['current_price']:.2f}")
    if dip_data.get("ref_high"):
        context_parts.append(f"Recent High: ${dip_data['ref_high']:.2f}")
    if dip_data.get("dip_pct"):
        context_parts.append(f"Current Dip: -{dip_data['dip_pct']:.1f}% from high")
    if dip_data.get("days_below"):
        context_parts.append(f"Days in Dip: {dip_data['days_below']}")

    # Fundamentals if provided
    if dip_data.get("fundamentals"):
        fundamentals = dip_data["fundamentals"]
        if fundamentals.get("pe_ratio"):
            context_parts.append(f"P/E Ratio: {fundamentals['pe_ratio']:.1f}")
        if fundamentals.get("market_cap"):
            cap = fundamentals["market_cap"]
            if cap > 1e12:
                context_parts.append(f"Market Cap: ${cap / 1e12:.1f}T")
            elif cap > 1e9:
                context_parts.append(f"Market Cap: ${cap / 1e9:.1f}B")
            else:
                context_parts.append(f"Market Cap: ${cap / 1e6:.1f}M")
        if fundamentals.get("dividend_yield"):
            context_parts.append(
                f"Dividend Yield: {fundamentals['dividend_yield']:.2f}%"
            )
        if fundamentals.get("52_week_change"):
            context_parts.append(
                f"52-Week Change: {fundamentals['52_week_change']:.1f}%"
            )

    context = "\n".join(context_parts)

    prompt = f"""Analyze this stock dip and provide an investment rating.

{context}

Based on this information, provide:
1. A rating: one of 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
2. Brief reasoning (2-3 sentences)
3. Confidence score (1-10)

Important: Consider the company's fundamentals, the size of the dip, and general market context.
Be balanced - not every dip is a buying opportunity.

Respond in JSON format:
{{"rating": "...", "reasoning": "...", "confidence": N}}"""

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a balanced financial analyst. Provide objective stock ratings based on available data. Always respond in valid JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,  # Lower temp for more consistent ratings
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if content:
            result = json.loads(content)
            logger.info(f"Generated dip rating for {symbol}: {result.get('rating')}")
            return result
        return None

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to rate dip for {symbol}: {e}")
        return None


async def generate_tinder_bio_for_dip(
    symbol: str,
    dip_data: dict[str, Any],
) -> Optional[str]:
    """
    Generate a Tinder-style bio specifically for a stock in a dip.

    This version emphasizes the dip aspect with more dramatic/humorous tone.
    """
    client = await _get_client()
    if not client:
        return None

    # Build context
    context_parts = [f"Stock: {symbol}"]

    if dip_data.get("name"):
        context_parts.append(f"Company: {dip_data['name']}")
    if dip_data.get("sector"):
        context_parts.append(f"Sector: {dip_data['sector']}")
    if dip_data.get("summary"):
        context_parts.append(f"Business: {dip_data['summary'][:300]}")
    if dip_data.get("current_price"):
        context_parts.append(f"Current Price: ${dip_data['current_price']:.2f}")
    if dip_data.get("dip_pct"):
        context_parts.append(f"DOWN {dip_data['dip_pct']:.1f}% from recent high")
    if dip_data.get("days_below"):
        context_parts.append(f"Been down for {dip_data['days_below']} days")

    context = "\n".join(context_parts)

    prompt = f"""Generate a Tinder-style dating bio for this stock that's currently in a DIP.
The bio should be from the stock's perspective, acknowledging it's going through a rough patch.
Be dramatic, funny, maybe a bit self-deprecating about the price drop.
Use dating/investing puns. Think: "Looking for someone who sees my true value" energy.
Max 3-4 sentences.

{context}

Write the bio (no intro, just the bio):"""

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative copywriter who writes fun, dramatic Tinder bios for stocks going through price dips. Be witty and use self-aware humor about market struggles.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.85,
        )

        bio = response.choices[0].message.content
        logger.info(f"Generated dip bio for {symbol}")
        return bio.strip() if bio else None

    except Exception as e:
        logger.error(f"Failed to generate dip bio for {symbol}: {e}")
        return None


async def check_api_key_valid() -> tuple[bool, Optional[str]]:
    """
    Check if the OpenAI API key is valid.

    Returns:
        Tuple of (is_valid, error_message)
    """
    client = await _get_client()
    if not client:
        return False, "API key not configured"

    try:
        # Make a minimal API call to verify the key
        await client.models.list()
        return True, None
    except Exception as e:
        return False, str(e)
