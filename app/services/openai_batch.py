"""OpenAI Batch API service for cost-efficient AI generation.

Uses the OpenAI Batch API for non-urgent tasks (weekly updates) and
real-time API for immediate needs (new stock additions).

Tracks API usage and costs for admin dashboard.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Any
from enum import Enum

from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.database.connection import get_db, fetch_one, execute
from app.repositories import api_keys as api_keys_repo

logger = get_logger("openai_batch")

# Preferred model - use gpt-5-mini for best cost/performance
PREFERRED_MODEL = "gpt-5-mini"
FALLBACK_MODEL = "gpt-4o-mini"  # Fallback if gpt-5-mini not available

# Cache for model info and pricing
_model_cache: dict[str, Any] = {}
_pricing_cache: dict[str, dict[str, float]] = {}
_cache_timestamp: Optional[datetime] = None
CACHE_TTL_DAYS = 7


class BatchJobType(str, Enum):
    """Types of batch jobs we run."""
    DIP_ANALYSIS = "dip_analysis"
    SUGGESTION_BIO = "suggestion_bio"
    TINDER_CARDS = "tinder_cards"


# Optimized system prompts
SYSTEM_PROMPTS = {
    "dip_bio": """You are a witty financial copywriter creating Tinder-style dating profiles for stocks in price dips.

STYLE:
- Write from the stock's perspective (first person)
- Acknowledge the price dip with self-aware humor
- Use dating/investing double meanings
- Be dramatic but charming
- 2-3 sentences max

EXAMPLES:
"Down 15% but my fundamentals are still intact 💪 Looking for patient investors who appreciate a discount. Swipe right if you believe in comebacks."
"Just got ghosted by the market, but my revenue keeps growing. Seeking someone who sees past short-term drama 📉→📈"
""",

    "dip_rating": """You are a balanced financial analyst providing stock dip ratings.

TASK: Analyze the dip and provide a JSON response with:
- rating: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
- reasoning: 2-3 sentence explanation
- confidence: 1-10 score

GUIDELINES:
- Consider dip magnitude vs historical volatility
- Factor in company fundamentals if provided
- Not every dip is a buying opportunity
- Be conservative with "strong_buy" or "strong_sell"
- Confidence reflects data quality, not conviction

Respond ONLY with valid JSON.""",

    "suggestion_bio": """You are a creative copywriter writing Tinder-style profiles for stock suggestions.

STYLE:
- First person from the stock's perspective  
- Highlight what makes this stock interesting
- Be playful and use investing puns
- Show personality based on the company's business
- 2-3 sentences max

Focus on what would make investors curious about this stock.""",
}


async def _get_client() -> Optional[AsyncOpenAI]:
    """Get OpenAI client with API key from database."""
    async with get_db() as conn:
        api_key = await api_keys_repo.get_decrypted_key_async(conn, api_keys_repo.OPENAI_API_KEY)
    
    if not api_key:
        logger.warning("OpenAI API key not configured")
        return None
    
    return AsyncOpenAI(api_key=api_key)


async def get_available_models() -> list[dict[str, Any]]:
    """
    Fetch available models from OpenAI API.
    Results are cached for 7 days.
    """
    global _model_cache, _cache_timestamp
    
    # Check cache
    if _model_cache and _cache_timestamp:
        if datetime.utcnow() - _cache_timestamp < timedelta(days=CACHE_TTL_DAYS):
            return list(_model_cache.values())
    
    client = await _get_client()
    if not client:
        return []
    
    try:
        models_response = await client.models.list()
        models = []
        
        for model in models_response.data:
            model_info = {
                "id": model.id,
                "created": model.created,
                "owned_by": model.owned_by,
            }
            models.append(model_info)
            _model_cache[model.id] = model_info
        
        _cache_timestamp = datetime.utcnow()
        logger.info(f"Cached {len(models)} models from OpenAI API")
        
        # Store in database for persistence
        await _persist_model_cache()
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return list(_model_cache.values()) if _model_cache else []


async def _persist_model_cache() -> None:
    """Persist model cache to database."""
    try:
        await execute(
            """
            INSERT INTO api_usage (service, endpoint, request_metadata, recorded_at)
            VALUES ('openai', 'models_cache', $1, NOW())
            ON CONFLICT DO NOTHING
            """,
            json.dumps({
                "models": list(_model_cache.keys()),
                "cached_at": _cache_timestamp.isoformat() if _cache_timestamp else None,
            }),
        )
    except Exception as e:
        logger.debug(f"Could not persist model cache: {e}")


async def _load_pricing_cache() -> None:
    """Load cached pricing from database if available."""
    global _pricing_cache, _cache_timestamp
    
    try:
        row = await fetch_one(
            """
            SELECT request_metadata, recorded_at
            FROM api_usage
            WHERE service = 'openai' AND endpoint = 'pricing_cache'
            ORDER BY recorded_at DESC
            LIMIT 1
            """
        )
        
        if row and row["request_metadata"]:
            data = row["request_metadata"] if isinstance(row["request_metadata"], dict) else json.loads(row["request_metadata"])
            cached_at = row["recorded_at"]
            
            # Check if cache is still valid (7 days)
            if cached_at and datetime.utcnow() - cached_at < timedelta(days=CACHE_TTL_DAYS):
                _pricing_cache = data.get("pricing", {})
                logger.info(f"Loaded cached pricing for {len(_pricing_cache)} models")
                
    except Exception as e:
        logger.debug(f"Could not load pricing cache: {e}")


async def get_model_pricing(model: str) -> dict[str, float]:
    """
    Get pricing for a model.
    
    OpenAI doesn't expose pricing via API, so we maintain a cache
    that can be updated via admin dashboard or fetched from public sources.
    
    Returns dict with 'input' and 'output' prices per 1M tokens.
    """
    global _pricing_cache
    
    # Load from DB if empty
    if not _pricing_cache:
        await _load_pricing_cache()
    
    # Check cache
    if model in _pricing_cache:
        return _pricing_cache[model]
    
    # Default pricing estimates (updated as of Dec 2024)
    # These can be overridden via admin dashboard
    defaults = {
        # GPT-5 series
        "gpt-5-mini": {"input": 0.10, "output": 0.40},
        "gpt-5-mini-batch": {"input": 0.05, "output": 0.20},
        # GPT-4o series
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini-batch": {"input": 0.075, "output": 0.30},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        # GPT-4 Turbo
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        # GPT-3.5
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    }
    
    if model in defaults:
        _pricing_cache[model] = defaults[model]
        return defaults[model]
    
    # Check for batch variant
    if model.endswith("-batch"):
        base_model = model.replace("-batch", "")
        if base_model in defaults:
            # Batch is 50% cheaper
            base_pricing = defaults[base_model]
            batch_pricing = {
                "input": base_pricing["input"] * 0.5,
                "output": base_pricing["output"] * 0.5,
            }
            _pricing_cache[model] = batch_pricing
            return batch_pricing
    
    # Unknown model - estimate based on naming
    logger.warning(f"Unknown model pricing for {model}, using default estimate")
    return {"input": 0.15, "output": 0.60}


async def update_model_pricing(pricing: dict[str, dict[str, float]]) -> None:
    """
    Update cached model pricing (admin function).
    
    Args:
        pricing: Dict of model_id -> {"input": float, "output": float}
    """
    global _pricing_cache
    
    _pricing_cache.update(pricing)
    
    # Persist to database
    try:
        await execute(
            """
            INSERT INTO api_usage (service, endpoint, request_metadata, recorded_at)
            VALUES ('openai', 'pricing_cache', $1, NOW())
            """,
            json.dumps({"pricing": _pricing_cache}),
        )
        logger.info(f"Updated pricing cache for {len(pricing)} models")
    except Exception as e:
        logger.error(f"Failed to persist pricing cache: {e}")


async def get_best_available_model() -> str:
    """
    Get the best available model, preferring gpt-5-mini.
    
    Falls back to gpt-4o-mini if preferred model isn't available.
    """
    global _model_cache
    
    # Try to get models if cache empty
    if not _model_cache:
        await get_available_models()
    
    # Check if preferred model is available
    if PREFERRED_MODEL in _model_cache or not _model_cache:
        return PREFERRED_MODEL
    
    # Check for fallback
    if FALLBACK_MODEL in _model_cache:
        logger.info(f"Using fallback model {FALLBACK_MODEL}")
        return FALLBACK_MODEL
    
    # Return preferred anyway (API will error if not available)
    return PREFERRED_MODEL


def _estimate_tokens(text: str) -> int:
    """Rough estimate of tokens (4 chars per token on average)."""
    return len(text) // 4


async def _calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
    is_batch: bool = False,
) -> float:
    """Calculate cost in USD for token usage."""
    model_key = f"{model}-batch" if is_batch else model
    pricing = await get_model_pricing(model_key)
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return round(input_cost + output_cost, 6)


# --- Real-time API (for immediate needs) ---

async def generate_dip_bio_realtime(
    symbol: str,
    current_price: Optional[float] = None,
    ath_price: Optional[float] = None,
    dip_percentage: Optional[float] = None,
) -> Optional[str]:
    """
    Generate a dip bio in real-time.
    
    Returns:
        Bio text or None if failed
    """
    client = await _get_client()
    if not client:
        return None
    
    model = await get_best_available_model()
    
    prompt = f"""Stock: {symbol}
Current Price: ${current_price:.2f if current_price else 'N/A'}
All-Time High: ${ath_price:.2f if ath_price else 'N/A'}
Dip from ATH: {dip_percentage:.1f if dip_percentage else 'N/A'}%

Write a witty, self-aware Tinder bio for this stock that's currently in a dip."""
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["dip_bio"]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.8,
        )
        
        bio = response.choices[0].message.content
        cost = await _calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            model,
            is_batch=False,
        )
        
        logger.info(f"Generated realtime bio for {symbol} using {model}, cost: ${cost:.4f}")
        return bio.strip() if bio else None
        
    except Exception as e:
        logger.error(f"Failed to generate bio for {symbol}: {e}")
        return None


async def rate_dip_realtime(
    symbol: str,
    current_price: Optional[float] = None,
    ath_price: Optional[float] = None,
    dip_percentage: Optional[float] = None,
) -> Optional[dict]:
    """
    Rate a dip in real-time.
    
    Returns:
        Rating result dict or None if failed
    """
    client = await _get_client()
    if not client:
        return None
    
    model = await get_best_available_model()
    
    prompt = f"""Analyze this stock dip:

Stock: {symbol}
Current Price: ${current_price:.2f if current_price else 'N/A'}
All-Time High: ${ath_price:.2f if ath_price else 'N/A'}
Dip from ATH: {dip_percentage:.1f if dip_percentage else 'N/A'}%

Provide your analysis as JSON with rating, reasoning, and confidence."""
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["dip_rating"]},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        cost = await _calculate_cost(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            model,
            is_batch=False,
        )
        
        if content:
            result = json.loads(content)
            logger.info(f"Generated realtime rating for {symbol}: {result.get('rating')}, cost: ${cost:.4f}")
            return result
        
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse AI response for {symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to rate dip for {symbol}: {e}")
        return None


# --- Batch API (for weekly updates) ---

async def create_batch_job(
    job_type: BatchJobType,
    requests: list[dict[str, Any]],
) -> Optional[str]:
    """
    Create a batch job for processing multiple items.
    
    Args:
        job_type: Type of batch job
        requests: List of request items with custom_id and data
        
    Returns:
        Batch job ID or None if failed
    """
    client = await _get_client()
    if not client:
        return None
    
    if not requests:
        logger.warning(f"No requests for batch job {job_type}")
        return None
    
    model = await get_best_available_model()
    
    # Build JSONL content for batch
    jsonl_lines = []
    for req in requests:
        custom_id = req.get("custom_id", f"{job_type.value}_{req.get('symbol', 'unknown')}")
        
        if job_type == BatchJobType.DIP_ANALYSIS:
            prompt = f"""Analyze this stock dip:

Stock: {req.get('symbol')}
Current Price: ${req.get('current_price', 'N/A')}
All-Time High: ${req.get('ath_price', 'N/A')}
Dip from ATH: {req.get('dip_percentage', 'N/A')}%

Provide your analysis as JSON with:
- bio: A witty Tinder-style bio (2-3 sentences)
- rating: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
- reasoning: 2-3 sentence explanation"""
            
            system_prompt = SYSTEM_PROMPTS["dip_bio"] + "\n\n" + SYSTEM_PROMPTS["dip_rating"]
            
        elif job_type == BatchJobType.SUGGESTION_BIO:
            prompt = f"""Write a Tinder-style bio for this stock suggestion:

Stock: {req.get('symbol')}
Company: {req.get('company_name', 'Unknown')}
Why suggested: {req.get('reason', 'No reason provided')}"""
            
            system_prompt = SYSTEM_PROMPTS["suggestion_bio"]
        else:
            continue
        
        batch_request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 300,
                "temperature": 0.7,
            },
        }
        
        jsonl_lines.append(json.dumps(batch_request))
    
    if not jsonl_lines:
        return None
    
    # Create temp file with JSONL
    jsonl_content = "\n".join(jsonl_lines)
    
    try:
        # Upload batch file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            temp_path = f.name
        
        with open(temp_path, 'rb') as f:
            batch_file = await client.files.create(
                file=f,
                purpose="batch",
            )
        
        # Create batch job
        batch = await client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "job_type": job_type.value,
                "model": model,
                "item_count": str(len(requests)),
                "created_at": datetime.utcnow().isoformat(),
            },
        )
        
        # Cleanup temp file
        Path(temp_path).unlink(missing_ok=True)
        
        logger.info(f"Created batch job {batch.id} for {job_type.value} with {len(requests)} items using {model}")
        return batch.id
        
    except Exception as e:
        logger.error(f"Failed to create batch job: {e}")
        return None


async def check_batch_status(batch_id: str) -> Optional[dict]:
    """Check status of a batch job."""
    client = await _get_client()
    if not client:
        return None
    
    try:
        batch = await client.batches.retrieve(batch_id)
        
        return {
            "batch_id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "completed_at": batch.completed_at,
            "failed_at": batch.failed_at,
            "expired_at": batch.expired_at,
            "request_counts": {
                "total": batch.request_counts.total if batch.request_counts else 0,
                "completed": batch.request_counts.completed if batch.request_counts else 0,
                "failed": batch.request_counts.failed if batch.request_counts else 0,
            },
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
        }
        
    except Exception as e:
        logger.error(f"Failed to check batch status: {e}")
        return None


async def retrieve_batch_results(batch_id: str) -> Optional[list[dict]]:
    """Retrieve results from a completed batch job."""
    client = await _get_client()
    if not client:
        return None
    
    try:
        # Get batch info
        batch = await client.batches.retrieve(batch_id)
        
        if batch.status != "completed" or not batch.output_file_id:
            logger.warning(f"Batch {batch_id} not ready: status={batch.status}")
            return None
        
        # Download output file
        output_content = await client.files.content(batch.output_file_id)
        
        # Parse JSONL results
        results = []
        for line in output_content.text.strip().split("\n"):
            if line:
                results.append(json.loads(line))
        
        logger.info(f"Retrieved {len(results)} results from batch {batch_id}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to retrieve batch results: {e}")
        return None


async def get_pricing_info() -> dict[str, Any]:
    """Get current pricing info for admin dashboard."""
    global _pricing_cache, _cache_timestamp
    
    if not _pricing_cache:
        await _load_pricing_cache()
    
    # Get available models
    models = await get_available_models()
    
    return {
        "preferred_model": PREFERRED_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "available_models": [m["id"] for m in models if "gpt" in m["id"].lower()],
        "pricing": _pricing_cache,
        "cache_timestamp": _cache_timestamp.isoformat() if _cache_timestamp else None,
        "cache_ttl_days": CACHE_TTL_DAYS,
    }
