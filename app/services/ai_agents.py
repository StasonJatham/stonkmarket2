"""
AI Agents Service - Investment analysis using AI personas.

Each agent represents a famous investor's philosophy:
- Warren Buffett: Value investing, moat analysis
- Peter Lynch: Growth at reasonable price (GARP)
- Cathie Wood: Disruptive innovation, high growth
- Michael Burry: Contrarian deep value

The service:
1. Fetches fundamentals and price data from our DB (via yfinance)
2. Runs each agent's analysis using OpenAI
3. Stores results for frontend display
4. Aggregates signals into an overall recommendation

Usage:
    from app.services.ai_agents import run_agent_analysis
    
    # Analyze a single stock
    result = await run_agent_analysis("AAPL")
    
    # Analyze all tracked stocks
    results = await run_all_agent_analyses()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import AiAgentAnalysis, AnalysisVersion, Symbol
from app.services import stock_info
from app.services.fundamentals import get_fundamentals_for_analysis
from app.services.openai_client import TaskType, generate


logger = get_logger("ai_agents")


def _compute_input_hash(fundamentals: dict[str, Any], stock_data: dict[str, Any]) -> str:
    """
    Compute hash of input data for change detection.
    
    Combines fundamentals and stock info into a single hash.
    Used to skip AI analysis if input data hasn't changed.
    
    IMPORTANT: Include ALL fields used in _format_metrics_for_prompt()
    to ensure cache invalidation when any input changes.
    """
    # Combine ALL data points used in prompts for accurate change detection
    key_data = {
        "fundamentals": {
            k: v for k, v in fundamentals.items()
            if v is not None and k in (
                # Valuation metrics
                "pe_ratio", "forward_pe", "peg_ratio", "price_to_book", "ev_to_ebitda",
                # Profitability metrics
                "profit_margin", "operating_margin", "return_on_equity", "return_on_assets",
                # Financial health
                "debt_to_equity", "current_ratio", "free_cash_flow",
                # Growth metrics
                "revenue_growth", "earnings_growth",
                # Risk metrics
                "beta", "short_percent_of_float",
                # Analyst data
                "recommendation", "target_mean_price",
            )
        },
        "stock": {
            k: v for k, v in stock_data.items()
            if v is not None and k in ("name", "sector", "industry")
        },
    }
    content = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def _get_stored_input_hash(symbol: str) -> str | None:
    """Get the stored input hash for a symbol's agent analysis.
    
    Only returns the hash if the symbol actually has valid (non-expired)
    analysis in ai_agent_analysis. This prevents skipping symbols where
    a batch was submitted but never completed.
    """
    async with get_session() as session:
        # First check if symbol has valid analysis
        has_analysis = await session.execute(
            select(AiAgentAnalysis.symbol).where(
                AiAgentAnalysis.symbol == symbol,
                AiAgentAnalysis.expires_at > datetime.now(UTC),
            )
        )
        if not has_analysis.scalar_one_or_none():
            # No valid analysis - don't trust stored hash
            return None
            
        result = await session.execute(
            select(AnalysisVersion.input_version_hash).where(
                AnalysisVersion.symbol == symbol,
                AnalysisVersion.analysis_type == "agent_analysis",
            )
        )
        row = result.scalar_one_or_none()
        return row if row else None


async def _store_analysis_version(
    symbol: str,
    input_hash: str,
    batch_job_id: str | None = None,
) -> None:
    """Store/update the analysis version after successful analysis."""
    expires_at = datetime.now(UTC) + timedelta(days=7)

    async with get_session() as session:
        stmt = insert(AnalysisVersion).values(
            symbol=symbol,
            analysis_type="agent_analysis",
            input_version_hash=input_hash,
            generated_at=datetime.now(UTC),
            expires_at=expires_at,
            batch_job_id=batch_job_id,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_analysis_version_symbol_type",
            set_={
                "input_version_hash": stmt.excluded.input_version_hash,
                "generated_at": stmt.excluded.generated_at,
                "expires_at": stmt.excluded.expires_at,
                "batch_job_id": stmt.excluded.batch_job_id,
            },
        )
        await session.execute(stmt)
        await session.commit()


# Import all personas from hedge_fund module - the authoritative source
from app.hedge_fund.agents.investor_persona import PERSONAS

# Build AGENTS dict from PERSONAS for backward compatibility
AGENTS = {
    key: {
        "name": persona.name,
        "philosophy": persona.philosophy,
        "focus": persona.focus_areas,
        "is_llm": True,  # LLM-based persona agents
    }
    for key, persona in PERSONAS.items()
}

# Calculation agents (non-LLM) - DISABLED for now
# These need proper data conversion to work with the hedge_fund Fundamentals schema
# which requires specific fields and numeric values (not formatted strings)
# TODO: Fix data conversion before re-enabling
# CALCULATION_AGENTS = {
#     "fundamentals": {...},
#     "technicals": {...},
#     ...
# }
# AGENTS.update(CALCULATION_AGENTS)

# Standard 5-value signal taxonomy (matches hedge_fund.schemas.Signal)
SignalType = Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]

# Mapping for legacy 3-value signals (backward compatibility)
LEGACY_SIGNAL_MAP: dict[str, SignalType] = {
    "bullish": "buy",
    "bearish": "sell",
    "neutral": "hold",
}


def _normalize_signal(signal: str) -> SignalType:
    """Normalize signal to standard 5-value taxonomy."""
    signal = signal.lower().strip()
    # Already standard format
    if signal in ("strong_buy", "buy", "hold", "sell", "strong_sell"):
        return signal  # type: ignore
    # Legacy 3-value format
    if signal in LEGACY_SIGNAL_MAP:
        return LEGACY_SIGNAL_MAP[signal]
    # Default to hold for unknown
    return "hold"


@dataclass
class AgentVerdict:
    """An individual agent's verdict on a stock."""
    agent_id: str
    agent_name: str
    signal: SignalType  # strong_buy | buy | hold | sell | strong_sell
    confidence: int  # 0-100
    reasoning: str
    key_factors: list[str]


@dataclass
class AgentAnalysisResult:
    """Complete agent analysis for a stock."""
    symbol: str
    verdicts: list[AgentVerdict]
    overall_signal: SignalType
    overall_confidence: int
    summary: str
    analyzed_at: datetime


def _format_metrics_for_prompt(fundamentals: dict[str, Any], stock_data: dict[str, Any]) -> str:
    """Format fundamentals and stock data into a concise prompt section."""
    lines = []

    # Basic info
    if stock_data.get("name"):
        lines.append(f"Company: {stock_data['name']}")
    if stock_data.get("sector"):
        lines.append(f"Sector: {stock_data['sector']}")
    if stock_data.get("industry"):
        lines.append(f"Industry: {stock_data['industry']}")

    # Valuation metrics
    valuation = []
    if fundamentals.get("pe_ratio"):
        valuation.append(f"P/E: {fundamentals['pe_ratio']}")
    if fundamentals.get("forward_pe"):
        valuation.append(f"Fwd P/E: {fundamentals['forward_pe']}")
    if fundamentals.get("peg_ratio"):
        valuation.append(f"PEG: {fundamentals['peg_ratio']}")
    if fundamentals.get("price_to_book"):
        valuation.append(f"P/B: {fundamentals['price_to_book']}")
    if fundamentals.get("ev_to_ebitda"):
        valuation.append(f"EV/EBITDA: {fundamentals['ev_to_ebitda']}")
    if valuation:
        lines.append(f"Valuation: {', '.join(valuation)}")

    # Profitability
    profit = []
    if fundamentals.get("profit_margin"):
        profit.append(f"Profit Margin: {fundamentals['profit_margin']}")
    if fundamentals.get("operating_margin"):
        profit.append(f"Op Margin: {fundamentals['operating_margin']}")
    if fundamentals.get("return_on_equity"):
        profit.append(f"ROE: {fundamentals['return_on_equity']}")
    if fundamentals.get("return_on_assets"):
        profit.append(f"ROA: {fundamentals['return_on_assets']}")
    if profit:
        lines.append(f"Profitability: {', '.join(profit)}")

    # Financial health
    health = []
    if fundamentals.get("debt_to_equity"):
        health.append(f"D/E: {fundamentals['debt_to_equity']}")
    if fundamentals.get("current_ratio"):
        health.append(f"Current: {fundamentals['current_ratio']}")
    if fundamentals.get("free_cash_flow"):
        health.append(f"FCF: {fundamentals['free_cash_flow']}")
    if health:
        lines.append(f"Financial Health: {', '.join(health)}")

    # Growth
    growth = []
    if fundamentals.get("revenue_growth"):
        growth.append(f"Revenue: {fundamentals['revenue_growth']}")
    if fundamentals.get("earnings_growth"):
        growth.append(f"Earnings: {fundamentals['earnings_growth']}")
    if growth:
        lines.append(f"Growth: {', '.join(growth)}")

    # Risk
    risk = []
    if fundamentals.get("beta"):
        risk.append(f"Beta: {fundamentals['beta']}")
    if fundamentals.get("short_percent_of_float"):
        risk.append(f"Short %: {fundamentals['short_percent_of_float']}")
    if risk:
        lines.append(f"Risk: {', '.join(risk)}")

    # Analyst consensus
    if fundamentals.get("recommendation"):
        lines.append(f"Analyst Consensus: {fundamentals['recommendation']}")
    if fundamentals.get("target_mean_price"):
        lines.append(f"Price Target: ${fundamentals['target_mean_price']}")

    return "\n".join(lines)


async def _run_calculation_agent(
    agent_id: str,
    symbol: str,
    fundamentals: dict[str, Any],
    stock_data: dict[str, Any],
) -> AgentVerdict | None:
    """Run a calculation-based agent from hedge_fund module."""
    from app.hedge_fund.agents import (
        get_fundamentals_agent,
        get_technicals_agent,
        get_valuation_agent,
        get_sentiment_agent,
        get_risk_agent,
    )
    from app.hedge_fund.schemas import Fundamentals, MarketData, PriceSeries

    agent_map = {
        "fundamentals": get_fundamentals_agent,
        "technicals": get_technicals_agent,
        "valuation": get_valuation_agent,
        "sentiment": get_sentiment_agent,
        "risk": get_risk_agent,
    }

    agent_factory = agent_map.get(agent_id)
    if not agent_factory:
        return None

    try:
        agent = agent_factory()

        # Build Fundamentals object from dict
        fund = Fundamentals(
            pe_ratio=fundamentals.get("pe_ratio"),
            forward_pe=fundamentals.get("forward_pe"),
            peg_ratio=fundamentals.get("peg_ratio"),
            price_to_book=fundamentals.get("price_to_book"),
            price_to_sales=fundamentals.get("price_to_sales"),
            ev_to_ebitda=fundamentals.get("ev_to_ebitda"),
            enterprise_value=fundamentals.get("enterprise_value"),
            market_cap=fundamentals.get("market_cap"),
            roe=fundamentals.get("return_on_equity"),
            roa=fundamentals.get("return_on_assets"),
            roic=fundamentals.get("roic"),
            profit_margin=fundamentals.get("profit_margin"),
            operating_margin=fundamentals.get("operating_margin"),
            gross_margin=fundamentals.get("gross_margin"),
            debt_to_equity=fundamentals.get("debt_to_equity"),
            current_ratio=fundamentals.get("current_ratio"),
            quick_ratio=fundamentals.get("quick_ratio"),
            interest_coverage=fundamentals.get("interest_coverage"),
            revenue_growth=fundamentals.get("revenue_growth"),
            earnings_growth=fundamentals.get("earnings_growth"),
            free_cash_flow=fundamentals.get("free_cash_flow"),
            fcf_yield=fundamentals.get("fcf_yield"),
            dividend_yield=fundamentals.get("dividend_yield"),
            payout_ratio=fundamentals.get("payout_ratio"),
            beta=fundamentals.get("beta"),
            shares_outstanding=fundamentals.get("shares_outstanding"),
            float_shares=fundamentals.get("float_shares"),
            insider_ownership=fundamentals.get("insider_ownership"),
            institutional_ownership=fundamentals.get("institutional_ownership"),
            short_ratio=fundamentals.get("short_ratio"),
            short_percent_of_float=fundamentals.get("short_percent_of_float"),
            analyst_rating=fundamentals.get("recommendation"),
            target_mean_price=fundamentals.get("target_mean_price"),
            target_high_price=fundamentals.get("target_high_price"),
            target_low_price=fundamentals.get("target_low_price"),
        )

        # Build MarketData object - minimal for now
        market_data = MarketData(
            symbol=symbol,
            current_price=stock_data.get("current_price") or stock_data.get("regularMarketPrice"),
            fundamentals=fund,
            prices=PriceSeries(symbol=symbol, prices=[]),  # Would need price history
        )

        # Run the calculation agent
        result = await agent.calculate(symbol, market_data)

        # Convert AgentSignal to AgentVerdict
        return AgentVerdict(
            agent_id=agent_id,
            agent_name=AGENTS[agent_id]["name"],
            signal=_normalize_signal(result.signal.value if hasattr(result.signal, 'value') else result.signal),
            confidence=int(result.confidence * 100) if result.confidence <= 1 else int(result.confidence),
            reasoning=result.reasoning,
            key_factors=result.key_factors or [],
        )

    except Exception as e:
        logger.warning(f"Calculation agent {agent_id} failed for {symbol}: {e}")
        return None


async def _run_single_agent(
    agent_id: str,
    symbol: str,
    metrics_text: str,
    fundamentals: dict[str, Any] | None = None,
    stock_data: dict[str, Any] | None = None,
) -> AgentVerdict | None:
    """Run a single agent's analysis - LLM or calculation based."""
    agent = AGENTS.get(agent_id)
    if not agent:
        return None

    # Check if this is a calculation agent (non-LLM)
    if not agent.get("is_llm", True):
        if fundamentals is None or stock_data is None:
            logger.warning(f"Cannot run calculation agent {agent_id} without fundamentals/stock_data")
            return None
        return await _run_calculation_agent(agent_id, symbol, fundamentals, stock_data)

    # LLM-based persona agent
    prompt = f"""You are {agent['name']}, the legendary investor. Analyze this stock using your investment philosophy.

YOUR PHILOSOPHY: {agent['philosophy']}
KEY FACTORS YOU FOCUS ON: {', '.join(agent['focus'])}

STOCK: {symbol}

FINANCIAL DATA:
{metrics_text}

Provide your analysis as JSON with these exact fields:
{{
    "rating": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
    "confidence": 1-10,
    "reasoning": "2-3 sentence explanation in your voice",
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Be specific about what the numbers tell you. If data is missing, factor that into your confidence level."""

    try:
        result = await generate(
            task=TaskType.RATING,  # Use rating task type for structured output
            context={"prompt": prompt},
            json_output=True,
            max_tokens=300,
        )

        if not result:
            return None

        # Parse the response
        if isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

        # Extract signal from "rating" field (RATING_SCHEMA uses "rating", not "signal")
        # Also support "signal" for backward compat with any legacy responses
        raw_signal = data.get("rating") or data.get("signal", "hold")
        signal = _normalize_signal(raw_signal)

        # Scale confidence from 1-10 to 0-100 (RATING_SCHEMA uses 1-10)
        raw_confidence = data.get("confidence", 5)
        confidence = min(100, max(0, int(raw_confidence) * 10))

        return AgentVerdict(
            agent_id=agent_id,
            agent_name=agent["name"],
            signal=signal,
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
        )
    except Exception as e:
        logger.warning(f"Agent {agent_id} failed for {symbol}: {e}")
        return None


def _aggregate_signals(verdicts: list[AgentVerdict]) -> tuple[SignalType, int, str]:
    """Aggregate individual agent verdicts into overall signal."""
    if not verdicts:
        return "hold", 0, "No agent analysis available"

    # Numeric scores for 5-value signals
    SIGNAL_SCORES = {
        "strong_buy": 2,
        "buy": 1,
        "hold": 0,
        "sell": -1,
        "strong_sell": -2,
    }

    # Weight signals by confidence
    weighted_sum = 0.0
    total_weight = 0.0
    total_confidence = 0

    bullish_agents = []  # strong_buy or buy
    bearish_agents = []  # sell or strong_sell

    for v in verdicts:
        weight = v.confidence / 100
        total_weight += weight
        total_confidence += v.confidence
        weighted_sum += SIGNAL_SCORES.get(v.signal, 0) * weight

        if v.signal in ("strong_buy", "buy"):
            bullish_agents.append(v.agent_name)
        elif v.signal in ("sell", "strong_sell"):
            bearish_agents.append(v.agent_name)

    # Determine overall signal from weighted average
    if total_weight > 0:
        avg_score = weighted_sum / total_weight
        if avg_score >= 1.5:
            overall_signal: SignalType = "strong_buy"
        elif avg_score >= 0.5:
            overall_signal = "buy"
        elif avg_score > -0.5:
            overall_signal = "hold"
        elif avg_score > -1.5:
            overall_signal = "sell"
        else:
            overall_signal = "strong_sell"
    else:
        overall_signal = "hold"

    # Calculate overall confidence
    overall_confidence = int(total_confidence / len(verdicts)) if verdicts else 0

    # Generate summary
    summary_parts = []
    if bullish_agents:
        summary_parts.append(f"Bullish: {', '.join(bullish_agents)}")
    if bearish_agents:
        summary_parts.append(f"Bearish: {', '.join(bearish_agents)}")

    summary = ". ".join(summary_parts) if summary_parts else "Mixed signals from analysts"

    return overall_signal, overall_confidence, summary


async def queue_agent_analysis(symbol: str, force: bool = False) -> str | None:
    """
    Queue agent analysis for a symbol via OpenAI Batch API.
    
    Sets agent_pending=True and returns batch_id.
    50% cheaper than real-time API calls.
    
    Args:
        symbol: Stock symbol to analyze
        force: Force re-analysis even if input unchanged
        
    Returns:
        batch_id if queued, None if skipped or failed
    """
    symbol = symbol.upper()
    
    # Check if already pending
    async with get_session() as session:
        result = await session.execute(
            select(AiAgentAnalysis.agent_pending).where(AiAgentAnalysis.symbol == symbol)
        )
        row = result.scalar_one_or_none()
        if row is True and not force:
            logger.info(f"Agent analysis already pending for {symbol}")
            return None
    
    # Submit batch
    batch_result = await submit_agent_batch([symbol])
    if not batch_result:
        logger.warning(f"Failed to queue agent analysis for {symbol}")
        return None
    
    batch_id, _ = batch_result
    
    # Mark as pending in DB (create placeholder if needed)
    async with get_session() as session:
        stmt = insert(AiAgentAnalysis).values(
            symbol=symbol,
            verdicts=[],
            overall_signal="hold",
            overall_confidence=0,
            summary="Analysis pending...",
            analyzed_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(days=7),
            agent_pending=True,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={"agent_pending": True},
        )
        await session.execute(stmt)
        await session.commit()
    
    logger.info(f"Queued agent analysis for {symbol}: batch {batch_id}")
    return batch_id


async def run_agent_analysis(
    symbol: str,
    agents: list[str] | None = None,
    store_result: bool = True,
    force: bool = False,
) -> AgentAnalysisResult | None:
    """
    Run AI agent analysis on a single stock.
    
    Uses input version tracking to skip analysis if data hasn't changed.
    
    Args:
        symbol: Stock symbol
        agents: List of agent IDs to run (default: all agents)
        store_result: Whether to store result in database
        force: Force re-analysis even if input hasn't changed
        
    Returns:
        AgentAnalysisResult with all verdicts and aggregated signal,
        or None if skipped or failed
    """
    symbol = symbol.upper()
    logger.info(f"Running agent analysis for {symbol}")

    # Get fundamentals
    fundamentals = await get_fundamentals_for_analysis(symbol)
    if not fundamentals:
        logger.warning(f"No fundamentals available for {symbol}")
        fundamentals = {}

    # Get stock info
    info = await stock_info.get_stock_info_async(symbol)
    stock_data = info if info else {}

    # Format metrics for prompts
    metrics_text = _format_metrics_for_prompt(fundamentals, stock_data)

    if not metrics_text.strip():
        logger.warning(f"No data available for {symbol}")
        return None

    # Compute input hash for version checking
    input_hash = _compute_input_hash(fundamentals, stock_data)

    # Check if we can skip (input unchanged)
    if not force:
        stored_hash = await _get_stored_input_hash(symbol)
        if stored_hash == input_hash:
            logger.info(f"Skipping {symbol}: input data unchanged (hash={input_hash[:8]})")
            return None  # Signal that we skipped

    # Run agents
    agent_ids = agents or list(AGENTS.keys())
    verdicts = []

    for agent_id in agent_ids:
        verdict = await _run_single_agent(agent_id, symbol, metrics_text, fundamentals, stock_data)
        if verdict:
            verdicts.append(verdict)

    if not verdicts:
        logger.warning(f"No agent verdicts generated for {symbol}")
        return None

    # Aggregate signals
    overall_signal, overall_confidence, summary = _aggregate_signals(verdicts)

    result = AgentAnalysisResult(
        symbol=symbol,
        verdicts=verdicts,
        overall_signal=overall_signal,
        overall_confidence=overall_confidence,
        summary=summary,
        analyzed_at=datetime.now(UTC),
    )

    # Store in database
    if store_result:
        await _store_agent_analysis(result)
        # Store the input hash for future change detection
        await _store_analysis_version(symbol, input_hash)

    logger.info(f"Agent analysis complete for {symbol}: {overall_signal} ({overall_confidence}%)")
    return result


async def _store_agent_analysis(result: AgentAnalysisResult, agent_pending: bool = False) -> None:
    """Store agent analysis result in database."""
    # Serialize verdicts to JSON
    verdicts_json = [
        {
            "agent_id": v.agent_id,
            "agent_name": v.agent_name,
            "signal": v.signal,
            "confidence": v.confidence,
            "reasoning": v.reasoning,
            "key_factors": v.key_factors,
        }
        for v in result.verdicts
    ]

    expires_at = datetime.now(UTC) + timedelta(days=7)

    async with get_session() as session:
        stmt = insert(AiAgentAnalysis).values(
            symbol=result.symbol,
            verdicts=verdicts_json,
            overall_signal=result.overall_signal,
            overall_confidence=result.overall_confidence,
            summary=result.summary,
            analyzed_at=result.analyzed_at,
            expires_at=expires_at,
            agent_pending=agent_pending,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "verdicts": stmt.excluded.verdicts,
                "overall_signal": stmt.excluded.overall_signal,
                "overall_confidence": stmt.excluded.overall_confidence,
                "summary": stmt.excluded.summary,
                "analyzed_at": stmt.excluded.analyzed_at,
                "expires_at": stmt.excluded.expires_at,
                "agent_pending": stmt.excluded.agent_pending,
            },
        )
        await session.execute(stmt)
        await session.commit()


async def get_agent_analysis(symbol: str, max_age_hours: int = 168) -> dict[str, Any] | None:
    """
    Get stored agent analysis for a symbol.
    
    Args:
        symbol: Stock symbol
        max_age_hours: Max age in hours (default 7 days)
        
    Returns:
        Dict with analysis data or None if not found/expired
    """
    symbol = symbol.upper()

    async with get_session() as session:
        result = await session.execute(
            select(AiAgentAnalysis).where(
                AiAgentAnalysis.symbol == symbol,
                AiAgentAnalysis.expires_at > datetime.now(UTC),
            )
        )
        row = result.scalar_one_or_none()

    if not row:
        return None

    # Parse verdicts JSON
    verdicts = row.verdicts
    if isinstance(verdicts, str):
        verdicts = json.loads(verdicts)

    return {
        "symbol": row.symbol,
        "verdicts": verdicts,
        "overall_signal": row.overall_signal,
        "overall_confidence": row.overall_confidence,
        "summary": row.summary,
        "analyzed_at": row.analyzed_at.isoformat() if row.analyzed_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "agent_pending": row.agent_pending,
    }


async def get_symbols_needing_analysis() -> list[str]:
    """Get symbols that need agent analysis (new or expired)."""
    from sqlalchemy import exists, not_

    async with get_session() as session:
        # Subquery to check for non-expired analysis
        has_valid_analysis = exists(
            select(AiAgentAnalysis.symbol).where(
                AiAgentAnalysis.symbol == Symbol.symbol,
                AiAgentAnalysis.expires_at > datetime.now(UTC),
            )
        )

        result = await session.execute(
            select(Symbol.symbol)
            .where(
                Symbol.is_active == True,
                not_(has_valid_analysis),
            )
            .order_by(Symbol.symbol)
        )
        return [r[0] for r in result.all()]


async def run_all_agent_analyses(force: bool = False) -> dict[str, Any]:
    """
    Run agent analysis for all symbols needing it.
    
    Uses input version checking to skip symbols whose input data
    hasn't changed since last analysis.
    
    Args:
        force: Force re-analysis even if input hasn't changed
    
    Returns:
        Dict with counts of analyzed/skipped/failed symbols
    """
    symbols = await get_symbols_needing_analysis()

    if not symbols:
        logger.info("No symbols need agent analysis")
        return {"analyzed": 0, "skipped": 0, "failed": 0, "symbols": []}

    logger.info(f"Running agent analysis for {len(symbols)} symbols")

    analyzed = []
    skipped = []
    failed = []

    for symbol in symbols:
        try:
            result = await run_agent_analysis(symbol, force=force)
            if result:
                analyzed.append(symbol)
            else:
                # None result means skipped (input unchanged) or no data
                skipped.append(symbol)
        except Exception as e:
            logger.error(f"Agent analysis failed for {symbol}: {e}")
            failed.append(symbol)

    return {
        "analyzed": len(analyzed),
        "skipped": len(skipped),
        "failed": len(failed),
        "symbols": analyzed,
        "skipped_symbols": skipped,
        "failed_symbols": failed,
    }


# =============================================================================
# BATCH API INTEGRATION
# =============================================================================


def _build_agent_prompt(agent_id: str, symbol: str, metrics_text: str) -> str:
    """Build prompt for a single agent analysis."""
    agent = AGENTS.get(agent_id)
    if not agent:
        return ""

    return f"""You are {agent['name']}, the legendary investor. Analyze this stock using your investment philosophy.

YOUR PHILOSOPHY: {agent['philosophy']}
KEY FACTORS YOU FOCUS ON: {', '.join(agent['focus'])}

STOCK: {symbol}

FINANCIAL DATA:
{metrics_text}

Provide your analysis as JSON with these exact fields:
{{
    "rating": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
    "confidence": 1-10,
    "reasoning": "2-3 sentence explanation in your voice",
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Be specific about what the numbers tell you. If data is missing, factor that into your confidence level."""


async def prepare_batch_items(symbols: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Prepare batch items for agent analysis.
    
    For each symbol, creates 5 items (one per agent).
    Returns items and a map of symbol -> input_hash for version tracking.
    
    Args:
        symbols: List of symbols to analyze
        
    Returns:
        Tuple of (batch_items, input_hashes)
        - batch_items: List of dicts with prompt, symbol, agent_id
        - input_hashes: Map of symbol -> computed input hash
    """
    items = []
    input_hashes: dict[str, str] = {}

    for symbol in symbols:
        symbol = symbol.upper()

        # Get fundamentals
        fundamentals = await get_fundamentals_for_analysis(symbol)
        if not fundamentals:
            fundamentals = {}

        # Get stock info
        info = await stock_info.get_stock_info_async(symbol)
        stock_data = info if info else {}

        # Format metrics
        metrics_text = _format_metrics_for_prompt(fundamentals, stock_data)

        if not metrics_text.strip():
            logger.warning(f"No data for {symbol}, skipping batch item")
            continue

        # Compute and store input hash
        input_hash = _compute_input_hash(fundamentals, stock_data)
        input_hashes[symbol] = input_hash

        # Create item for each agent
        for agent_id in AGENTS:
            prompt = _build_agent_prompt(agent_id, symbol, metrics_text)
            items.append({
                "symbol": symbol,
                "agent_id": agent_id,
                "prompt": prompt,
                "input_hash": input_hash,
            })

    return items, input_hashes


async def submit_agent_batch(symbols: list[str]) -> tuple[str, dict[str, str]] | None:
    """
    Submit a batch job for agent analysis on multiple symbols.
    
    Each symbol gets 5 prompts (one per agent). Batch API provides
    50% cost savings compared to real-time API.
    
    Args:
        symbols: List of symbols to analyze
        
    Returns:
        Tuple of (batch_job_id, input_hashes) or None on failure
    """
    from app.services.openai_client import TaskType, submit_batch

    # Prepare batch items
    items, input_hashes = await prepare_batch_items(symbols)

    if not items:
        logger.warning("No valid items for batch")
        return None

    logger.info(f"Submitting batch with {len(items)} items for {len(input_hashes)} symbols")

    # Submit using AGENT task type for proper batch routing
    batch_id = await submit_batch(
        task=TaskType.AGENT,  # Use AGENT for persona analysis batches
        items=items,
    )

    if not batch_id:
        return None

    # Store batch job ID for each symbol
    for symbol, input_hash in input_hashes.items():
        await _store_analysis_version(symbol, input_hash, batch_job_id=batch_id)

    return batch_id, input_hashes


async def collect_agent_batch(batch_id: str) -> dict[str, AgentAnalysisResult]:
    """
    Collect results from a completed agent analysis batch.
    
    Processes the batch results, aggregates verdicts per symbol,
    and stores the final analysis in the database.
    
    Args:
        batch_id: The batch job ID
        
    Returns:
        Dict of symbol -> AgentAnalysisResult
    """
    from app.services.openai_client import collect_batch

    results = await collect_batch(batch_id)
    if not results:
        return {}

    # Group results by symbol
    symbol_verdicts: dict[str, list[AgentVerdict]] = {}

    for item in results:
        if item.get("failed"):
            logger.warning(f"Batch item failed: {item.get('custom_id')}")
            continue

        custom_id = item.get("custom_id", "")
        result_data = item.get("result")

        # Parse custom_id format: "batch_run_id:symbol:agent_id:task"
        # Colon-delimited to avoid ambiguity with multi-part agent IDs like "warren_buffett"
        parts = custom_id.split(":")
        if len(parts) >= 3:
            # parts[0] = batch_run_id, parts[1] = symbol, parts[2] = agent_id
            symbol = parts[1]
            agent_id = parts[2]
        else:
            # Legacy underscore format fallback: "agent_{agent_id}_{symbol}_{batch_run_id}"
            underscore_parts = custom_id.split("_")
            if len(underscore_parts) >= 4 and underscore_parts[0] == "agent":
                # Find the symbol by looking for uppercase after the agent parts
                symbol = None
                for i, part in enumerate(underscore_parts[2:], start=2):
                    if part.isupper() or (part.isalnum() and part[0].isupper()):
                        symbol = part
                        agent_id = "_".join(underscore_parts[1:i])
                        break
                if not symbol:
                    logger.warning(f"Could not parse symbol from custom_id: {custom_id}")
                    continue
            else:
                logger.warning(f"Unknown custom_id format: {custom_id}")
                continue

        if not result_data or not isinstance(result_data, dict):
            continue

        # Get agent info
        agent = AGENTS.get(agent_id)
        agent_name = agent["name"] if agent else agent_id

        # Initialize symbol's verdict list
        if symbol not in symbol_verdicts:
            symbol_verdicts[symbol] = []

        # Build verdict from result - use "rating" field (RATING_SCHEMA) with "signal" fallback
        raw_signal = result_data.get("rating") or result_data.get("signal", "hold")
        signal = _normalize_signal(raw_signal)

        # Scale confidence from 1-10 to 0-100
        raw_confidence = result_data.get("confidence", 5)
        confidence = min(100, max(0, int(raw_confidence) * 10))

        verdict = AgentVerdict(
            agent_id=agent_id,
            agent_name=agent_name,
            signal=signal,
            confidence=confidence,
            reasoning=result_data.get("reasoning", ""),
            key_factors=result_data.get("key_factors", []),
        )
        symbol_verdicts[symbol].append(verdict)

    # Build final results
    final_results: dict[str, AgentAnalysisResult] = {}

    for symbol, verdicts in symbol_verdicts.items():
        if not verdicts:
            continue

        overall_signal, overall_confidence, summary = _aggregate_signals(verdicts)

        result = AgentAnalysisResult(
            symbol=symbol,
            verdicts=verdicts,
            overall_signal=overall_signal,
            overall_confidence=overall_confidence,
            summary=summary,
            analyzed_at=datetime.now(UTC),
        )

        # Store in database and clear pending flag
        await _store_agent_analysis(result, agent_pending=False)
        final_results[symbol] = result

    return final_results


async def run_all_agent_analyses_batch() -> dict[str, Any]:
    """
    Run agent analysis for all symbols using Batch API.
    
    This is the batch version of run_all_agent_analyses().
    Provides ~50% cost savings by using OpenAI's Batch API.
    
    Workflow:
    1. Get symbols needing analysis
    2. Check input hashes to skip unchanged symbols
    3. Submit batch job for remaining symbols
    4. Return batch_id for later polling
    
    Returns:
        Dict with batch_id, symbol counts, etc.
    """
    symbols = await get_symbols_needing_analysis()

    if not symbols:
        logger.info("No symbols need agent analysis")
        return {
            "batch_id": None,
            "submitted": 0,
            "skipped": 0,
            "message": "No symbols need analysis",
        }

    logger.info(f"Preparing batch analysis for {len(symbols)} symbols")

    # Filter symbols where input has changed
    symbols_to_analyze = []
    skipped = []

    for symbol in symbols:
        symbol = symbol.upper()

        # Get current data to compute hash
        fundamentals = await get_fundamentals_for_analysis(symbol)
        info = await stock_info.get_stock_info_async(symbol)

        if not fundamentals and not info:
            skipped.append(symbol)
            continue

        current_hash = _compute_input_hash(fundamentals or {}, info or {})
        stored_hash = await _get_stored_input_hash(symbol)

        if stored_hash == current_hash:
            logger.debug(f"Skipping {symbol}: input unchanged")
            skipped.append(symbol)
        else:
            symbols_to_analyze.append(symbol)

    if not symbols_to_analyze:
        return {
            "batch_id": None,
            "submitted": 0,
            "skipped": len(skipped),
            "message": "All symbols have unchanged input data",
        }

    # Submit batch
    result = await submit_agent_batch(symbols_to_analyze)

    if not result:
        return {
            "batch_id": None,
            "submitted": 0,
            "skipped": len(skipped),
            "error": "Failed to submit batch",
        }

    batch_id, input_hashes = result

    return {
        "batch_id": batch_id,
        "submitted": len(symbols_to_analyze),
        "skipped": len(skipped),
        "symbols": symbols_to_analyze,
        "skipped_symbols": skipped,
        "message": f"Batch {batch_id} submitted with {len(symbols_to_analyze)} symbols",
    }


# Export agent info for frontend
def get_agent_info() -> list[dict[str, Any]]:
    """Get info about all available agents for frontend display."""
    return [
        {
            "id": agent_id,
            "name": agent["name"],
            "philosophy": agent["philosophy"],
            "focus": agent["focus"],
        }
        for agent_id, agent in AGENTS.items()
    ]
