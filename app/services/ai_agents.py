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

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Literal
from dataclasses import dataclass

from app.core.logging import get_logger
from app.database.connection import fetch_one, fetch_all, execute
from app.services.fundamentals import get_fundamentals, get_fundamentals_for_analysis
from app.services import stock_info
from app.services.openai_client import generate, TaskType

logger = get_logger("ai_agents")

# Agent definitions with their investment philosophy
AGENTS = {
    "warren_buffett": {
        "name": "Warren Buffett",
        "philosophy": "Value investing focused on companies with strong moats, consistent earnings, and fair valuations",
        "focus": ["ROE", "debt levels", "profit margins", "consistent earnings", "intrinsic value"],
    },
    "peter_lynch": {
        "name": "Peter Lynch",
        "philosophy": "Growth at a Reasonable Price (GARP) - finding undervalued growth companies",
        "focus": ["PEG ratio", "earnings growth", "revenue growth", "market opportunity"],
    },
    "cathie_wood": {
        "name": "Cathie Wood",
        "philosophy": "Disruptive innovation investing - betting on transformative technologies",
        "focus": ["innovation potential", "market disruption", "growth trajectory", "technology moat"],
    },
    "michael_burry": {
        "name": "Michael Burry",
        "philosophy": "Deep value and contrarian investing - finding opportunities others miss",
        "focus": ["balance sheet strength", "cash flow", "undervaluation", "market sentiment"],
    },
    "ben_graham": {
        "name": "Ben Graham",
        "philosophy": "Defensive value investing - focus on margin of safety and asset protection",
        "focus": ["net current assets", "earnings stability", "dividend record", "moderate P/E"],
    },
}

SignalType = Literal["bullish", "bearish", "neutral"]


@dataclass
class AgentVerdict:
    """An individual agent's verdict on a stock."""
    agent_id: str
    agent_name: str
    signal: SignalType
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


async def _run_single_agent(
    agent_id: str,
    symbol: str,
    metrics_text: str,
) -> Optional[AgentVerdict]:
    """Run a single agent's analysis using OpenAI."""
    agent = AGENTS.get(agent_id)
    if not agent:
        return None
    
    prompt = f"""You are {agent['name']}, the legendary investor. Analyze this stock using your investment philosophy.

YOUR PHILOSOPHY: {agent['philosophy']}
KEY FACTORS YOU FOCUS ON: {', '.join(agent['focus'])}

STOCK: {symbol}

FINANCIAL DATA:
{metrics_text}

Provide your analysis as JSON with these exact fields:
{{
    "signal": "bullish" | "bearish" | "neutral",
    "confidence": 0-100,
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
        
        return AgentVerdict(
            agent_id=agent_id,
            agent_name=agent["name"],
            signal=data.get("signal", "neutral"),
            confidence=min(100, max(0, int(data.get("confidence", 50)))),
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
        )
    except Exception as e:
        logger.warning(f"Agent {agent_id} failed for {symbol}: {e}")
        return None


def _aggregate_signals(verdicts: list[AgentVerdict]) -> tuple[SignalType, int, str]:
    """Aggregate individual agent verdicts into overall signal."""
    if not verdicts:
        return "neutral", 0, "No agent analysis available"
    
    # Weight signals by confidence
    bullish_score = 0
    bearish_score = 0
    neutral_score = 0
    total_confidence = 0
    
    for v in verdicts:
        weight = v.confidence / 100
        total_confidence += v.confidence
        if v.signal == "bullish":
            bullish_score += weight
        elif v.signal == "bearish":
            bearish_score += weight
        else:
            neutral_score += weight
    
    # Determine overall signal
    max_score = max(bullish_score, bearish_score, neutral_score)
    if max_score == bullish_score and bullish_score > bearish_score + neutral_score * 0.5:
        overall_signal: SignalType = "bullish"
    elif max_score == bearish_score and bearish_score > bullish_score + neutral_score * 0.5:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"
    
    # Calculate overall confidence
    overall_confidence = int(total_confidence / len(verdicts)) if verdicts else 0
    
    # Generate summary
    bullish_agents = [v.agent_name for v in verdicts if v.signal == "bullish"]
    bearish_agents = [v.agent_name for v in verdicts if v.signal == "bearish"]
    
    summary_parts = []
    if bullish_agents:
        summary_parts.append(f"Bullish: {', '.join(bullish_agents)}")
    if bearish_agents:
        summary_parts.append(f"Bearish: {', '.join(bearish_agents)}")
    
    summary = ". ".join(summary_parts) if summary_parts else "Mixed signals from analysts"
    
    return overall_signal, overall_confidence, summary


async def run_agent_analysis(
    symbol: str,
    agents: Optional[list[str]] = None,
    store_result: bool = True,
) -> Optional[AgentAnalysisResult]:
    """
    Run AI agent analysis on a single stock.
    
    Args:
        symbol: Stock symbol
        agents: List of agent IDs to run (default: all agents)
        store_result: Whether to store result in database
        
    Returns:
        AgentAnalysisResult with all verdicts and aggregated signal
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
    
    # Run agents
    agent_ids = agents or list(AGENTS.keys())
    verdicts = []
    
    for agent_id in agent_ids:
        verdict = await _run_single_agent(agent_id, symbol, metrics_text)
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
        analyzed_at=datetime.now(timezone.utc),
    )
    
    # Store in database
    if store_result:
        await _store_agent_analysis(result)
    
    logger.info(f"Agent analysis complete for {symbol}: {overall_signal} ({overall_confidence}%)")
    return result


async def _store_agent_analysis(result: AgentAnalysisResult) -> None:
    """Store agent analysis result in database."""
    # Serialize verdicts to JSON
    verdicts_json = json.dumps([
        {
            "agent_id": v.agent_id,
            "agent_name": v.agent_name,
            "signal": v.signal,
            "confidence": v.confidence,
            "reasoning": v.reasoning,
            "key_factors": v.key_factors,
        }
        for v in result.verdicts
    ])
    
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    
    await execute(
        """
        INSERT INTO ai_agent_analysis (
            symbol, verdicts, overall_signal, overall_confidence,
            summary, analyzed_at, expires_at
        )
        VALUES ($1, $2::jsonb, $3, $4, $5, $6, $7)
        ON CONFLICT (symbol) DO UPDATE SET
            verdicts = EXCLUDED.verdicts,
            overall_signal = EXCLUDED.overall_signal,
            overall_confidence = EXCLUDED.overall_confidence,
            summary = EXCLUDED.summary,
            analyzed_at = EXCLUDED.analyzed_at,
            expires_at = EXCLUDED.expires_at
        """,
        result.symbol,
        verdicts_json,
        result.overall_signal,
        result.overall_confidence,
        result.summary,
        result.analyzed_at,
        expires_at,
    )


async def get_agent_analysis(symbol: str, max_age_hours: int = 168) -> Optional[dict[str, Any]]:
    """
    Get stored agent analysis for a symbol.
    
    Args:
        symbol: Stock symbol
        max_age_hours: Max age in hours (default 7 days)
        
    Returns:
        Dict with analysis data or None if not found/expired
    """
    symbol = symbol.upper()
    
    row = await fetch_one(
        """
        SELECT symbol, verdicts, overall_signal, overall_confidence,
               summary, analyzed_at, expires_at
        FROM ai_agent_analysis
        WHERE symbol = $1 AND expires_at > NOW()
        """,
        symbol,
    )
    
    if not row:
        return None
    
    # Parse verdicts JSON
    verdicts = row["verdicts"]
    if isinstance(verdicts, str):
        verdicts = json.loads(verdicts)
    
    return {
        "symbol": row["symbol"],
        "verdicts": verdicts,
        "overall_signal": row["overall_signal"],
        "overall_confidence": row["overall_confidence"],
        "summary": row["summary"],
        "analyzed_at": row["analyzed_at"].isoformat() if row["analyzed_at"] else None,
        "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
    }


async def get_symbols_needing_analysis() -> list[str]:
    """Get symbols that need agent analysis (new or expired)."""
    rows = await fetch_all(
        """
        SELECT s.symbol
        FROM symbols s
        WHERE s.is_active = TRUE
          AND NOT EXISTS (
              SELECT 1 FROM ai_agent_analysis a
              WHERE a.symbol = s.symbol
                AND a.expires_at > NOW()
          )
        ORDER BY s.symbol
        """,
    )
    return [r["symbol"] for r in rows]


async def run_all_agent_analyses() -> dict[str, Any]:
    """
    Run agent analysis for all symbols needing it.
    
    Returns:
        Dict with counts of analyzed/failed symbols
    """
    symbols = await get_symbols_needing_analysis()
    
    if not symbols:
        logger.info("No symbols need agent analysis")
        return {"analyzed": 0, "failed": 0, "symbols": []}
    
    logger.info(f"Running agent analysis for {len(symbols)} symbols")
    
    analyzed = []
    failed = []
    
    for symbol in symbols:
        try:
            result = await run_agent_analysis(symbol)
            if result:
                analyzed.append(symbol)
            else:
                failed.append(symbol)
        except Exception as e:
            logger.error(f"Agent analysis failed for {symbol}: {e}")
            failed.append(symbol)
    
    return {
        "analyzed": len(analyzed),
        "failed": len(failed),
        "symbols": analyzed,
        "failed_symbols": failed,
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
