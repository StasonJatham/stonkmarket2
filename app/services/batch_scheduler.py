"""Batch job scheduler for AI analysis updates."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import (
    AnalysisVersion,
    BatchJob,
    DipAIAnalysis,
    DipfinderSignal,
    DipState,
    StockFundamentals,
    Symbol,
)
from app.repositories import api_usage_orm as api_usage
from app.services.openai_client import (
    TaskType,
    check_batch,
    collect_batch,
    generate,
    submit_batch,
)
from app.services.statistical_rating import calculate_rating


logger = get_logger("batch_scheduler")


# =============================================================================
# INPUT HASH FUNCTIONS FOR DIP AI ANALYSIS
# =============================================================================


def _compute_dip_input_hash(dip_data: dict[str, Any]) -> str:
    """
    Compute hash of dip data for change detection.
    
    Used to skip AI analysis if input data hasn't changed.
    Includes all fields that affect AI rating output.
    """
    key_data = {
        # Price and dip metrics
        "dip_percentage": round(float(dip_data.get("dip_percentage") or 0), 2),
        # Fundamentals (all used in AI prompt)
        "pe_ratio": round(float(dip_data.get("pe_ratio") or 0), 2) if dip_data.get("pe_ratio") else None,
        "forward_pe": round(float(dip_data.get("forward_pe") or 0), 2) if dip_data.get("forward_pe") else None,
        "peg_ratio": round(float(dip_data.get("peg_ratio") or 0), 2) if dip_data.get("peg_ratio") else None,
        "price_to_book": round(float(dip_data.get("price_to_book") or 0), 2) if dip_data.get("price_to_book") else None,
        "profit_margin": round(float(dip_data.get("profit_margin") or 0), 4) if dip_data.get("profit_margin") else None,
        "return_on_equity": round(float(dip_data.get("return_on_equity") or 0), 4) if dip_data.get("return_on_equity") else None,
        "debt_to_equity": round(float(dip_data.get("debt_to_equity") or 0), 2) if dip_data.get("debt_to_equity") else None,
        "recommendation": dip_data.get("recommendation"),
        # Signal data
        "dip_class": dip_data.get("classification"),
        "quality_score": round(float(dip_data.get("quality_score") or 0), 2) if dip_data.get("quality_score") else None,
        "stability_score": round(float(dip_data.get("stability_score") or 0), 2) if dip_data.get("stability_score") else None,
    }
    content = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def _get_stored_dip_hash(symbol: str) -> str | None:
    """Get the stored input hash for a symbol's dip AI analysis."""
    async with get_session() as session:
        result = await session.execute(
            select(AnalysisVersion.input_version_hash).where(
                AnalysisVersion.symbol == symbol,
                AnalysisVersion.analysis_type == "dip_rating",
            )
        )
        row = result.scalar_one_or_none()
        return row if row else None


async def _store_dip_analysis_version(
    symbol: str,
    input_hash: str,
    batch_job_id: str | None = None,
) -> None:
    """Store/update the analysis version after successful dip analysis."""
    expires_at = datetime.now(UTC) + timedelta(days=7)

    async with get_session() as session:
        stmt = insert(AnalysisVersion).values(
            symbol=symbol,
            analysis_type="dip_rating",
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


async def get_all_dip_symbols() -> list[str]:
    """Get all current dip symbols."""
    async with get_session() as session:
        result = await session.execute(select(DipState.symbol))
        return [r[0] for r in result.all()]


async def get_dips_needing_analysis() -> list[dict]:
    """Get dips that need AI analysis (new or expired), with fundamentals."""
    async with get_session() as session:
        # Build subquery for latest dipfinder signal
        signal_subq = (
            select(
                DipfinderSignal.ticker,
                DipfinderSignal.dip_class,
                DipfinderSignal.excess_dip,
                DipfinderSignal.quality_score,
                DipfinderSignal.stability_score,
            )
            .order_by(DipfinderSignal.as_of_date.desc())
            .limit(1)
            .subquery()
        )

        # Main query joining dip_state with symbols, fundamentals, signals, and analysis
        result = await session.execute(
            select(
                DipState.symbol,
                DipState.current_price,
                DipState.ath_price,
                DipState.dip_percentage,
                DipState.first_seen,
                Symbol.name,
                Symbol.sector,
                Symbol.summary_ai,
                StockFundamentals.pe_ratio,
                StockFundamentals.forward_pe,
                StockFundamentals.peg_ratio,
                StockFundamentals.price_to_book,
                StockFundamentals.ev_to_ebitda,
                StockFundamentals.profit_margin,
                StockFundamentals.gross_margin,
                StockFundamentals.return_on_equity,
                StockFundamentals.revenue_growth,
                StockFundamentals.earnings_growth,
                StockFundamentals.debt_to_equity,
                StockFundamentals.current_ratio,
                StockFundamentals.free_cash_flow,
                StockFundamentals.recommendation,
                StockFundamentals.target_mean_price,
                StockFundamentals.num_analyst_opinions,
                StockFundamentals.beta,
                StockFundamentals.short_percent_of_float,
                StockFundamentals.held_percent_institutions,
                DipAIAnalysis.symbol.label("ai_symbol"),
                DipAIAnalysis.expires_at,
                DipAIAnalysis.generated_at,
            )
            .outerjoin(Symbol, Symbol.symbol == DipState.symbol)
            .outerjoin(StockFundamentals, StockFundamentals.symbol == DipState.symbol)
            .outerjoin(DipAIAnalysis, DipAIAnalysis.symbol == DipState.symbol)
            .where(
                (DipAIAnalysis.symbol == None)
                | (DipAIAnalysis.expires_at < datetime.now(UTC))
                | (DipAIAnalysis.generated_at < datetime.now(UTC) - timedelta(days=7))
            )
        )

        rows = result.all()
        return [
            {
                "symbol": r[0],
                "current_price": r[1],
                "ath_price": r[2],
                "dip_percentage": r[3],
                "first_seen": r[4],
                "name": r[5],
                "sector": r[6],
                "summary_ai": r[7],
                "pe_ratio": r[8],
                "forward_pe": r[9],
                "peg_ratio": r[10],
                "price_to_book": r[11],
                "ev_to_ebitda": r[12],
                "profit_margin": r[13],
                "gross_margin": r[14],
                "return_on_equity": r[15],
                "revenue_growth": r[16],
                "earnings_growth": r[17],
                "debt_to_equity": r[18],
                "current_ratio": r[19],
                "free_cash_flow": r[20],
                "recommendation": r[21],
                "target_mean_price": r[22],
                "num_analyst_opinions": r[23],
                "beta": r[24],
                "short_percent_of_float": r[25],
                "held_percent_institutions": r[26],
            }
            for r in rows
        ]


async def schedule_batch_dip_analysis() -> str | None:
    """
    Schedule a batch job to analyze all current dips.
    
    Uses input-hash filtering to skip dips whose data hasn't changed.
    
    Returns:
        Batch job ID if created, None if no dips to analyze
    """
    dips = await get_dips_needing_analysis()

    if not dips:
        logger.info("No dips need AI analysis")
        return None

    logger.info(f"Checking {len(dips)} dips for AI analysis")

    # Filter dips by input hash - skip if data unchanged
    dips_to_analyze = []
    skipped_count = 0
    input_hashes: dict[str, str] = {}
    
    for dip in dips:
        symbol = dip["symbol"]
        current_hash = _compute_dip_input_hash(dip)
        stored_hash = await _get_stored_dip_hash(symbol)
        
        if stored_hash == current_hash:
            logger.debug(f"Skipping {symbol}: input unchanged (hash={current_hash[:8]}...)")
            skipped_count += 1
            continue
            
        dips_to_analyze.append(dip)
        input_hashes[symbol] = current_hash
    
    if not dips_to_analyze:
        logger.info(f"All {skipped_count} dips have unchanged input data - skipping batch")
        return None

    logger.info(f"Scheduling batch AI analysis for {len(dips_to_analyze)} dips ({skipped_count} skipped)")

    # Prepare items for batch - use context keys expected by _build_prompt
    items = []
    for dip in dips_to_analyze:
        # Calculate days in dip
        days_below = 0
        if dip.get("first_seen"):
            first_seen = dip["first_seen"]
            if hasattr(first_seen, 'tzinfo') and first_seen.tzinfo is None:
                first_seen = first_seen.replace(tzinfo=UTC)
            days_below = (datetime.now(UTC) - first_seen).days

        # Helper to format percentages
        def fmt_pct(val):
            return f"{val * 100:.1f}%" if val else None

        def fmt_ratio(val):
            return f"{val:.2f}" if val else None

        def fmt_large_num(val):
            if val is None:
                return None
            if val >= 1e12:
                return f"${val / 1e12:.1f}T"
            if val >= 1e9:
                return f"${val / 1e9:.1f}B"
            if val >= 1e6:
                return f"${val / 1e6:.1f}M"
            return f"${val:,.0f}"

        items.append(
            {
                "symbol": dip["symbol"],
                "name": dip.get("name"),
                "sector": dip.get("sector"),
                "summary": dip.get("summary_ai"),
                "current_price": float(dip["current_price"])
                if dip["current_price"]
                else None,
                "ref_high": float(dip["ath_price"]) if dip["ath_price"] else None,
                "dip_pct": float(dip["dip_percentage"])
                if dip["dip_percentage"]
                else None,
                "days_below": days_below,
                # Dip classification and scores
                "dip_classification": dip.get("classification"),
                "excess_dip": float(dip["excess_dip"]) if dip.get("excess_dip") else None,
                "quality_score": float(dip["quality_score"]) if dip.get("quality_score") else None,
                "stability_score": float(dip["stability_score"]) if dip.get("stability_score") else None,
                # Fundamentals from database
                "pe_ratio": float(dip["pe_ratio"]) if dip.get("pe_ratio") else None,
                "forward_pe": float(dip["forward_pe"]) if dip.get("forward_pe") else None,
                "peg_ratio": fmt_ratio(dip.get("peg_ratio")),
                "price_to_book": fmt_ratio(dip.get("price_to_book")),
                "ev_to_ebitda": fmt_ratio(dip.get("ev_to_ebitda")),
                "profit_margin": fmt_pct(dip.get("profit_margin")),
                "gross_margin": fmt_pct(dip.get("gross_margin")),
                "return_on_equity": fmt_pct(dip.get("return_on_equity")),
                "revenue_growth": fmt_pct(dip.get("revenue_growth")),
                "earnings_growth": fmt_pct(dip.get("earnings_growth")),
                "debt_to_equity": fmt_ratio(dip.get("debt_to_equity")),
                "current_ratio": fmt_ratio(dip.get("current_ratio")),
                "free_cash_flow": fmt_large_num(dip.get("free_cash_flow")),
                "recommendation": dip.get("recommendation"),
                "target_mean_price": float(dip["target_mean_price"]) if dip.get("target_mean_price") else None,
                "num_analyst_opinions": dip.get("num_analyst_opinions"),
                "beta": fmt_ratio(dip.get("beta")),
                "short_percent_of_float": fmt_pct(dip.get("short_percent_of_float")),
                "institutional_ownership": fmt_pct(dip.get("held_percent_institutions")),
            }
        )

    # Submit batch job for RATING (includes reasoning)
    batch_id = await submit_batch(
        task=TaskType.RATING,
        items=items,
    )

    if batch_id:
        # Record batch job
        await api_usage.record_batch_job(
            batch_id=batch_id,
            job_type=TaskType.RATING.value,
            total_requests=len(items),
        )
        
        # Store input hashes for all symbols in batch
        for symbol, input_hash in input_hashes.items():
            await _store_dip_analysis_version(symbol, input_hash, batch_job_id=batch_id)

        logger.info(f"Created batch job {batch_id} for {len(dips_to_analyze)} dips")

    return batch_id


async def schedule_batch_suggestion_bios() -> str | None:
    """
    Schedule a batch job to generate bios for pending suggestions.

    NOTE: This is intentionally disabled. Pending suggestions don't need AI bios -
    they get AI content generated when approved and processed via process_approved_symbol.
    Batch bios for suggestions were creating noise in the batch jobs panel without value.

    Returns:
        Always None - batch bios for suggestions are disabled
    """
    logger.info("Skipping batch bios for pending suggestions (AI content generated on approval)")
    return None


# Alias for swipe bios
schedule_batch_swipe_bios = schedule_batch_suggestion_bios


async def process_completed_batch_jobs() -> int:
    """
    Check for and process completed batch jobs.

    Returns:
        Number of jobs processed
    """
    # Get pending batch jobs
    async with get_session() as session:
        result = await session.execute(
            select(BatchJob.batch_id, BatchJob.job_type, BatchJob.status).where(
                BatchJob.status.in_(["pending", "validating", "in_progress", "finalizing"])
            )
        )
        rows = result.all()

    processed = 0

    for row in rows:
        batch_id = row[0]
        job_type = row[1]

        # Check status
        status_info = await check_batch(batch_id)

        if not status_info:
            continue

        new_status = status_info.get("status", "unknown")

        # Update status in database
        async with get_session() as session:
            await session.execute(
                update(BatchJob)
                .where(BatchJob.batch_id == batch_id)
                .values(
                    status=new_status,
                    completed_requests=status_info.get("completed_count", 0),
                    failed_requests=status_info.get("failed_count", 0),
                    output_file_id=status_info.get("output_file_id"),
                    error_file_id=status_info.get("error_file_id"),
                )
            )
            await session.commit()

        # If completed, retrieve and process results
        if new_status == "completed":
            results = await collect_batch(batch_id)

            if results:
                await _process_batch_results(batch_id, job_type, results)
                processed += 1

                # Update completion time and cost
                async with get_session() as session:
                    await session.execute(
                        update(BatchJob)
                        .where(BatchJob.batch_id == batch_id)
                        .values(
                            completed_at=datetime.now(UTC),
                            actual_cost_usd=Decimal(str(status_info.get("total_cost", 0))),
                        )
                    )
                    await session.commit()

    return processed


async def _process_batch_results(
    batch_id: str,
    job_type: str,
    results: list[dict],
) -> None:
    """Process results from a completed batch job."""
    logger.info(f"Processing {len(results)} results from batch {batch_id} (type: {job_type})")

    # AGENT batches are handled by collect_agent_batch which does its own processing
    if job_type == TaskType.AGENT.value:
        from app.services.ai_agents import collect_agent_batch
        
        agent_results = await collect_agent_batch(batch_id)
        logger.info(f"Processed AGENT batch {batch_id}: {len(agent_results)} symbols analyzed")
        return

    for result in results:
        custom_id = result.get("custom_id", "")
        # Results from new collect_batch have "result" directly
        content = result.get("result", "")
        symbol = result.get("symbol", "")

        try:
            if job_type == TaskType.RATING.value:
                # RATING batch returns parsed JSON with rating/reasoning/confidence
                if isinstance(content, dict) and not result.get("failed"):
                    await _store_dip_analysis(symbol, content, batch_id)
                else:
                    logger.warning(f"Skipping failed RATING result for {symbol}: {result.get('error')}")

            elif job_type == TaskType.BIO.value:
                # BIO batch for suggestions is deprecated - AI content generated on approval
                logger.debug(f"Skipping BIO result for {symbol} - suggestion bios no longer used")

            elif job_type == TaskType.PORTFOLIO.value:
                # Portfolio analysis - store result in portfolio
                if not result.get("failed"):
                    await _store_portfolio_analysis(custom_id, content, batch_id)
                else:
                    logger.warning(f"Skipping failed PORTFOLIO result for {custom_id}: {result.get('error')}")

        except Exception as e:
            logger.error(f"Error processing batch result for {custom_id}: {e}")


async def _store_dip_analysis(symbol: str, content: dict | str, batch_id: str) -> None:
    """Store AI rating analysis for a dip and clear pending flag."""
    # Handle both dict (from RATING batch) and legacy string formats
    if isinstance(content, dict):
        rating = content.get("rating")  # String like "buy", "hold", "sell"
        reasoning = content.get("reasoning", "")
        # Note: Bio is not included in RATING - generate separately if needed
    else:
        # Legacy: plain text (shouldn't happen with RATING task)
        rating = None
        reasoning = str(content)

    expires_at = datetime.now(UTC) + timedelta(days=7)

    async with get_session() as session:
        stmt = insert(DipAIAnalysis).values(
            symbol=symbol.upper(),
            ai_rating=rating,
            rating_reasoning=reasoning,
            model_used="gpt-5-mini",
            is_batch_generated=True,
            batch_job_id=batch_id,
            ai_pending=False,  # Clear pending flag
            generated_at=datetime.now(UTC),
            expires_at=expires_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "ai_rating": stmt.excluded.ai_rating,
                "rating_reasoning": stmt.excluded.rating_reasoning,
                "is_batch_generated": True,
                "batch_job_id": stmt.excluded.batch_job_id,
                "ai_pending": False,  # Clear pending flag
                "generated_at": datetime.now(UTC),
                "expires_at": stmt.excluded.expires_at,
            },
        )
        await session.execute(stmt)
        await session.commit()

    logger.debug(f"Stored AI rating for {symbol}: {rating}")


async def _store_portfolio_analysis(
    custom_id: str,
    content: str,
    batch_id: str,
) -> None:
    """Store AI analysis for a portfolio.
    
    Validates that content is valid JSON matching AIPortfolioAnalysis schema.
    On validation failure, immediately retries with synchronous generate() call.
    If retry also fails, skips storage (marks for next scheduled run).
    
    Args:
        custom_id: Portfolio identifier (portfolio_{id})
        content: Raw AI response content
        batch_id: Source batch job ID
    """
    import json
    from app.database.orm import Portfolio, PortfolioHolding
    from app.schemas.portfolio import AIPortfolioAnalysis
    from pydantic import ValidationError
    
    # Parse portfolio_id from custom_id
    if not custom_id.startswith("portfolio_"):
        logger.warning(f"Invalid portfolio custom_id: {custom_id}")
        return
    
    try:
        portfolio_id = int(custom_id.replace("portfolio_", ""))
    except ValueError:
        logger.warning(f"Cannot parse portfolio_id from: {custom_id}")
        return
    
    def _clean_and_validate(raw: str) -> str | None:
        """Strip markdown, parse JSON, validate schema. Returns validated JSON string or None."""
        clean = raw.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()
        
        parsed = json.loads(clean)
        validated = AIPortfolioAnalysis.model_validate(parsed)
        return validated.model_dump_json()
    
    # First attempt: validate batch result
    validated_content = None
    first_attempt_failed = False
    
    try:
        validated_content = _clean_and_validate(content)
        logger.debug(f"Portfolio {portfolio_id} AI analysis validated successfully")
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"Portfolio {portfolio_id} AI analysis validation failed: {e}")
        first_attempt_failed = True
    
    # Retry once with synchronous generate() call if first attempt failed
    if first_attempt_failed:
        logger.info(f"Portfolio {portfolio_id}: retrying AI generation...")
        try:
            # Build minimal context for retry
            retry_context = await _build_portfolio_retry_context(portfolio_id)
            
            if retry_context:
                retry_result = await generate(
                    task=TaskType.PORTFOLIO,
                    context=retry_context,
                    json_output=True,
                )
                
                if retry_result and isinstance(retry_result, dict):
                    # Already parsed as dict, validate with Pydantic
                    validated = AIPortfolioAnalysis.model_validate(retry_result)
                    validated_content = validated.model_dump_json()
                    logger.info(f"Portfolio {portfolio_id}: retry succeeded")
                elif retry_result and isinstance(retry_result, str):
                    validated_content = _clean_and_validate(retry_result)
                    logger.info(f"Portfolio {portfolio_id}: retry succeeded")
                else:
                    logger.warning(f"Portfolio {portfolio_id}: retry returned empty result")
            else:
                logger.warning(f"Portfolio {portfolio_id}: could not build retry context")
                    
        except Exception as retry_err:
            logger.warning(f"Portfolio {portfolio_id}: retry also failed: {retry_err}")
    
    if not validated_content:
        logger.info(f"Portfolio {portfolio_id} marked for re-analysis on next scheduled run")
        return
    
    async with get_session() as session:
        result = await session.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            logger.warning(f"Portfolio {portfolio_id} not found")
            return
        
        holdings_result = await session.execute(
            select(
                PortfolioHolding.symbol,
                PortfolioHolding.quantity,
                PortfolioHolding.avg_cost,
            ).where(PortfolioHolding.portfolio_id == portfolio_id)
        )
        holdings = [
            {
                "symbol": r[0],
                "quantity": str(r[1]) if r[1] else "0",
                "avg_cost": str(r[2]) if r[2] else "0",
            }
            for r in holdings_result.all()
        ]
        holdings_hash = _compute_portfolio_holdings_hash(holdings)
        
        # Update portfolio
        portfolio.ai_analysis_summary = validated_content
        portfolio.ai_analysis_hash = holdings_hash
        portfolio.ai_analysis_at = datetime.now(UTC)
        
        await session.commit()
    
    logger.info(f"Stored AI analysis for portfolio {portfolio_id}")


async def _build_portfolio_retry_context(portfolio_id: int) -> dict[str, Any] | None:
    """Build minimal context for portfolio AI analysis retry.
    
    Fetches portfolio data from database and builds context dict.
    Returns None if portfolio not found or has no holdings.
    """
    from app.database.orm import Portfolio, PortfolioHolding
    
    async with get_session() as session:
        # Get portfolio
        result = await session.execute(
            select(Portfolio).where(Portfolio.id == portfolio_id)
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            return None
        
        # Get holdings
        holdings_result = await session.execute(
            select(PortfolioHolding).where(PortfolioHolding.portfolio_id == portfolio_id)
        )
        holdings = holdings_result.scalars().all()
        
        if not holdings:
            return None
        
        # Build holdings list
        holdings_data = []
        total_value = 0
        total_cost = 0
        
        for h in holdings:
            qty = float(h.quantity) if h.quantity else 0
            avg = float(h.avg_cost) if h.avg_cost else 0
            cost = qty * avg
            # Use avg_cost as proxy for current price (real price fetch is expensive)
            value = cost  # Will be slightly off but good enough for retry
            total_cost += cost
            total_value += value
            
            holdings_data.append({
                "symbol": h.symbol,
                "weight": 0,  # Will calculate after
                "gain_pct": 0,  # Unknown without current prices
            })
        
        # Calculate weights
        for hd in holdings_data:
            # Equal weight as fallback
            hd["weight"] = 100 / len(holdings_data) if holdings_data else 0
        
        return {
            "portfolio_name": portfolio.name,
            "total_value": total_value,
            "total_gain": 0,
            "total_gain_pct": 0,
            "holdings": holdings_data,
            "sector_weights": {},
            "performance": {},
            "risk": {},
        }


async def run_realtime_analysis_for_new_stock(
    symbol: str,
    current_price: float | None = None,
    ath_price: float | None = None,
    dip_percentage: float | None = None,
) -> dict:
    """
    Queue AI analysis for a newly added stock via batch API.

    Called when a stock is approved from suggestions and added to dips.
    Uses batch API (50% cheaper) - caller should handle pending UI state.
    
    Returns immediately with pending status. AI content will be available
    after batch processing completes (typically within 15-30 minutes).
    """
    from app.services import stock_info

    logger.info(f"Queueing batch AI analysis for new stock: {symbol}")

    try:
        # Fetch stock info for context
        info = await stock_info.get_stock_info_async(symbol)
        name = info.get("name") if info else None
        sector = info.get("sector") if info else None
        summary = info.get("summary") if info else None
        pe_ratio = info.get("pe_ratio") if info else None

        # Mark as pending in database
        async with get_session() as session:
            stmt = insert(DipAIAnalysis).values(
                symbol=symbol.upper(),
                ai_pending=True,
                is_batch_generated=True,
                generated_at=datetime.now(UTC),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_={
                    "ai_pending": True,
                    "is_batch_generated": True,
                },
            )
            await session.execute(stmt)
            await session.commit()

        # Prepare batch item
        fundamentals = {}
        try:
            from app.services.fundamentals import get_fundamentals_for_analysis
            fundamentals = await get_fundamentals_for_analysis(symbol)
        except Exception:
            pass

        batch_item = {
            "symbol": symbol,
            "name": name,
            "sector": sector,
            "summary": summary,
            "current_price": current_price,
            "ref_high": ath_price,
            "dip_pct": dip_percentage,
            "days_below": 0,
            "pe_ratio": pe_ratio,
            **fundamentals,
        }

        # Submit to batch API
        batch_id = await submit_batch(task=TaskType.RATING, items=[batch_item])
        bio_batch_id = await submit_batch(task=TaskType.BIO, items=[batch_item])

        logger.info(f"Queued batch AI for {symbol}: rating={batch_id}, bio={bio_batch_id}")

        return {
            "symbol": symbol,
            "status": "pending",
            "batch_id": batch_id,
            "bio_batch_id": bio_batch_id,
        }

    except Exception as e:
        logger.error(f"Failed to queue batch analysis for {symbol}: {e}")
        return {"symbol": symbol, "status": "error", "error": str(e)}


async def queue_ai_for_symbols(symbols: list[str]) -> dict:
    """
    Queue AI analysis for multiple symbols via batch API.
    
    This is the primary entry point for queueing AI content generation.
    Marks symbols as pending and submits to OpenAI Batch API.
    
    Args:
        symbols: List of symbols to queue for AI analysis
        
    Returns:
        Dict with batch_ids and pending count
    """
    from app.services import stock_info
    from app.services.fundamentals import get_fundamentals_for_analysis
    from app.repositories import dip_state_repo
    
    if not symbols:
        return {"pending": 0, "batch_ids": []}
    
    logger.info(f"Queueing batch AI analysis for {len(symbols)} symbols")
    
    # Mark all as pending
    async with get_session() as session:
        for symbol in symbols:
            stmt = insert(DipAIAnalysis).values(
                symbol=symbol.upper(),
                ai_pending=True,
                is_batch_generated=True,
                generated_at=datetime.now(UTC),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["symbol"],
                set_={
                    "ai_pending": True,
                },
            )
            await session.execute(stmt)
        await session.commit()
    
    # Build batch items
    batch_items = []
    for symbol in symbols:
        try:
            dip_state = await dip_state_repo.get_dip_state(symbol)
            info = await stock_info.get_stock_info_async(symbol)
            fundamentals = await get_fundamentals_for_analysis(symbol)
            
            batch_items.append({
                "symbol": symbol,
                "name": info.get("name") if info else None,
                "sector": info.get("sector") if info else None,
                "summary": info.get("summary") if info else None,
                "current_price": float(dip_state.current_price) if dip_state and dip_state.current_price else None,
                "ref_high": float(dip_state.ath_price) if dip_state and dip_state.ath_price else None,
                "dip_pct": float(dip_state.dip_percentage) if dip_state and dip_state.dip_percentage else None,
                "days_below": 0,
                **fundamentals,
            })
        except Exception as e:
            logger.warning(f"Failed to prepare batch item for {symbol}: {e}")
    
    if not batch_items:
        return {"pending": 0, "batch_ids": []}
    
    # Submit batches
    batch_ids = []
    try:
        rating_batch_id = await submit_batch(task=TaskType.RATING, items=batch_items)
        if rating_batch_id:
            batch_ids.append(rating_batch_id)
        
        bio_batch_id = await submit_batch(task=TaskType.BIO, items=batch_items)
        if bio_batch_id:
            batch_ids.append(bio_batch_id)
    except Exception as e:
        logger.error(f"Failed to submit batch: {e}")
    
    return {
        "pending": len(batch_items),
        "batch_ids": batch_ids,
    }


# ============================================================================
# Cron Job Functions (called by scheduler)
# ============================================================================


async def cron_batch_ai_suggestions() -> dict:
    """
    Cron job: Schedule weekly batch AI bios for suggestions.

    Runs Sunday 4 AM UTC by default.
    """
    logger.info("Running weekly batch AI bios for suggestions")

    batch_id = await schedule_batch_suggestion_bios()

    return {
        "job": "batch_ai_suggestions",
        "batch_id": batch_id,
        "status": "scheduled" if batch_id else "no_suggestions",
    }


async def cron_sync_batch_jobs() -> dict:
    """
    Cron job: Check and process completed batch jobs.

    Runs every 15 minutes.
    """
    logger.info("Syncing batch job statuses")

    processed = await process_completed_batch_jobs()

    return {
        "job": "sync_batch_jobs",
        "processed": processed,
    }


async def cron_cleanup_expired() -> dict:
    """
    Cron job: Clean up expired data.

    Runs daily at midnight.
    """
    from sqlalchemy import text

    from app.repositories.dip_history_orm import cleanup_old_history

    logger.info("Running daily cleanup")

    # Clean up old dip history (keep 90 days)
    history_deleted = await cleanup_old_history(days=90)

    # Clean up old rate limit entries
    async with get_session() as session:
        await session.execute(text("SELECT cleanup_rate_limits()"))
        await session.commit()

    return {
        "job": "cleanup_expired",
        "history_deleted": history_deleted,
    }


# ============================================================================
# Portfolio AI Analysis
# ============================================================================


def _compute_portfolio_holdings_hash(holdings: list[dict[str, Any]]) -> str:
    """
    Compute hash of portfolio holdings for change detection.
    
    Only includes fields that affect analysis output.
    """
    key_data = [
        {
            "symbol": h.get("symbol"),
            "quantity": str(h.get("quantity", 0)),
            "avg_cost": str(h.get("avg_cost", 0)),
        }
        for h in sorted(holdings, key=lambda x: x.get("symbol", ""))
    ]
    content = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


async def get_portfolios_needing_ai_analysis() -> list[dict[str, Any]]:
    """
    Get portfolios where holdings have changed since last AI analysis.
    
    Returns portfolios with their holdings for analysis.
    """
    from app.database.orm import Portfolio, PortfolioHolding
    
    async with get_session() as session:
        # Get all portfolios with their holdings
        result = await session.execute(
            select(
                Portfolio.id,
                Portfolio.user_id,
                Portfolio.name,
                Portfolio.ai_analysis_hash,
            )
        )
        portfolios = result.all()
        
        portfolios_to_analyze = []
        
        for portfolio_id, user_id, name, stored_hash in portfolios:
            # Get holdings for this portfolio
            holdings_result = await session.execute(
                select(
                    PortfolioHolding.symbol,
                    PortfolioHolding.quantity,
                    PortfolioHolding.avg_cost,
                ).where(PortfolioHolding.portfolio_id == portfolio_id)
            )
            holdings = [
                {
                    "symbol": r[0],
                    "quantity": float(r[1]) if r[1] else 0,
                    "avg_cost": float(r[2]) if r[2] else 0,
                }
                for r in holdings_result.all()
            ]
            
            if not holdings:
                continue  # Skip empty portfolios
            
            current_hash = _compute_portfolio_holdings_hash(holdings)
            
            if stored_hash == current_hash:
                logger.debug(f"Skipping portfolio {name}: holdings unchanged")
                continue
            
            portfolios_to_analyze.append({
                "portfolio_id": portfolio_id,
                "user_id": user_id,
                "portfolio_name": name,
                "holdings": holdings,
                "holdings_hash": current_hash,
            })
        
        return portfolios_to_analyze


async def _enrich_portfolio_holdings(
    holdings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float], float]:
    """
    Enrich holdings with sector/country data, current prices, and compute weights.
    
    Returns:
        - Enriched holdings list
        - Sector weights dict
        - Total portfolio value
    """
    from app.repositories import price_history_orm as price_history_repo
    from app.services import financedatabase_service
    
    symbols = [h["symbol"] for h in holdings]
    
    # Get current prices in ONE efficient query
    current_prices = await price_history_repo.get_latest_prices(symbols)
    
    # Get sector/country from local universe first
    symbol_info: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        try:
            universe_data = await financedatabase_service.get_by_symbol(symbol)
            if universe_data:
                symbol_info[symbol] = {
                    "sector": universe_data.get("sector"),
                    "country": universe_data.get("country"),
                }
        except Exception:
            pass
    
    # Calculate values and weights
    total_value = 0.0
    for h in holdings:
        quantity = h.get("quantity", 0)
        symbol = h["symbol"]
        
        # Use price from price_history, fallback to avg_cost
        price = float(current_prices.get(symbol, 0)) or h.get("avg_cost", 0)
        h["current_price"] = price
        h["market_value"] = quantity * price
        total_value += h["market_value"]
        
        # Add sector/country from universe
        info = symbol_info.get(symbol, {})
        h["sector"] = info.get("sector")
        h["country"] = info.get("country")
        
        # Calculate gain
        avg_cost = h.get("avg_cost", 0)
        if avg_cost > 0 and price > 0:
            h["gain_pct"] = ((price - avg_cost) / avg_cost) * 100
        else:
            h["gain_pct"] = 0
    
    # Calculate weights and sector breakdown
    sector_weights: dict[str, float] = {}
    for h in holdings:
        if total_value > 0:
            h["weight"] = (h["market_value"] / total_value) * 100
        else:
            h["weight"] = 0
        
        sector = h.get("sector") or "Other"
        sector_weights[sector] = sector_weights.get(sector, 0) + h["weight"]
    
    return holdings, sector_weights, total_value


async def schedule_batch_portfolio_analysis() -> str | None:
    """
    Schedule batch AI analysis for portfolios with changed holdings.
    
    Enriches portfolio data with:
    - Quantstats/pyfolio performance metrics (CAGR, Sharpe, Sortino, etc.)
    - Risk analytics (VaR, CVaR, max drawdown)
    - Sector/country weights
    - Individual position details
    
    Returns:
        Batch job ID if created, None if no portfolios need analysis.
    """
    from datetime import date, timedelta
    from app.portfolio.service import build_portfolio_context, run_quantstats, run_pyfolio
    from app.quant_engine import analyze_portfolio as run_risk_analytics
    from app.dipfinder.service import DatabasePriceProvider
    
    portfolios = await get_portfolios_needing_ai_analysis()
    
    if not portfolios:
        logger.info("No portfolios need AI analysis")
        return None
    
    logger.info(f"Scheduling AI analysis for {len(portfolios)} portfolios")
    
    # Prepare items for batch
    items = []
    price_provider = DatabasePriceProvider()
    
    for p in portfolios:
        try:
            holdings, sector_weights, total_value = await _enrich_portfolio_holdings(
                p["holdings"]
            )
            
            # Calculate total gain
            total_cost = sum(
                h.get("quantity", 0) * h.get("avg_cost", 0) for h in holdings
            )
            total_gain = total_value - total_cost
            total_gain_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0
            
            # Build context for quantstats/pyfolio
            performance_metrics = {}
            risk_metrics = {}
            
            try:
                context = await build_portfolio_context(
                    portfolio_id=p["portfolio_id"],
                    user_id=p["user_id"],
                    benchmark="SPY",
                )
                
                # Get quantstats metrics
                qs_result = run_quantstats(context)
                if qs_result.get("status") in ("ok", "partial"):
                    qs_data = qs_result.get("data", {})
                    performance_metrics = {
                        "cagr": qs_data.get("cagr"),
                        "sharpe": qs_data.get("sharpe"),
                        "sortino": qs_data.get("sortino"),
                        "volatility": qs_data.get("volatility"),
                        "max_drawdown": qs_data.get("max_drawdown"),
                        "beta": qs_data.get("beta"),
                    }
                
                # Get pyfolio metrics (as backup/additional)
                pf_result = run_pyfolio(context)
                if pf_result.get("status") in ("ok", "partial"):
                    pf_data = pf_result.get("data", {})
                    # Only add if not already present
                    if not performance_metrics.get("cagr"):
                        performance_metrics["cagr"] = pf_data.get("cagr")
                    if not performance_metrics.get("sharpe"):
                        performance_metrics["sharpe"] = pf_data.get("sharpe")
                    if not performance_metrics.get("max_drawdown"):
                        performance_metrics["max_drawdown"] = pf_data.get("max_drawdown")
                        
            except Exception as e:
                logger.warning(f"Could not get performance metrics for portfolio {p['portfolio_id']}: {e}")
            
            # Try to get risk analytics
            try:
                import pandas as pd
                
                symbols = [h["symbol"] for h in holdings]
                
                # Get price history
                end_dt = date.today()
                start_dt = end_dt - timedelta(days=365)
                
                prices_dict = {}
                for symbol in symbols:
                    price_df = await price_provider.get_prices(symbol, start_dt, end_dt)
                    if price_df is not None and not price_df.empty:
                        prices_dict[symbol] = price_df["Close"]
                
                if prices_dict:
                    prices_df = pd.DataFrame(prices_dict).dropna()
                    if len(prices_df) >= 60:
                        returns = prices_df.pct_change().dropna()
                        
                        # Get weights
                        weights = {h["symbol"]: h["weight"] / 100 for h in holdings if h.get("weight")}
                        
                        # Run risk analytics
                        analytics = run_risk_analytics(
                            holdings=weights,
                            returns=returns,
                            total_value=total_value,
                        )
                        
                        risk_metrics = {
                            "risk_score": analytics.overall_risk_score,
                            "portfolio_volatility": float(analytics.risk_decomposition.portfolio_volatility) if analytics.risk_decomposition else None,
                            "var_95_daily": float(analytics.tail_risk.var_95_daily) if analytics.tail_risk else None,
                            "cvar_95_daily": float(analytics.tail_risk.cvar_95_daily) if analytics.tail_risk else None,
                            "effective_n": float(analytics.diversification.effective_n) if analytics.diversification else None,
                            "diversification_ratio": float(analytics.diversification.diversification_ratio) if analytics.diversification else None,
                            "market_regime": analytics.regime.regime if analytics.regime else None,
                            "top_risk_contributors": dict(list(analytics.risk_decomposition.risk_contribution_pct.items())[:3]) if analytics.risk_decomposition else {},
                        }
            except Exception as e:
                logger.warning(f"Could not get risk analytics for portfolio {p['portfolio_id']}: {e}")
            
            items.append({
                "custom_id": f"portfolio_{p['portfolio_id']}",
                "portfolio_id": p["portfolio_id"],
                "portfolio_name": p["portfolio_name"],
                "holdings_hash": p["holdings_hash"],
                "total_value": total_value,
                "total_gain": total_gain,
                "total_gain_pct": total_gain_pct,
                "holdings": [
                    {
                        "symbol": h["symbol"],
                        "weight": h["weight"],
                        "gain_pct": h["gain_pct"],
                        "market_value": h.get("market_value"),
                        "sector": h.get("sector"),
                        "country": h.get("country"),
                    }
                    for h in sorted(holdings, key=lambda x: -x.get("weight", 0))
                ],
                "sector_weights": sector_weights,
                "performance": performance_metrics,
                "risk": risk_metrics,
            })
            
        except Exception as e:
            logger.error(f"Failed to prepare portfolio {p['portfolio_id']} for analysis: {e}")
            continue
    
    if not items:
        logger.info("No portfolios could be prepared for AI analysis")
        return None
    
    # Submit batch job
    batch_id = await submit_batch(
        task=TaskType.PORTFOLIO,
        items=items,
    )
    
    if batch_id:
        await api_usage.record_batch_job(
            batch_id=batch_id,
            job_type=TaskType.PORTFOLIO.value,
            total_requests=len(items),
        )
        logger.info(f"Created portfolio analysis batch {batch_id} for {len(items)} portfolios")
    
    return batch_id


async def process_portfolio_analysis_results(batch_id: str) -> int:
    """
    Process completed portfolio analysis batch results.
    
    Returns number of portfolios updated.
    """
    from app.database.orm import Portfolio
    
    results = await collect_batch(batch_id)
    
    if not results:
        logger.warning(f"No results for portfolio batch {batch_id}")
        return 0
    
    updated = 0
    
    async with get_session() as session:
        for result in results:
            custom_id = result.get("custom_id", "")
            if not custom_id.startswith("portfolio_"):
                continue
            
            portfolio_id = int(custom_id.replace("portfolio_", ""))
            content = result.get("content", "")
            holdings_hash = result.get("holdings_hash")
            
            if content:
                await session.execute(
                    update(Portfolio)
                    .where(Portfolio.id == portfolio_id)
                    .values(
                        ai_analysis_summary=content,
                        ai_analysis_hash=holdings_hash,
                        ai_analysis_at=datetime.now(UTC),
                    )
                )
                updated += 1
                logger.info(f"Updated AI analysis for portfolio {portfolio_id}")
        
        await session.commit()
    
    return updated


async def cron_portfolio_ai_analysis() -> dict:
    """
    Cron job: Schedule portfolio AI analysis for changed portfolios.
    
    Runs daily at 6 AM UTC.
    """
    logger.info("Running portfolio AI analysis job")
    
    batch_id = await schedule_batch_portfolio_analysis()
    
    return {
        "job": "portfolio_ai_analysis",
        "batch_id": batch_id,
        "status": "scheduled" if batch_id else "no_changes",
    }
