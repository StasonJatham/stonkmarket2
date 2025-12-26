"""Symbol processing workflows for background tasks."""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta

from app.cache.cache import Cache
from app.core.logging import get_logger
from app.repositories import dip_state_orm as dip_state_repo
from app.repositories import dip_votes_orm as dip_votes_repo
from app.repositories import suggestions_orm as suggestions_repo
from app.repositories import symbols_orm as symbols_repo
from app.services.openai_client import generate_bio, rate_dip, summarize_company
from app.services.runtime_settings import get_runtime_setting
from app.services.stock_info import get_stock_info_async


logger = get_logger("services.symbol_processing")


async def process_new_symbol(symbol: str) -> None:
    """Process a newly added symbol (fetch data, AI content, cache invalidation)."""
    from app.dipfinder.service import get_dipfinder_service
    from app.services.fundamentals import get_fundamentals_for_analysis

    logger.info(f"[NEW SYMBOL] Starting background processing for: {symbol}")

    # Wait for the main transaction to commit (race condition prevention)
    await asyncio.sleep(0.5)

    # Verify symbol exists before proceeding
    exists = await symbols_repo.symbol_exists(symbol.upper())
    if not exists:
        logger.error(f"[NEW SYMBOL] FAILED: Symbol {symbol} not found in database - aborting")
        return

    # Set fetch_status to 'fetching' so UI shows loading state
    await symbols_repo.update_fetch_status(
        symbol.upper(),
        fetch_status="fetching",
        fetch_error=None,
    )

    steps_completed: list[str] = []

    try:
        # Step 1: Fetch Yahoo Finance info
        logger.info(f"[NEW SYMBOL] Step 1: Fetching Yahoo Finance data for {symbol}")
        info = await get_stock_info_async(symbol)
        if not info:
            logger.error(f"[NEW SYMBOL] FAILED: Could not fetch Yahoo data for {symbol} - aborting")
            await symbols_repo.update_fetch_status(
                symbol.upper(),
                fetch_status="error",
                fetch_error="Could not fetch data from Yahoo Finance",
            )
            try:
                from app.services.task_tracking import clear_symbol_task
                await clear_symbol_task(symbol)
            except Exception:
                pass
            try:
                ranking_cache = Cache(prefix="ranking", default_ttl=3600)
                await ranking_cache.invalidate_pattern("*")
            except Exception:
                pass
            return

        steps_completed.append("yahoo_fetch")

        name = info.get("name") or info.get("short_name")
        sector = info.get("sector")
        full_summary = info.get("summary")  # longBusinessSummary from yfinance
        current_price = info.get("current_price", 0)
        ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
        dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0

        logger.info(
            f"[NEW SYMBOL] Fetched: {symbol} - name='{name}', price=${current_price}, "
            f"ATH=${ath_price}, dip={dip_pct:.1f}%"
        )

        # Step 2: Update symbols with name/sector
        if name or sector:
            try:
                await symbols_repo.update_symbol_info(
                    symbol.upper(),
                    name=name,
                    sector=sector,
                )
                steps_completed.append("symbol_update")
                logger.info(f"[NEW SYMBOL] Step 2: Updated symbol table for {symbol}")
            except Exception as exc:
                logger.error(f"[NEW SYMBOL] Step 2 FAILED: Could not update symbol {symbol}: {exc}")

        # Step 3: Add to dip_state
        try:
            await dip_state_repo.upsert_dip_state(
                symbol=symbol.upper(),
                current_price=current_price,
                ath_price=ath_price,
                dip_percentage=dip_pct,
            )
            steps_completed.append("dip_state")
            logger.info(f"[NEW SYMBOL] Step 3: Added to dip_state with dip={dip_pct:.1f}%")
        except Exception as exc:
            logger.error(f"[NEW SYMBOL] Step 3 FAILED: Could not add to dip_state for {symbol}: {exc}")

        # Step 4: Fetch price history
        try:
            service = get_dipfinder_service()
            prices = await service.price_provider.get_prices(
                symbol.upper(),
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
            )
            if prices is not None and not prices.empty:
                steps_completed.append("price_history")
                logger.info(f"[NEW SYMBOL] Step 4: Fetched {len(prices)} days of price history")
            else:
                logger.warning(f"[NEW SYMBOL] Step 4: No price history returned for {symbol}")
        except Exception as exc:
            logger.warning(f"[NEW SYMBOL] Step 4 FAILED: Could not fetch price history for {symbol}: {exc}")

        # Step 5: Generate AI summary
        ai_summary = None
        if full_summary and len(full_summary) > 100:
            try:
                ai_summary = await summarize_company(
                    symbol=symbol,
                    name=name,
                    description=full_summary,
                )
                if ai_summary:
                    await symbols_repo.update_symbol_info(
                        symbol.upper(),
                        summary_ai=ai_summary,
                    )
                    steps_completed.append("ai_summary")
                    logger.info(f"[NEW SYMBOL] Step 5: Generated AI summary ({len(ai_summary)} chars)")
                else:
                    logger.warning("[NEW SYMBOL] Step 5: No AI summary generated (OpenAI not configured?)")
            except Exception as exc:
                logger.warning(f"[NEW SYMBOL] Step 5 FAILED: AI summary error for {symbol}: {exc}")
        else:
            logger.info("[NEW SYMBOL] Step 5: Skipped AI summary (no description or too short)")

        # Step 6: Generate AI bio
        bio = None
        try:
            bio = await generate_bio(
                symbol=symbol,
                name=name,
                sector=sector,
                summary=full_summary,
                dip_pct=dip_pct,
            )
            if bio:
                steps_completed.append("ai_bio")
                logger.info("[NEW SYMBOL] Step 6: Generated AI bio")
            else:
                logger.warning("[NEW SYMBOL] Step 6: No AI bio generated (OpenAI not configured?)")
        except Exception as exc:
            logger.warning(f"[NEW SYMBOL] Step 6 FAILED: AI bio error for {symbol}: {exc}")

        # Step 7: Generate AI rating
        rating_data = None
        try:
            fundamentals = await get_fundamentals_for_analysis(symbol)
            rating_data = await rate_dip(
                symbol=symbol,
                current_price=current_price,
                ref_high=ath_price,
                dip_pct=dip_pct,
                days_below=0,
                name=name,
                sector=sector,
                summary=full_summary,
                **fundamentals,
            )
            if rating_data:
                steps_completed.append("ai_rating")
                logger.info("[NEW SYMBOL] Step 7: Generated AI rating")
        except Exception as exc:
            logger.warning(f"[NEW SYMBOL] Step 7 FAILED: AI rating error for {symbol}: {exc}")

        # Step 8: Store AI analysis
        if bio or rating_data:
            try:
                await dip_votes_repo.upsert_ai_analysis(
                    symbol=symbol.upper(),
                    swipe_bio=bio,
                    ai_rating=rating_data.get("rating") if rating_data else None,
                    ai_reasoning=rating_data.get("reasoning") if rating_data else None,
                    is_batch=False,
                )
                steps_completed.append("ai_store")
                logger.info("[NEW SYMBOL] Step 8: Stored AI analysis")
            except Exception as exc:
                logger.warning(f"[NEW SYMBOL] Step 8 FAILED: Could not store AI analysis: {exc}")

        # Step 8.5: Run AI agent analysis
        try:
            from app.services.ai_agents import run_agent_analysis

            agent_result = await run_agent_analysis(symbol)
            if agent_result:
                steps_completed.append("ai_agents")
                logger.info(
                    "[NEW SYMBOL] Step 8.5: AI agents: %s (%s%%)",
                    agent_result.overall_signal,
                    agent_result.overall_confidence,
                )
            else:
                logger.warning("[NEW SYMBOL] Step 8.5: No agent analysis generated")
        except Exception as exc:
            logger.warning(f"[NEW SYMBOL] Step 8.5 FAILED: AI agents error for {symbol}: {exc}")

        # Step 9: Invalidate caches
        try:
            ranking_cache = Cache(prefix="ranking", default_ttl=3600)
            deleted = await ranking_cache.invalidate_pattern("*")
            stockinfo_cache = Cache(prefix="stockinfo", default_ttl=3600)
            await stockinfo_cache.delete(symbol.upper())
            symbols_cache = Cache(prefix="symbols", default_ttl=3600)
            await symbols_cache.invalidate_pattern("*")
            steps_completed.append("cache_invalidated")
            logger.info(
                f"[NEW SYMBOL] Step 9: Invalidated caches (ranking: {deleted} keys, stockinfo, symbols)"
            )
        except Exception as exc:
            logger.warning(f"[NEW SYMBOL] Step 9 FAILED: Could not invalidate cache: {exc}")

        # Step 10: Mark as fetched
        await symbols_repo.update_fetch_status(
            symbol.upper(),
            fetch_status="fetched",
            fetched_at=datetime.now(UTC),
        )
        from app.services.task_tracking import clear_symbol_task
        await clear_symbol_task(symbol)

        logger.info(
            f"[NEW SYMBOL] COMPLETED processing {symbol}. "
            f"Steps completed: {', '.join(steps_completed)}"
        )
    except Exception as exc:
        logger.error(f"[NEW SYMBOL] FATAL ERROR processing {symbol}: {exc}", exc_info=True)
        try:
            await symbols_repo.update_fetch_status(
                symbol.upper(),
                fetch_status="error",
                fetch_error=str(exc)[:500],
            )
        except Exception:
            pass
        try:
            from app.services.task_tracking import clear_symbol_task
            await clear_symbol_task(symbol)
        except Exception:
            pass
        try:
            ranking_cache = Cache(prefix="ranking", default_ttl=3600)
            await ranking_cache.invalidate_pattern("*")
        except Exception:
            pass


async def process_approved_symbol(symbol: str) -> None:
    """Process an approved suggestion (fetch data, AI content, update statuses)."""
    from app.dipfinder.service import get_dipfinder_service

    logger.info(f"Processing newly approved symbol: {symbol}")

    # Set fetch_status to 'fetching' so admin UI shows loading state
    await suggestions_repo.set_suggestion_fetching(symbol.upper())
    await symbols_repo.update_fetch_status(
        symbol.upper(),
        fetch_status="fetching",
        fetch_error=None,
    )

    try:
        info = await get_stock_info_async(symbol)
        if not info:
            logger.warning(f"Could not fetch Yahoo data for {symbol}")
            await suggestions_repo.set_suggestion_error(
                symbol.upper(), "Could not fetch data from Yahoo Finance"
            )
            await symbols_repo.update_fetch_status(
                symbol.upper(),
                fetch_status="error",
                fetch_error="Could not fetch data from Yahoo Finance",
            )
            try:
                from app.services.task_tracking import clear_symbol_task
                await clear_symbol_task(symbol)
            except Exception:
                pass
            return

        current_price = info.get("current_price", 0)
        ath_price = info.get("ath_price") or info.get("fifty_two_week_high", 0)
        dip_pct = ((ath_price - current_price) / ath_price * 100) if ath_price > 0 else 0

        logger.info(
            f"Fetched data for {symbol}: price=${current_price}, ATH=${ath_price}, dip={dip_pct:.1f}%"
        )

        name = info.get("name") or info.get("short_name")
        sector = info.get("sector")
        full_summary = info.get("summary")

        ai_summary = None
        ai_enabled = get_runtime_setting("ai_enrichment_enabled", True)
        if ai_enabled and full_summary and len(full_summary) > 100:
            existing_summary = await symbols_repo.get_symbol_summary_ai(symbol.upper())
            if not existing_summary:
                ai_summary = await summarize_company(
                    symbol=symbol,
                    name=name,
                    description=full_summary,
                )
                if ai_summary:
                    logger.info(f"Generated AI summary for {symbol}: {len(ai_summary)} chars")

        if name or sector or ai_summary:
            await symbols_repo.update_symbol_info(
                symbol.upper(),
                name=name,
                sector=sector,
                summary_ai=ai_summary,
            )
            logger.info(
                f"Updated symbol info for {symbol}: "
                f"name='{name}', sector='{sector}', summary_ai={'yes' if ai_summary else 'no'}"
            )

        await suggestions_repo.update_suggestion_stock_info(
            symbol=symbol.upper(),
            company_name=name,
            sector=sector,
            summary=full_summary[:1000] if full_summary else None,
            current_price=current_price,
            ath_price=ath_price,
        )

        # Fetch 365 days of price history
        try:
            service = get_dipfinder_service()
            prices = await service.price_provider.get_prices(
                symbol.upper(),
                start_date=date.today() - timedelta(days=365),
                end_date=date.today(),
            )
            if prices is not None and not prices.empty:
                logger.info(f"Fetched {len(prices)} days of price history for {symbol}")
            else:
                logger.warning(f"No price history returned for {symbol}")
        except Exception as exc:
            logger.warning(f"Failed to fetch price history for {symbol}: {exc}")

        await dip_state_repo.upsert_dip_state(
            symbol=symbol.upper(),
            current_price=current_price,
            ath_price=ath_price,
            dip_percentage=dip_pct,
        )

        ai_enabled = get_runtime_setting("ai_enrichment_enabled", True)
        if ai_enabled:
            bio = await generate_bio(
                symbol=symbol,
                dip_pct=dip_pct,
            )

            rating_data = await rate_dip(
                symbol=symbol,
                current_price=current_price,
                ref_high=ath_price,
                dip_pct=dip_pct,
            )

            if bio or rating_data:
                await dip_votes_repo.upsert_ai_analysis(
                    symbol=symbol,
                    swipe_bio=bio,
                    ai_rating=rating_data.get("rating") if rating_data else None,
                    ai_reasoning=rating_data.get("reasoning") if rating_data else None,
                    is_batch=False,
                )
                logger.info(
                    "Generated AI content for %s: bio=%s, rating=%s",
                    symbol,
                    "yes" if bio else "no",
                    rating_data.get("rating") if rating_data else "none",
                )
            else:
                logger.warning(f"No AI content generated for {symbol}")

            try:
                from app.services.ai_agents import run_agent_analysis

                agent_result = await run_agent_analysis(symbol)
                if agent_result:
                    logger.info(
                        "AI agents for %s: %s (%s%%)",
                        symbol,
                        agent_result.overall_signal,
                        agent_result.overall_confidence,
                    )
                else:
                    logger.warning(f"No agent analysis generated for {symbol}")
            except Exception as exc:
                logger.warning(f"AI agents error for {symbol}: {exc}")
        else:
            logger.info(f"AI enrichment disabled, skipping AI content generation for {symbol}")

        await suggestions_repo.set_suggestion_fetched(symbol.upper())
        await symbols_repo.update_fetch_status(
            symbol.upper(),
            fetch_status="fetched",
            fetch_error=None,
        )
        from app.services.task_tracking import clear_symbol_task
        await clear_symbol_task(symbol)

        ranking_cache = Cache(prefix="ranking", default_ttl=3600)
        deleted = await ranking_cache.invalidate_pattern("*")
        logger.info(
            f"Completed processing {symbol}, invalidated {deleted} ranking cache keys"
        )
    except Exception as exc:
        logger.error(f"Error processing approved symbol {symbol}: {exc}")
        await suggestions_repo.set_suggestion_error(symbol.upper(), str(exc))
        try:
            from app.services.task_tracking import clear_symbol_task
            await clear_symbol_task(symbol)
        except Exception:
            pass
