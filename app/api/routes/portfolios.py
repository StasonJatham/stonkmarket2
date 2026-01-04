"""Portfolio API routes."""

from __future__ import annotations

import secrets
from datetime import date, datetime

from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, status
from pydantic import BaseModel

from app.api.dependencies import (
    CurrentAdmin,
    EditPortfolio,
    SharePortfolio,
    ViewPortfolio,
    require_user,
)
from app.core.config import settings
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import TokenData
from app.portfolio.service import (
    get_cached_tool_result,
    invalidate_portfolio_analytics_cache,
    is_cached_result_stale,
    run_portfolio_tools,
    split_tools,
)
from app.repositories import auth_user_orm as auth_repo
from app.repositories import portfolio_analytics_jobs_orm as analytics_jobs_repo
from app.repositories import portfolios_orm as portfolios_repo
from app.schemas.bulk_import import (
    BulkImportRequest,
    BulkImportResponse,
    ImageExtractionResponse,
    ImportPositionResult,
    ImportResultStatus,
)
from app.schemas.portfolio import (
    HoldingInput,
    HoldingResponse,
    PortfolioAnalyticsJobResponse,
    PortfolioAnalyticsRequest,
    PortfolioAnalyticsResponse,
    PortfolioCreateRequest,
    PortfolioDetailResponse,
    PortfolioResponse,
    PortfolioUpdateRequest,
    PortfolioVisibility,
    PublicPortfolioSummary,
    ShareLinkResponse,
    ToolResult,
    TransactionInput,
    TransactionResponse,
    VisibilityUpdateRequest,
)


router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


async def _get_user_id(user: TokenData) -> int:
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    return record.id


@router.get("", response_model=list[PortfolioResponse])
async def list_portfolios(
    user: TokenData = Depends(require_user),
) -> list[PortfolioResponse]:
    user_id = await _get_user_id(user)
    rows = await portfolios_repo.list_portfolios(user_id)
    return [PortfolioResponse(**r) for r in rows]


@router.post("", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_portfolio(
    payload: PortfolioCreateRequest,
    user: TokenData = Depends(require_user),
) -> PortfolioResponse:
    user_id = await _get_user_id(user)
    created = await portfolios_repo.create_portfolio(
        user_id,
        payload.name,
        description=payload.description,
        base_currency=payload.base_currency,
        visibility=payload.visibility.value,
    )
    return PortfolioResponse(**created)


# ============================================================================
# Public/Shared Endpoints (MUST be before /{portfolio_id} to avoid route conflicts)
# ============================================================================


@router.get("/public", response_model=list[PublicPortfolioSummary])
async def list_public_portfolios(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[PublicPortfolioSummary]:
    """List all public portfolios for discovery."""
    rows = await portfolios_repo.list_public_portfolios(limit=limit, offset=offset)
    return [PublicPortfolioSummary(**r) for r in rows]


@router.get("/shared/{share_token}", response_model=PortfolioDetailResponse)
async def get_shared_portfolio(
    share_token: str,
) -> PortfolioDetailResponse:
    """Access a portfolio via share link (no authentication required)."""
    portfolio = await portfolios_repo.get_portfolio_by_share_token(share_token)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found or sharing expired")
    holdings = await portfolios_repo.list_holdings(portfolio["id"])
    transactions = await portfolios_repo.list_transactions(portfolio["id"])
    return PortfolioDetailResponse(
        **portfolio,
        holdings=[HoldingResponse(**h) for h in holdings],
        transactions=[TransactionResponse(**t) for t in transactions],
    )


# ============================================================================
# Authenticated Portfolio Endpoints
# ============================================================================


@router.get("/{portfolio_id}", response_model=PortfolioDetailResponse)
async def get_portfolio(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> PortfolioDetailResponse:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    transactions = await portfolios_repo.list_transactions(portfolio_id)
    return PortfolioDetailResponse(
        **portfolio,
        holdings=[HoldingResponse(**h) for h in holdings],
        transactions=[TransactionResponse(**t) for t in transactions],
    )


@router.patch("/{portfolio_id}", response_model=PortfolioResponse)
async def update_portfolio(
    portfolio_id: int,
    payload: PortfolioUpdateRequest,
    user: TokenData = Depends(require_user),
) -> PortfolioResponse:
    user_id = await _get_user_id(user)
    updated = await portfolios_repo.update_portfolio(
        portfolio_id,
        user_id,
        name=payload.name,
        description=payload.description,
        base_currency=payload.base_currency,
        cash_balance=payload.cash_balance,
        is_active=payload.is_active,
    )
    if not updated:
        raise NotFoundError(message="Portfolio not found")
    await invalidate_portfolio_analytics_cache(portfolio_id)
    return PortfolioResponse(**updated)


@router.delete("/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_portfolio(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> None:
    user_id = await _get_user_id(user)
    success = await portfolios_repo.archive_portfolio(portfolio_id, user_id)
    if not success:
        raise NotFoundError(message="Portfolio not found")
    await invalidate_portfolio_analytics_cache(portfolio_id)


# ============================================================================
# Visibility & Sharing Endpoints
# ============================================================================


@router.patch("/{portfolio_id}/visibility", response_model=PortfolioResponse)
async def update_visibility(
    payload: VisibilityUpdateRequest,
    portfolio_id: SharePortfolio,
) -> PortfolioResponse:
    """Update portfolio visibility (private, public, shared_link)."""
    updated = await portfolios_repo.update_portfolio_visibility(
        portfolio_id,
        visibility=payload.visibility.value,
    )
    if not updated:
        raise NotFoundError(message="Portfolio not found")
    return PortfolioResponse(**updated)


@router.post("/{portfolio_id}/share", response_model=ShareLinkResponse)
async def create_share_link(
    portfolio_id: SharePortfolio,
) -> ShareLinkResponse:
    """Generate a share link for the portfolio."""
    # Generate a URL-safe token
    token = secrets.token_urlsafe(32)
    
    updated = await portfolios_repo.set_share_token(portfolio_id, token)
    if not updated:
        raise NotFoundError(message="Portfolio not found")
    
    # Construct the share URL
    base_url = settings.oauth_redirect_url or "https://stonkmarket.app"
    share_url = f"{base_url}/portfolios/shared/{token}"
    
    return ShareLinkResponse(
        share_token=token,
        share_url=share_url,
        shared_at=updated["shared_at"],
    )


@router.delete("/{portfolio_id}/share", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_share_link(
    portfolio_id: SharePortfolio,
) -> None:
    """Revoke the share link for the portfolio."""
    await portfolios_repo.set_share_token(portfolio_id, None)


@router.get("/{portfolio_id}/holdings", response_model=list[HoldingResponse])
async def list_holdings(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> list[HoldingResponse]:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    return [HoldingResponse(**h) for h in holdings]


@router.post("/{portfolio_id}/holdings", response_model=HoldingResponse)
async def upsert_holding(
    portfolio_id: int,
    payload: HoldingInput,
    user: TokenData = Depends(require_user),
) -> HoldingResponse:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    holding = await portfolios_repo.upsert_holding(
        portfolio_id,
        payload.symbol,
        quantity=payload.quantity,
        avg_cost=payload.avg_cost,
        target_weight=payload.target_weight,
    )
    await invalidate_portfolio_analytics_cache(portfolio_id)
    return HoldingResponse(**holding)


@router.delete("/{portfolio_id}/holdings/{symbol}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_holding(
    portfolio_id: int,
    symbol: str,
    user: TokenData = Depends(require_user),
) -> None:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    success = await portfolios_repo.delete_holding(portfolio_id, symbol)
    if not success:
        raise NotFoundError(message="Holding not found")
    await invalidate_portfolio_analytics_cache(portfolio_id)


@router.get("/{portfolio_id}/transactions", response_model=list[TransactionResponse])
async def list_transactions(
    portfolio_id: int,
    limit: int = Query(200, ge=1, le=500),
    user: TokenData = Depends(require_user),
) -> list[TransactionResponse]:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    rows = await portfolios_repo.list_transactions(portfolio_id, limit=limit)
    return [TransactionResponse(**r) for r in rows]


@router.post("/{portfolio_id}/transactions", response_model=TransactionResponse)
async def add_transaction(
    portfolio_id: int,
    payload: TransactionInput,
    user: TokenData = Depends(require_user),
) -> TransactionResponse:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")

    side = payload.side.lower()
    if side not in {"buy", "sell", "dividend", "split", "deposit", "withdrawal"}:
        raise ValidationError(message="Invalid transaction side")

    transaction = await portfolios_repo.add_transaction(
        portfolio_id,
        payload.symbol,
        side=side,
        quantity=payload.quantity,
        price=payload.price,
        fees=payload.fees,
        trade_date=payload.trade_date,
        notes=payload.notes,
    )

    if side in {"buy", "sell"} and payload.quantity is not None and payload.price is not None:
        await portfolios_repo.apply_transaction_to_holdings(
            portfolio_id,
            symbol=payload.symbol,
            side=side,
            quantity=payload.quantity,
            price=payload.price,
        )

    await invalidate_portfolio_analytics_cache(portfolio_id)
    return TransactionResponse(**transaction)


@router.delete("/{portfolio_id}/transactions/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transaction(
    portfolio_id: int,
    transaction_id: int,
    user: TokenData = Depends(require_user),
) -> None:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    success = await portfolios_repo.delete_transaction(portfolio_id, transaction_id)
    if not success:
        raise NotFoundError(message="Transaction not found")
    await invalidate_portfolio_analytics_cache(portfolio_id)


@router.post("/{portfolio_id}/analytics", response_model=PortfolioAnalyticsResponse)
async def run_analytics(
    portfolio_id: int,
    payload: PortfolioAnalyticsRequest,
    user: TokenData = Depends(require_user),
) -> PortfolioAnalyticsResponse:
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    holdings = await portfolios_repo.list_holdings(portfolio_id)
    if not holdings:
        raise ValidationError(message="Portfolio has no holdings")

    tools = payload.tools or [
        "quantstats",
        "skfolio",
        "arch",
        "prophet",
    ]
    light_tools, heavy_tools = split_tools(tools)
    params = payload.params or {}
    results = []
    scheduled_tools: list[str] = []

    if light_tools:
        try:
            results.extend(
                await run_portfolio_tools(
                    portfolio_id,
                    user_id=user_id,
                    tools=light_tools,
                    window=payload.window,
                    start_date=payload.start_date,
                    end_date=payload.end_date,
                    benchmark=payload.benchmark,
                    params=params,
                    force_refresh=payload.force_refresh,
                )
            )
        except ValueError as exc:
            raise ValidationError(message=str(exc))

    if heavy_tools:
        for tool in heavy_tools:
            if payload.force_refresh:
                scheduled_tools.append(tool)
                continue
            cached = await get_cached_tool_result(
                portfolio_id,
                tool=tool,
                window=payload.window,
                start_date=payload.start_date,
                end_date=payload.end_date,
                params=params,
            )
            if cached:
                if cached.get("source") == "db" and is_cached_result_stale(
                    tool,
                    cached.get("generated_at"),
                ):
                    cached.setdefault("warnings", []).append(
                        "Cached result is stale; refresh scheduled"
                    )
                    scheduled_tools.append(tool)
                results.append(cached)
            else:
                scheduled_tools.append(tool)

    job_id = None
    job_status = None
    if scheduled_tools:
        job = await analytics_jobs_repo.create_job(
            portfolio_id,
            user_id,
            tools=scheduled_tools,
            window=payload.window,
            start_date=payload.start_date,
            end_date=payload.end_date,
            benchmark=payload.benchmark,
            params=params,
            force_refresh=payload.force_refresh,
        )
        job_id = job.get("job_id")
        job_status = job.get("status")

    tool_order = {name.lower(): idx for idx, name in enumerate(tools)}
    results.sort(key=lambda item: tool_order.get(item.get("tool", ""), 999))

    return PortfolioAnalyticsResponse(
        portfolio_id=portfolio_id,
        as_of_date=payload.end_date or date.today(),
        results=[ToolResult(**r) for r in results],
        job_id=job_id,
        job_status=job_status,
        scheduled_tools=scheduled_tools,
    )


@router.get("/{portfolio_id}/analytics/jobs/{job_id}", response_model=PortfolioAnalyticsJobResponse)
async def get_analytics_job_status(
    portfolio_id: int,
    job_id: str,
    user: TokenData = Depends(require_user),
) -> PortfolioAnalyticsJobResponse:
    user_id = await _get_user_id(user)
    job = await analytics_jobs_repo.get_job(job_id, user_id)
    if not job or job["portfolio_id"] != portfolio_id:
        raise NotFoundError(message="Analytics job not found")
    return PortfolioAnalyticsJobResponse(**job)


# =============================================================================
# Bulk Import Endpoints
# =============================================================================


@router.post(
    "/{portfolio_id}/import/extract-image",
    response_model=ImageExtractionResponse,
    summary="Extract positions from portfolio screenshot",
    description="""
    Upload a screenshot of your portfolio from any broker/app.
    AI will analyze the image and extract stock positions.
    
    Supported formats: PNG, JPEG, WebP, GIF
    Max size: 10MB
    
    Returns extracted positions with confidence scores.
    User should review and edit before importing.
    """,
)
async def extract_positions_from_image(
    portfolio_id: int,
    file: UploadFile = File(..., description="Portfolio screenshot image"),
    user: TokenData = Depends(require_user),
) -> ImageExtractionResponse:
    """Extract portfolio positions from an uploaded image."""
    import base64
    
    from app.services.portfolio_image_extractor import (
        extract_positions_from_image as extract,
        MAX_IMAGE_SIZE,
        SUPPORTED_MIME_TYPES,
    )
    
    # Verify user owns portfolio
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    
    # Validate file type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in SUPPORTED_MIME_TYPES:
        raise ValidationError(
            message=f"Unsupported file type: {content_type}. Supported: PNG, JPEG, WebP, GIF, HEIC"
        )
    
    # Read file content
    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise ValidationError(
            message=f"File too large: {len(content) / 1024 / 1024:.1f}MB (max: 10MB)"
        )
    
    # Convert to base64 and extract
    image_base64 = base64.b64encode(content).decode("utf-8")
    result = await extract(image_base64, content_type)
    
    return result


@router.post(
    "/{portfolio_id}/import/bulk",
    response_model=BulkImportResponse,
    summary="Bulk import positions into portfolio",
    description="""
    Import multiple positions into a portfolio at once.
    
    Typically used after extracting positions from an image,
    where the user has reviewed and edited the data.
    
    Duplicate detection: If a position with the same symbol exists,
    it will be skipped (if skip_duplicates=True) or updated.
    """,
)
async def bulk_import_positions(
    portfolio_id: int,
    payload: BulkImportRequest,
    user: TokenData = Depends(require_user),
) -> BulkImportResponse:
    """Bulk import positions into a portfolio."""
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    
    # Get existing holdings to detect duplicates
    existing_holdings = await portfolios_repo.list_holdings(portfolio_id)
    existing_symbols = {h["symbol"] for h in existing_holdings}
    
    results: list[ImportPositionResult] = []
    created = 0
    updated = 0
    skipped = 0
    failed = 0
    
    for pos in payload.positions:
        symbol = pos.symbol.upper()
        
        try:
            # Check for duplicate
            is_duplicate = symbol in existing_symbols
            
            if is_duplicate and payload.skip_duplicates:
                results.append(ImportPositionResult(
                    symbol=symbol,
                    status=ImportResultStatus.SKIPPED,
                    message="Position already exists",
                ))
                skipped += 1
                continue
            
            # Create or update holding
            holding = await portfolios_repo.upsert_holding(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=pos.quantity,
                avg_cost=pos.avg_cost,
            )
            
            if is_duplicate:
                results.append(ImportPositionResult(
                    symbol=symbol,
                    status=ImportResultStatus.UPDATED,
                    message="Position updated",
                    holding_id=holding["id"],
                ))
                updated += 1
            else:
                results.append(ImportPositionResult(
                    symbol=symbol,
                    status=ImportResultStatus.CREATED,
                    message="Position created",
                    holding_id=holding["id"],
                ))
                created += 1
                existing_symbols.add(symbol)  # Track for remaining imports
                
        except Exception as e:
            results.append(ImportPositionResult(
                symbol=symbol,
                status=ImportResultStatus.FAILED,
                message=str(e),
            ))
            failed += 1
    
    # Invalidate analytics cache
    await invalidate_portfolio_analytics_cache(portfolio_id)
    
    return BulkImportResponse(
        success=failed == 0,
        total=len(payload.positions),
        created=created,
        updated=updated,
        skipped=skipped,
        failed=failed,
        results=results,
    )


# =============================================================================
# Sparkline Data for Holdings
# =============================================================================


class SparklinePoint(BaseModel):
    """A single point in sparkline data."""
    date: str
    close: float


class TradeMarker(BaseModel):
    """A trade marker to overlay on sparkline."""
    date: str
    side: str  # buy, sell
    price: float


class HoldingSparklineResponse(BaseModel):
    """Sparkline data with trade markers for a holding."""
    symbol: str
    prices: list[SparklinePoint]
    trades: list[TradeMarker]
    change_pct: float | None  # Overall % change from start to end


class BatchSparklineRequest(BaseModel):
    """Request for batch sparkline data."""
    symbols: list[str]
    days: int = 180  # 6 months default


class BatchSparklineResponse(BaseModel):
    """Response with sparklines for multiple holdings."""
    sparklines: dict[str, HoldingSparklineResponse]


@router.post(
    "/{portfolio_id}/sparklines",
    response_model=BatchSparklineResponse,
    summary="Get sparkline data for holdings",
    description="Get mini price charts with trade markers for portfolio holdings.",
)
async def get_holdings_sparklines(
    portfolio_id: int,
    payload: BatchSparklineRequest,
    user: TokenData = Depends(require_user),
) -> BatchSparklineResponse:
    """
    Get sparkline chart data for multiple holdings.
    
    Returns price history (min 6 months) and trade entry points for overlay.
    """
    from datetime import timedelta
    from app.repositories import price_history_orm as price_history_repo
    
    user_id = await _get_user_id(user)
    portfolio = await portfolios_repo.get_portfolio(portfolio_id, user_id)
    if not portfolio:
        raise NotFoundError(message="Portfolio not found")
    
    # Get transactions for markers
    transactions = await portfolios_repo.list_transactions(portfolio_id)
    trades_by_symbol: dict[str, list[TradeMarker]] = {}
    for tx in transactions:
        symbol = tx["symbol"]
        if symbol not in payload.symbols:
            continue
        if tx["side"] not in ("buy", "sell"):
            continue
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(TradeMarker(
            date=tx["trade_date"].isoformat(),
            side=tx["side"],
            price=float(tx["price"]) if tx.get("price") else 0,
        ))
    
    # Fetch price data
    end_date = date.today()
    start_date = end_date - timedelta(days=payload.days)
    
    sparklines: dict[str, HoldingSparklineResponse] = {}
    
    for symbol in payload.symbols:
        df = await price_history_repo.get_prices_as_dataframe(symbol, start_date, end_date)
        
        if df is None or df.empty or "Close" not in df.columns:
            # Try PriceService fallback
            from app.services.prices import get_price_service
            price_service = get_price_service()
            try:
                df = await price_service.get_prices(symbol, start_date, end_date)
            except Exception:
                pass
        
        prices: list[SparklinePoint] = []
        change_pct = None
        
        if df is not None and not df.empty and "Close" in df.columns:
            # Sample down to ~60 points for performance (every 3rd day for 6 months)
            step = max(1, len(df) // 60)
            sampled = df.iloc[::step]
            
            for idx, row in sampled.iterrows():
                dt = idx if isinstance(idx, date) else idx.date() if hasattr(idx, 'date') else idx
                prices.append(SparklinePoint(
                    date=str(dt),
                    close=float(row["Close"]),
                ))
            
            # Calculate overall change
            if len(prices) >= 2:
                first_price = prices[0].close
                last_price = prices[-1].close
                if first_price > 0:
                    change_pct = ((last_price - first_price) / first_price) * 100
        
        sparklines[symbol] = HoldingSparklineResponse(
            symbol=symbol,
            prices=prices,
            trades=trades_by_symbol.get(symbol, []),
            change_pct=change_pct,
        )
    
    return BatchSparklineResponse(sparklines=sparklines)
