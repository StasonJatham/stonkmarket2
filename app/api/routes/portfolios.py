"""Portfolio API routes."""

from __future__ import annotations

from datetime import date
from typing import List

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import require_user
from app.core.exceptions import NotFoundError, ValidationError
from app.core.security import TokenData
from app.repositories import auth_user_orm as auth_repo
from app.repositories import portfolios_orm as portfolios_repo
from app.repositories import portfolio_analytics_jobs_orm as analytics_jobs_repo
from app.portfolio.service import (
    run_portfolio_tools,
    split_tools,
    get_cached_tool_result,
    is_cached_result_stale,
    invalidate_portfolio_analytics_cache,
)
from app.schemas.portfolio import (
    PortfolioCreateRequest,
    PortfolioUpdateRequest,
    PortfolioResponse,
    PortfolioDetailResponse,
    HoldingInput,
    HoldingResponse,
    TransactionInput,
    TransactionResponse,
    PortfolioAnalyticsRequest,
    PortfolioAnalyticsResponse,
    PortfolioAnalyticsJobResponse,
    ToolResult,
)

router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


async def _get_user_id(user: TokenData) -> int:
    record = await auth_repo.get_user(user.sub)
    if not record:
        raise NotFoundError(message="User not found")
    return record.id


@router.get("", response_model=List[PortfolioResponse])
async def list_portfolios(
    user: TokenData = Depends(require_user),
) -> List[PortfolioResponse]:
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
        cash_balance=payload.cash_balance or 0,
    )
    return PortfolioResponse(**created)


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


@router.get("/{portfolio_id}/holdings", response_model=List[HoldingResponse])
async def list_holdings(
    portfolio_id: int,
    user: TokenData = Depends(require_user),
) -> List[HoldingResponse]:
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


@router.get("/{portfolio_id}/transactions", response_model=List[TransactionResponse])
async def list_transactions(
    portfolio_id: int,
    limit: int = Query(200, ge=1, le=500),
    user: TokenData = Depends(require_user),
) -> List[TransactionResponse]:
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
