from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ...api.deps import get_db, require_admin
from ...models import ChartPoint, DipStateResponse, RankingEntry
from ...repositories import dips as dip_repo
from ...repositories import symbols as symbol_repo
from ...services import dip_service

router = APIRouter(prefix="/dips", tags=["dips"])


@router.post("/refresh", response_model=List[RankingEntry], dependencies=[Depends(require_admin)])
def refresh(conn=Depends(get_db)):
    return dip_service.compute_ranking_details(conn, force_refresh=True)


@router.get("/ranking", response_model=List[RankingEntry])
def ranking(conn=Depends(get_db)):
    return dip_service.compute_ranking_details(conn)


@router.get("/states", response_model=List[DipStateResponse])
def states(conn=Depends(get_db)):
    states = dip_repo.load_states(conn)
    symbols = symbol_repo.list_symbols(conn)
    thresholds = {s.symbol: s.min_dip_pct for s in symbols}
    return [
        DipStateResponse(
            symbol=sym,
            ref_high=state.ref_high,
            days_below=state.days_below,
            last_price=state.last_price,
            dip_depth=dip_service.dip_depth(state),
            updated_at=state.updated_at,
        )
        for sym, state in states.items()
        if sym in thresholds
    ]


@router.get("/{symbol}/history", response_model=List[ChartPoint])
def chart(symbol: str, days: int = 180, conn=Depends(get_db)):
    sym = symbol.upper()
    symbol_cfg = symbol_repo.get_symbol(conn, sym)
    if symbol_cfg is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    return dip_service.get_chart_points(sym, symbol_cfg.min_dip_pct, days=days)


def _ranking(conn) -> List[RankingEntry]:
    ranked = dip_service.compute_ranking_details(conn)
    return ranked
