from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ...api.deps import get_db, require_user
from ...models import SymbolPayload, SymbolResponse, SymbolUpdatePayload
from ...repositories import dips as dip_repo
from ...repositories import symbols as symbol_repo
from ...services import dip_service

router = APIRouter(prefix="/symbols", tags=["symbols"], dependencies=[Depends(require_user)])


@router.get("", response_model=List[SymbolResponse])
def list_all(conn=Depends(get_db)):
    symbols = symbol_repo.list_symbols(conn)
    return [
        SymbolResponse(symbol=s.symbol, min_dip_pct=s.min_dip_pct, min_days=s.min_days)
        for s in symbols
    ]


@router.post("", response_model=SymbolResponse, status_code=status.HTTP_201_CREATED)
def add_symbol(payload: SymbolPayload, conn=Depends(get_db)):
    created = symbol_repo.upsert_symbol(
        conn, payload.symbol.upper(), payload.min_dip_pct, payload.min_days
    )
    dip_repo.delete_state(conn, created.symbol)
    dip_service.refresh_symbol(conn, created.symbol)
    return SymbolResponse(
        symbol=created.symbol, min_dip_pct=created.min_dip_pct, min_days=created.min_days
    )


@router.put("/{symbol}", response_model=SymbolResponse)
def update_symbol(symbol: str, payload: SymbolUpdatePayload, conn=Depends(get_db)):
    sym = symbol.upper()
    existing = symbol_repo.get_symbol(conn, sym)
    if existing is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    updated = symbol_repo.upsert_symbol(
        conn, sym, payload.min_dip_pct, payload.min_days
    )
    dip_repo.delete_state(conn, sym)
    dip_service.refresh_symbol(conn, sym)
    return SymbolResponse(
        symbol=updated.symbol,
        min_dip_pct=updated.min_dip_pct,
        min_days=updated.min_days,
    )


@router.delete("/{symbol}", status_code=status.HTTP_204_NO_CONTENT)
def delete_symbol(symbol: str, conn=Depends(get_db)):
    sym = symbol.upper()
    existing = symbol_repo.get_symbol(conn, sym)
    if existing is None:
        raise HTTPException(status_code=404, detail="Symbol not found")
    symbol_repo.delete_symbol(conn, sym)
    dip_repo.delete_state(conn, sym)
    return None
