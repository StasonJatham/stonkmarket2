from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import time

from ..config import settings
from ..models import ChartPoint, DipState, RankingEntry
from ..repositories import dips as dip_repo
from ..repositories import symbols as symbol_repo


def dip_depth(state: DipState) -> float:
    return (state.last_price - state.ref_high) / state.ref_high


def rank_dips(
    states: Dict[str, DipState], thresholds: Dict[str, Tuple[float, int]]
) -> List[Tuple[str, float]]:
    ranked: List[Tuple[str, float]] = []

    for symbol, state in states.items():
        min_dip_pct, min_days = thresholds[symbol]
        if dip_depth(state) <= -min_dip_pct and state.days_below >= min_days:
            ranked.append((symbol, dip_depth(state)))

    ranked.sort(key=lambda x: x[1])
    return ranked


_DOWNLOAD_CACHE: Dict[Tuple[str, int], Tuple[float, pd.DataFrame]] = {}
_CACHE_TTL_SECONDS = 120


def _download(symbols: Sequence[str], period_days: int) -> pd.DataFrame:
    tickers = " ".join(symbols)
    key = (tickers, period_days)
    now = time.time()
    cached = _DOWNLOAD_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    try:
        df = yf.download(
            tickers,
            period=f"{period_days}d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
        )
        _DOWNLOAD_CACHE[key] = (now, df)
        return df
    except Exception:
        return pd.DataFrame()


def _extract_close_frame(data: pd.DataFrame, symbol: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        return data[symbol]["Close"].dropna()
    return data["Close"].dropna()


def _state_from_series(close: pd.Series, min_dip_pct: float) -> DipState:
    prices = close.to_numpy(dtype=np.float64)
    if len(prices) == 0:
        return DipState(0.0, 0, 0.0)
    ref_high = float(np.max(prices))
    threshold = ref_high * (1.0 - min_dip_pct)
    below = prices <= threshold
    if not below[-1]:
        days_below = 0
    else:
        rev = below[::-1]
        days_below = int(np.argmax(~rev)) if (~rev).any() else int(len(prices))
    last_price = float(prices[-1])
    return DipState(ref_high=ref_high, days_below=days_below, last_price=last_price)


def refresh_states(conn) -> Dict[str, DipState]:
    symbols = symbol_repo.list_symbols(conn)
    if not symbols:
        return {}

    thresholds = {s.symbol: s.min_dip_pct for s in symbols}
    hist = _download([s.symbol for s in symbols], settings.history_days)
    states: Dict[str, DipState] = {}
    for sym in symbols:
        close = _extract_close_frame(hist, sym.symbol)
        if close.empty:
            single = _download([sym.symbol], settings.history_days)
            close = _extract_close_frame(single, sym.symbol)
        if close.empty:
            continue
        close_window = close.iloc[-min(len(close), settings.history_days) :]
        states[sym.symbol] = _state_from_series(close_window, thresholds[sym.symbol])

    dip_repo.save_states_batch(conn, states)
    return states


def refresh_symbol(conn, symbol: str) -> None:
    sym = symbol.upper()
    cfg = symbol_repo.get_symbol(conn, sym)
    if cfg is None:
        return
    data = _download([sym], settings.history_days)
    close = _extract_close_frame(data, sym)
    if close.empty:
        return
    close_window = close.iloc[-min(len(close), settings.history_days) :]
    state = _state_from_series(close_window, cfg.min_dip_pct)
    dip_repo.save_states_batch(conn, {sym: state})


def compute_ranking(conn) -> List[Tuple[str, float]]:
    states = refresh_states(conn)
    symbols = symbol_repo.list_symbols(conn)
    thresholds = {s.symbol: (s.min_dip_pct, s.min_days) for s in symbols}
    return rank_dips(states, thresholds)

_RANK_CACHE: Tuple[float, List[RankingEntry]] | None = None
_RANK_CACHE_TTL = 60.0


def compute_ranking_details(conn, force_refresh: bool = False) -> List[RankingEntry]:
    global _RANK_CACHE
    now = time.time()
    if not force_refresh and _RANK_CACHE and now - _RANK_CACHE[0] < _RANK_CACHE_TTL:
        return _RANK_CACHE[1]

    symbols = symbol_repo.list_symbols(conn)
    if not symbols:
        return []

    thresholds = {s.symbol: (s.min_dip_pct, s.min_days) for s in symbols}

    hist = _download([s.symbol for s in symbols], settings.history_days)
    entries: List[RankingEntry] = []
    states: Dict[str, DipState] = {}

    for sym in symbols:
        min_dip_pct, min_days = thresholds[sym.symbol]
        close = _extract_close_frame(hist, sym.symbol)
        if close.empty:
            single = _download([sym.symbol], settings.history_days)
            close = _extract_close_frame(single, sym.symbol)
        if close.empty:
            continue

        close_window = close.iloc[-min(len(close), settings.history_days) :]
        prices = close_window.to_numpy(dtype=np.float64)

        state = _state_from_series(close_window, min_dip_pct)
        states[sym.symbol] = state
        depth = dip_depth(state)
        if depth > -min_dip_pct or state.days_below < min_days:
            continue

        high_52w = float(np.max(prices))
        low_52w = float(np.min(prices))
        last_price = float(prices[-1])

        idx_max = int(np.argmax(prices))
        threshold = state.ref_high * (1.0 - min_dip_pct)
        dip_start_idx = None
        for j in range(idx_max + 1, len(prices)):
            if prices[j] <= threshold:
                dip_start_idx = j
                break
        days_since_dip = int(len(prices) - 1 - dip_start_idx) if dip_start_idx is not None else None

        entries.append(
            RankingEntry(
                symbol=sym.symbol,
                depth=depth,
                last_price=last_price,
                days_since_dip=days_since_dip,
                high_52w=high_52w,
                low_52w=low_52w,
            )
        )

    if states:
        dip_repo.save_states_batch(conn, states)

    entries.sort(key=lambda x: x.depth)
    _RANK_CACHE = (now, entries)
    return entries


def get_chart_points(
    symbol: str, min_dip_pct: float, days: int = settings.chart_days
) -> List[ChartPoint]:
    days = max(7, min(days, 365))

    # Use the full history window to anchor ref_high and dip start, then trim for chart display.
    history = _download([symbol], settings.history_days)
    close_full = _extract_close_frame(history, symbol)
    if close_full.empty:
        return []

    close_window = close_full.iloc[-min(len(close_full), settings.history_days) :]
    prices_window = close_window.to_numpy(dtype=np.float64)
    idx_max = int(np.argmax(prices_window))
    ref_high = float(prices_window[idx_max])
    ref_high_date = str(close_window.index[idx_max].date())
    threshold = ref_high * (1.0 - min_dip_pct)

    dip_start_idx = None
    for j in range(idx_max + 1, len(prices_window)):
        if prices_window[j] <= threshold:
            dip_start_idx = j
            break
    base_price = prices_window[dip_start_idx] if dip_start_idx is not None else None
    dip_start_date = (
        str(close_window.index[dip_start_idx].date()) if dip_start_idx is not None else None
    )

    # Trim to the requested chart range but keep metrics anchored to full-window high/dip.
    close_trimmed = close_window.iloc[-days:]
    prices_trimmed = close_trimmed.to_numpy(dtype=np.float64)

    chart_points: List[ChartPoint] = []
    for idx, price in zip(close_trimmed.index, prices_trimmed):
        drawdown = (price - ref_high) / ref_high if ref_high else 0.0
        since_dip = (price - base_price) / base_price if base_price else None
        chart_points.append(
            ChartPoint(
                date=str(idx.date()),
                close=float(price),
                threshold=float(threshold),
                ref_high=float(ref_high),
                drawdown=float(drawdown),
                since_dip=float(since_dip) if since_dip is not None else None,
                ref_high_date=ref_high_date,
                dip_start_date=dip_start_date,
            )
        )

    return chart_points
