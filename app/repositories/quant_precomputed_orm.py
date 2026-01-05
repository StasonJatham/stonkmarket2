"""Repository for QuantPrecomputed table operations."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import QuantPrecomputed

logger = get_logger("repositories.quant_precomputed")


async def get_precomputed(symbol: str) -> QuantPrecomputed | None:
    """Get precomputed quant data for a symbol."""
    async with get_session() as session:
        result = await session.execute(
            select(QuantPrecomputed).where(QuantPrecomputed.symbol == symbol)
        )
        return result.scalar_one_or_none()


async def upsert_backtest_results(
    symbol: str,
    signal_name: str | None,
    n_trades: int,
    win_rate: float,
    total_return_pct: float,
    avg_return_per_trade: float,
    holding_days: int,
    buy_hold_return_pct: float,
    edge_pct: float,
    outperformed: bool,
    data_start: date | None = None,
    data_end: date | None = None,
) -> None:
    """Upsert signal backtest results for a symbol."""
    async with get_session() as session:
        stmt = insert(QuantPrecomputed).values(
            symbol=symbol,
            backtest_signal_name=signal_name,
            backtest_n_trades=n_trades,
            backtest_win_rate=Decimal(str(win_rate)),
            backtest_total_return_pct=Decimal(str(total_return_pct)),
            backtest_avg_return_per_trade=Decimal(str(avg_return_per_trade)),
            backtest_holding_days=holding_days,
            backtest_buy_hold_return_pct=Decimal(str(buy_hold_return_pct)),
            backtest_edge_pct=Decimal(str(edge_pct)),
            backtest_outperformed=outperformed,
            data_start=data_start,
            data_end=data_end,
            computed_at=datetime.now(),
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "backtest_signal_name": signal_name,
                "backtest_n_trades": n_trades,
                "backtest_win_rate": Decimal(str(win_rate)),
                "backtest_total_return_pct": Decimal(str(total_return_pct)),
                "backtest_avg_return_per_trade": Decimal(str(avg_return_per_trade)),
                "backtest_holding_days": holding_days,
                "backtest_buy_hold_return_pct": Decimal(str(buy_hold_return_pct)),
                "backtest_edge_pct": Decimal(str(edge_pct)),
                "backtest_outperformed": outperformed,
                "data_start": data_start,
                "data_end": data_end,
                "computed_at": datetime.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()


async def upsert_trade_strategy(
    symbol: str,
    entry_signal: str | None,
    entry_threshold: float | None,
    exit_signal: str | None,
    exit_threshold: float | None,
    n_trades: int,
    win_rate: float,
    total_return_pct: float,
    avg_return_pct: float,
    sharpe_ratio: float,
    buy_hold_return_pct: float,
    spy_return_pct: float,
    beats_both: bool,
) -> None:
    """Upsert full trade strategy results for a symbol."""
    async with get_session() as session:
        stmt = insert(QuantPrecomputed).values(
            symbol=symbol,
            trade_entry_signal=entry_signal,
            trade_entry_threshold=Decimal(str(entry_threshold)) if entry_threshold else None,
            trade_exit_signal=exit_signal,
            trade_exit_threshold=Decimal(str(exit_threshold)) if exit_threshold else None,
            trade_n_trades=n_trades,
            trade_win_rate=Decimal(str(win_rate)),
            trade_total_return_pct=Decimal(str(total_return_pct)),
            trade_avg_return_pct=Decimal(str(avg_return_pct)),
            trade_sharpe_ratio=Decimal(str(sharpe_ratio)),
            trade_buy_hold_return_pct=Decimal(str(buy_hold_return_pct)),
            trade_spy_return_pct=Decimal(str(spy_return_pct)),
            trade_beats_both=beats_both,
            computed_at=datetime.now(),
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "trade_entry_signal": entry_signal,
                "trade_entry_threshold": Decimal(str(entry_threshold)) if entry_threshold else None,
                "trade_exit_signal": exit_signal,
                "trade_exit_threshold": Decimal(str(exit_threshold)) if exit_threshold else None,
                "trade_n_trades": n_trades,
                "trade_win_rate": Decimal(str(win_rate)),
                "trade_total_return_pct": Decimal(str(total_return_pct)),
                "trade_avg_return_pct": Decimal(str(avg_return_pct)),
                "trade_sharpe_ratio": Decimal(str(sharpe_ratio)),
                "trade_buy_hold_return_pct": Decimal(str(buy_hold_return_pct)),
                "trade_spy_return_pct": Decimal(str(spy_return_pct)),
                "trade_beats_both": beats_both,
                "computed_at": datetime.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()


async def upsert_signal_combinations(
    symbol: str,
    combinations: list[dict[str, Any]],
) -> None:
    """Upsert signal combinations for a symbol."""
    async with get_session() as session:
        stmt = insert(QuantPrecomputed).values(
            symbol=symbol,
            signal_combinations=combinations,
            computed_at=datetime.now(),
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "signal_combinations": combinations,
                "computed_at": datetime.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()


async def upsert_dip_analysis(
    symbol: str,
    current_drawdown_pct: float,
    typical_pct: float,
    max_historical_pct: float,
    zscore: float,
    dip_type: str,
    action: str,
    confidence: float,
    reasoning: str,
    recovery_probability: float,
) -> None:
    """Upsert dip analysis results for a symbol."""
    async with get_session() as session:
        stmt = insert(QuantPrecomputed).values(
            symbol=symbol,
            dip_current_drawdown_pct=Decimal(str(current_drawdown_pct)),
            dip_typical_pct=Decimal(str(typical_pct)),
            dip_max_historical_pct=Decimal(str(max_historical_pct)),
            dip_zscore=Decimal(str(zscore)),
            dip_type=dip_type,
            dip_action=action,
            dip_confidence=Decimal(str(confidence)),
            dip_reasoning=reasoning,
            dip_recovery_probability=Decimal(str(recovery_probability)),
            computed_at=datetime.now(),
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "dip_current_drawdown_pct": Decimal(str(current_drawdown_pct)),
                "dip_typical_pct": Decimal(str(typical_pct)),
                "dip_max_historical_pct": Decimal(str(max_historical_pct)),
                "dip_zscore": Decimal(str(zscore)),
                "dip_type": dip_type,
                "dip_action": action,
                "dip_confidence": Decimal(str(confidence)),
                "dip_reasoning": reasoning,
                "dip_recovery_probability": Decimal(str(recovery_probability)),
                "computed_at": datetime.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()


async def upsert_current_signals(
    symbol: str,
    signals: dict[str, Any],
) -> None:
    """Upsert current signals for a symbol."""
    async with get_session() as session:
        stmt = insert(QuantPrecomputed).values(
            symbol=symbol,
            current_signals=signals,
            computed_at=datetime.now(),
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "current_signals": signals,
                "computed_at": datetime.now(),
            },
        )
        await session.execute(stmt)
        await session.commit()


async def upsert_all_quant_data(
    symbol: str,
    backtest: dict[str, Any] | None = None,
    trade_strategy: dict[str, Any] | None = None,
    combinations: list[dict[str, Any]] | None = None,
    dip_analysis: dict[str, Any] | None = None,
    current_signals: dict[str, Any] | None = None,
    dip_entry: dict[str, Any] | None = None,
    signal_triggers: dict[str, Any] | None = None,
    data_start: date | None = None,
    data_end: date | None = None,
) -> None:
    """Upsert all quant precomputed data for a symbol in one operation."""
    values: dict[str, Any] = {
        "symbol": symbol,
        "computed_at": datetime.now(),
        "data_start": data_start,
        "data_end": data_end,
    }
    
    if backtest:
        values.update({
            "backtest_signal_name": backtest.get("signal_name"),
            "backtest_n_trades": backtest.get("n_trades"),
            "backtest_win_rate": Decimal(str(backtest["win_rate"])) if backtest.get("win_rate") is not None else None,
            "backtest_total_return_pct": Decimal(str(backtest["total_return_pct"])) if backtest.get("total_return_pct") is not None else None,
            "backtest_avg_return_per_trade": Decimal(str(backtest["avg_return_per_trade"])) if backtest.get("avg_return_per_trade") is not None else None,
            "backtest_holding_days": backtest.get("holding_days"),
            "backtest_buy_hold_return_pct": Decimal(str(backtest["buy_hold_return_pct"])) if backtest.get("buy_hold_return_pct") is not None else None,
            "backtest_edge_pct": Decimal(str(backtest["edge_pct"])) if backtest.get("edge_pct") is not None else None,
            "backtest_outperformed": backtest.get("outperformed", False),
        })
    
    if trade_strategy:
        values.update({
            "trade_entry_signal": trade_strategy.get("entry_signal"),
            "trade_entry_threshold": Decimal(str(trade_strategy["entry_threshold"])) if trade_strategy.get("entry_threshold") is not None else None,
            "trade_exit_signal": trade_strategy.get("exit_signal"),
            "trade_exit_threshold": Decimal(str(trade_strategy["exit_threshold"])) if trade_strategy.get("exit_threshold") is not None else None,
            "trade_n_trades": trade_strategy.get("n_trades"),
            "trade_win_rate": Decimal(str(trade_strategy["win_rate"])) if trade_strategy.get("win_rate") is not None else None,
            "trade_total_return_pct": Decimal(str(trade_strategy["total_return_pct"])) if trade_strategy.get("total_return_pct") is not None else None,
            "trade_avg_return_pct": Decimal(str(trade_strategy["avg_return_pct"])) if trade_strategy.get("avg_return_pct") is not None else None,
            "trade_sharpe_ratio": Decimal(str(trade_strategy["sharpe_ratio"])) if trade_strategy.get("sharpe_ratio") is not None else None,
            "trade_buy_hold_return_pct": Decimal(str(trade_strategy["buy_hold_return_pct"])) if trade_strategy.get("buy_hold_return_pct") is not None else None,
            "trade_spy_return_pct": Decimal(str(trade_strategy["spy_return_pct"])) if trade_strategy.get("spy_return_pct") is not None else None,
            "trade_beats_both": trade_strategy.get("beats_both", False),
        })
    
    if combinations is not None:
        values["signal_combinations"] = combinations
    
    if dip_analysis:
        values.update({
            "dip_current_drawdown_pct": Decimal(str(dip_analysis["current_drawdown_pct"])) if dip_analysis.get("current_drawdown_pct") is not None else None,
            "dip_typical_pct": Decimal(str(dip_analysis["typical_pct"])) if dip_analysis.get("typical_pct") is not None else None,
            "dip_max_historical_pct": Decimal(str(dip_analysis["max_historical_pct"])) if dip_analysis.get("max_historical_pct") is not None else None,
            "dip_zscore": Decimal(str(dip_analysis["zscore"])) if dip_analysis.get("zscore") is not None else None,
            "dip_type": dip_analysis.get("type"),
            "dip_action": dip_analysis.get("action"),
            "dip_confidence": Decimal(str(dip_analysis["confidence"])) if dip_analysis.get("confidence") is not None else None,
            "dip_reasoning": dip_analysis.get("reasoning"),
            "dip_recovery_probability": Decimal(str(dip_analysis["recovery_probability"])) if dip_analysis.get("recovery_probability") is not None else None,
        })
    
    if current_signals is not None:
        values["current_signals"] = current_signals
    
    if dip_entry:
        values.update({
            "dip_entry_optimal_threshold": Decimal(str(dip_entry["optimal_threshold"])) if dip_entry.get("optimal_threshold") is not None else None,
            "dip_entry_optimal_price": Decimal(str(dip_entry["optimal_price"])) if dip_entry.get("optimal_price") is not None else None,
            "dip_entry_is_buy_now": dip_entry.get("is_buy_now", False),
            "dip_entry_signal_strength": Decimal(str(dip_entry["signal_strength"])) if dip_entry.get("signal_strength") is not None else None,
            "dip_entry_signal_reason": dip_entry.get("signal_reason"),
            "dip_entry_recovery_days": dip_entry.get("recovery_days"),
            "dip_entry_threshold_analysis": dip_entry.get("threshold_analysis"),
            "dip_entry_signal_triggers": dip_entry.get("signal_triggers"),
        })
    
    if signal_triggers is not None:
        values["signal_triggers"] = signal_triggers
    
    async with get_session() as session:
        # Build the set_ dict excluding symbol
        set_dict = {k: v for k, v in values.items() if k != "symbol"}
        
        stmt = insert(QuantPrecomputed).values(**values).on_conflict_do_update(
            index_elements=["symbol"],
            set_=set_dict,
        )
        await session.execute(stmt)
        await session.commit()


async def update_dip_entry(
    symbol: str,
    optimal_threshold: float,
    optimal_price: float,
    is_buy_now: bool,
    signal_strength: float,
    signal_reason: str,
    recovery_days: float,
    threshold_analysis: list[dict[str, Any]],
    max_profit_threshold: float | None = None,
    max_profit_price: float | None = None,
    max_profit_total_return: float | None = None,
    signal_triggers: dict[str, Any] | None = None,
) -> None:
    """Update dip entry data for a single symbol."""
    async with get_session() as session:
        values = {
            "symbol": symbol,
            "dip_entry_optimal_threshold": Decimal(str(optimal_threshold)),
            "dip_entry_optimal_price": Decimal(str(optimal_price)),
            "dip_entry_max_profit_threshold": Decimal(str(max_profit_threshold)) if max_profit_threshold else None,
            "dip_entry_max_profit_price": Decimal(str(max_profit_price)) if max_profit_price else None,
            "dip_entry_max_profit_total_return": Decimal(str(max_profit_total_return)) if max_profit_total_return else None,
            "dip_entry_is_buy_now": is_buy_now,
            "dip_entry_signal_strength": Decimal(str(signal_strength)),
            "dip_entry_signal_reason": signal_reason,
            "dip_entry_recovery_days": int(recovery_days),
            "dip_entry_threshold_analysis": threshold_analysis,
            "dip_entry_signal_triggers": signal_triggers,
            "computed_at": datetime.now(),
        }
        
        set_dict = {k: v for k, v in values.items() if k != "symbol"}
        
        stmt = insert(QuantPrecomputed).values(**values).on_conflict_do_update(
            index_elements=["symbol"],
            set_=set_dict,
        )
        await session.execute(stmt)
        await session.commit()
