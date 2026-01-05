#!/usr/bin/env python
"""Run strategy optimization for a single symbol."""
import asyncio
import sys
from decimal import Decimal
from datetime import date, timedelta, datetime, UTC

async def run_single_symbol(symbol: str):
    from sqlalchemy.dialects.postgresql import insert
    from app.database.connection import get_session
    from app.database.orm import StrategySignal, StockFundamentals
    from app.repositories import price_history_orm as price_history_repo
    from app.quant_engine.backtest_v2.baseline_strategies import (
        BaselineEngine,
        RecommendationType,
    )
    from app.quant_engine.dip_entry_optimizer import DipEntryOptimizer
    
    print(f"Running strategy optimization for {symbol}...")
    
    # Get price data - use 3 years to match hero chart, 5 years for established stocks
    end_date = date.today()
    start_3y = end_date - timedelta(days=1095)  # 3 years
    start_5y = end_date - timedelta(days=1825)  # 5 years
    
    # First check how much data the stock has
    df_all = await price_history_repo.get_prices_as_dataframe(symbol, start_5y, end_date)
    
    if df_all is None or len(df_all) < 200:
        print(f"Insufficient data for {symbol}")
        return
    
    # Use 3 years by default (matches hero chart)
    # Only use 5 years if stock has 5+ years of data
    if len(df_all) >= 1250:  # ~5 years of trading days
        df = df_all
        print(f"Using 5 years of data ({len(df)} days)")
    else:
        # Use 3 years - trim to match hero chart period
        df = await price_history_repo.get_prices_as_dataframe(symbol, start_3y, end_date)
        if df is None or len(df) < 200:
            # Fallback to all available data if 3y is too short
            df = df_all
            print(f"Using all available data ({len(df)} days)")
        else:
            print(f"Using 3 years of data ({len(df)} days)")
    
    # Get SPY for the same period
    stock_start = df.index[0]
    spy_df = await price_history_repo.get_prices_as_dataframe("SPY", stock_start.date(), end_date)
    
    print(f"Got {len(df)} days of price data")
    
    # Run BaselineEngine comparison
    engine = BaselineEngine(
        prices=df,
        spy_prices=spy_df,
        symbol=symbol,
        initial_capital=10_000.0,
        monthly_contribution=1_000.0,
    )
    comparison = engine.run_all()
    
    rec = comparison.recommendation
    
    # Map recommendation type to strategy name
    strategy_name_map = {
        RecommendationType.DCA: "dollar_cost_average",
        RecommendationType.BUY_AND_HOLD: "buy_and_hold",
        RecommendationType.BUY_DIPS: "buy_dips_hold",
        RecommendationType.OPTIMIZED_STRATEGY: "optimized_technical",
        RecommendationType.SPY_DCA: "spy_dca",
        RecommendationType.SWITCH_TO_SPY: "switch_to_spy",
    }
    strategy_name = strategy_name_map.get(rec.recommendation, "dca")
    
    # Get best result
    results_map = {
        "DCA Monthly": comparison.dca,
        "Buy & Hold": comparison.buy_hold,
        "Buy Dips & Hold": comparison.buy_dips,
    }
    
    best_name = comparison.ranked_by_return[0] if comparison.ranked_by_return else "DCA Monthly"
    best_result = results_map.get(best_name, comparison.dca)
    
    # Build strategy comparison
    strategy_comparison = {
        "initial_capital": 10_000.0,
        "monthly_contribution": 1_000.0,
        "backtest_days": len(df),
        "strategies": {
            "dca": {
                "name": "DCA Monthly",
                "description": "Invest $1k every month regardless of price",
                "total_invested": comparison.dca.total_invested,
                "final_value": comparison.dca.final_value,
                "total_return_pct": comparison.dca.total_return_pct,
                "n_buys": comparison.dca.total_buys,
            },
            "buy_dips": {
                "name": "Buy Dips & Hold",
                "description": "Accumulate $1k/mo cash, deploy only on dips",
                "total_invested": comparison.buy_dips.total_invested,
                "final_value": comparison.buy_dips.final_value,
                "total_return_pct": comparison.buy_dips.total_return_pct,
                "n_buys": comparison.buy_dips.total_buys,
            },
            "buy_hold": {
                "name": "Buy & Hold",
                "description": "Invest $10k on day 1, hold forever",
                "total_invested": comparison.buy_hold.total_invested,
                "final_value": comparison.buy_hold.final_value,
                "total_return_pct": comparison.buy_hold.total_return_pct,
                "n_buys": 1,
            },
            "spy_dca": {
                "name": "SPY DCA",
                "description": "Same schedule but buying SPY instead",
                "total_invested": comparison.spy_dca.total_invested,
                "final_value": comparison.spy_dca.final_value,
                "total_return_pct": comparison.spy_dca.total_return_pct,
                "n_buys": comparison.spy_dca.total_buys,
            },
        },
        "ranked_by_return": comparison.ranked_by_return[:5],
        "winner": comparison.ranked_by_return[0] if comparison.ranked_by_return else "DCA Monthly",
    }
    
    print(f"\n=== Strategy Comparison for {symbol} ===")
    print(f"Winner: {strategy_comparison['winner']}")
    for key, strat in strategy_comparison["strategies"].items():
        print(f"  {strat['name']}: ${strat['total_invested']:,.0f} -> ${strat['final_value']:,.0f} ({strat['total_return_pct']:+.1f}%)")
    
    # Generate synthetic trades for chart
    best_trades = []
    if strategy_name == "buy_dips_hold":
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        close_prices = df[close_col]
        dip_threshold = engine.dip_threshold_pct / 100 if engine.dip_threshold_pct else -0.10
        rolling_high = close_prices.rolling(window=63).max()
        drawdown = (close_prices - rolling_high) / rolling_high
        dip_mask = drawdown <= dip_threshold
        dip_dates = close_prices[dip_mask].index
        step = max(1, len(dip_dates) // 20)
        for i in range(0, len(dip_dates), step):
            dip_date = dip_dates[i]
            dip_price = float(close_prices.loc[dip_date])
            best_trades.append({
                "entry_date": dip_date.strftime("%Y-%m-%d"),
                "exit_date": None,
                "entry_price": dip_price,
                "exit_price": None,
                "pnl_pct": 0,
                "exit_reason": f"dip_{abs(dip_threshold*100):.0f}pct",
                "holding_days": None,
            })
        best_trades = best_trades[-20:]
    
    # Calculate vs buy & hold
    vs_buy_hold = best_result.total_return_pct - comparison.buy_hold.total_return_pct
    
    # Upsert to database
    async with get_session() as session:
        stmt = insert(StrategySignal).values(
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_params={"winner": best_name},
            signal_type="BUY" if rec.recommendation in (RecommendationType.DCA, RecommendationType.BUY_DIPS) else "HOLD",
            signal_reason=rec.headline,
            has_active_signal=True,
            total_return_pct=Decimal(str(best_result.total_return_pct)),
            sharpe_ratio=Decimal(str(best_result.sharpe_ratio)),
            win_rate=Decimal(str(best_result.win_rate_pct)),
            max_drawdown_pct=Decimal(str(best_result.max_drawdown_pct)),
            n_trades=best_result.total_buys,
            recency_weighted_return=Decimal(str(best_result.annualized_return_pct)),
            current_year_return_pct=Decimal(str(best_result.total_return_pct)),
            current_year_win_rate=Decimal(str(best_result.win_rate_pct)),
            current_year_trades=0,
            vs_buy_hold_pct=Decimal(str(vs_buy_hold)),
            vs_spy_pct=Decimal(str(best_result.total_return_pct - comparison.spy_dca.total_return_pct)),
            beats_buy_hold=vs_buy_hold > 0,
            beats_spy=best_result.total_return_pct > comparison.spy_dca.total_return_pct,
            fundamentals_healthy=True,
            fundamental_concerns=[],
            is_statistically_valid=True,
            recent_trades=best_trades,
            strategy_comparison=strategy_comparison,
            indicators_used=["price", "dip_detection"],
            typical_recovery_days=None,
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "strategy_name": strategy_name,
                "strategy_params": {"winner": best_name},
                "signal_type": "BUY" if rec.recommendation in (RecommendationType.DCA, RecommendationType.BUY_DIPS) else "HOLD",
                "signal_reason": rec.headline,
                "has_active_signal": True,
                "total_return_pct": Decimal(str(best_result.total_return_pct)),
                "sharpe_ratio": Decimal(str(best_result.sharpe_ratio)),
                "win_rate": Decimal(str(best_result.win_rate_pct)),
                "max_drawdown_pct": Decimal(str(best_result.max_drawdown_pct)),
                "n_trades": best_result.total_buys,
                "recency_weighted_return": Decimal(str(best_result.annualized_return_pct)),
                "vs_buy_hold_pct": Decimal(str(vs_buy_hold)),
                "vs_spy_pct": Decimal(str(best_result.total_return_pct - comparison.spy_dca.total_return_pct)),
                "beats_buy_hold": vs_buy_hold > 0,
                "beats_spy": best_result.total_return_pct > comparison.spy_dca.total_return_pct,
                "recent_trades": best_trades,
                "strategy_comparison": strategy_comparison,
                "optimized_at": datetime.now(UTC),
            }
        )
        await session.execute(stmt)
        await session.commit()
    
    print(f"\n✅ Saved strategy data for {symbol}")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "HOOD"
    asyncio.run(run_single_symbol(symbol))
