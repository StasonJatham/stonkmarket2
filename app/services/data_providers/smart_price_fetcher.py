"""Smart batch price fetcher with intelligent grouping.

Groups stocks by missing days and fetches in optimal batches:
- Stocks missing 5-9 days → batch with 10 days
- Stocks missing 2-4 days → batch with 5 days
- Stocks missing 50+ days → fetch individually to avoid over-fetching

This prevents:
1. Over-fetching (getting 200 days for all stocks when only 1 needs it)
2. Under-fetching (missing recent data due to small windows)
3. Corrupted data (validates prices before saving)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

import pandas as pd

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import PriceHistory
from app.repositories import price_history_orm as price_history_repo
from app.services.data_providers.yfinance_service import get_yfinance_service


logger = get_logger("services.smart_price_fetcher")


# Fetch window buckets (max missing days → fetch window days)
FETCH_BUCKETS = [
    (3, 5),      # 1-3 days missing → fetch 5 days
    (7, 10),     # 4-7 days missing → fetch 10 days  
    (14, 20),    # 8-14 days missing → fetch 20 days
    (30, 40),    # 15-30 days missing → fetch 40 days
    (60, 80),    # 31-60 days missing → fetch 80 days
    (120, 150),  # 61-120 days missing → fetch 150 days
]

# Maximum symbols per batch call (yfinance limit is ~100)
MAX_BATCH_SIZE = 50

# Maximum change allowed per day before flagging as suspicious
MAX_DAILY_CHANGE_PCT = 0.30  # 30% max daily move (high but seen in volatile stocks)

# Maximum change that causes immediate rejection (data corruption)
# Historical max: AMD +52%, SNAP +58%, NFLX -41% - real moves can be 60%+
# Stock splits are handled by yfinance adjusted prices, so this catches corruption
MAX_REJECTION_CHANGE_PCT = 0.75  # 75% = almost certainly corrupt data

# Maximum missing days before individual fetch (avoid giant downloads)
INDIVIDUAL_FETCH_THRESHOLD = 180


@dataclass
class SymbolGap:
    """Information about missing price data for a symbol."""
    symbol: str
    last_date: date | None  # Last date we have data for
    missing_days: int       # Trading days missing
    end_date: date          # Target end date


@dataclass
class FetchGroup:
    """Group of symbols to fetch with the same date range."""
    symbols: list[str]
    start_date: date
    end_date: date
    fetch_window: int  # How many days we're fetching


@dataclass 
class PriceValidation:
    """Result of price validation."""
    valid: bool
    error: str | None = None
    suspicious_dates: list[date] | None = None


class SmartPriceFetcher:
    """Smart batch price fetcher with intelligent grouping."""

    def __init__(self) -> None:
        self._yf_service = get_yfinance_service()

    async def analyze_gaps(
        self,
        symbols: list[str],
        target_end_date: date | None = None,
    ) -> list[SymbolGap]:
        """
        Analyze price data gaps for multiple symbols.
        
        Args:
            symbols: List of symbols to check
            target_end_date: End date to check gaps against (default: today)
        
        Returns:
            List of SymbolGap objects with gap information
        """
        target_end_date = target_end_date or date.today()
        gaps: list[SymbolGap] = []
        
        # Get latest dates for all symbols in one query
        latest_dates = await price_history_repo.get_latest_price_dates(symbols)
        
        for symbol in symbols:
            symbol_upper = symbol.upper()
            last_date = latest_dates.get(symbol_upper)
            
            if last_date is None:
                # No data at all - need full history
                missing_days = 365  # Flag as needing substantial data
            else:
                # Calculate trading days missing (rough: 5 of 7 days are trading)
                calendar_days = (target_end_date - last_date).days
                missing_days = max(0, int(calendar_days * 5 / 7))
            
            gaps.append(SymbolGap(
                symbol=symbol_upper,
                last_date=last_date,
                missing_days=missing_days,
                end_date=target_end_date,
            ))
        
        return gaps

    def group_by_fetch_window(
        self,
        gaps: list[SymbolGap],
    ) -> list[FetchGroup]:
        """
        Group symbols by optimal fetch window.
        
        Symbols with similar missing data get grouped together
        to minimize API calls while not over-fetching.
        """
        today = date.today()
        
        # Separate symbols needing individual fetch (too much missing)
        individual_fetch: list[SymbolGap] = []
        bucketed: dict[int, list[SymbolGap]] = defaultdict(list)
        
        for gap in gaps:
            if gap.missing_days <= 0:
                # Already up to date, skip
                continue
            
            if gap.missing_days > INDIVIDUAL_FETCH_THRESHOLD:
                individual_fetch.append(gap)
                continue
            
            # Find the right bucket
            bucket_window = None
            for max_missing, fetch_window in FETCH_BUCKETS:
                if gap.missing_days <= max_missing:
                    bucket_window = fetch_window
                    break
            
            if bucket_window is None:
                # Fallback for gaps between last bucket and threshold
                bucket_window = INDIVIDUAL_FETCH_THRESHOLD
            
            bucketed[bucket_window].append(gap)
        
        # Build fetch groups
        groups: list[FetchGroup] = []
        
        # Process bucketed symbols
        for fetch_window, symbol_gaps in bucketed.items():
            symbols = [g.symbol for g in symbol_gaps]
            start_date = today - timedelta(days=fetch_window)
            
            # Split into batches if too many symbols
            for i in range(0, len(symbols), MAX_BATCH_SIZE):
                batch = symbols[i:i + MAX_BATCH_SIZE]
                groups.append(FetchGroup(
                    symbols=batch,
                    start_date=start_date,
                    end_date=today,
                    fetch_window=fetch_window,
                ))
        
        # Process individual fetches (large gaps)
        for gap in individual_fetch:
            # For large gaps, start from last known date + 1
            if gap.last_date:
                start_date = gap.last_date + timedelta(days=1)
            else:
                # No data - fetch 1 year
                start_date = today - timedelta(days=365)
            
            groups.append(FetchGroup(
                symbols=[gap.symbol],
                start_date=start_date,
                end_date=today,
                fetch_window=(today - start_date).days,
            ))
        
        return groups

    def validate_prices(
        self,
        symbol: str,
        df: pd.DataFrame,
        existing_last_price: Decimal | None = None,
    ) -> PriceValidation:
        """
        Validate fetched price data for anomalies.
        
        Checks:
        1. No extreme daily jumps (>25% in one day)
        2. Continuity with existing data
        3. No obviously wrong values (negative, zero, etc.)
        """
        if df is None or df.empty:
            return PriceValidation(valid=True)  # Nothing to validate
        
        suspicious: list[date] = []
        
        # Check for invalid values
        if "Close" not in df.columns:
            return PriceValidation(
                valid=False,
                error="No Close column in data",
            )
        
        closes = df["Close"].dropna()
        if closes.empty:
            return PriceValidation(
                valid=False,
                error="All Close values are NaN",
            )
        
        # Check for zero or negative prices
        if (closes <= 0).any():
            return PriceValidation(
                valid=False,
                error=f"Invalid prices (zero or negative) for {symbol}",
            )
        
        # Check daily changes
        pct_changes = closes.pct_change().dropna()
        for idx, change in pct_changes.items():
            dt = idx.date() if hasattr(idx, "date") else idx
            
            # Reject data with massive jumps (>50%) - definitely corrupt
            if abs(change) > MAX_REJECTION_CHANGE_PCT:
                logger.error(
                    f"REJECTING {symbol}: corrupt price change on {dt}: "
                    f"{change*100:.1f}% (threshold: {MAX_REJECTION_CHANGE_PCT*100}%)"
                )
                return PriceValidation(
                    valid=False,
                    error=f"Corrupt data: {change*100:.1f}% change on {dt}",
                )
            
            # Flag suspicious but not corrupt (25-50%)
            if abs(change) > MAX_DAILY_CHANGE_PCT:
                suspicious.append(dt)
                logger.warning(
                    f"Suspicious price change for {symbol} on {dt}: "
                    f"{change*100:.1f}%"
                )
        
        # Check continuity with existing data
        if existing_last_price is not None and len(closes) > 0:
            first_new_price = float(closes.iloc[0])
            existing_float = float(existing_last_price)
            
            if existing_float > 0:
                gap_change = (first_new_price - existing_float) / existing_float
                
                # Reject massive gaps
                if abs(gap_change) > MAX_REJECTION_CHANGE_PCT:
                    logger.error(
                        f"REJECTING {symbol}: corrupt gap from existing ${existing_float:.2f} → "
                        f"new ${first_new_price:.2f} ({gap_change*100:.1f}%)"
                    )
                    return PriceValidation(
                        valid=False,
                        error=f"Corrupt data: {gap_change*100:.1f}% gap from existing data",
                    )
                
                if abs(gap_change) > MAX_DAILY_CHANGE_PCT:
                    first_date = closes.index[0]
                    dt = first_date.date() if hasattr(first_date, "date") else first_date
                    suspicious.append(dt)
                    logger.warning(
                        f"Suspicious gap for {symbol}: existing ${existing_float:.2f} → "
                        f"new ${first_new_price:.2f} ({gap_change*100:.1f}%)"
                    )
        
        return PriceValidation(
            valid=True,  # We don't reject, just warn
            suspicious_dates=suspicious if suspicious else None,
        )

    async def fetch_and_save(
        self,
        symbols: list[str],
        validate: bool = True,
    ) -> dict[str, int]:
        """
        Smart fetch and save prices for multiple symbols.
        
        1. Analyzes gaps for each symbol
        2. Groups by fetch window
        3. Batch fetches each group
        4. Validates and saves
        
        Args:
            symbols: List of symbols to update
            validate: If True, validate prices before saving
        
        Returns:
            Dict mapping symbol to number of new records saved
        """
        if not symbols:
            return {}
        
        logger.info(f"Smart fetch starting for {len(symbols)} symbols")
        
        # Step 1: Analyze gaps
        gaps = await self.analyze_gaps(symbols)
        
        # Filter to only symbols needing updates
        symbols_needing_update = [g for g in gaps if g.missing_days > 0]
        
        if not symbols_needing_update:
            logger.info("All symbols up to date, no fetching needed")
            return {}
        
        logger.info(
            f"{len(symbols_needing_update)} symbols need updates, "
            f"{len(symbols) - len(symbols_needing_update)} already current"
        )
        
        # Step 2: Group by fetch window
        groups = self.group_by_fetch_window(symbols_needing_update)
        
        logger.info(f"Created {len(groups)} fetch groups:")
        for group in groups:
            logger.info(
                f"  - {len(group.symbols)} symbols, "
                f"{group.fetch_window} days ({group.start_date} to {group.end_date})"
            )
        
        # Step 3: Get existing prices for validation
        existing_prices: dict[str, Decimal] = {}
        if validate:
            all_symbols = [g.symbol for g in symbols_needing_update]
            existing_prices = await price_history_repo.get_latest_prices(all_symbols)
        
        # Step 4: Fetch and save each group
        results: dict[str, int] = {}
        
        for group in groups:
            try:
                batch_results = await self._yf_service.get_price_history_batch(
                    group.symbols,
                    group.start_date,
                    group.end_date,
                )
                
                for symbol, (df, version) in batch_results.items():
                    if df is None or df.empty:
                        logger.debug(f"No data returned for {symbol}")
                        results[symbol] = 0
                        continue
                    
                    # Validate
                    if validate:
                        validation = self.validate_prices(
                            symbol,
                            df,
                            existing_prices.get(symbol),
                        )
                        if not validation.valid:
                            logger.error(
                                f"Skipping {symbol}: {validation.error}"
                            )
                            results[symbol] = 0
                            continue
                        
                        if validation.suspicious_dates:
                            logger.warning(
                                f"{symbol} has {len(validation.suspicious_dates)} "
                                f"suspicious price points - saving anyway"
                            )
                    
                    # Save to database
                    count = await price_history_repo.save_prices(symbol, df)
                    results[symbol] = count
                    
                    # Invalidate related caches
                    if count > 0:
                        await self._invalidate_caches(symbol)
                    
            except Exception as e:
                logger.error(f"Failed to fetch group: {e}")
                for symbol in group.symbols:
                    results[symbol] = 0
        
        # Summary
        total_saved = sum(results.values())
        logger.info(
            f"Smart fetch complete: {total_saved} total records for "
            f"{len([r for r in results.values() if r > 0])} symbols"
        )
        
        return results

    async def _invalidate_caches(self, symbol: str) -> None:
        """Invalidate all price-related caches for a symbol."""
        from app.cache.client import get_valkey_client
        
        try:
            client = await get_valkey_client()
            
            # Find and delete all price-related cache keys for this symbol
            patterns = [
                f"stonkmarket:v1:prices:{symbol}*",
                f"stonkmarket:v1:chart:{symbol}*",
                f"stonkmarket:v1:dipfinder:{symbol}*",
                f"stonkmarket:v1:quant_prices:*",  # Quant uses hash keys
            ]
            
            deleted = 0
            for pattern in patterns:
                keys = await client.keys(pattern)
                if keys:
                    await client.delete(*keys)
                    deleted += len(keys)
            
            if deleted > 0:
                logger.debug(f"Invalidated {deleted} cache keys for {symbol}")
                
        except Exception as e:
            logger.warning(f"Failed to invalidate caches for {symbol}: {e}")

    async def get_gaps_summary(
        self,
        symbols: list[str],
    ) -> dict[str, Any]:
        """
        Get a summary of price data gaps for reporting.
        
        Returns:
            Dict with gap statistics
        """
        gaps = await self.analyze_gaps(symbols)
        
        up_to_date = sum(1 for g in gaps if g.missing_days == 0)
        slight_gaps = sum(1 for g in gaps if 0 < g.missing_days <= 7)
        moderate_gaps = sum(1 for g in gaps if 7 < g.missing_days <= 30)
        large_gaps = sum(1 for g in gaps if 30 < g.missing_days <= 120)
        no_data = sum(1 for g in gaps if g.missing_days > 120)
        
        return {
            "total_symbols": len(symbols),
            "up_to_date": up_to_date,
            "slight_gaps_1_7_days": slight_gaps,
            "moderate_gaps_8_30_days": moderate_gaps,
            "large_gaps_31_120_days": large_gaps,
            "no_data_or_stale": no_data,
            "gaps": [
                {
                    "symbol": g.symbol,
                    "last_date": str(g.last_date) if g.last_date else None,
                    "missing_days": g.missing_days,
                }
                for g in gaps
                if g.missing_days > 0
            ][:50],  # Limit to first 50 for readability
        }


# Singleton instance
_instance: SmartPriceFetcher | None = None


def get_smart_price_fetcher() -> SmartPriceFetcher:
    """Get singleton SmartPriceFetcher instance."""
    global _instance
    if _instance is None:
        _instance = SmartPriceFetcher()
    return _instance
