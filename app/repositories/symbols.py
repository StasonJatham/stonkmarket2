"""Symbol repository - PostgreSQL async."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from app.database.connection import fetch_all, fetch_one, execute, fetch_val
from app.core.logging import get_logger

logger = get_logger("repositories.symbols")


async def invalidate_symbol_caches(symbol: str | None = None) -> None:
    """Invalidate caches affected by symbol changes.
    
    Args:
        symbol: If provided, invalidate caches for specific symbol.
                If None, invalidate all ranking/chart caches.
    """
    from app.cache.cache import Cache
    
    try:
        # Ranking cache uses "all:True" and "all:False" keys
        ranking_cache = Cache(prefix="ranking")
        await ranking_cache.invalidate_pattern("*")
        
        if symbol:
            # Invalidate chart cache for specific symbol
            chart_cache = Cache(prefix="chart")
            await chart_cache.invalidate_pattern(f"{symbol}:*")
            logger.debug(f"Invalidated caches for symbol {symbol}")
        else:
            # Invalidate all chart caches
            chart_cache = Cache(prefix="chart")
            await chart_cache.invalidate_pattern("*")
            logger.debug("Invalidated all symbol caches")
    except Exception as e:
        logger.warning(f"Failed to invalidate symbol caches: {e}")


class SymbolConfig:
    """Symbol configuration."""

    def __init__(
        self,
        symbol: str,
        min_dip_pct: float = 0.15,
        min_days: int = 5,
        name: str | None = None,
        fetch_status: str | None = None,
        fetch_error: str | None = None,
        created_at: datetime | None = None,
    ):
        self.symbol = symbol
        self.min_dip_pct = min_dip_pct
        self.min_days = min_days
        self.name = name
        self.fetch_status = fetch_status
        self.fetch_error = fetch_error
        self.created_at = created_at

    @classmethod
    def from_row(cls, row) -> "SymbolConfig":
        return cls(
            symbol=row["symbol"],
            min_dip_pct=float(row.get("min_dip_pct", 0.15)),
            min_days=int(row.get("min_days", 5)),
            name=row.get("name"),
            fetch_status=row.get("fetch_status"),
            fetch_error=row.get("fetch_error"),
            created_at=row.get("created_at"),
        )


async def list_symbols() -> List[SymbolConfig]:
    """List all symbols."""
    rows = await fetch_all(
        """
        SELECT symbol, 
               name,
               COALESCE(min_dip_pct, 0.15) as min_dip_pct,
               COALESCE(min_days, 5) as min_days,
               fetch_status,
               fetch_error,
               added_at as created_at
        FROM symbols 
        WHERE is_active = TRUE
        ORDER BY symbol ASC
        """
    )
    return [SymbolConfig.from_row(row) for row in rows]


async def get_symbol(symbol: str) -> Optional[SymbolConfig]:
    """Get a symbol by ticker."""
    row = await fetch_one(
        """
        SELECT symbol, 
               name,
               COALESCE(min_dip_pct, 0.15) as min_dip_pct,
               COALESCE(min_days, 5) as min_days,
               fetch_status,
               fetch_error,
               added_at as created_at
        FROM symbols 
        WHERE symbol = $1
        """,
        symbol.upper(),
    )
    return SymbolConfig.from_row(row) if row else None


async def upsert_symbol(
    symbol: str,
    min_dip_pct: float = 0.15,
    min_days: int = 5,
) -> SymbolConfig:
    """Create or update a symbol."""
    await execute(
        """
        INSERT INTO symbols (symbol, min_dip_pct, min_days, is_active, added_at, updated_at)
        VALUES ($1, $2, $3, TRUE, NOW(), NOW())
        ON CONFLICT (symbol) DO UPDATE SET
            min_dip_pct = EXCLUDED.min_dip_pct,
            min_days = EXCLUDED.min_days,
            is_active = TRUE,
            updated_at = NOW()
        """,
        symbol.upper(),
        float(min_dip_pct),
        int(min_days),
    )
    
    # Invalidate caches since symbol config changed
    await invalidate_symbol_caches(symbol.upper())
    
    return await get_symbol(symbol.upper())  # type: ignore


async def create_symbol(
    symbol: str,
    min_dip_pct: float = 0.15,
    min_days: int = 5,
) -> SymbolConfig:
    """Create a new symbol (alias for upsert)."""
    return await upsert_symbol(symbol, min_dip_pct, min_days)


async def update_symbol(
    symbol: str,
    min_dip_pct: float | None = None,
    min_days: int | None = None,
) -> Optional[SymbolConfig]:
    """Update a symbol's configuration."""
    existing = await get_symbol(symbol)
    if not existing:
        return None

    new_min_dip_pct = min_dip_pct if min_dip_pct is not None else existing.min_dip_pct
    new_min_days = min_days if min_days is not None else existing.min_days

    return await upsert_symbol(symbol, new_min_dip_pct, new_min_days)


async def delete_symbol(symbol: str) -> bool:
    """Delete a symbol."""
    result = await execute(
        "DELETE FROM symbols WHERE symbol = $1",
        symbol.upper(),
    )
    
    # Invalidate caches since symbol was removed
    await invalidate_symbol_caches(symbol.upper())
    
    # Parse the result to check if any rows were affected
    return "DELETE" in result and not result.endswith(" 0")


async def symbol_exists(symbol: str) -> bool:
    """Check if a symbol exists."""
    count = await fetch_val(
        "SELECT COUNT(*) FROM symbols WHERE symbol = $1",
        symbol.upper(),
    )
    return count > 0


async def count_symbols() -> int:
    """Count total symbols."""
    return await fetch_val("SELECT COUNT(*) FROM symbols") or 0
