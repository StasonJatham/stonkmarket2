"""YFinance info cache repository using SQLAlchemy ORM.

Repository for managing yfinance info cache data.

Usage:
    from app.repositories import yfinance_cache_orm as yfinance_cache_repo
    
    info = await yfinance_cache_repo.get_cached_info("AAPL")
    await yfinance_cache_repo.save_info("AAPL", info_dict, ttl_seconds=3600)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert

from app.database.connection import get_session
from app.database.orm import YfinanceInfoCache
from app.core.logging import get_logger

logger = get_logger("repositories.yfinance_cache_orm")


async def get_cached_info(ticker: str) -> Optional[dict[str, Any]]:
    """Get cached info from database if not expired.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Cached info dict or None if not found/expired
    """
    async with get_session() as session:
        result = await session.execute(
            select(YfinanceInfoCache)
            .where(YfinanceInfoCache.symbol == ticker.upper())
            .where(YfinanceInfoCache.expires_at > func.now())
        )
        cache_entry = result.scalar_one_or_none()
        
        if cache_entry and cache_entry.data:
            data = cache_entry.data
            if isinstance(data, dict):
                return data
            # Handle case where data is stored as JSON string
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return None
        
        return None


async def save_info(ticker: str, info: dict[str, Any], ttl_seconds: int) -> bool:
    """Save info to database cache.
    
    Args:
        ticker: Stock ticker symbol
        info: Info dict to cache
        ttl_seconds: Time-to-live in seconds
    
    Returns:
        True if saved successfully
    """
    async with get_session() as session:
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)
        
        # Serialize info to JSON string
        data_json = json.dumps(info)
        
        stmt = insert(YfinanceInfoCache).values(
            symbol=ticker.upper(),
            data=data_json,
            fetched_at=now,
            expires_at=expires_at,
        ).on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "data": data_json,
                "fetched_at": now,
                "expires_at": expires_at,
            }
        )
        
        await session.execute(stmt)
        await session.commit()
        return True
