"""Database models (dataclasses for ORM-like usage)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from decimal import Decimal


@dataclass
class DipState:
    """Dip state for a symbol.
    
    Maps to: dip_state table
    Columns: id, symbol, current_price, ath_price, dip_percentage, first_seen, last_updated
    """

    id: int
    symbol: str
    current_price: float
    ath_price: float
    dip_percentage: float
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "DipState":
        """Create from database row."""
        keys = list(row.keys())
        
        def parse_dt(key: str) -> Optional[datetime]:
            if key in keys and row[key]:
                val = row[key]
                return val if isinstance(val, datetime) else datetime.fromisoformat(val)
            return None
        
        def to_float(key: str, default: float = 0.0) -> float:
            if key not in keys:
                return default
            val = row[key]
            if val is None:
                return default
            if isinstance(val, (float, int)):
                return float(val)
            if isinstance(val, Decimal):
                return float(val)
            return default
        
        return cls(
            id=row["id"],
            symbol=row["symbol"],
            current_price=to_float("current_price"),
            ath_price=to_float("ath_price"),
            dip_percentage=to_float("dip_percentage"),
            first_seen=parse_dt("first_seen"),
            last_updated=parse_dt("last_updated"),
        )


@dataclass
class SymbolConfig:
    """Symbol configuration."""

    symbol: str
    min_dip_pct: float
    min_days: int
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "SymbolConfig":
        """Create from database row."""
        keys = list(row.keys())
        created = None
        if "created_at" in keys and row["created_at"]:
            val = row["created_at"]
            created = val if isinstance(val, datetime) else datetime.fromisoformat(val)
        return cls(
            symbol=row["symbol"],
            min_dip_pct=row["min_dip_pct"],
            min_days=row["min_days"],
            created_at=created,
        )


@dataclass
class CronJobConfig:
    """Cron job configuration."""

    name: str
    cron: str
    description: Optional[str] = None

    @classmethod
    def from_row(cls, row) -> "CronJobConfig":
        """Create from database row."""
        keys = list(row.keys())
        return cls(
            name=row["name"],
            cron=row["cron"],
            description=row["description"] if "description" in keys else None,
        )


@dataclass
class CronJobLog:
    """Cron job log entry."""

    id: int
    name: str
    status: str
    message: Optional[str]
    created_at: datetime

    @classmethod
    def from_row(cls, row) -> "CronJobLog":
        """Create from database row."""
        keys = list(row.keys())
        val = row["created_at"]
        created = val if isinstance(val, datetime) else datetime.fromisoformat(val)
        return cls(
            id=row["id"] if "id" in keys else 0,
            name=row["name"],
            status=row["status"],
            message=row["message"] if "message" in keys else None,
            created_at=created,
        )


@dataclass
class AuthUser:
    """Authentication user with MFA support."""

    id: int
    username: str
    password_hash: str
    is_admin: bool = False
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False
    mfa_backup_codes: Optional[str] = None  # JSON list of hashed backup codes
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "AuthUser":
        """Create from database row."""
        # Materialize keys to list - asyncpg returns iterator that gets consumed
        keys = list(row.keys())
        updated = None
        if "updated_at" in keys and row["updated_at"]:
            val = row["updated_at"]
            # asyncpg returns datetime objects directly
            updated = val if isinstance(val, datetime) else datetime.fromisoformat(val)
        return cls(
            id=row["id"],
            username=row["username"],
            password_hash=row["password_hash"],
            is_admin=bool(row["is_admin"]) if "is_admin" in keys else False,
            mfa_secret=row["mfa_secret"] if "mfa_secret" in keys else None,
            mfa_enabled=bool(row["mfa_enabled"]) if "mfa_enabled" in keys else False,
            mfa_backup_codes=row["mfa_backup_codes"]
            if "mfa_backup_codes" in keys
            else None,
            updated_at=updated,
        )


@dataclass
class StockSuggestion:
    """User-submitted stock suggestion."""

    id: int
    symbol: str
    status: str  # pending, approved, rejected, removed, fetching, fetch_failed
    vote_count: int = 0
    name: Optional[str] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    summary: Optional[str] = None
    last_price: Optional[float] = None
    price_90d_ago: Optional[float] = None
    price_change_90d: Optional[float] = None
    rejection_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    fetched_at: Optional[datetime] = None
    removed_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "StockSuggestion":
        """Create from database row."""
        # Materialize keys to list - asyncpg returns iterator that gets consumed
        keys = list(row.keys())

        def get_val(key: str, default=None):
            return row[key] if key in keys else default

        def parse_dt(key: str) -> Optional[datetime]:
            if key in keys and row[key]:
                val = row[key]
                return val if isinstance(val, datetime) else datetime.fromisoformat(val)
            return None

        return cls(
            id=row["id"],
            symbol=row["symbol"],
            status=row["status"],
            vote_count=get_val("vote_count", 0),
            name=get_val("name"),
            sector=get_val("sector"),
            industry=get_val("industry"),
            summary=get_val("summary"),
            last_price=get_val("last_price"),
            price_90d_ago=get_val("price_90d_ago"),
            price_change_90d=get_val("price_change_90d"),
            rejection_reason=get_val("rejection_reason"),
            created_at=parse_dt("created_at"),
            updated_at=parse_dt("updated_at"),
            fetched_at=parse_dt("fetched_at"),
            removed_at=parse_dt("removed_at"),
        )


@dataclass
class SuggestionVote:
    """Vote for a stock suggestion (tracks unique voters)."""

    id: int
    suggestion_id: int
    voter_hash: str  # Hashed IP or session for deduplication
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "SuggestionVote":
        """Create from database row."""
        keys = list(row.keys())
        created = None
        if "created_at" in keys and row["created_at"]:
            val = row["created_at"]
            created = val if isinstance(val, datetime) else datetime.fromisoformat(val)
        return cls(
            id=row["id"],
            suggestion_id=row["suggestion_id"],
            voter_hash=row["voter_hash"],
            created_at=created,
        )


@dataclass
class SecureApiKey:
    """Encrypted API key storage."""

    id: int
    key_name: str
    encrypted_key: str
    key_hint: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: str = ""

    @classmethod
    def from_row(cls, row) -> "SecureApiKey":
        """Create from database row."""
        # Materialize keys to list - asyncpg returns iterator that gets consumed
        keys = list(row.keys())

        def parse_dt(key: str) -> Optional[datetime]:
            if key in keys and row[key]:
                val = row[key]
                return val if isinstance(val, datetime) else datetime.fromisoformat(val)
            return None

        return cls(
            id=row["id"],
            key_name=row["key_name"],
            encrypted_key=row["encrypted_key"],
            key_hint=row["key_hint"] if "key_hint" in keys else None,
            created_at=parse_dt("created_at"),
            updated_at=parse_dt("updated_at"),
            created_by=row["created_by"] if "created_by" in keys else "",
        )


@dataclass
class DipVote:
    """User vote on a dip (buy/sell).
    
    Maps to: dip_votes table
    Columns: id, symbol, fingerprint, vote_type, vote_weight, api_key_id, created_at
    """

    id: int
    symbol: str
    fingerprint: str
    vote_type: str  # 'buy' or 'sell'
    vote_weight: int = 1
    api_key_id: Optional[int] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "DipVote":
        """Create from database row."""
        keys = list(row.keys())
        created = None
        if "created_at" in keys and row["created_at"]:
            val = row["created_at"]
            created = val if isinstance(val, datetime) else datetime.fromisoformat(val)
        return cls(
            id=row["id"],
            symbol=row["symbol"],
            fingerprint=row["fingerprint"],
            vote_type=row["vote_type"],
            vote_weight=row["vote_weight"] if "vote_weight" in keys else 1,
            api_key_id=row["api_key_id"] if "api_key_id" in keys else None,
            created_at=created,
        )


@dataclass
class DipAIAnalysis:
    """Cached AI analysis for a dip.
    
    Maps to: dip_ai_analysis table
    Columns: id, symbol, tinder_bio, ai_rating, rating_reasoning, model_used,
             tokens_used, is_batch_generated, batch_job_id, generated_at, expires_at
    """

    id: int
    symbol: str
    tinder_bio: Optional[str] = None
    ai_rating: Optional[str] = None  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    rating_reasoning: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None
    is_batch_generated: bool = False
    batch_job_id: Optional[str] = None
    generated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "DipAIAnalysis":
        """Create from database row."""
        keys = list(row.keys())

        def parse_dt(key: str) -> Optional[datetime]:
            if key in keys and row[key]:
                val = row[key]
                return val if isinstance(val, datetime) else datetime.fromisoformat(val)
            return None
        
        def get_val(key: str, default=None):
            return row[key] if key in keys else default

        return cls(
            id=row["id"],
            symbol=row["symbol"],
            tinder_bio=get_val("tinder_bio"),
            ai_rating=get_val("ai_rating"),
            rating_reasoning=get_val("rating_reasoning"),
            model_used=get_val("model_used"),
            tokens_used=get_val("tokens_used"),
            is_batch_generated=bool(get_val("is_batch_generated", False)),
            batch_job_id=get_val("batch_job_id"),
            generated_at=parse_dt("generated_at"),
            expires_at=parse_dt("expires_at"),
        )
