"""Database models (dataclasses for ORM-like usage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DipState:
    """Dip state for a symbol."""

    symbol: str
    ref_high: float
    days_below: int
    last_price: float
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "DipState":
        """Create from database row."""
        updated = datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        return cls(
            symbol=row["symbol"],
            ref_high=row["ref_high"],
            days_below=row["days_below"],
            last_price=row["last_price"],
            updated_at=updated,
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
        created = None
        if "created_at" in row.keys() and row["created_at"]:
            created = datetime.fromisoformat(row["created_at"])
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
        return cls(
            name=row["name"],
            cron=row["cron"],
            description=row["description"] if "description" in row.keys() else None,
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
        return cls(
            id=row["id"] if "id" in row.keys() else 0,
            name=row["name"],
            status=row["status"],
            message=row["message"] if "message" in row.keys() else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )


@dataclass
class AuthUser:
    """Authentication user with MFA support."""

    username: str
    password_hash: str
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False
    mfa_backup_codes: Optional[str] = None  # JSON list of hashed backup codes
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "AuthUser":
        """Create from database row."""
        keys = row.keys()
        updated = None
        if "updated_at" in keys and row["updated_at"]:
            updated = datetime.fromisoformat(row["updated_at"])
        return cls(
            username=row["username"],
            password_hash=row["password_hash"],
            mfa_secret=row["mfa_secret"] if "mfa_secret" in keys else None,
            mfa_enabled=bool(row["mfa_enabled"]) if "mfa_enabled" in keys else False,
            mfa_backup_codes=row["mfa_backup_codes"] if "mfa_backup_codes" in keys else None,
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
        keys = row.keys()
        
        def get_val(key: str, default=None):
            return row[key] if key in keys else default
        
        def parse_dt(key: str) -> Optional[datetime]:
            if key in keys and row[key]:
                return datetime.fromisoformat(row[key])
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
        created = None
        if "created_at" in row.keys() and row["created_at"]:
            created = datetime.fromisoformat(row["created_at"])
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
        keys = row.keys()
        
        def parse_dt(key: str) -> Optional[datetime]:
            if key in keys and row[key]:
                return datetime.fromisoformat(row[key])
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
    """User vote on a dip (buy/sell/skip)."""

    id: int
    symbol: str
    voter_hash: str
    vote_type: str  # 'buy', 'sell', 'skip'
    created_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "DipVote":
        """Create from database row."""
        created = None
        if "created_at" in row.keys() and row["created_at"]:
            created = datetime.fromisoformat(row["created_at"])
        return cls(
            id=row["id"],
            symbol=row["symbol"],
            voter_hash=row["voter_hash"],
            vote_type=row["vote_type"],
            created_at=created,
        )


@dataclass
class DipAIAnalysis:
    """Cached AI analysis for a dip."""

    symbol: str
    tinder_bio: Optional[str] = None
    ai_rating: Optional[str] = None  # 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
    ai_reasoning: Optional[str] = None
    analysis_data: Optional[str] = None  # JSON blob of data used for analysis
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "DipAIAnalysis":
        """Create from database row."""
        keys = row.keys()
        
        def parse_dt(key: str) -> Optional[datetime]:
            if key in keys and row[key]:
                return datetime.fromisoformat(row[key])
            return None
        
        return cls(
            symbol=row["symbol"],
            tinder_bio=row["tinder_bio"] if "tinder_bio" in keys else None,
            ai_rating=row["ai_rating"] if "ai_rating" in keys else None,
            ai_reasoning=row["ai_reasoning"] if "ai_reasoning" in keys else None,
            analysis_data=row["analysis_data"] if "analysis_data" in keys else None,
            created_at=parse_dt("created_at"),
            expires_at=parse_dt("expires_at"),
        )
