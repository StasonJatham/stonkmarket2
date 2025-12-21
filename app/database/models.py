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
    """Authentication user."""

    username: str
    password_hash: str
    updated_at: Optional[datetime] = None

    @classmethod
    def from_row(cls, row) -> "AuthUser":
        """Create from database row."""
        updated = None
        if "updated_at" in row.keys() and row["updated_at"]:
            updated = datetime.fromisoformat(row["updated_at"])
        return cls(
            username=row["username"],
            password_hash=row["password_hash"],
            updated_at=updated,
        )
