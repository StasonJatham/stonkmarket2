from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


@dataclass
class DipState:
    ref_high: float
    days_below: int
    last_price: float
    updated_at: Optional[datetime] = None


@dataclass
class SymbolConfig:
    symbol: str
    min_dip_pct: float
    min_days: int


@dataclass
class CronJobConfig:
    name: str
    cron: str
    description: Optional[str] = None


class SymbolPayload(BaseModel):
    symbol: str = Field(..., min_length=1)
    min_dip_pct: float = Field(..., gt=0, lt=1)
    min_days: int = Field(..., ge=0)


class SymbolUpdatePayload(BaseModel):
    min_dip_pct: float = Field(..., gt=0, lt=1)
    min_days: int = Field(..., ge=0)


class SymbolResponse(BaseModel):
    symbol: str
    min_dip_pct: float
    min_days: int

    class Config:
        from_attributes = True


class DipStateResponse(BaseModel):
    symbol: str
    ref_high: float
    days_below: int
    last_price: float
    dip_depth: float
    updated_at: Optional[datetime]


class RankingEntry(BaseModel):
    symbol: str
    depth: float
    last_price: float
    days_since_dip: Optional[int] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None


class CronJobResponse(BaseModel):
    name: str
    cron: str
    description: Optional[str] = None


class CronJobUpdatePayload(BaseModel):
    cron: str


class CronJobLogPayload(BaseModel):
    status: str = Field(..., min_length=1)
    message: Optional[str] = None


class CronJobLogResponse(BaseModel):
    name: str
    status: str
    message: Optional[str] = None
    created_at: datetime


class ChartPoint(BaseModel):
    date: str
    close: float
    threshold: Optional[float] = None
    ref_high: Optional[float] = None
    drawdown: Optional[float] = None  # (close - ref_high)/ref_high
    since_dip: Optional[float] = None  # (close - base_dip_price)/base_dip_price
    ref_high_date: Optional[str] = None
    dip_start_date: Optional[str] = None
