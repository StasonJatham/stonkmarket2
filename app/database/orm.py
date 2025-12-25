"""SQLAlchemy ORM models for Stonkmarket.

This module defines all database tables using SQLAlchemy 2.0 ORM style.
Uses async support via asyncpg driver.

Usage:
    from app.database.orm import Symbol, DipState, AuthUser
    from app.database.connection import get_session
    
    async with get_session() as session:
        symbol = await session.get(Symbol, "AAPL")
        symbol.name = "Apple Inc."
        await session.commit()
"""

from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Any

from sqlalchemy import (
    String, Integer, BigInteger, Boolean, Text, DateTime, Date,
    Numeric, ForeignKey, Index, CheckConstraint, UniqueConstraint,
    LargeBinary, func, text, MetaData,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship,
)


# Naming convention for constraints and indexes (deterministic names for Alembic)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for all ORM models with naming convention."""
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# =============================================================================
# AUTH & SECURITY
# =============================================================================


class AuthUser(Base):
    """Authentication user with MFA support."""
    __tablename__ = "auth_user"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(64))
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_backup_codes: Mapped[Optional[str]] = mapped_column(Text)  # JSON array of hashed codes
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    api_keys: Mapped[List["UserApiKey"]] = relationship(back_populates="user")
    secure_keys: Mapped[List["SecureApiKey"]] = relationship(back_populates="created_by_user")

    __table_args__ = (
        Index("idx_auth_user_username", "username"),
    )


class SecureApiKey(Base):
    """Admin-managed API keys for secure storage (OpenAI, etc.)."""
    __tablename__ = "secure_api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    service_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    key_hint: Mapped[Optional[str]] = mapped_column(String(20))
    created_by_id: Mapped[Optional[int]] = mapped_column("created_by", ForeignKey("auth_user.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    created_by_user: Mapped[Optional["AuthUser"]] = relationship(back_populates="secure_keys")

    __table_args__ = (
        Index("idx_secure_api_keys_service", "service_name"),
    )


class UserApiKey(Base):
    """User API keys for public API access."""
    __tablename__ = "user_api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)  # SHA-256 hash
    key_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("auth_user.id"))
    vote_weight: Mapped[int] = mapped_column(Integer, default=10)
    rate_limit_bypass: Mapped[bool] = mapped_column(Boolean, default=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    user: Mapped[Optional["AuthUser"]] = relationship(back_populates="api_keys")
    dip_votes: Mapped[List["DipVote"]] = relationship(back_populates="api_key")
    suggestion_votes: Mapped[List["SuggestionVote"]] = relationship(back_populates="api_key")

    __table_args__ = (
        Index("idx_user_api_keys_hash", "key_hash"),
        Index("idx_user_api_keys_active", "is_active", postgresql_where=text("is_active = TRUE")),
    )


# =============================================================================
# STOCK SYMBOLS
# =============================================================================


class Symbol(Base):
    """Stock symbols being tracked."""
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap: Mapped[Optional[int]] = mapped_column(BigInteger)
    summary_ai: Mapped[Optional[str]] = mapped_column(String(500))  # AI-generated summary (300-400 target, 500 max)
    symbol_type: Mapped[str] = mapped_column(String(20), default="stock")  # stock, etf, index
    min_dip_pct: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("0.15"))
    min_days: Mapped[int] = mapped_column(Integer, default=5)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Logo caching
    logo_light: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    logo_dark: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    logo_fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    logo_source: Mapped[Optional[str]] = mapped_column(String(50))
    # Fetch status
    fetch_status: Mapped[str] = mapped_column(String(20), default="pending")
    fetch_error: Mapped[Optional[str]] = mapped_column(Text)
    fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    dip_state: Mapped[Optional["DipState"]] = relationship(back_populates="symbol_ref", uselist=False)
    dipfinder_config: Mapped[Optional["DipfinderConfig"]] = relationship(back_populates="symbol_ref", uselist=False)

    __table_args__ = (
        Index("idx_symbols_symbol", "symbol"),
        Index("idx_symbols_sector", "sector"),
        Index("idx_symbols_type", "symbol_type"),
        Index("idx_symbols_logo_fetched_at", "logo_fetched_at"),
    )


class DipState(Base):
    """Current dip state for tracked symbols."""
    __tablename__ = "dip_state"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), ForeignKey("symbols.symbol", ondelete="CASCADE"), unique=True, nullable=False)
    current_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ath_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    dip_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    dip_start_date: Mapped[Optional[date]] = mapped_column(Date)
    # Legacy columns
    ref_high: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    last_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    days_below: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    symbol_ref: Mapped["Symbol"] = relationship(back_populates="dip_state")

    __table_args__ = (
        Index("idx_dip_state_symbol", "symbol"),
        Index("idx_dip_state_percentage", "dip_percentage", postgresql_ops={"dip_percentage": "DESC"}),
        Index("idx_dip_state_updated", "last_updated", postgresql_ops={"last_updated": "DESC"}),
        Index("idx_dip_state_start_date", "dip_start_date"),
    )


class DipHistory(Base):
    """History of dip state changes."""
    __tablename__ = "dip_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    action: Mapped[str] = mapped_column(String(10), nullable=False)  # added, removed, updated
    current_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ath_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    dip_percentage: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("action IN ('added', 'removed', 'updated')", name="ck_dip_history_action"),
        Index("idx_dip_history_symbol", "symbol"),
        Index("idx_dip_history_action", "action"),
        Index("idx_dip_history_recorded", "recorded_at", postgresql_ops={"recorded_at": "DESC"}),
    )


# =============================================================================
# STOCK SUGGESTIONS
# =============================================================================


class StockSuggestion(Base):
    """User-submitted stock suggestions."""
    __tablename__ = "stock_suggestions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    company_name: Mapped[Optional[str]] = mapped_column(String(255))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    summary: Mapped[Optional[str]] = mapped_column(Text)
    website: Mapped[Optional[str]] = mapped_column(String(255))
    ipo_year: Mapped[Optional[int]] = mapped_column(Integer)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, approved, rejected
    vote_score: Mapped[int] = mapped_column(Integer, default=0)
    fetch_status: Mapped[str] = mapped_column(String(20), default="pending")
    fetch_error: Mapped[Optional[str]] = mapped_column(Text)
    fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    current_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ath_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    approved_by_id: Mapped[Optional[int]] = mapped_column("approved_by", ForeignKey("auth_user.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationships
    votes: Mapped[List["SuggestionVote"]] = relationship(back_populates="suggestion")

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_suggestion_status"),
        Index("idx_suggestions_status", "status"),
        Index("idx_suggestions_score", "vote_score", postgresql_ops={"vote_score": "DESC"}),
        Index("idx_suggestions_symbol", "symbol"),
    )


class SuggestionVote(Base):
    """Votes on stock suggestions."""
    __tablename__ = "suggestion_votes"

    id: Mapped[int] = mapped_column(primary_key=True)
    suggestion_id: Mapped[int] = mapped_column(ForeignKey("stock_suggestions.id", ondelete="CASCADE"), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    vote_type: Mapped[str] = mapped_column(String(10), nullable=False)  # up, down
    vote_weight: Mapped[int] = mapped_column(Integer, default=1)
    api_key_id: Mapped[Optional[int]] = mapped_column(ForeignKey("user_api_keys.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    suggestion: Mapped["StockSuggestion"] = relationship(back_populates="votes")
    api_key: Mapped[Optional["UserApiKey"]] = relationship(back_populates="suggestion_votes")

    __table_args__ = (
        CheckConstraint("vote_type IN ('up', 'down')", name="ck_suggestion_vote_type"),
        UniqueConstraint("suggestion_id", "fingerprint", name="uq_suggestion_vote"),
        Index("idx_suggestion_votes_suggestion", "suggestion_id"),
    )


# =============================================================================
# DIP VOTING (SWIPE)
# =============================================================================


class DipVote(Base):
    """Votes on current dips (buy/sell)."""
    __tablename__ = "dip_votes"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    vote_type: Mapped[str] = mapped_column(String(10), nullable=False)  # buy, sell
    vote_weight: Mapped[int] = mapped_column(Integer, default=1)
    api_key_id: Mapped[Optional[int]] = mapped_column(ForeignKey("user_api_keys.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    api_key: Mapped[Optional["UserApiKey"]] = relationship(back_populates="dip_votes")

    __table_args__ = (
        CheckConstraint("vote_type IN ('buy', 'sell')", name="ck_dip_vote_type"),
        UniqueConstraint("symbol", "fingerprint", name="uq_dip_vote"),
        Index("idx_dip_votes_symbol", "symbol"),
        Index("idx_dip_votes_fingerprint", "fingerprint"),
        Index("idx_dip_votes_created", "created_at", postgresql_ops={"created_at": "DESC"}),
    )


class DipAIAnalysis(Base):
    """AI-generated analysis for dips."""
    __tablename__ = "dip_ai_analysis"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    swipe_bio: Mapped[Optional[str]] = mapped_column(Text)
    ai_rating: Mapped[Optional[str]] = mapped_column(String(20))  # strong_buy, buy, hold, sell, strong_sell
    rating_reasoning: Mapped[Optional[str]] = mapped_column(Text)
    model_used: Mapped[Optional[str]] = mapped_column(String(50))
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer)
    is_batch_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    batch_job_id: Mapped[Optional[str]] = mapped_column(String(100))
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "ai_rating IS NULL OR ai_rating IN ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')",
            name="ck_ai_rating"
        ),
        Index("idx_dip_ai_analysis_symbol", "symbol"),
        Index("idx_dip_ai_analysis_rating", "ai_rating"),
        Index("idx_dip_ai_analysis_expires", "expires_at"),
    )


# =============================================================================
# API USAGE & BATCH JOBS
# =============================================================================


class ApiUsage(Base):
    """Track API usage for cost monitoring."""
    __tablename__ = "api_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    service: Mapped[str] = mapped_column(String(50), nullable=False)  # openai, yfinance, etc.
    endpoint: Mapped[Optional[str]] = mapped_column(String(100))
    model: Mapped[Optional[str]] = mapped_column(String(50))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    is_batch: Mapped[bool] = mapped_column(Boolean, default=False)
    request_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_api_usage_service", "service"),
        Index("idx_api_usage_recorded", "recorded_at", postgresql_ops={"recorded_at": "DESC"}),
    )


class BatchJob(Base):
    """Batch job tracking."""
    __tablename__ = "batch_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    total_requests: Mapped[int] = mapped_column(Integer, default=0)
    completed_requests: Mapped[int] = mapped_column(Integer, default=0)
    failed_requests: Mapped[int] = mapped_column(Integer, default=0)
    input_file_id: Mapped[Optional[str]] = mapped_column(String(100))
    output_file_id: Mapped[Optional[str]] = mapped_column(String(100))
    error_file_id: Mapped[Optional[str]] = mapped_column(String(100))
    estimated_cost_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    actual_cost_usd: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    job_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSONB)  # Named 'metadata' in DB
    task_custom_ids: Mapped[Optional[dict]] = mapped_column(JSONB)  # Map custom_id → task metadata

    # Relationship to errors
    errors: Mapped[List["BatchTaskError"]] = relationship(back_populates="batch_job")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'validating', 'in_progress', 'finalizing', 'completed', 'failed', 'expired', 'cancelled')",
            name="ck_batch_job_status"
        ),
        Index("idx_batch_jobs_status", "status"),
        Index("idx_batch_jobs_type", "job_type"),
        Index("idx_batch_jobs_created", "created_at", postgresql_ops={"created_at": "DESC"}),
    )


class BatchTaskError(Base):
    """Failed batch task for retry tracking."""
    __tablename__ = "batch_task_errors"

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(100), ForeignKey("batch_jobs.batch_id"), nullable=False)
    custom_id: Mapped[str] = mapped_column(String(200), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'agent_analysis', 'rating', 'bio'
    agent_id: Mapped[Optional[str]] = mapped_column(String(50))  # For agent batch tasks
    error_type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'api_error', 'validation_error', 'timeout'
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    original_request: Mapped[Optional[dict]] = mapped_column(JSONB)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, retrying, resolved, abandoned
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    # Relationship
    batch_job: Mapped["BatchJob"] = relationship(back_populates="errors")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'retrying', 'resolved', 'abandoned')",
            name="ck_batch_task_errors_status"
        ),
        Index("idx_batch_task_errors_batch", "batch_id"),
        Index("idx_batch_task_errors_symbol", "symbol"),
        Index("idx_batch_task_errors_status", "status"),
        Index("idx_batch_task_errors_pending", "status", "created_at", postgresql_where=text("status = 'pending'")),
    )


# =============================================================================
# SCHEDULER / CRON JOBS
# =============================================================================


class CronJob(Base):
    """Cron job configuration and status."""
    __tablename__ = "cronjobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    cron_expression: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    next_run: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[Optional[str]] = mapped_column(String(20))
    last_duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    config: Mapped[Optional[dict]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_cronjobs_active", "is_active", postgresql_where=text("is_active = TRUE")),
        Index("idx_cronjobs_next_run", "next_run"),
    )


class RuntimeSetting(Base):
    """Runtime settings (persisted key-value store)."""
    __tablename__ = "runtime_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# =============================================================================
# RATE LIMITING
# =============================================================================


class RateLimitLog(Base):
    """Rate limit tracking."""
    __tablename__ = "rate_limit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    identifier: Mapped[str] = mapped_column(String(64), nullable=False)  # IP hash or fingerprint
    endpoint: Mapped[str] = mapped_column(String(100), nullable=False)
    request_count: Mapped[int] = mapped_column(Integer, default=1)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_request_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("idx_rate_limit_identifier", "identifier", "endpoint"),
        Index("idx_rate_limit_window", "window_start"),
    )


# =============================================================================
# DIPFINDER ENGINE
# =============================================================================


class PriceHistory(Base):
    """Price history cache from yfinance."""
    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 6))
    high: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 6))
    low: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 6))
    close: Mapped[Decimal] = mapped_column(Numeric(16, 6), nullable=False)
    adj_close: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 6))
    volume: Mapped[Optional[int]] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_price_history"),
        Index("idx_price_history_symbol", "symbol"),
        Index("idx_price_history_date", "date", postgresql_ops={"date": "DESC"}),
        Index("idx_price_history_symbol_date", "symbol", "date"),
    )


class DipfinderSignal(Base):
    """DipFinder computed signals."""
    __tablename__ = "dipfinder_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    benchmark: Mapped[str] = mapped_column(String(20), nullable=False)
    window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    as_of_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    # Dip metrics
    dip_stock: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    peak_stock: Mapped[Optional[Decimal]] = mapped_column(Numeric(16, 6))
    dip_pctl: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    dip_vs_typical: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    persist_days: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Market context
    dip_mkt: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    excess_dip: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    dip_class: Mapped[Optional[str]] = mapped_column(String(20))
    
    # Scores
    quality_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    stability_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    dip_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    final_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    
    # Alert
    alert_level: Mapped[Optional[str]] = mapped_column(String(20))
    should_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[Optional[str]] = mapped_column(Text)
    
    # Contributing factors
    quality_factors: Mapped[Optional[dict]] = mapped_column(JSONB)
    stability_factors: Mapped[Optional[dict]] = mapped_column(JSONB)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("ticker", "benchmark", "window_days", "as_of_date", name="uq_dipfinder_signal"),
        Index("idx_dipfinder_signals_ticker", "ticker"),
        Index("idx_dipfinder_signals_date", "as_of_date", postgresql_ops={"as_of_date": "DESC"}),
        Index("idx_dipfinder_signals_lookup", "ticker", "benchmark", "window_days", "as_of_date"),
        Index("idx_dipfinder_signals_alert", "should_alert", postgresql_where=text("should_alert = TRUE")),
        Index("idx_dipfinder_signals_final", "final_score", postgresql_ops={"final_score": "DESC"}),
        Index("idx_dipfinder_signals_expires", "expires_at"),
    )


class DipfinderConfig(Base):
    """DipFinder per-symbol configuration overrides."""
    __tablename__ = "dipfinder_config"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    min_dip_abs: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("0.10"))
    min_persist_days: Mapped[int] = mapped_column(Integer, default=2)
    dip_percentile_threshold: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("0.80"))
    dip_vs_typical_threshold: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("1.5"))
    quality_gate: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=Decimal("60"))
    stability_gate: Mapped[Decimal] = mapped_column(Numeric(5, 2), default=Decimal("60"))
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    symbol_ref: Mapped[Optional["Symbol"]] = relationship(back_populates="dipfinder_config")

    __table_args__ = (
        Index("idx_dipfinder_config_symbol", "symbol"),
        Index("idx_dipfinder_config_enabled", "is_enabled", postgresql_where=text("is_enabled = TRUE")),
    )


class DipfinderHistory(Base):
    """DipFinder event history."""
    __tablename__ = "dipfinder_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)  # entered_dip, exited_dip, deepened, recovered, alert_triggered
    window_days: Mapped[int] = mapped_column(Integer, nullable=False)
    dip_pct: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 6))
    final_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    dip_class: Mapped[Optional[str]] = mapped_column(String(20))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('entered_dip', 'exited_dip', 'deepened', 'recovered', 'alert_triggered')",
            name="ck_dipfinder_history_event"
        ),
        Index("idx_dipfinder_history_ticker", "ticker"),
        Index("idx_dipfinder_history_event", "event_type"),
        Index("idx_dipfinder_history_recorded", "recorded_at", postgresql_ops={"recorded_at": "DESC"}),
    )


class YfinanceInfoCache(Base):
    """YFinance info cache (L3 persistent cache)."""
    __tablename__ = "yfinance_info_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_yfinance_cache_symbol", "symbol"),
        Index("idx_yfinance_cache_expires", "expires_at"),
    )


# =============================================================================
# AI AGENT ANALYSIS
# =============================================================================


class AiAgentAnalysis(Base):
    """AI investor persona analysis."""
    __tablename__ = "ai_agent_analysis"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    verdicts: Mapped[dict] = mapped_column(JSONB, nullable=False, default=list)
    overall_signal: Mapped[str] = mapped_column(String(20), nullable=False)  # bullish, bearish, neutral
    overall_confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(Text)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("overall_signal IN ('bullish', 'bearish', 'neutral')", name="ck_ai_overall_signal"),
        CheckConstraint("overall_confidence >= 0 AND overall_confidence <= 100", name="ck_ai_confidence_range"),
        Index("idx_ai_agent_analysis_symbol", "symbol"),
        Index("idx_ai_agent_analysis_expires", "expires_at"),
        Index("idx_ai_agent_analysis_signal", "overall_signal"),
    )


# =============================================================================
# STOCK FUNDAMENTALS
# =============================================================================


class StockFundamentals(Base):
    """Stock fundamental metrics from Yahoo Finance."""
    __tablename__ = "stock_fundamentals"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    
    # Valuation
    pe_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    forward_pe: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    peg_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    price_to_book: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    price_to_sales: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    enterprise_value: Mapped[Optional[int]] = mapped_column(BigInteger)
    ev_to_ebitda: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    ev_to_revenue: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    
    # Profitability
    profit_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    operating_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    gross_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    ebitda_margin: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    return_on_equity: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    return_on_assets: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    
    # Financial Health
    debt_to_equity: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    current_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    quick_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    total_cash: Mapped[Optional[int]] = mapped_column(BigInteger)
    total_debt: Mapped[Optional[int]] = mapped_column(BigInteger)
    free_cash_flow: Mapped[Optional[int]] = mapped_column(BigInteger)
    operating_cash_flow: Mapped[Optional[int]] = mapped_column(BigInteger)
    
    # Per Share
    book_value: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    eps_trailing: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    eps_forward: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    revenue_per_share: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    
    # Growth
    revenue_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    earnings_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    earnings_quarterly_growth: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    
    # Shares & Ownership
    shares_outstanding: Mapped[Optional[int]] = mapped_column(BigInteger)
    float_shares: Mapped[Optional[int]] = mapped_column(BigInteger)
    held_percent_insiders: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    held_percent_institutions: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    short_ratio: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    short_percent_of_float: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 6))
    
    # Risk
    beta: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 4))
    
    # Analyst Ratings
    recommendation: Mapped[Optional[str]] = mapped_column(String(20))
    recommendation_mean: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    num_analyst_opinions: Mapped[Optional[int]] = mapped_column(Integer)
    target_high_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    target_low_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    target_mean_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    target_median_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    
    # Revenue & Earnings
    revenue: Mapped[Optional[int]] = mapped_column(BigInteger)
    ebitda: Mapped[Optional[int]] = mapped_column(BigInteger)
    net_income: Mapped[Optional[int]] = mapped_column(BigInteger)
    
    # Earnings Calendar
    next_earnings_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    earnings_estimate_high: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    earnings_estimate_low: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    earnings_estimate_avg: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 4))
    
    # Timestamps
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("idx_stock_fundamentals_symbol", "symbol"),
        Index("idx_stock_fundamentals_expires", "expires_at"),
    )


# =============================================================================
# SEARCH CACHE
# =============================================================================


class SymbolSearchResult(Base):
    """Cached symbol search results from yfinance."""
    __tablename__ = "symbol_search_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    exchange: Mapped[Optional[str]] = mapped_column(String(50))
    quote_type: Mapped[Optional[str]] = mapped_column(String(20))
    sector: Mapped[Optional[str]] = mapped_column(String(100))
    industry: Mapped[Optional[str]] = mapped_column(String(100))
    market_cap: Mapped[Optional[int]] = mapped_column(BigInteger)
    search_query: Mapped[Optional[str]] = mapped_column(String(100))
    relevance_score: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2))
    confidence_score: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(4, 3), comment="Combined score (0-1) from relevance, recency, and data quality"
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), comment="Last time this result was returned in a search"
    )
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", name="uq_symbol_search_result"),
        Index("idx_search_results_symbol", "symbol"),
        Index("idx_search_results_query", "search_query"),
        Index("idx_search_results_expires", "expires_at"),
        # Note: trigram index on name and cursor index are created via migration
    )


class SymbolSearchLog(Base):
    """Log all search queries for analytics."""
    __tablename__ = "symbol_search_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    query: Mapped[str] = mapped_column(String(100), nullable=False)
    query_normalized: Mapped[str] = mapped_column(String(100), nullable=False)  # Uppercase, trimmed
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # local, api, mixed
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    user_fingerprint: Mapped[Optional[str]] = mapped_column(String(64))
    searched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("source IN ('local', 'api', 'mixed')", name="ck_search_log_source"),
        Index("idx_search_log_query", "query_normalized"),
        Index("idx_search_log_searched", "searched_at", postgresql_ops={"searched_at": "DESC"}),
    )


# =============================================================================
# DATA VERSIONING & CHANGE DETECTION
# =============================================================================


class DataVersion(Base):
    """Track data versions for change detection (prices, fundamentals, calendar)."""
    __tablename__ = "data_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)  # prices, fundamentals, calendar
    version_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("source IN ('prices', 'fundamentals', 'calendar')", name="ck_data_version_source"),
        UniqueConstraint("symbol", "source", name="uq_data_version_symbol_source"),
        Index("idx_data_versions_symbol", "symbol"),
        Index("idx_data_versions_source", "source"),
        Index("idx_data_versions_updated", "updated_at", postgresql_ops={"updated_at": "DESC"}),
    )


class AnalysisVersion(Base):
    """Track analysis versions to skip unchanged symbols."""
    __tablename__ = "analysis_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)  # bio, rating, agent_buffett, etc.
    input_version_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # Combined hash of input data
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    batch_job_id: Mapped[Optional[str]] = mapped_column(String(100))

    __table_args__ = (
        UniqueConstraint("symbol", "analysis_type", name="uq_analysis_version_symbol_type"),
        Index("idx_analysis_versions_symbol", "symbol"),
        Index("idx_analysis_versions_type", "analysis_type"),
        Index("idx_analysis_versions_expires", "expires_at"),
    )


# =============================================================================
# SYMBOL INGEST QUEUE
# =============================================================================


class SymbolIngestQueue(Base):
    """Queue for newly added symbols awaiting initial data fetch."""
    __tablename__ = "symbol_ingest_queue"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, processing, completed, failed
    priority: Mapped[int] = mapped_column(Integer, default=0)  # Higher = more urgent
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    last_error: Mapped[Optional[str]] = mapped_column(Text)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processing_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name="ck_ingest_queue_status"),
        Index("idx_ingest_queue_status", "status"),
        Index("idx_ingest_queue_pending", "status", "queued_at", postgresql_where=text("status = 'pending'")),
    )


# =============================================================================
# SCHEMA MIGRATIONS (for tracking)
# =============================================================================


class SchemaMigration(Base):
    """Track applied schema migrations."""
    __tablename__ = "schema_migrations"

    version: Mapped[str] = mapped_column(String(50), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
