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

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
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
    mfa_secret: Mapped[str | None] = mapped_column(String(64))
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_backup_codes: Mapped[str | None] = mapped_column(Text)  # JSON array of hashed codes
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    api_keys: Mapped[list[UserApiKey]] = relationship(back_populates="user")
    secure_keys: Mapped[list[SecureApiKey]] = relationship(back_populates="created_by_user")
    portfolios: Mapped[list[Portfolio]] = relationship(back_populates="user")
    portfolio_analytics_jobs: Mapped[list[PortfolioAnalyticsJob]] = relationship(back_populates="user")

    __table_args__ = (
        Index("idx_auth_user_username", "username"),
    )


class SecureApiKey(Base):
    """Admin-managed API keys for secure storage (OpenAI, etc.)."""
    __tablename__ = "secure_api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    service_name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    encrypted_key: Mapped[str] = mapped_column(Text, nullable=False)
    key_hint: Mapped[str | None] = mapped_column(String(20))
    created_by_id: Mapped[int | None] = mapped_column("created_by", ForeignKey("auth_user.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    created_by_user: Mapped[AuthUser | None] = relationship(back_populates="secure_keys")

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
    description: Mapped[str | None] = mapped_column(Text)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("auth_user.id"))
    vote_weight: Mapped[int] = mapped_column(Integer, default=10)
    rate_limit_bypass: Mapped[bool] = mapped_column(Boolean, default=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    user: Mapped[AuthUser | None] = relationship(back_populates="api_keys")
    dip_votes: Mapped[list[DipVote]] = relationship(back_populates="api_key")
    suggestion_votes: Mapped[list[SuggestionVote]] = relationship(back_populates="api_key")

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
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    market_cap: Mapped[int | None] = mapped_column(BigInteger)
    summary_ai: Mapped[str | None] = mapped_column(String(500))  # AI-generated summary (300-400 target, 500 max)
    symbol_type: Mapped[str] = mapped_column(String(20), default="stock")  # stock, etf, index
    min_dip_pct: Mapped[Decimal] = mapped_column(Numeric(5, 4), default=Decimal("0.15"))
    min_days: Mapped[int] = mapped_column(Integer, default=5)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # Cached stock info (refreshed daily by scheduled job)
    fifty_two_week_low: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    fifty_two_week_high: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    previous_close: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    avg_volume: Mapped[int | None] = mapped_column(BigInteger)
    pe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    stock_info_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Logo caching
    logo_light: Mapped[bytes | None] = mapped_column(LargeBinary)
    logo_dark: Mapped[bytes | None] = mapped_column(LargeBinary)
    logo_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    logo_source: Mapped[str | None] = mapped_column(String(50))
    # Fetch status
    fetch_status: Mapped[str] = mapped_column(String(20), default="pending")
    fetch_error: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    dip_state: Mapped[DipState | None] = relationship(back_populates="symbol_ref", uselist=False)
    dipfinder_config: Mapped[DipfinderConfig | None] = relationship(back_populates="symbol_ref", uselist=False)

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
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    ath_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    dip_percentage: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    dip_start_date: Mapped[date | None] = mapped_column(Date)
    opportunity_type: Mapped[str | None] = mapped_column(String(20), default="NONE")  # OUTLIER, BOUNCE, BOTH, NONE
    # Extreme Value Analysis (EVA) fields
    is_tail_event: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")  # Is this an extreme tail event?
    return_period_years: Mapped[Decimal | None] = mapped_column(Numeric(6, 1))  # How rare? (years between similar events)
    regime_dip_percentile: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))  # Percentile within normal regime
    # Legacy columns
    ref_high: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    last_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    days_below: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    symbol_ref: Mapped[Symbol] = relationship(back_populates="dip_state")

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
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    ath_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    dip_percentage: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
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
    company_name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(100))
    summary: Mapped[str | None] = mapped_column(Text)
    website: Mapped[str | None] = mapped_column(String(255))
    ipo_year: Mapped[int | None] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, approved, rejected
    vote_score: Mapped[int] = mapped_column(Integer, default=0)
    fetch_status: Mapped[str] = mapped_column(String(20), default="pending")
    fetch_error: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    ath_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    approved_by_id: Mapped[int | None] = mapped_column("approved_by", ForeignKey("auth_user.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    votes: Mapped[list[SuggestionVote]] = relationship(back_populates="suggestion")

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
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("user_api_keys.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    suggestion: Mapped[StockSuggestion] = relationship(back_populates="votes")
    api_key: Mapped[UserApiKey | None] = relationship(back_populates="suggestion_votes")

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
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("user_api_keys.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    api_key: Mapped[UserApiKey | None] = relationship(back_populates="dip_votes")

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
    swipe_bio: Mapped[str | None] = mapped_column(Text)
    ai_rating: Mapped[str | None] = mapped_column(String(20))  # strong_buy, buy, hold, sell, strong_sell
    rating_reasoning: Mapped[str | None] = mapped_column(Text)
    model_used: Mapped[str | None] = mapped_column(String(50))
    tokens_used: Mapped[int | None] = mapped_column(Integer)
    is_batch_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    batch_job_id: Mapped[str | None] = mapped_column(String(100))
    ai_pending: Mapped[bool] = mapped_column(Boolean, default=False)  # True when queued for batch, not yet completed
    generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

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
    endpoint: Mapped[str | None] = mapped_column(String(100))
    model: Mapped[str | None] = mapped_column(String(50))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    is_batch: Mapped[bool] = mapped_column(Boolean, default=False)
    request_metadata: Mapped[dict | None] = mapped_column(JSONB)
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
    input_file_id: Mapped[str | None] = mapped_column(String(100))
    output_file_id: Mapped[str | None] = mapped_column(String(100))
    error_file_id: Mapped[str | None] = mapped_column(String(100))
    estimated_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    actual_cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    job_metadata: Mapped[dict | None] = mapped_column("metadata", JSONB)  # Named 'metadata' in DB
    task_custom_ids: Mapped[dict | None] = mapped_column(JSONB)  # Map custom_id → task metadata

    # Relationship to errors
    errors: Mapped[list[BatchTaskError]] = relationship(back_populates="batch_job")

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
    agent_id: Mapped[str | None] = mapped_column(String(50))  # For agent batch tasks
    error_type: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., 'api_error', 'validation_error', 'timeout'
    error_message: Mapped[str | None] = mapped_column(Text)
    original_request: Mapped[dict | None] = mapped_column(JSONB)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, retrying, resolved, abandoned
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationship
    batch_job: Mapped[BatchJob] = relationship(back_populates="errors")

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
    cron: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    next_run: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(20))
    last_duration_ms: Mapped[int | None] = mapped_column(Integer)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    config: Mapped[dict | None] = mapped_column(JSONB)
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


class SettingsChangeHistory(Base):
    """Audit log for settings changes with revert capability."""
    __tablename__ = "settings_change_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    setting_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'runtime', 'cronjob', etc.
    setting_key: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., 'ai_enrichment_enabled' or cronjob name
    old_value: Mapped[dict | None] = mapped_column(JSONB)  # Previous value (JSON)
    new_value: Mapped[dict | None] = mapped_column(JSONB)  # New value (JSON)
    changed_by: Mapped[int] = mapped_column(ForeignKey("auth_user.id", ondelete="SET NULL"), nullable=True)
    changed_by_username: Mapped[str | None] = mapped_column(String(100))  # Denormalized for display
    change_reason: Mapped[str | None] = mapped_column(Text)  # Optional reason for change
    reverted: Mapped[bool] = mapped_column(Boolean, default=False)  # Whether this change was reverted
    reverted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reverted_by: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped[AuthUser | None] = relationship(foreign_keys=[changed_by])

    __table_args__ = (
        Index("idx_settings_history_type", "setting_type"),
        Index("idx_settings_history_key", "setting_key"),
        Index("idx_settings_history_created", "created_at", postgresql_ops={"created_at": "DESC"}),
    )


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
    open: Mapped[Decimal | None] = mapped_column(Numeric(16, 6))
    high: Mapped[Decimal | None] = mapped_column(Numeric(16, 6))
    low: Mapped[Decimal | None] = mapped_column(Numeric(16, 6))
    close: Mapped[Decimal] = mapped_column(Numeric(16, 6), nullable=False)
    adj_close: Mapped[Decimal | None] = mapped_column(Numeric(16, 6))
    volume: Mapped[int | None] = mapped_column(BigInteger)
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
    dip_stock: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    peak_stock: Mapped[Decimal | None] = mapped_column(Numeric(16, 6))
    dip_pctl: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    dip_vs_typical: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    persist_days: Mapped[int | None] = mapped_column(Integer)

    # Market context
    dip_mkt: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    excess_dip: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    dip_class: Mapped[str | None] = mapped_column(String(20))

    # Scores
    quality_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    stability_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    dip_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    final_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))

    # Alert
    alert_level: Mapped[str | None] = mapped_column(String(20))
    should_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[str | None] = mapped_column(Text)

    # Contributing factors
    quality_factors: Mapped[dict | None] = mapped_column(JSONB)
    stability_factors: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

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
    symbol: Mapped[str] = mapped_column(String(20), ForeignKey("symbols.symbol", ondelete="CASCADE"), unique=True, nullable=False)
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
    symbol_ref: Mapped[Symbol | None] = relationship(back_populates="dipfinder_config")

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
    dip_pct: Mapped[Decimal | None] = mapped_column(Numeric(8, 6))
    final_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    dip_class: Mapped[str | None] = mapped_column(String(20))
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
    overall_signal: Mapped[str] = mapped_column(String(20), nullable=False)  # strong_buy, buy, hold, sell, strong_sell
    overall_confidence: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    agent_pending: Mapped[bool] = mapped_column(Boolean, default=False)  # True if batch queued but not completed
    analyzed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("overall_signal IN ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')", name="ck_ai_overall_signal"),
        CheckConstraint("overall_confidence >= 0 AND overall_confidence <= 100", name="ck_ai_confidence_range"),
        Index("idx_ai_agent_analysis_symbol", "symbol"),
        Index("idx_ai_agent_analysis_expires", "expires_at"),
        Index("idx_ai_agent_analysis_signal", "overall_signal"),
    )


# =============================================================================
# AI PERSONA CONFIGURATION
# =============================================================================


class AIPersona(Base):
    """AI investor persona configuration with avatar images.
    
    Stores persona details like Warren Buffett, Peter Lynch, etc.
    with customizable avatar images for display in the UI.
    """
    __tablename__ = "ai_persona"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)  # e.g., "warren_buffett"
    name: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "Warren Buffett"
    description: Mapped[str | None] = mapped_column(Text)  # Short bio/description
    philosophy: Mapped[str | None] = mapped_column(Text)  # Investment philosophy
    avatar_data: Mapped[bytes | None] = mapped_column(LargeBinary)  # Optimized WebP avatar image
    avatar_mime_type: Mapped[str | None] = mapped_column(String(50))  # e.g., "image/webp"
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0)  # For UI ordering
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_ai_persona_key", "key"),
        Index("idx_ai_persona_active", "is_active", "display_order"),
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
    pe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    forward_pe: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    peg_ratio: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    price_to_book: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    price_to_sales: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    enterprise_value: Mapped[int | None] = mapped_column(BigInteger)
    ev_to_ebitda: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    ev_to_revenue: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    # Profitability
    profit_margin: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    operating_margin: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    gross_margin: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    ebitda_margin: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    return_on_equity: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    return_on_assets: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    # Financial Health
    debt_to_equity: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    current_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    quick_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    total_cash: Mapped[int | None] = mapped_column(BigInteger)
    total_debt: Mapped[int | None] = mapped_column(BigInteger)
    free_cash_flow: Mapped[int | None] = mapped_column(BigInteger)
    operating_cash_flow: Mapped[int | None] = mapped_column(BigInteger)

    # Per Share
    book_value: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    eps_trailing: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    eps_forward: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    revenue_per_share: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    # Growth
    revenue_growth: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    earnings_growth: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    earnings_quarterly_growth: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    # Shares & Ownership
    shares_outstanding: Mapped[int | None] = mapped_column(BigInteger)
    float_shares: Mapped[int | None] = mapped_column(BigInteger)
    held_percent_insiders: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    held_percent_institutions: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    short_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    short_percent_of_float: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    # Risk
    beta: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))

    # Analyst Ratings
    recommendation: Mapped[str | None] = mapped_column(String(20))
    recommendation_mean: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    num_analyst_opinions: Mapped[int | None] = mapped_column(Integer)
    target_high_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    target_low_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    target_mean_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    target_median_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    # Revenue & Earnings
    revenue: Mapped[int | None] = mapped_column(BigInteger)
    ebitda: Mapped[int | None] = mapped_column(BigInteger)
    net_income: Mapped[int | None] = mapped_column(BigInteger)

    # Earnings Calendar
    next_earnings_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    earnings_estimate_high: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    earnings_estimate_low: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    earnings_estimate_avg: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    earnings_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # Last earnings date
    
    # Dividend Calendar
    dividend_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # Next dividend pay date
    ex_dividend_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # Ex-dividend date

    # Domain Classification
    domain: Mapped[str | None] = mapped_column(String(20))  # bank, reit, insurer, utility, biotech, etf, stock

    # Financial Statements (JSONB - quarterly and annual income stmt, balance sheet, cash flow)
    income_stmt_quarterly: Mapped[dict | None] = mapped_column(JSONB)
    income_stmt_annual: Mapped[dict | None] = mapped_column(JSONB)
    balance_sheet_quarterly: Mapped[dict | None] = mapped_column(JSONB)
    balance_sheet_annual: Mapped[dict | None] = mapped_column(JSONB)
    cash_flow_quarterly: Mapped[dict | None] = mapped_column(JSONB)
    cash_flow_annual: Mapped[dict | None] = mapped_column(JSONB)

    # Domain-Specific Metrics (pre-calculated from financial statements)
    # Banks
    net_interest_income: Mapped[int | None] = mapped_column(BigInteger)  # NII from income stmt
    net_interest_margin: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))  # NIM = NII / Interest-earning assets
    interest_income: Mapped[int | None] = mapped_column(BigInteger)  # Total interest income
    interest_expense: Mapped[int | None] = mapped_column(BigInteger)  # Total interest expense

    # REITs
    ffo: Mapped[int | None] = mapped_column(BigInteger)  # Funds From Operations = Net Income + D&A
    ffo_per_share: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    p_ffo: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))  # Price / FFO per share

    # Insurers
    loss_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))  # Loss adj expense / Premiums
    expense_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))  # Total expenses / Revenue
    combined_ratio: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))  # Loss ratio + expense ratio

    # Timestamps
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    financials_fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # When statements were fetched

    __table_args__ = (
        Index("idx_stock_fundamentals_symbol", "symbol"),
        Index("idx_stock_fundamentals_expires", "expires_at"),
        Index("idx_stock_fundamentals_domain", "domain"),
        Index("idx_stock_fundamentals_next_earnings", "next_earnings_date"),
    )


# =============================================================================
# SEARCH CACHE
# =============================================================================


class SymbolSearchResult(Base):
    """Cached symbol search results from yfinance."""
    __tablename__ = "symbol_search_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    exchange: Mapped[str | None] = mapped_column(String(50))
    quote_type: Mapped[str | None] = mapped_column(String(20))
    sector: Mapped[str | None] = mapped_column(String(100))
    industry: Mapped[str | None] = mapped_column(String(100))
    market_cap: Mapped[int | None] = mapped_column(BigInteger)
    search_query: Mapped[str | None] = mapped_column(String(100))
    relevance_score: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    confidence_score: Mapped[Decimal | None] = mapped_column(
        Numeric(4, 3), comment="Combined score (0-1) from relevance, recency, and data quality"
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
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
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    user_fingerprint: Mapped[str | None] = mapped_column(String(64))
    searched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        CheckConstraint("source IN ('local', 'api', 'mixed')", name="ck_search_log_source"),
        Index("idx_search_log_query", "query_normalized"),
        Index("idx_search_log_searched", "searched_at", postgresql_ops={"searched_at": "DESC"}),
    )


# =============================================================================
# FINANCIAL UNIVERSE (from FinanceDatabase)
# =============================================================================


class FinancialUniverse(Base):
    """Comprehensive financial instrument universe from FinanceDatabase.
    
    Contains ~130K symbols across asset classes:
    - Equities (~24K) - stocks with sector/industry/country metadata
    - ETFs (~3K) - category/family groupings
    - Funds (~31K) - mutual funds with category info
    - Indices (~62K) - market indices
    - Cryptos (~3K) - cryptocurrencies
    - Currencies (~3K) - forex pairs
    - Moneymarkets (~1K) - money market instruments
    
    Used for:
    - Fast local autocomplete/search before API calls
    - Symbol validation without external API
    - Sector/industry/country faceting
    - ISIN/CUSIP/FIGI identifier resolution
    
    Synced weekly from financedatabase Python package.
    """
    __tablename__ = "financial_universe"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    name: Mapped[str | None] = mapped_column(String(255))
    
    # Asset classification
    asset_class: Mapped[str] = mapped_column(String(20), nullable=False)  # equity, etf, fund, index, crypto, currency, moneymarket
    
    # Sector/Industry (primarily for equities)
    sector: Mapped[str | None] = mapped_column(String(100))  # e.g., "Technology", "Health Care"
    industry_group: Mapped[str | None] = mapped_column(String(150))  # e.g., "Software & Services"
    industry: Mapped[str | None] = mapped_column(String(150))  # e.g., "Systems Software"
    
    # For ETFs/Funds: category groupings
    category_group: Mapped[str | None] = mapped_column(String(100))  # ETF/Fund category group
    category: Mapped[str | None] = mapped_column(String(100))  # ETF/Fund category
    family: Mapped[str | None] = mapped_column(String(100))  # Fund family (e.g., "Vanguard")
    
    # Market/Exchange info
    exchange: Mapped[str | None] = mapped_column(String(50))  # Exchange code (e.g., "NYQ", "NMS")
    market: Mapped[str | None] = mapped_column(String(100))  # Market name (e.g., "NASDAQ")
    country: Mapped[str | None] = mapped_column(String(100))  # Country of domicile
    currency: Mapped[str | None] = mapped_column(String(10))  # Trading currency
    
    # Unique identifiers for cross-referencing
    isin: Mapped[str | None] = mapped_column(String(12))  # International Securities Identification Number
    cusip: Mapped[str | None] = mapped_column(String(9))  # US/Canada identifier
    figi: Mapped[str | None] = mapped_column(String(12))  # Bloomberg FIGI
    composite_figi: Mapped[str | None] = mapped_column(String(12))  # Composite FIGI
    
    # Size classification
    market_cap_category: Mapped[str | None] = mapped_column(String(20))  # mega, large, mid, small, micro
    
    # Business summary (truncated for space efficiency)
    summary: Mapped[str | None] = mapped_column(Text)
    
    # Metadata
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # When FinanceDB data was updated
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint(
            "asset_class IN ('equity', 'etf', 'fund', 'index', 'crypto', 'currency', 'moneymarket')",
            name="ck_universe_asset_class",
        ),
        Index("idx_universe_symbol", "symbol"),
        Index("idx_universe_name", "name"),  # For LIKE searches; trigram GIN added via migration
        Index("idx_universe_asset_class", "asset_class"),
        Index("idx_universe_sector", "sector"),
        Index("idx_universe_industry", "industry"),
        Index("idx_universe_country", "country"),
        Index("idx_universe_exchange", "exchange"),
        Index("idx_universe_isin", "isin", postgresql_where=text("isin IS NOT NULL")),
        Index("idx_universe_cusip", "cusip", postgresql_where=text("cusip IS NOT NULL")),
        Index("idx_universe_figi", "figi", postgresql_where=text("figi IS NOT NULL")),
        Index("idx_universe_active", "is_active", "asset_class"),
    )


# =============================================================================
# PORTFOLIOS & HOLDINGS
# =============================================================================


class Portfolio(Base):
    """User-managed investment portfolio."""
    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("auth_user.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    base_currency: Mapped[str] = mapped_column(String(10), default="USD")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # AI analysis tracking
    ai_analysis_summary: Mapped[str | None] = mapped_column(Text)
    ai_analysis_hash: Mapped[str | None] = mapped_column(String(32))  # MD5 of holdings state
    ai_analysis_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    user: Mapped[AuthUser] = relationship(back_populates="portfolios")
    holdings: Mapped[list[PortfolioHolding]] = relationship(back_populates="portfolio")
    transactions: Mapped[list[PortfolioTransaction]] = relationship(back_populates="portfolio")
    analytics: Mapped[list[PortfolioAnalytics]] = relationship(back_populates="portfolio")
    analytics_jobs: Mapped[list[PortfolioAnalyticsJob]] = relationship(back_populates="portfolio")

    __table_args__ = (
        Index("idx_portfolios_user", "user_id"),
        Index("idx_portfolios_active", "is_active", postgresql_where=text("is_active = TRUE")),
    )


class PortfolioHolding(Base):
    """Current holdings for a portfolio."""
    __tablename__ = "portfolio_holdings"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(18, 6), nullable=False)
    avg_cost: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    target_weight: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "symbol", name="uq_portfolio_holdings_symbol"),
        Index("idx_portfolio_holdings_portfolio", "portfolio_id"),
        Index("idx_portfolio_holdings_symbol", "symbol"),
    )


class PortfolioTransaction(Base):
    """Ledger of portfolio transactions."""
    __tablename__ = "portfolio_transactions"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    side: Mapped[str] = mapped_column(String(20), nullable=False)  # buy, sell, dividend, split, deposit, withdrawal
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(18, 6))
    price: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    fees: Mapped[Decimal | None] = mapped_column(Numeric(18, 4))
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    portfolio: Mapped[Portfolio] = relationship(back_populates="transactions")

    __table_args__ = (
        CheckConstraint(
            "side IN ('buy', 'sell', 'dividend', 'split', 'deposit', 'withdrawal')",
            name="ck_portfolio_transactions_side",
        ),
        Index("idx_portfolio_transactions_portfolio", "portfolio_id"),
        Index("idx_portfolio_transactions_symbol", "symbol"),
        Index("idx_portfolio_transactions_date", "trade_date", postgresql_ops={"trade_date": "DESC"}),
    )


class PortfolioAnalytics(Base):
    """Stored analytics output for portfolio tools."""
    __tablename__ = "portfolio_analytics"

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    tool: Mapped[str] = mapped_column(String(50), nullable=False)
    as_of_date: Mapped[date | None] = mapped_column(Date)
    window: Mapped[str | None] = mapped_column(String(50))
    params: Mapped[dict | None] = mapped_column(JSONB)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="ok")  # ok, error, partial
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    portfolio: Mapped[Portfolio] = relationship(back_populates="analytics")

    __table_args__ = (
        CheckConstraint("status IN ('ok', 'error', 'partial')", name="ck_portfolio_analytics_status"),
        Index("idx_portfolio_analytics_portfolio", "portfolio_id"),
        Index("idx_portfolio_analytics_tool", "tool"),
        Index("idx_portfolio_analytics_created", "created_at", postgresql_ops={"created_at": "DESC"}),
    )


class PortfolioAnalyticsJob(Base):
    """Background analytics job for heavy portfolio tools."""
    __tablename__ = "portfolio_analytics_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    portfolio_id: Mapped[int] = mapped_column(ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("auth_user.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    tools: Mapped[list] = mapped_column(JSONB, nullable=False)
    params: Mapped[dict | None] = mapped_column(JSONB)
    window: Mapped[str | None] = mapped_column(String(50))
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)
    benchmark: Mapped[str | None] = mapped_column(String(20))
    force_refresh: Mapped[bool] = mapped_column(Boolean, default=False)
    results_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    portfolio: Mapped[Portfolio] = relationship(back_populates="analytics_jobs")
    user: Mapped[AuthUser] = relationship(back_populates="portfolio_analytics_jobs")

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_portfolio_analytics_jobs_status",
        ),
        Index("idx_portfolio_analytics_jobs_portfolio", "portfolio_id"),
        Index("idx_portfolio_analytics_jobs_user", "user_id"),
        Index("idx_portfolio_analytics_jobs_status", "status"),
        Index("idx_portfolio_analytics_jobs_created", "created_at", postgresql_ops={"created_at": "DESC"}),
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
    version_metadata: Mapped[dict | None] = mapped_column(JSONB)
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
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    batch_job_id: Mapped[str | None] = mapped_column(String(100))

    __table_args__ = (
        UniqueConstraint("symbol", "analysis_type", name="uq_analysis_version_symbol_type"),
        Index("idx_analysis_versions_symbol", "symbol"),
        Index("idx_analysis_versions_type", "analysis_type"),
        Index("idx_analysis_versions_expires", "expires_at"),
    )


# =============================================================================
# STRATEGY SIGNALS
# =============================================================================


class StrategySignal(Base):
    """Optimized trading strategy signals computed nightly.
    
    Stores the best strategy for each symbol with performance metrics.
    Updated by the strategy_nightly job.
    """
    __tablename__ = "strategy_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    
    # Strategy info
    strategy_name: Mapped[str] = mapped_column(String(50), nullable=False)
    strategy_params: Mapped[dict] = mapped_column(JSONB, default=dict)
    
    # Current signal
    signal_type: Mapped[str] = mapped_column(String(20), nullable=False)  # BUY, SELL, HOLD, WAIT, WATCH
    signal_reason: Mapped[str] = mapped_column(Text)
    has_active_signal: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Performance metrics
    total_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    win_rate: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    max_drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    n_trades: Mapped[int] = mapped_column(Integer, default=0)
    
    # Recency-weighted metrics (prioritize recent performance)
    recency_weighted_return: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    current_year_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    current_year_win_rate: Mapped[Decimal | None] = mapped_column(Numeric(8, 2))
    current_year_trades: Mapped[int] = mapped_column(Integer, default=0)
    
    # Benchmark comparison
    vs_buy_hold_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))  # Excess return
    vs_spy_pct: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    beats_buy_hold: Mapped[bool] = mapped_column(Boolean, default=False)
    beats_spy: Mapped[bool] = mapped_column(Boolean, default=False)  # Strategy beats SPY benchmark
    
    # Fundamental status
    fundamentals_healthy: Mapped[bool] = mapped_column(Boolean, default=False)
    fundamental_concerns: Mapped[list] = mapped_column(JSONB, default=list)
    
    # Validation
    is_statistically_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Recovery time (from dip entry optimizer)
    typical_recovery_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    # Recent trades (last 5)
    recent_trades: Mapped[list] = mapped_column(JSONB, default=list)
    
    # Indicators used
    indicators_used: Mapped[list] = mapped_column(JSONB, default=list)
    
    # Timestamps
    optimized_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_strategy_signals_symbol", "symbol"),
        Index("idx_strategy_signals_signal", "signal_type"),
        Index("idx_strategy_signals_active", "has_active_signal", postgresql_where=text("has_active_signal = TRUE")),
        Index("idx_strategy_signals_beats", "beats_buy_hold", postgresql_where=text("beats_buy_hold = TRUE")),
    )


# =============================================================================
# QUANT PRECOMPUTED RESULTS (Nightly Job Cache)
# =============================================================================


class QuantPrecomputed(Base):
    """
    Pre-computed quant analysis results for each symbol.
    
    Populated nightly by quant_analysis_job. API endpoints read from here
    instead of computing heavy operations inline.
    """
    __tablename__ = "quant_precomputed"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    
    # Signal Backtest Results
    backtest_signal_name: Mapped[str | None] = mapped_column(String(100))
    backtest_n_trades: Mapped[int | None] = mapped_column(Integer)
    backtest_win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    backtest_total_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    backtest_avg_return_per_trade: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    backtest_holding_days: Mapped[int | None] = mapped_column(Integer)
    backtest_buy_hold_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    backtest_edge_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    backtest_outperformed: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Full Trade Strategy Results
    trade_entry_signal: Mapped[str | None] = mapped_column(String(100))
    trade_entry_threshold: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    trade_exit_signal: Mapped[str | None] = mapped_column(String(100))
    trade_exit_threshold: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    trade_n_trades: Mapped[int | None] = mapped_column(Integer)
    trade_win_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    trade_total_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    trade_avg_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    trade_sharpe_ratio: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    trade_buy_hold_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    trade_spy_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    trade_beats_both: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Signal Combinations (JSONB for flexibility)
    signal_combinations: Mapped[dict | None] = mapped_column(JSONB)
    
    # Dip Analysis Results
    dip_current_drawdown_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    dip_typical_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    dip_max_historical_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    dip_zscore: Mapped[Decimal | None] = mapped_column(Numeric(8, 4))
    dip_type: Mapped[str | None] = mapped_column(String(30))  # OVERREACTION, NORMAL_VOLATILITY, FUNDAMENTAL_DECLINE
    dip_action: Mapped[str | None] = mapped_column(String(20))  # STRONG_BUY, BUY, WAIT, AVOID
    dip_confidence: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    dip_reasoning: Mapped[str | None] = mapped_column(Text)
    dip_recovery_probability: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))
    
    # Current Signals (JSONB for flexibility)
    current_signals: Mapped[dict | None] = mapped_column(JSONB)
    
    # Dip Entry Analysis Results - Risk-Adjusted Optimal
    dip_entry_optimal_threshold: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    dip_entry_optimal_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    # Dip Entry Analysis Results - Max Profit Optimal
    dip_entry_max_profit_threshold: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    dip_entry_max_profit_price: Mapped[Decimal | None] = mapped_column(Numeric(14, 4))
    dip_entry_max_profit_total_return: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    # Dip Entry Analysis Results - Common fields
    dip_entry_is_buy_now: Mapped[bool] = mapped_column(Boolean, default=False)
    dip_entry_signal_strength: Mapped[Decimal | None] = mapped_column(Numeric(5, 2))
    dip_entry_signal_reason: Mapped[str | None] = mapped_column(Text)
    dip_entry_recovery_days: Mapped[int | None] = mapped_column(Integer)
    dip_entry_threshold_analysis: Mapped[dict | None] = mapped_column(JSONB)
    
    # Signal Triggers (for chart overlays)
    signal_triggers: Mapped[dict | None] = mapped_column(JSONB)
    
    # Metadata
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    data_start: Mapped[date | None] = mapped_column(Date)
    data_end: Mapped[date | None] = mapped_column(Date)

    __table_args__ = (
        Index("idx_quant_precomputed_symbol", "symbol"),
        Index("idx_quant_precomputed_computed", "computed_at", postgresql_ops={"computed_at": "DESC"}),
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
    last_error: Mapped[str | None] = mapped_column(Text)
    queued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name="ck_ingest_queue_status"),
        Index("idx_ingest_queue_status", "status"),
        Index("idx_ingest_queue_pending", "status", "queued_at", postgresql_where=text("status = 'pending'")),
    )


# =============================================================================
# QUANT SCORES (APUS + DOUS Daily Scoring)
# =============================================================================


class QuantScore(Base):
    """
    Daily dual-mode scoring results for each symbol.
    
    Implements the APUS (Certified Buy) + DOUS (Dip Entry) scoring pipeline.
    Persisted for auditability and historical tracking.
    """
    __tablename__ = "quant_scores"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Final score and mode
    best_score: Mapped[Decimal] = mapped_column(Numeric(6, 2), nullable=False)  # 0-100
    mode: Mapped[str] = mapped_column(String(20), nullable=False)  # CERTIFIED_BUY or DIP_ENTRY
    score_a: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))  # Mode A score
    score_b: Mapped[Decimal | None] = mapped_column(Numeric(6, 2))  # Mode B score
    gate_pass: Mapped[bool] = mapped_column(Boolean, default=False)  # Did Mode A gate pass?
    
    # Statistical validation (Mode A)
    p_outperf: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))  # P(edge > 0)
    ci_low: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))  # 95% CI lower bound
    ci_high: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))  # 95% CI upper bound
    dsr: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))  # Deflated Sharpe Ratio
    
    # Edge metrics
    median_edge: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    edge_vs_stock: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    edge_vs_spy: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    worst_regime_edge: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    cvar_5: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))  # CVaR at 5%
    
    # Fundamental metrics (Mode B)
    fund_mom: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))  # Fundamental momentum z
    val_z: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))  # Valuation z-score
    event_risk: Mapped[bool] = mapped_column(Boolean, default=False)  # Earnings/dividend within 7 days
    
    # Dip metrics (Mode B)
    p_recovery: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))  # P(recovery within H days)
    expected_value: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))  # EV of dip entry
    sector_relative: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))  # Sector relative drawdown
    
    # Metadata
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # Hash of scoring config
    scoring_version: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., "1.0.0"
    data_start: Mapped[date] = mapped_column(Date, nullable=False)  # Start of data used
    data_end: Mapped[date] = mapped_column(Date, nullable=False)  # End of data used
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Full evidence block (JSONB for flexibility)
    evidence: Mapped[dict | None] = mapped_column(JSONB)

    __table_args__ = (
        UniqueConstraint("symbol", name="uq_quant_scores_symbol"),
        Index("idx_quant_scores_symbol", "symbol"),
        Index("idx_quant_scores_symbol_date", "symbol", "computed_at"),
        Index("idx_quant_scores_latest", "symbol", "computed_at", postgresql_using="btree"),
        Index("idx_quant_scores_best_score", "best_score", postgresql_ops={"best_score": "DESC"}),
    )


# =============================================================================
# SCHEMA MIGRATIONS (for tracking)
# =============================================================================


class SchemaMigration(Base):
    """Track applied schema migrations."""
    __tablename__ = "schema_migrations"

    version: Mapped[str] = mapped_column(String(50), primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
