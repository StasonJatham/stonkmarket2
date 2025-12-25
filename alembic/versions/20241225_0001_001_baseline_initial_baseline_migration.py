"""Initial baseline migration matching existing schema.

Revision ID: 001_baseline
Revises: 
Create Date: 2024-12-25

This migration represents the existing database schema from init.sql.
It should be marked as applied for existing databases without running.

For new databases, this creates the full schema.
For existing databases, run: alembic stamp 001_baseline
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_baseline"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables from the baseline schema.
    
    Note: For existing databases that were created with init.sql,
    run 'alembic stamp 001_baseline' to mark this as applied without running it.
    """
    # Enable PostgreSQL extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')
    
    # ==========================================================================
    # AUTH & SECURITY TABLES
    # ==========================================================================
    
    op.create_table(
        "auth_user",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("is_admin", sa.Boolean(), default=False),
        sa.Column("mfa_secret", sa.String(64)),
        sa.Column("mfa_enabled", sa.Boolean(), default=False),
        sa.Column("mfa_backup_codes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_auth_user_username", "auth_user", ["username"])
    
    op.create_table(
        "secure_api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("service_name", sa.String(100), unique=True, nullable=False),
        sa.Column("encrypted_key", sa.Text(), nullable=False),
        sa.Column("key_hint", sa.String(20)),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("auth_user.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_secure_api_keys_service", "secure_api_keys", ["service_name"])
    
    op.create_table(
        "user_api_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("key_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("key_prefix", sa.String(16), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("auth_user.id")),
        sa.Column("vote_weight", sa.Integer(), default=10),
        sa.Column("rate_limit_bypass", sa.Boolean(), default=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("usage_count", sa.Integer(), default=0),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_user_api_keys_hash", "user_api_keys", ["key_hash"])
    op.execute("CREATE INDEX idx_user_api_keys_active ON user_api_keys(is_active) WHERE is_active = TRUE")
    
    # ==========================================================================
    # STOCK SYMBOL TABLES
    # ==========================================================================
    
    op.create_table(
        "symbols",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("name", sa.String(255)),
        sa.Column("sector", sa.String(100)),
        sa.Column("market_cap", sa.BigInteger()),
        sa.Column("summary_ai", sa.String(350)),
        sa.Column("symbol_type", sa.String(20), default="stock"),
        sa.Column("min_dip_pct", sa.Numeric(5, 4), default=0.15),
        sa.Column("min_days", sa.Integer(), default=5),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("logo_light", sa.LargeBinary()),
        sa.Column("logo_dark", sa.LargeBinary()),
        sa.Column("logo_fetched_at", sa.DateTime(timezone=True)),
        sa.Column("logo_source", sa.String(50)),
        sa.Column("fetch_status", sa.String(20), default="pending"),
        sa.Column("fetch_error", sa.Text()),
        sa.Column("fetched_at", sa.DateTime(timezone=True)),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_symbols_symbol", "symbols", ["symbol"])
    op.create_index("idx_symbols_sector", "symbols", ["sector"])
    op.create_index("idx_symbols_type", "symbols", ["symbol_type"])
    op.create_index("idx_symbols_logo_fetched_at", "symbols", ["logo_fetched_at"])
    
    op.create_table(
        "dip_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), sa.ForeignKey("symbols.symbol", ondelete="CASCADE"), unique=True, nullable=False),
        sa.Column("current_price", sa.Numeric(12, 4)),
        sa.Column("ath_price", sa.Numeric(12, 4)),
        sa.Column("dip_percentage", sa.Numeric(8, 4)),
        sa.Column("dip_start_date", sa.Date()),
        sa.Column("ref_high", sa.Numeric(12, 4)),
        sa.Column("last_price", sa.Numeric(12, 4)),
        sa.Column("days_below", sa.Integer(), default=0),
        sa.Column("first_seen", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_updated", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_dip_state_symbol", "dip_state", ["symbol"])
    op.execute("CREATE INDEX idx_dip_state_percentage ON dip_state(dip_percentage DESC)")
    op.execute("CREATE INDEX idx_dip_state_updated ON dip_state(last_updated DESC)")
    op.create_index("idx_dip_state_start_date", "dip_state", ["dip_start_date"])
    
    op.create_table(
        "dip_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("action", sa.String(10), nullable=False),
        sa.Column("current_price", sa.Numeric(12, 4)),
        sa.Column("ath_price", sa.Numeric(12, 4)),
        sa.Column("dip_percentage", sa.Numeric(8, 4)),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("action IN ('added', 'removed', 'updated')", name="ck_dip_history_action"),
    )
    op.create_index("idx_dip_history_symbol", "dip_history", ["symbol"])
    op.create_index("idx_dip_history_action", "dip_history", ["action"])
    op.execute("CREATE INDEX idx_dip_history_recorded ON dip_history(recorded_at DESC)")
    
    # ==========================================================================
    # STOCK SUGGESTIONS
    # ==========================================================================
    
    op.create_table(
        "stock_suggestions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("company_name", sa.String(255)),
        sa.Column("sector", sa.String(100)),
        sa.Column("summary", sa.Text()),
        sa.Column("website", sa.String(255)),
        sa.Column("ipo_year", sa.Integer()),
        sa.Column("reason", sa.Text()),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("status", sa.String(20), default="pending"),
        sa.Column("vote_score", sa.Integer(), default=0),
        sa.Column("fetch_status", sa.String(20), default="pending"),
        sa.Column("fetch_error", sa.Text()),
        sa.Column("fetched_at", sa.DateTime(timezone=True)),
        sa.Column("current_price", sa.Numeric(12, 4)),
        sa.Column("ath_price", sa.Numeric(12, 4)),
        sa.Column("approved_by", sa.Integer(), sa.ForeignKey("auth_user.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("reviewed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("status IN ('pending', 'approved', 'rejected')", name="ck_suggestion_status"),
    )
    op.create_index("idx_suggestions_status", "stock_suggestions", ["status"])
    op.execute("CREATE INDEX idx_suggestions_score ON stock_suggestions(vote_score DESC)")
    op.create_index("idx_suggestions_symbol", "stock_suggestions", ["symbol"])
    
    op.create_table(
        "suggestion_votes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("suggestion_id", sa.Integer(), sa.ForeignKey("stock_suggestions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("vote_type", sa.String(10), nullable=False),
        sa.Column("vote_weight", sa.Integer(), default=1),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("user_api_keys.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("suggestion_id", "fingerprint", name="uq_suggestion_vote"),
        sa.CheckConstraint("vote_type IN ('up', 'down')", name="ck_suggestion_vote_type"),
    )
    op.create_index("idx_suggestion_votes_suggestion", "suggestion_votes", ["suggestion_id"])
    
    # ==========================================================================
    # DIP VOTING (SWIPE)
    # ==========================================================================
    
    op.create_table(
        "dip_votes",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("vote_type", sa.String(10), nullable=False),
        sa.Column("vote_weight", sa.Integer(), default=1),
        sa.Column("api_key_id", sa.Integer(), sa.ForeignKey("user_api_keys.id")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("symbol", "fingerprint", name="uq_dip_vote"),
        sa.CheckConstraint("vote_type IN ('buy', 'sell')", name="ck_dip_vote_type"),
    )
    op.create_index("idx_dip_votes_symbol", "dip_votes", ["symbol"])
    op.create_index("idx_dip_votes_fingerprint", "dip_votes", ["fingerprint"])
    op.execute("CREATE INDEX idx_dip_votes_created ON dip_votes(created_at DESC)")
    
    op.create_table(
        "dip_ai_analysis",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("swipe_bio", sa.Text()),
        sa.Column("ai_rating", sa.String(20)),
        sa.Column("rating_reasoning", sa.Text()),
        sa.Column("model_used", sa.String(50)),
        sa.Column("tokens_used", sa.Integer()),
        sa.Column("is_batch_generated", sa.Boolean(), default=False),
        sa.Column("batch_job_id", sa.String(100)),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "ai_rating IS NULL OR ai_rating IN ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')",
            name="ck_ai_rating"
        ),
    )
    op.create_index("idx_dip_ai_analysis_symbol", "dip_ai_analysis", ["symbol"])
    op.create_index("idx_dip_ai_analysis_rating", "dip_ai_analysis", ["ai_rating"])
    op.create_index("idx_dip_ai_analysis_expires", "dip_ai_analysis", ["expires_at"])
    
    # ==========================================================================
    # API USAGE & BATCH JOBS
    # ==========================================================================
    
    op.create_table(
        "api_usage",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("service", sa.String(50), nullable=False),
        sa.Column("endpoint", sa.String(100)),
        sa.Column("model", sa.String(50)),
        sa.Column("input_tokens", sa.Integer(), default=0),
        sa.Column("output_tokens", sa.Integer(), default=0),
        sa.Column("cost_usd", sa.Numeric(10, 6)),
        sa.Column("is_batch", sa.Boolean(), default=False),
        sa.Column("request_metadata", postgresql.JSONB()),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_api_usage_service", "api_usage", ["service"])
    op.execute("CREATE INDEX idx_api_usage_recorded ON api_usage(recorded_at DESC)")
    
    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("batch_id", sa.String(100), unique=True, nullable=False),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(20), default="pending"),
        sa.Column("total_requests", sa.Integer(), default=0),
        sa.Column("completed_requests", sa.Integer(), default=0),
        sa.Column("failed_requests", sa.Integer(), default=0),
        sa.Column("input_file_id", sa.String(100)),
        sa.Column("output_file_id", sa.String(100)),
        sa.Column("error_file_id", sa.String(100)),
        sa.Column("estimated_cost_usd", sa.Numeric(10, 6)),
        sa.Column("actual_cost_usd", sa.Numeric(10, 6)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", postgresql.JSONB()),
        sa.CheckConstraint(
            "status IN ('pending', 'validating', 'in_progress', 'finalizing', 'completed', 'failed', 'expired', 'cancelled')",
            name="ck_batch_job_status"
        ),
    )
    op.create_index("idx_batch_jobs_status", "batch_jobs", ["status"])
    op.create_index("idx_batch_jobs_type", "batch_jobs", ["job_type"])
    op.execute("CREATE INDEX idx_batch_jobs_created ON batch_jobs(created_at DESC)")
    
    # ==========================================================================
    # SCHEDULER / CRON JOBS
    # ==========================================================================
    
    op.create_table(
        "cronjobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("cron_expression", sa.String(50), nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("last_run", sa.DateTime(timezone=True)),
        sa.Column("next_run", sa.DateTime(timezone=True)),
        sa.Column("last_status", sa.String(20)),
        sa.Column("last_duration_ms", sa.Integer()),
        sa.Column("run_count", sa.Integer(), default=0),
        sa.Column("error_count", sa.Integer(), default=0),
        sa.Column("last_error", sa.Text()),
        sa.Column("config", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.execute("CREATE INDEX idx_cronjobs_active ON cronjobs(is_active) WHERE is_active = TRUE")
    op.create_index("idx_cronjobs_next_run", "cronjobs", ["next_run"])
    
    op.create_table(
        "runtime_settings",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("value", postgresql.JSONB(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # ==========================================================================
    # RATE LIMITING
    # ==========================================================================
    
    op.create_table(
        "rate_limit_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("identifier", sa.String(64), nullable=False),
        sa.Column("endpoint", sa.String(100), nullable=False),
        sa.Column("request_count", sa.Integer(), default=1),
        sa.Column("window_start", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("last_request_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_rate_limit_identifier", "rate_limit_log", ["identifier", "endpoint"])
    op.create_index("idx_rate_limit_window", "rate_limit_log", ["window_start"])
    
    # ==========================================================================
    # DIPFINDER ENGINE
    # ==========================================================================
    
    op.create_table(
        "price_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("open", sa.Numeric(16, 6)),
        sa.Column("high", sa.Numeric(16, 6)),
        sa.Column("low", sa.Numeric(16, 6)),
        sa.Column("close", sa.Numeric(16, 6), nullable=False),
        sa.Column("adj_close", sa.Numeric(16, 6)),
        sa.Column("volume", sa.BigInteger()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("symbol", "date", name="uq_price_history"),
    )
    op.create_index("idx_price_history_symbol", "price_history", ["symbol"])
    op.execute("CREATE INDEX idx_price_history_date ON price_history(date DESC)")
    op.create_index("idx_price_history_symbol_date", "price_history", ["symbol", "date"])
    
    op.create_table(
        "dipfinder_signals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("benchmark", sa.String(20), nullable=False),
        sa.Column("window_days", sa.Integer(), nullable=False),
        sa.Column("as_of_date", sa.Date(), nullable=False),
        sa.Column("dip_stock", sa.Numeric(8, 6)),
        sa.Column("peak_stock", sa.Numeric(16, 6)),
        sa.Column("dip_pctl", sa.Numeric(5, 2)),
        sa.Column("dip_vs_typical", sa.Numeric(8, 4)),
        sa.Column("persist_days", sa.Integer()),
        sa.Column("dip_mkt", sa.Numeric(8, 6)),
        sa.Column("excess_dip", sa.Numeric(8, 6)),
        sa.Column("dip_class", sa.String(20)),
        sa.Column("quality_score", sa.Numeric(5, 2)),
        sa.Column("stability_score", sa.Numeric(5, 2)),
        sa.Column("dip_score", sa.Numeric(5, 2)),
        sa.Column("final_score", sa.Numeric(5, 2)),
        sa.Column("alert_level", sa.String(20)),
        sa.Column("should_alert", sa.Boolean(), default=False),
        sa.Column("reason", sa.Text()),
        sa.Column("quality_factors", postgresql.JSONB()),
        sa.Column("stability_factors", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("ticker", "benchmark", "window_days", "as_of_date", name="uq_dipfinder_signal"),
    )
    op.create_index("idx_dipfinder_signals_ticker", "dipfinder_signals", ["ticker"])
    op.execute("CREATE INDEX idx_dipfinder_signals_date ON dipfinder_signals(as_of_date DESC)")
    op.create_index("idx_dipfinder_signals_lookup", "dipfinder_signals", ["ticker", "benchmark", "window_days", "as_of_date"])
    op.execute("CREATE INDEX idx_dipfinder_signals_alert ON dipfinder_signals(should_alert) WHERE should_alert = TRUE")
    op.execute("CREATE INDEX idx_dipfinder_signals_final ON dipfinder_signals(final_score DESC)")
    op.create_index("idx_dipfinder_signals_expires", "dipfinder_signals", ["expires_at"])
    
    op.create_table(
        "dipfinder_config",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("min_dip_abs", sa.Numeric(5, 4), default=0.10),
        sa.Column("min_persist_days", sa.Integer(), default=2),
        sa.Column("dip_percentile_threshold", sa.Numeric(5, 4), default=0.80),
        sa.Column("dip_vs_typical_threshold", sa.Numeric(5, 4), default=1.5),
        sa.Column("quality_gate", sa.Numeric(5, 2), default=60),
        sa.Column("stability_gate", sa.Numeric(5, 2), default=60),
        sa.Column("is_enabled", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_dipfinder_config_symbol", "dipfinder_config", ["symbol"])
    op.execute("CREATE INDEX idx_dipfinder_config_enabled ON dipfinder_config(is_enabled) WHERE is_enabled = TRUE")
    
    op.create_table(
        "dipfinder_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(20), nullable=False),
        sa.Column("event_type", sa.String(20), nullable=False),
        sa.Column("window_days", sa.Integer(), nullable=False),
        sa.Column("dip_pct", sa.Numeric(8, 6)),
        sa.Column("final_score", sa.Numeric(5, 2)),
        sa.Column("dip_class", sa.String(20)),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            "event_type IN ('entered_dip', 'exited_dip', 'deepened', 'recovered', 'alert_triggered')",
            name="ck_dipfinder_history_event"
        ),
    )
    op.create_index("idx_dipfinder_history_ticker", "dipfinder_history", ["ticker"])
    op.create_index("idx_dipfinder_history_event", "dipfinder_history", ["event_type"])
    op.execute("CREATE INDEX idx_dipfinder_history_recorded ON dipfinder_history(recorded_at DESC)")
    
    op.create_table(
        "yfinance_info_cache",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("info_data", postgresql.JSONB(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_yfinance_cache_symbol", "yfinance_info_cache", ["symbol"])
    op.create_index("idx_yfinance_cache_expires", "yfinance_info_cache", ["expires_at"])
    
    # ==========================================================================
    # AI AGENT ANALYSIS
    # ==========================================================================
    
    op.create_table(
        "ai_agent_analysis",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("verdicts", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("overall_signal", sa.String(20), nullable=False),
        sa.Column("overall_confidence", sa.Integer(), nullable=False),
        sa.Column("summary", sa.Text()),
        sa.Column("analyzed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("overall_signal IN ('bullish', 'bearish', 'neutral')", name="ck_ai_overall_signal"),
        sa.CheckConstraint("overall_confidence >= 0 AND overall_confidence <= 100", name="ck_ai_confidence_range"),
    )
    op.create_index("idx_ai_agent_analysis_symbol", "ai_agent_analysis", ["symbol"])
    op.create_index("idx_ai_agent_analysis_expires", "ai_agent_analysis", ["expires_at"])
    op.create_index("idx_ai_agent_analysis_signal", "ai_agent_analysis", ["overall_signal"])
    
    # ==========================================================================
    # STOCK FUNDAMENTALS
    # ==========================================================================
    
    op.create_table(
        "stock_fundamentals",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), unique=True, nullable=False),
        sa.Column("pe_ratio", sa.Numeric(12, 4)),
        sa.Column("forward_pe", sa.Numeric(12, 4)),
        sa.Column("peg_ratio", sa.Numeric(12, 4)),
        sa.Column("price_to_book", sa.Numeric(12, 4)),
        sa.Column("price_to_sales", sa.Numeric(12, 4)),
        sa.Column("enterprise_value", sa.BigInteger()),
        sa.Column("ev_to_ebitda", sa.Numeric(12, 4)),
        sa.Column("ev_to_revenue", sa.Numeric(12, 4)),
        sa.Column("profit_margin", sa.Numeric(10, 6)),
        sa.Column("operating_margin", sa.Numeric(10, 6)),
        sa.Column("gross_margin", sa.Numeric(10, 6)),
        sa.Column("ebitda_margin", sa.Numeric(10, 6)),
        sa.Column("return_on_equity", sa.Numeric(10, 6)),
        sa.Column("return_on_assets", sa.Numeric(10, 6)),
        sa.Column("debt_to_equity", sa.Numeric(12, 4)),
        sa.Column("current_ratio", sa.Numeric(10, 4)),
        sa.Column("quick_ratio", sa.Numeric(10, 4)),
        sa.Column("total_cash", sa.BigInteger()),
        sa.Column("total_debt", sa.BigInteger()),
        sa.Column("free_cash_flow", sa.BigInteger()),
        sa.Column("operating_cash_flow", sa.BigInteger()),
        sa.Column("book_value", sa.Numeric(12, 4)),
        sa.Column("eps_trailing", sa.Numeric(12, 4)),
        sa.Column("eps_forward", sa.Numeric(12, 4)),
        sa.Column("revenue_per_share", sa.Numeric(12, 4)),
        sa.Column("revenue_growth", sa.Numeric(10, 6)),
        sa.Column("earnings_growth", sa.Numeric(10, 6)),
        sa.Column("earnings_quarterly_growth", sa.Numeric(10, 6)),
        sa.Column("shares_outstanding", sa.BigInteger()),
        sa.Column("float_shares", sa.BigInteger()),
        sa.Column("held_percent_insiders", sa.Numeric(10, 6)),
        sa.Column("held_percent_institutions", sa.Numeric(10, 6)),
        sa.Column("short_ratio", sa.Numeric(10, 4)),
        sa.Column("short_percent_of_float", sa.Numeric(10, 6)),
        sa.Column("beta", sa.Numeric(10, 4)),
        sa.Column("recommendation", sa.String(20)),
        sa.Column("recommendation_mean", sa.Numeric(5, 2)),
        sa.Column("num_analyst_opinions", sa.Integer()),
        sa.Column("target_high_price", sa.Numeric(12, 4)),
        sa.Column("target_low_price", sa.Numeric(12, 4)),
        sa.Column("target_mean_price", sa.Numeric(12, 4)),
        sa.Column("target_median_price", sa.Numeric(12, 4)),
        sa.Column("revenue", sa.BigInteger()),
        sa.Column("ebitda", sa.BigInteger()),
        sa.Column("net_income", sa.BigInteger()),
        sa.Column("next_earnings_date", sa.DateTime(timezone=True)),
        sa.Column("earnings_estimate_high", sa.Numeric(12, 4)),
        sa.Column("earnings_estimate_low", sa.Numeric(12, 4)),
        sa.Column("earnings_estimate_avg", sa.Numeric(12, 4)),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_stock_fundamentals_symbol", "stock_fundamentals", ["symbol"])
    op.create_index("idx_stock_fundamentals_expires", "stock_fundamentals", ["expires_at"])
    
    # ==========================================================================
    # SEARCH CACHE
    # ==========================================================================
    
    op.create_table(
        "symbol_search_results",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("name", sa.String(255)),
        sa.Column("exchange", sa.String(50)),
        sa.Column("quote_type", sa.String(20)),
        sa.Column("sector", sa.String(100)),
        sa.Column("industry", sa.String(100)),
        sa.Column("market_cap", sa.BigInteger()),
        sa.Column("search_query", sa.String(100), nullable=False),
        sa.Column("relevance_score", sa.Numeric(5, 2)),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("symbol", "search_query", name="uq_search_result"),
    )
    op.create_index("idx_search_results_symbol", "symbol_search_results", ["symbol"])
    op.create_index("idx_search_results_query", "symbol_search_results", ["search_query"])
    op.create_index("idx_search_results_expires", "symbol_search_results", ["expires_at"])
    
    # ==========================================================================
    # SCHEMA MIGRATIONS TRACKING
    # ==========================================================================
    
    op.create_table(
        "schema_migrations",
        sa.Column("version", sa.String(50), primary_key=True),
        sa.Column("applied_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # ==========================================================================
    # TRIGGERS & FUNCTIONS
    # ==========================================================================
    
    # Auto-update updated_at trigger function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Apply triggers to tables with updated_at
    for table in ["auth_user", "symbols", "dip_state", "secure_api_keys", "cronjobs", "dipfinder_config"]:
        op.execute(f"""
            CREATE TRIGGER tr_{table}_updated
                BEFORE UPDATE ON {table}
                FOR EACH ROW EXECUTE FUNCTION update_updated_at();
        """)
    
    # Dip history logging trigger
    op.execute("""
        CREATE OR REPLACE FUNCTION log_dip_change()
        RETURNS TRIGGER AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                INSERT INTO dip_history (symbol, action, current_price, ath_price, dip_percentage)
                VALUES (NEW.symbol, 'added', NEW.current_price, NEW.ath_price, NEW.dip_percentage);
            ELSIF TG_OP = 'DELETE' THEN
                INSERT INTO dip_history (symbol, action, current_price, ath_price, dip_percentage)
                VALUES (OLD.symbol, 'removed', OLD.current_price, OLD.ath_price, OLD.dip_percentage);
            END IF;
            RETURN COALESCE(NEW, OLD);
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
        CREATE TRIGGER tr_dip_state_history
            AFTER INSERT OR DELETE ON dip_state
            FOR EACH ROW EXECUTE FUNCTION log_dip_change();
    """)
    
    # Rate limit cleanup function
    op.execute("""
        CREATE OR REPLACE FUNCTION cleanup_rate_limits() RETURNS void AS $$
        BEGIN
            DELETE FROM rate_limit_log WHERE window_start < NOW() - INTERVAL '1 day';
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # ==========================================================================
    # VIEWS
    # ==========================================================================
    
    op.execute("""
        CREATE OR REPLACE VIEW dip_vote_summary AS
        SELECT 
            symbol,
            COUNT(*) FILTER (WHERE vote_type = 'buy') as buy_votes,
            COUNT(*) FILTER (WHERE vote_type = 'sell') as sell_votes,
            SUM(vote_weight) FILTER (WHERE vote_type = 'buy') as weighted_buy,
            SUM(vote_weight) FILTER (WHERE vote_type = 'sell') as weighted_sell,
            SUM(vote_weight) FILTER (WHERE vote_type = 'buy') - 
                SUM(vote_weight) FILTER (WHERE vote_type = 'sell') as net_score
        FROM dip_votes
        GROUP BY symbol;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW swipe_cards AS
        SELECT 
            ds.symbol,
            ds.current_price,
            ds.ath_price,
            ds.dip_percentage,
            ds.first_seen,
            ds.last_updated,
            ai.swipe_bio,
            ai.ai_rating,
            ai.rating_reasoning,
            ai.generated_at as ai_generated_at,
            COALESCE(v.buy_votes, 0) as buy_votes,
            COALESCE(v.sell_votes, 0) as sell_votes,
            COALESCE(v.weighted_buy, 0) as weighted_buy,
            COALESCE(v.weighted_sell, 0) as weighted_sell,
            COALESCE(v.net_score, 0) as net_score
        FROM dip_state ds
        LEFT JOIN dip_ai_analysis ai ON ds.symbol = ai.symbol
        LEFT JOIN dip_vote_summary v ON ds.symbol = v.symbol;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW daily_api_costs AS
        SELECT 
            DATE(recorded_at) as date,
            service,
            COUNT(*) as request_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(cost_usd) as total_cost_usd,
            COUNT(*) FILTER (WHERE is_batch) as batch_requests
        FROM api_usage
        GROUP BY DATE(recorded_at), service
        ORDER BY date DESC, service;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW dipfinder_latest_signals AS
        SELECT DISTINCT ON (ticker, benchmark, window_days)
            *
        FROM dipfinder_signals
        ORDER BY ticker, benchmark, window_days, as_of_date DESC;
    """)
    
    op.execute("""
        CREATE OR REPLACE VIEW dipfinder_active_alerts AS
        SELECT 
            ds.*,
            s.name as company_name,
            s.sector,
            s.market_cap
        FROM dipfinder_signals ds
        JOIN symbols s ON ds.ticker = s.symbol
        WHERE ds.should_alert = TRUE
          AND ds.as_of_date >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY ds.final_score DESC;
    """)
    
    # ==========================================================================
    # DEFAULT DATA
    # ==========================================================================
    
    # Default cron jobs
    op.execute("""
        INSERT INTO cronjobs (name, cron_expression, is_active, config) VALUES
            ('data_grab', '0 23 * * 1-5', TRUE, '{"description": "Fetch stock data from yfinance Mon-Fri 11pm"}'),
            ('batch_ai_swipe', '0 3 * * 0', TRUE, '{"description": "Generate swipe bios weekly Sunday 3am"}'),
            ('batch_ai_analysis', '0 4 * * 0', TRUE, '{"description": "Generate dip analysis weekly Sunday 4am"}'),
            ('cleanup', '0 0 * * *', TRUE, '{"description": "Clean up expired data daily midnight"}')
        ON CONFLICT (name) DO NOTHING;
    """)
    
    # Default runtime settings
    op.execute("""
        INSERT INTO runtime_settings (key, value) VALUES
            ('signal_threshold_strong_buy', '80.0'),
            ('signal_threshold_buy', '60.0'),
            ('signal_threshold_hold', '40.0'),
            ('ai_enrichment_enabled', 'true'),
            ('ai_batch_size', '0'),
            ('ai_model', '"gpt-5-mini"'),
            ('suggestion_cleanup_days', '30'),
            ('auto_approve_votes', '10'),
            ('benchmarks', '[{"id": "SP500", "symbol": "^GSPC", "name": "S&P 500", "description": "US Large Cap Index"}, {"id": "MSCI_WORLD", "symbol": "URTH", "name": "MSCI World", "description": "Global Developed Markets"}]')
        ON CONFLICT (key) DO NOTHING;
    """)
    
    # Text search indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name_trgm ON symbols USING gin (name gin_trgm_ops)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_name_trgm ON stock_suggestions USING gin (company_name gin_trgm_ops)")


def downgrade() -> None:
    """Drop all tables (WARNING: destructive operation)."""
    # Drop views first
    op.execute("DROP VIEW IF EXISTS dipfinder_active_alerts CASCADE")
    op.execute("DROP VIEW IF EXISTS dipfinder_latest_signals CASCADE")
    op.execute("DROP VIEW IF EXISTS daily_api_costs CASCADE")
    op.execute("DROP VIEW IF EXISTS swipe_cards CASCADE")
    op.execute("DROP VIEW IF EXISTS dip_vote_summary CASCADE")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS cleanup_rate_limits() CASCADE")
    op.execute("DROP FUNCTION IF EXISTS log_dip_change() CASCADE")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at() CASCADE")
    
    # Drop tables in reverse dependency order
    tables = [
        "schema_migrations",
        "symbol_search_results",
        "stock_fundamentals",
        "ai_agent_analysis",
        "yfinance_info_cache",
        "dipfinder_history",
        "dipfinder_config",
        "dipfinder_signals",
        "price_history",
        "rate_limit_log",
        "runtime_settings",
        "cronjobs",
        "batch_jobs",
        "api_usage",
        "dip_ai_analysis",
        "dip_votes",
        "suggestion_votes",
        "stock_suggestions",
        "dip_history",
        "dip_state",
        "symbols",
        "user_api_keys",
        "secure_api_keys",
        "auth_user",
    ]
    
    for table in tables:
        op.drop_table(table)
