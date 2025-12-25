"""Add data versioning, ingest queue, and search logging tables

Revision ID: 4c41a63f2011
Revises: 001_baseline
Create Date: 2025-12-25 12:49:51.888726+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "4c41a63f2011"
down_revision: Union[str, None] = "001_baseline"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema - add new tables for data versioning system."""
    
    # === NEW TABLES FOR DATA VERSIONING ===
    
    # analysis_versions - track which input versions produced each analysis
    op.create_table(
        "analysis_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("analysis_type", sa.String(length=50), nullable=False),
        sa.Column("input_version_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("batch_job_id", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "analysis_type", name="uq_analysis_version_symbol_type"),
    )
    op.create_index(
        "idx_analysis_versions_expires", "analysis_versions", ["expires_at"], unique=False
    )
    op.create_index("idx_analysis_versions_symbol", "analysis_versions", ["symbol"], unique=False)
    op.create_index(
        "idx_analysis_versions_type", "analysis_versions", ["analysis_type"], unique=False
    )
    
    # data_versions - track hash of source data (prices, fundamentals, calendar)
    op.create_table(
        "data_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("version_hash", sa.String(length=64), nullable=False),
        sa.Column("version_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source IN ('prices', 'fundamentals', 'calendar')", name="ck_data_version_source"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol", "source", name="uq_data_version_symbol_source"),
    )
    op.create_index("idx_data_versions_source", "data_versions", ["source"], unique=False)
    op.create_index("idx_data_versions_symbol", "data_versions", ["symbol"], unique=False)
    op.create_index(
        "idx_data_versions_updated",
        "data_versions",
        ["updated_at"],
        unique=False,
        postgresql_ops={"updated_at": "DESC"},
    )
    
    # symbol_ingest_queue - queue new symbols for initial data fetch
    op.create_table(
        "symbol_ingest_queue",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "queued_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')",
            name="ck_ingest_queue_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("symbol"),
    )
    op.create_index(
        "idx_ingest_queue_pending",
        "symbol_ingest_queue",
        ["status", "queued_at"],
        unique=False,
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.create_index("idx_ingest_queue_status", "symbol_ingest_queue", ["status"], unique=False)
    
    # symbol_search_log - log search queries for analytics
    op.create_table(
        "symbol_search_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("query", sa.String(length=100), nullable=False),
        sa.Column("query_normalized", sa.String(length=100), nullable=False),
        sa.Column("result_count", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("user_fingerprint", sa.String(length=64), nullable=True),
        sa.Column(
            "searched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("source IN ('local', 'api', 'mixed')", name="ck_search_log_source"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_search_log_query", "symbol_search_log", ["query_normalized"], unique=False)
    op.create_index(
        "idx_search_log_searched",
        "symbol_search_log",
        ["searched_at"],
        unique=False,
        postgresql_ops={"searched_at": "DESC"},
    )
    
    # === MODIFICATIONS TO EXISTING TABLES ===
    
    # Rename yfinance_info_cache.info_data to .data
    op.alter_column(
        "yfinance_info_cache",
        "info_data",
        new_column_name="data",
    )


def downgrade() -> None:
    """Downgrade database schema."""
    # Reverse yfinance cache column rename
    op.alter_column(
        "yfinance_info_cache",
        "data",
        new_column_name="info_data",
    )
    
    # Drop tables in reverse order
    op.drop_index("idx_search_log_searched", table_name="symbol_search_log")
    op.drop_index("idx_search_log_query", table_name="symbol_search_log")
    op.drop_table("symbol_search_log")
    
    op.drop_index("idx_ingest_queue_status", table_name="symbol_ingest_queue")
    op.drop_index("idx_ingest_queue_pending", table_name="symbol_ingest_queue")
    op.drop_table("symbol_ingest_queue")
    
    op.drop_index("idx_data_versions_updated", table_name="data_versions")
    op.drop_index("idx_data_versions_symbol", table_name="data_versions")
    op.drop_index("idx_data_versions_source", table_name="data_versions")
    op.drop_table("data_versions")
    
    op.drop_index("idx_analysis_versions_type", table_name="analysis_versions")
    op.drop_index("idx_analysis_versions_symbol", table_name="analysis_versions")
    op.drop_index("idx_analysis_versions_expires", table_name="analysis_versions")
    op.drop_table("analysis_versions")
