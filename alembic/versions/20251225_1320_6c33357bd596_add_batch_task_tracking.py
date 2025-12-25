"""add_batch_task_tracking

Adds task_custom_ids JSONB column to batch_jobs for result mapping,
and batch_task_errors table for failed task retry.

Revision ID: 6c33357bd596
Revises: efd4cdb041fb
Create Date: 2025-12-25 13:20:54.389777+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "6c33357bd596"
down_revision: Union[str, None] = "efd4cdb041fb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade database schema.
    
    Changes:
    1. Add task_custom_ids JSONB column to batch_jobs for mapping custom_id → result
    2. Create batch_task_errors table to store failed tasks for retry
    """
    # Add task_custom_ids to batch_jobs for custom_id → symbol/agent mapping
    op.add_column(
        "batch_jobs",
        sa.Column(
            "task_custom_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            comment="Map of custom_id → task metadata for result mapping",
        ),
    )
    
    # Create batch_task_errors table for failed task retry
    op.create_table(
        "batch_task_errors",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("batch_id", sa.String(100), nullable=False),
        sa.Column("custom_id", sa.String(200), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("task_type", sa.String(50), nullable=False),  # e.g., 'agent_analysis', 'rating', 'bio'
        sa.Column("agent_id", sa.String(50), nullable=True),  # For agent batch tasks
        sa.Column("error_type", sa.String(50), nullable=False),  # e.g., 'api_error', 'validation_error', 'timeout'
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("original_request", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("retry_count", sa.Integer(), default=0, nullable=False),
        sa.Column("max_retries", sa.Integer(), default=3, nullable=False),
        sa.Column("status", sa.String(20), default="pending", nullable=False),  # pending, retrying, resolved, abandoned
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("last_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "status IN ('pending', 'retrying', 'resolved', 'abandoned')",
            name="ck_batch_task_errors_status",
        ),
    )
    op.create_index("idx_batch_task_errors_batch", "batch_task_errors", ["batch_id"])
    op.create_index("idx_batch_task_errors_symbol", "batch_task_errors", ["symbol"])
    op.create_index("idx_batch_task_errors_status", "batch_task_errors", ["status"])
    op.create_index(
        "idx_batch_task_errors_pending",
        "batch_task_errors",
        ["status", "created_at"],
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("idx_batch_task_errors_pending", table_name="batch_task_errors")
    op.drop_index("idx_batch_task_errors_status", table_name="batch_task_errors")
    op.drop_index("idx_batch_task_errors_symbol", table_name="batch_task_errors")
    op.drop_index("idx_batch_task_errors_batch", table_name="batch_task_errors")
    op.drop_table("batch_task_errors")
    op.drop_column("batch_jobs", "task_custom_ids")
