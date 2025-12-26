"""add_portfolio_analytics_jobs

Revision ID: 9b6f1d2c7a1e
Revises: f2b3c8e1a9d0
Create Date: 2025-12-28 09:00:00.000000+00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "9b6f1d2c7a1e"
down_revision: Union[str, None] = "f2b3c8e1a9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add portfolio analytics job queue."""
    op.create_table(
        "portfolio_analytics_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("job_id", sa.String(64), nullable=False, unique=True),
        sa.Column("portfolio_id", sa.Integer, sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("auth_user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("tools", JSONB, nullable=False),
        sa.Column("params", JSONB),
        sa.Column("window", sa.String(50)),
        sa.Column("start_date", sa.Date),
        sa.Column("end_date", sa.Date),
        sa.Column("benchmark", sa.String(20)),
        sa.Column("force_refresh", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
        sa.Column("results_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_portfolio_analytics_jobs_status",
        ),
    )
    op.create_index(
        "idx_portfolio_analytics_jobs_portfolio",
        "portfolio_analytics_jobs",
        ["portfolio_id"],
    )
    op.create_index(
        "idx_portfolio_analytics_jobs_user",
        "portfolio_analytics_jobs",
        ["user_id"],
    )
    op.create_index(
        "idx_portfolio_analytics_jobs_status",
        "portfolio_analytics_jobs",
        ["status"],
    )
    op.create_index(
        "idx_portfolio_analytics_jobs_created",
        "portfolio_analytics_jobs",
        ["created_at"],
        postgresql_ops={"created_at": "DESC"},
    )


def downgrade() -> None:
    """Drop portfolio analytics job queue."""
    op.drop_index("idx_portfolio_analytics_jobs_created", table_name="portfolio_analytics_jobs")
    op.drop_index("idx_portfolio_analytics_jobs_status", table_name="portfolio_analytics_jobs")
    op.drop_index("idx_portfolio_analytics_jobs_user", table_name="portfolio_analytics_jobs")
    op.drop_index("idx_portfolio_analytics_jobs_portfolio", table_name="portfolio_analytics_jobs")
    op.drop_table("portfolio_analytics_jobs")
