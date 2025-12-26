"""add_portfolio_tables

Revision ID: f2b3c8e1a9d0
Revises: 037035a6ec44
Create Date: 2025-12-27 12:00:00.000000+00:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB


# revision identifiers, used by Alembic.
revision: str = "f2b3c8e1a9d0"
down_revision: Union[str, None] = "037035a6ec44"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add portfolio tables for user holdings and analytics."""
    op.create_table(
        "portfolios",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("auth_user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("base_currency", sa.String(10), nullable=False, server_default="USD"),
        sa.Column("cash_balance", sa.Numeric(18, 4), nullable=True, server_default="0"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.text("TRUE")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_portfolios_user", "portfolios", ["user_id"])
    op.create_index(
        "idx_portfolios_active",
        "portfolios",
        ["is_active"],
        postgresql_where=sa.text("is_active = TRUE"),
    )

    op.create_table(
        "portfolio_holdings",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("portfolio_id", sa.Integer, sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 6), nullable=False),
        sa.Column("avg_cost", sa.Numeric(18, 4)),
        sa.Column("target_weight", sa.Numeric(6, 4)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.UniqueConstraint("portfolio_id", "symbol", name="uq_portfolio_holdings_symbol"),
    )
    op.create_index("idx_portfolio_holdings_portfolio", "portfolio_holdings", ["portfolio_id"])
    op.create_index("idx_portfolio_holdings_symbol", "portfolio_holdings", ["symbol"])

    op.create_table(
        "portfolio_transactions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("portfolio_id", sa.Integer, sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("side", sa.String(20), nullable=False),
        sa.Column("quantity", sa.Numeric(18, 6)),
        sa.Column("price", sa.Numeric(18, 4)),
        sa.Column("fees", sa.Numeric(18, 4)),
        sa.Column("trade_date", sa.Date, nullable=False),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "side IN ('buy', 'sell', 'dividend', 'split', 'deposit', 'withdrawal')",
            name="ck_portfolio_transactions_side",
        ),
    )
    op.create_index("idx_portfolio_transactions_portfolio", "portfolio_transactions", ["portfolio_id"])
    op.create_index("idx_portfolio_transactions_symbol", "portfolio_transactions", ["symbol"])
    op.create_index(
        "idx_portfolio_transactions_date",
        "portfolio_transactions",
        ["trade_date"],
        postgresql_ops={"trade_date": "DESC"},
    )

    op.create_table(
        "portfolio_analytics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("portfolio_id", sa.Integer, sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tool", sa.String(50), nullable=False),
        sa.Column("as_of_date", sa.Date),
        sa.Column("window", sa.String(50)),
        sa.Column("params", JSONB),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="ok"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint("status IN ('ok', 'error', 'partial')", name="ck_portfolio_analytics_status"),
    )
    op.create_index("idx_portfolio_analytics_portfolio", "portfolio_analytics", ["portfolio_id"])
    op.create_index("idx_portfolio_analytics_tool", "portfolio_analytics", ["tool"])
    op.create_index(
        "idx_portfolio_analytics_created",
        "portfolio_analytics",
        ["created_at"],
        postgresql_ops={"created_at": "DESC"},
    )


def downgrade() -> None:
    """Drop portfolio tables."""
    op.drop_index("idx_portfolio_analytics_created", table_name="portfolio_analytics")
    op.drop_index("idx_portfolio_analytics_tool", table_name="portfolio_analytics")
    op.drop_index("idx_portfolio_analytics_portfolio", table_name="portfolio_analytics")
    op.drop_table("portfolio_analytics")

    op.drop_index("idx_portfolio_transactions_date", table_name="portfolio_transactions")
    op.drop_index("idx_portfolio_transactions_symbol", table_name="portfolio_transactions")
    op.drop_index("idx_portfolio_transactions_portfolio", table_name="portfolio_transactions")
    op.drop_table("portfolio_transactions")

    op.drop_index("idx_portfolio_holdings_symbol", table_name="portfolio_holdings")
    op.drop_index("idx_portfolio_holdings_portfolio", table_name="portfolio_holdings")
    op.drop_table("portfolio_holdings")

    op.drop_index("idx_portfolios_active", table_name="portfolios")
    op.drop_index("idx_portfolios_user", table_name="portfolios")
    op.drop_table("portfolios")
