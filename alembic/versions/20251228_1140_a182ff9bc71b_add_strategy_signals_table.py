"""add_strategy_signals_table

Revision ID: a182ff9bc71b
Revises: 19d3bbf89488
Create Date: 2025-12-28 11:40:19.845649+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "a182ff9bc71b"
down_revision: Union[str, None] = "19d3bbf89488"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "strategy_signals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("strategy_name", sa.String(length=50), nullable=False),
        sa.Column("strategy_params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("signal_type", sa.String(length=20), nullable=False),
        sa.Column("signal_reason", sa.Text(), nullable=False),
        sa.Column("has_active_signal", sa.Boolean(), nullable=False),
        sa.Column("total_return_pct", sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column("sharpe_ratio", sa.Numeric(precision=8, scale=4), nullable=True),
        sa.Column("win_rate", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("max_drawdown_pct", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("n_trades", sa.Integer(), nullable=False),
        sa.Column("recency_weighted_return", sa.Numeric(precision=12, scale=4), nullable=True),
        sa.Column("current_year_return_pct", sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column("current_year_win_rate", sa.Numeric(precision=8, scale=2), nullable=True),
        sa.Column("current_year_trades", sa.Integer(), nullable=False),
        sa.Column("vs_buy_hold_pct", sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column("vs_spy_pct", sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column("beats_buy_hold", sa.Boolean(), nullable=False),
        sa.Column("fundamentals_healthy", sa.Boolean(), nullable=False),
        sa.Column("fundamental_concerns", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("is_statistically_valid", sa.Boolean(), nullable=False),
        sa.Column("recent_trades", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("indicators_used", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "optimized_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_strategy_signals")),
        sa.UniqueConstraint("symbol", name=op.f("uq_strategy_signals_symbol")),
    )
    op.create_index(
        "idx_strategy_signals_active",
        "strategy_signals",
        ["has_active_signal"],
        unique=False,
        postgresql_where=sa.text("has_active_signal = TRUE"),
    )
    op.create_index(
        "idx_strategy_signals_beats",
        "strategy_signals",
        ["beats_buy_hold"],
        unique=False,
        postgresql_where=sa.text("beats_buy_hold = TRUE"),
    )
    op.create_index(
        "idx_strategy_signals_signal", "strategy_signals", ["signal_type"], unique=False
    )
    op.create_index("idx_strategy_signals_symbol", "strategy_signals", ["symbol"], unique=False)


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("idx_strategy_signals_symbol", table_name="strategy_signals")
    op.drop_index("idx_strategy_signals_signal", table_name="strategy_signals")
    op.drop_index(
        "idx_strategy_signals_beats",
        table_name="strategy_signals",
        postgresql_where=sa.text("beats_buy_hold = TRUE"),
    )
    op.drop_index(
        "idx_strategy_signals_active",
        table_name="strategy_signals",
        postgresql_where=sa.text("has_active_signal = TRUE"),
    )
    op.drop_table("strategy_signals")
