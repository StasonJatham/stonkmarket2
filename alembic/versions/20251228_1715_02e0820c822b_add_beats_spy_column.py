"""add_beats_spy_column

Revision ID: 02e0820c822b
Revises: a182ff9bc71b
Create Date: 2025-12-28 17:15:12.130370+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "02e0820c822b"
down_revision: Union[str, None] = "a182ff9bc71b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add beats_spy column to strategy_signals table."""
    op.add_column(
        "strategy_signals",
        sa.Column("beats_spy", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )


def downgrade() -> None:
    """Remove beats_spy column from strategy_signals table."""
    op.drop_column("strategy_signals", "beats_spy")
