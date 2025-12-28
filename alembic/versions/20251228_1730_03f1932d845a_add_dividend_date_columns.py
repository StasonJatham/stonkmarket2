"""add_dividend_date_columns

Revision ID: 03f1932d845a
Revises: 02e0820c822b
Create Date: 2025-12-28 17:30:00.000000+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "03f1932d845a"
down_revision: Union[str, None] = "02e0820c822b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add dividend_date and ex_dividend_date columns to stock_fundamentals."""
    op.add_column(
        "stock_fundamentals",
        sa.Column("dividend_date", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "stock_fundamentals",
        sa.Column("ex_dividend_date", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Remove dividend date columns."""
    op.drop_column("stock_fundamentals", "ex_dividend_date")
    op.drop_column("stock_fundamentals", "dividend_date")
