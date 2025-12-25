"""expand_summary_ai_column

Expand summary_ai column from VARCHAR(350) to VARCHAR(500) to accommodate
longer AI-generated summaries. The prompt targets 300-400 chars but we need
buffer for edge cases.

Revision ID: 9a1752a5fd0d
Revises: 6c33357bd596
Create Date: 2025-12-25 13:24:55.644134+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a1752a5fd0d"
down_revision: Union[str, None] = "6c33357bd596"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Expand summary_ai column from VARCHAR(350) to VARCHAR(500).
    
    This allows AI summaries to be slightly longer while maintaining
    a reasonable UI display length.
    """
    op.alter_column(
        "symbols",
        "summary_ai",
        existing_type=sa.String(350),
        type_=sa.String(500),
        existing_nullable=True,
    )


def downgrade() -> None:
    """
    Shrink summary_ai back to VARCHAR(350).
    
    Note: Existing summaries longer than 350 chars will be truncated!
    """
    # First truncate any existing long summaries
    op.execute(
        "UPDATE symbols SET summary_ai = LEFT(summary_ai, 350) WHERE LENGTH(summary_ai) > 350"
    )
    op.alter_column(
        "symbols",
        "summary_ai",
        existing_type=sa.String(500),
        type_=sa.String(350),
        existing_nullable=True,
    )
