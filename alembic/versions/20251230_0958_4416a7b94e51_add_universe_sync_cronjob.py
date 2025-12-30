"""add_universe_sync_cronjob

Revision ID: 4416a7b94e51
Revises: e243f6a12158
Create Date: 2025-12-30 09:58:14.718399+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4416a7b94e51"
down_revision: Union[str, None] = "e243f6a12158"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add universe_sync cronjob."""
    conn = op.get_bind()
    
    # Check if job already exists
    result = conn.execute(
        sa.text("SELECT id FROM cronjobs WHERE name = 'universe_sync'")
    ).fetchone()
    
    if not result:
        # Insert universe_sync job - runs weekly on Sunday at 1:30 AM UTC (before weekly_ai_pipeline at 2 AM)
        conn.execute(
            sa.text("""
                INSERT INTO cronjobs (name, cron, description, is_active, run_count, error_count)
                VALUES (
                    'universe_sync',
                    '30 1 * * 0',
                    'Sync FinanceDatabase (~130K financial instruments: equities, ETFs, funds, indices, crypto, currencies)',
                    TRUE,
                    0,
                    0
                )
            """)
        )


def downgrade() -> None:
    """Remove universe_sync cronjob."""
    conn = op.get_bind()
    conn.execute(
        sa.text("DELETE FROM cronjobs WHERE name = 'universe_sync'")
    )
