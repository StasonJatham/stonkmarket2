"""add_missing_indexes

Revision ID: c14161221ecb
Revises: 9a1752a5fd0d
Create Date: 2025-12-25 13:41:04.076398+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c14161221ecb"
down_revision: Union[str, None] = "9a1752a5fd0d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add indexes for common query patterns identified in architecture review."""
    # B-tree index for dip_votes.fingerprint (frequent lookups by fingerprint)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dip_votes_fingerprint "
        "ON dip_votes (fingerprint)"
    )

    # B-tree DESC index for api_usage.recorded_at (ORDER BY recorded_at DESC queries)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_api_usage_recorded_at_desc "
        "ON api_usage (recorded_at DESC)"
    )

    # GIN index for batch_jobs.metadata JSONB column (containment queries)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_batch_jobs_metadata_gin "
        "ON batch_jobs USING GIN (metadata)"
    )

    # GIN index for dipfinder_signals.quality_factors JSONB column
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_dipfinder_signals_quality_gin "
        "ON dipfinder_signals USING GIN (quality_factors)"
    )


def downgrade() -> None:
    """Remove indexes added in upgrade."""
    op.execute("DROP INDEX IF EXISTS idx_dipfinder_signals_quality_gin")
    op.execute("DROP INDEX IF EXISTS idx_batch_jobs_metadata_gin")
    op.drop_index("idx_api_usage_recorded_at_desc", table_name="api_usage")
    op.drop_index("idx_dip_votes_fingerprint", table_name="dip_votes")
