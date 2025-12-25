"""search_improvements_trigram_cursor_pagination

Adds trigram similarity search on symbol_search_results.name,
confidence_score and last_seen_at columns for search result ranking,
and improved indexes for cursor-based pagination.

Revision ID: efd4cdb041fb
Revises: 4c41a63f2011
Create Date: 2025-12-25 13:10:39.628179+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "efd4cdb041fb"
down_revision: Union[str, None] = "4c41a63f2011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Upgrade database schema.
    
    Changes:
    1. Enable pg_trgm extension for fuzzy text search
    2. Add confidence_score (0.0-1.0) to symbol_search_results
    3. Add last_seen_at to track when result was last searched
    4. Add trigram GIN index on name for fuzzy search
    5. Add composite index for cursor-based pagination (confidence, id)
    """
    # Enable pg_trgm extension for trigram similarity
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    
    # Add confidence_score column (0.0 to 1.0, computed from relevance + recency)
    op.add_column(
        "symbol_search_results",
        sa.Column(
            "confidence_score",
            sa.Numeric(4, 3),
            nullable=True,
            comment="Combined score (0-1) from relevance, recency, and data quality",
        ),
    )
    
    # Add last_seen_at column for tracking when result was last searched
    op.add_column(
        "symbol_search_results",
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=True,
            comment="Last time this result was returned in a search",
        ),
    )
    
    # Create trigram GIN index on name for fuzzy search
    # This enables fast ILIKE and similarity() searches
    op.execute(
        """
        CREATE INDEX idx_search_results_name_trgm 
        ON symbol_search_results 
        USING gin (name gin_trgm_ops)
        """
    )
    
    # Create composite index for cursor-based pagination
    # Sorted by confidence DESC, id ASC for stable cursor ordering
    op.create_index(
        "idx_search_results_cursor",
        "symbol_search_results",
        [sa.text("confidence_score DESC NULLS LAST"), "id"],
        unique=False,
    )
    
    # Backfill confidence_score based on relevance_score
    # Normalize relevance_score (0-100) to confidence_score (0-1)
    op.execute(
        """
        UPDATE symbol_search_results 
        SET confidence_score = COALESCE(relevance_score / 100.0, 0.5),
            last_seen_at = fetched_at
        WHERE confidence_score IS NULL
        """
    )


def downgrade() -> None:
    """Downgrade database schema."""
    # Drop indexes first
    op.drop_index("idx_search_results_cursor", table_name="symbol_search_results")
    op.execute("DROP INDEX IF EXISTS idx_search_results_name_trgm")
    
    # Drop columns
    op.drop_column("symbol_search_results", "last_seen_at")
    op.drop_column("symbol_search_results", "confidence_score")
    
    # Note: pg_trgm extension is left in place as other things may depend on it
