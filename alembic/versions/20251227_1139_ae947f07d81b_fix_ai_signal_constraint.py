"""fix_ai_signal_constraint

Revision ID: ae947f07d81b
Revises: 9c7a7e67f5b1
Create Date: 2025-12-27 11:39:29.558016+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ae947f07d81b"
down_revision: Union[str, None] = "9c7a7e67f5b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    # Drop the old constraint that only allows bullish/bearish/neutral
    # Using raw SQL because the constraint name in DB differs from what alembic expects
    op.execute("ALTER TABLE ai_agent_analysis DROP CONSTRAINT IF EXISTS ck_ai_agent_analysis_ck_ai_overall_signal")
    
    # Create new constraint with proper signal values
    op.execute("""
        ALTER TABLE ai_agent_analysis 
        ADD CONSTRAINT ck_ai_agent_analysis_ck_ai_overall_signal 
        CHECK (overall_signal IN ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell'))
    """)


def downgrade() -> None:
    """Downgrade database schema."""
    op.execute("ALTER TABLE ai_agent_analysis DROP CONSTRAINT IF EXISTS ck_ai_agent_analysis_ck_ai_overall_signal")
    op.execute("""
        ALTER TABLE ai_agent_analysis 
        ADD CONSTRAINT ck_ai_agent_analysis_ck_ai_overall_signal 
        CHECK (overall_signal IN ('bullish', 'bearish', 'neutral'))
    """)
