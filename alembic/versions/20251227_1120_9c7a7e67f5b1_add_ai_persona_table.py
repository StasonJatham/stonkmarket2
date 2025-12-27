"""add_ai_persona_table

Revision ID: 9c7a7e67f5b1
Revises: d40200938afb
Create Date: 2025-12-27 11:20:42.055120+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9c7a7e67f5b1"
down_revision: Union[str, None] = "d40200938afb"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "ai_persona",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=50), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("philosophy", sa.Text(), nullable=True),
        sa.Column("avatar_data", sa.LargeBinary(), nullable=True),
        sa.Column("avatar_mime_type", sa.String(length=50), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("display_order", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "created_at",
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
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ai_persona")),
        sa.UniqueConstraint("key", name=op.f("uq_ai_persona_key")),
    )
    op.create_index(
        "idx_ai_persona_active", "ai_persona", ["is_active", "display_order"], unique=False
    )
    op.create_index("idx_ai_persona_key", "ai_persona", ["key"], unique=False)
    
    # Seed default personas
    op.execute("""
        INSERT INTO ai_persona (key, name, description, philosophy, display_order) VALUES
        ('warren_buffett', 'Warren Buffett', 'The Oracle of Omaha - legendary value investor and CEO of Berkshire Hathaway', 'Value investing focused on companies with strong moats, consistent earnings, and fair valuations', 1),
        ('peter_lynch', 'Peter Lynch', 'Former manager of Fidelity Magellan Fund, achieved 29% average annual returns', 'Growth at a reasonable price (GARP), invest in what you know, look for tenbaggers', 2),
        ('cathie_wood', 'Cathie Wood', 'Founder of ARK Invest, focused on disruptive innovation', 'Innovation investing - 5-year time horizon, exponential growth potential, disruptive technologies', 3),
        ('michael_burry', 'Michael Burry', 'Founder of Scion Capital, famously shorted the housing market in 2008', 'Deep value and contrarian investing, extensive fundamental research, willingness to go against the crowd', 4)
    """)


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("idx_ai_persona_key", table_name="ai_persona")
    op.drop_index("idx_ai_persona_active", table_name="ai_persona")
    op.drop_table("ai_persona")
