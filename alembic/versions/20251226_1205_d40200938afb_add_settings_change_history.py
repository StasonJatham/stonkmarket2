"""add_settings_change_history

Revision ID: d40200938afb
Revises: de8f24192475
Create Date: 2025-12-26 12:05:29.962825+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "d40200938afb"
down_revision: Union[str, None] = "de8f24192475"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "settings_change_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("setting_type", sa.String(length=50), nullable=False),
        sa.Column("setting_key", sa.String(length=100), nullable=False),
        sa.Column("old_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("new_value", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("changed_by", sa.Integer(), nullable=True),
        sa.Column("changed_by_username", sa.String(length=100), nullable=True),
        sa.Column("change_reason", sa.Text(), nullable=True),
        sa.Column("reverted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("reverted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reverted_by", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["changed_by"],
            ["auth_user.id"],
            name=op.f("fk_settings_change_history_changed_by_auth_user"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_settings_change_history")),
    )
    op.create_index(
        "idx_settings_history_created",
        "settings_change_history",
        [sa.text("created_at DESC")],
        unique=False,
    )
    op.create_index(
        "idx_settings_history_key", "settings_change_history", ["setting_key"], unique=False
    )
    op.create_index(
        "idx_settings_history_type", "settings_change_history", ["setting_type"], unique=False
    )


def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_index("idx_settings_history_type", table_name="settings_change_history")
    op.drop_index("idx_settings_history_key", table_name="settings_change_history")
    op.drop_index("idx_settings_history_created", table_name="settings_change_history")
    op.drop_table("settings_change_history")
