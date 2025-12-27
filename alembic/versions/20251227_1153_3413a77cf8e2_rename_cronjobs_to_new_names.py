"""rename_cronjobs_to_new_names

Revision ID: 3413a77cf8e2
Revises: ae947f07d81b
Create Date: 2025-12-27 11:53:23.556622+00:00

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3413a77cf8e2"
down_revision: Union[str, None] = "ae947f07d81b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Old name -> New name mapping
JOB_RENAMES = {
    "initial_data_ingest": "symbol_ingest",
    "process_new_symbols_batch": None,  # Delete - merged into symbol_ingest
    "data_grab": "prices_daily",
    "batch_ai_swipe": "ai_bios_weekly",
    "batch_ai_analysis": "ai_ratings_weekly",
    "batch_poll": "ai_batch_poll",
    "fundamentals_refresh": "fundamentals_monthly",
    "ai_agents_analysis": "ai_personas_weekly",
    "ai_agents_batch_submit": None,  # Delete - merged into ai_personas_weekly
    "ai_agents_batch_collect": None,  # Delete - merged into ai_batch_poll
    "cleanup": "cleanup_daily",
    "portfolio_analytics_worker": "portfolio_worker",
    "signal_scanner_daily": "signals_daily",
    "market_regime_daily": "regime_daily",
    "quant_engine_monthly": "quant_monthly",
}

# New schedules for renamed jobs
NEW_SCHEDULES = {
    "symbol_ingest": "*/15 * * * *",  # Every 15 min
    "prices_daily": "0 23 * * 1-5",  # Mon-Fri 11pm
    "ai_bios_weekly": "0 4 * * 0",  # Sunday 4am
    "ai_ratings_weekly": "0 5 * * 0",  # Sunday 5am
    "ai_batch_poll": "*/5 * * * *",  # Every 5 min
    "fundamentals_monthly": "0 2 1 * *",  # 1st of month 2am
    "ai_personas_weekly": "0 3 * * 0",  # Sunday 3am
    "cleanup_daily": "0 0 * * *",  # Midnight
    "portfolio_worker": "*/5 * * * *",  # Every 5 min
    "signals_daily": "0 22 * * 1-5",  # Mon-Fri 10pm
    "regime_daily": "30 22 * * 1-5",  # Mon-Fri 10:30pm
    "quant_monthly": "0 3 1 * *",  # 1st of month 3am
    "cache_warmup": "*/30 * * * *",  # Every 30 min
}


def upgrade() -> None:
    """Rename cronjobs and delete deprecated ones."""
    conn = op.get_bind()
    
    # Delete jobs that are being removed
    for old_name, new_name in JOB_RENAMES.items():
        if new_name is None:
            conn.execute(
                sa.text("DELETE FROM cronjobs WHERE name = :old_name"),
                {"old_name": old_name}
            )
    
    # Rename jobs
    for old_name, new_name in JOB_RENAMES.items():
        if new_name is not None:
            # Check if old job exists
            result = conn.execute(
                sa.text("SELECT id FROM cronjobs WHERE name = :old_name"),
                {"old_name": old_name}
            ).fetchone()
            
            if result:
                # Rename it
                new_cron = NEW_SCHEDULES.get(new_name, "0 * * * *")
                conn.execute(
                    sa.text("""
                        UPDATE cronjobs 
                        SET name = :new_name, cron = :new_cron
                        WHERE name = :old_name
                    """),
                    {"old_name": old_name, "new_name": new_name, "new_cron": new_cron}
                )


def downgrade() -> None:
    """Revert job renames (reverse the mapping)."""
    conn = op.get_bind()
    
    for old_name, new_name in JOB_RENAMES.items():
        if new_name is not None:
            result = conn.execute(
                sa.text("SELECT id FROM cronjobs WHERE name = :new_name"),
                {"new_name": new_name}
            ).fetchone()
            
            if result:
                conn.execute(
                    sa.text("UPDATE cronjobs SET name = :old_name WHERE name = :new_name"),
                    {"old_name": old_name, "new_name": new_name}
                )
