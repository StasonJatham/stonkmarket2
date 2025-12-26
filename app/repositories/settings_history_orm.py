"""Settings change history repository - SQLAlchemy ORM async."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select

from app.core.logging import get_logger
from app.database.connection import get_session
from app.database.orm import SettingsChangeHistory


logger = get_logger("repositories.settings_history")


async def log_change(
    setting_type: str,
    setting_key: str,
    old_value: Any,
    new_value: Any,
    changed_by: int | None = None,
    changed_by_username: str | None = None,
    change_reason: str | None = None,
) -> int:
    """Log a settings change to the history table.
    
    Args:
        setting_type: Type of setting (e.g., 'runtime', 'cronjob', 'api_key')
        setting_key: The specific setting key that changed
        old_value: Previous value (will be stored as JSONB)
        new_value: New value (will be stored as JSONB)
        changed_by: User ID who made the change
        changed_by_username: Username who made the change
        change_reason: Optional reason for the change
        
    Returns:
        The ID of the created history record
    """
    async with get_session() as session:
        record = SettingsChangeHistory(
            setting_type=setting_type,
            setting_key=setting_key,
            old_value=old_value if isinstance(old_value, dict) else {"value": old_value},
            new_value=new_value if isinstance(new_value, dict) else {"value": new_value},
            changed_by=changed_by,
            changed_by_username=changed_by_username,
            change_reason=change_reason,
        )
        session.add(record)
        await session.commit()
        await session.refresh(record)
        logger.info(f"Logged settings change: {setting_type}/{setting_key} by {changed_by_username}")
        return record.id


async def list_changes(
    setting_type: str | None = None,
    setting_key: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """List settings changes with optional filtering.
    
    Args:
        setting_type: Filter by setting type (optional)
        setting_key: Filter by setting key (optional)
        limit: Maximum number of records to return
        offset: Number of records to skip
        
    Returns:
        Tuple of (list of change records, total count)
    """
    async with get_session() as session:
        # Build base query
        query = select(SettingsChangeHistory)
        count_query = select(func.count()).select_from(SettingsChangeHistory)

        # Apply filters
        if setting_type:
            query = query.where(SettingsChangeHistory.setting_type == setting_type)
            count_query = count_query.where(SettingsChangeHistory.setting_type == setting_type)
        if setting_key:
            query = query.where(SettingsChangeHistory.setting_key == setting_key)
            count_query = count_query.where(SettingsChangeHistory.setting_key == setting_key)

        # Get total count
        count_result = await session.execute(count_query)
        total = count_result.scalar() or 0

        # Get records with pagination
        query = query.order_by(SettingsChangeHistory.created_at.desc()).offset(offset).limit(limit)
        result = await session.execute(query)
        rows = result.scalars().all()

        changes = [
            {
                "id": row.id,
                "setting_type": row.setting_type,
                "setting_key": row.setting_key,
                "old_value": row.old_value,
                "new_value": row.new_value,
                "changed_by": row.changed_by,
                "changed_by_username": row.changed_by_username,
                "change_reason": row.change_reason,
                "reverted": row.reverted,
                "reverted_at": row.reverted_at,
                "reverted_by": row.reverted_by,
                "created_at": row.created_at,
            }
            for row in rows
        ]

        return changes, total


async def get_change(change_id: int) -> dict | None:
    """Get a single settings change by ID.
    
    Args:
        change_id: The ID of the change record
        
    Returns:
        Change record dict or None if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(SettingsChangeHistory).where(SettingsChangeHistory.id == change_id)
        )
        row = result.scalar_one_or_none()

        if not row:
            return None

        return {
            "id": row.id,
            "setting_type": row.setting_type,
            "setting_key": row.setting_key,
            "old_value": row.old_value,
            "new_value": row.new_value,
            "changed_by": row.changed_by,
            "changed_by_username": row.changed_by_username,
            "change_reason": row.change_reason,
            "reverted": row.reverted,
            "reverted_at": row.reverted_at,
            "reverted_by": row.reverted_by,
            "created_at": row.created_at,
        }


async def mark_as_reverted(
    change_id: int,
    reverted_by: int | None = None,
) -> bool:
    """Mark a change as reverted.
    
    Args:
        change_id: The ID of the change to mark
        reverted_by: User ID who performed the revert
        
    Returns:
        True if the record was updated, False if not found
    """
    async with get_session() as session:
        result = await session.execute(
            select(SettingsChangeHistory).where(SettingsChangeHistory.id == change_id)
        )
        record = result.scalar_one_or_none()

        if not record:
            return False

        record.reverted = True
        record.reverted_at = datetime.utcnow()
        record.reverted_by = reverted_by
        await session.commit()

        logger.info(f"Marked change {change_id} as reverted by user {reverted_by}")
        return True


async def get_last_change_for_key(
    setting_type: str,
    setting_key: str,
    exclude_reverted: bool = True,
) -> dict | None:
    """Get the most recent change for a specific setting.
    
    Args:
        setting_type: The setting type
        setting_key: The setting key
        exclude_reverted: Whether to exclude already reverted changes
        
    Returns:
        Most recent change record or None
    """
    async with get_session() as session:
        query = (
            select(SettingsChangeHistory)
            .where(SettingsChangeHistory.setting_type == setting_type)
            .where(SettingsChangeHistory.setting_key == setting_key)
        )

        if exclude_reverted:
            query = query.where(SettingsChangeHistory.reverted == False)  # noqa: E712

        query = query.order_by(SettingsChangeHistory.created_at.desc()).limit(1)

        result = await session.execute(query)
        row = result.scalar_one_or_none()

        if not row:
            return None

        return {
            "id": row.id,
            "setting_type": row.setting_type,
            "setting_key": row.setting_key,
            "old_value": row.old_value,
            "new_value": row.new_value,
            "changed_by": row.changed_by,
            "changed_by_username": row.changed_by_username,
            "change_reason": row.change_reason,
            "reverted": row.reverted,
            "reverted_at": row.reverted_at,
            "reverted_by": row.reverted_by,
            "created_at": row.created_at,
        }
