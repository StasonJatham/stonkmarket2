"""Data access layer repositories.

Each repository module provides async functions for database operations
using raw SQL via asyncpg. New code should consider using SQLAlchemy ORM
models from `app.database.orm` with the `get_session()` context manager.

ORM-based repositories (recommended for new code):
- symbols_orm: SQLAlchemy ORM version of symbols repository

Legacy repositories (raw SQL via asyncpg):
- auth_user, cronjobs, symbols, api_keys, api_usage, etc.
"""

from . import auth_user
from . import cronjobs
from . import symbols
from . import symbols_orm  # ORM version (recommended for new code)
from . import api_keys
from . import api_usage
from . import dip_history
from . import dip_votes
from . import user_api_keys

__all__ = [
    "auth_user",
    "cronjobs",
    "symbols",
    "symbols_orm",
    "api_keys",
    "api_usage",
    "dip_history",
    "dip_votes",
    "user_api_keys",
]
