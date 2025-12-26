"""Data access layer repositories.

Each repository module provides async functions for database operations.
All code uses SQLAlchemy ORM models from `app.database.orm` with the
`get_session()` context manager.

ORM-based repositories:
- api_keys_orm: SQLAlchemy ORM for secure API keys
- api_usage_orm: SQLAlchemy ORM for API usage tracking
- auth_user_orm: SQLAlchemy ORM for auth users with MFA
- cronjobs_orm: SQLAlchemy ORM for cron jobs
- dip_history_orm: SQLAlchemy ORM for dip change history
- dip_votes_orm: SQLAlchemy ORM for dip voting
- portfolios_orm: SQLAlchemy ORM for portfolios
- symbols_orm: SQLAlchemy ORM for symbols
- user_api_keys_orm: SQLAlchemy ORM for user API keys
"""

from . import auth_user_orm
from . import cronjobs_orm
from . import symbols_orm
from . import dip_votes_orm
from . import api_keys_orm
from . import api_usage_orm
from . import dip_history_orm
from . import user_api_keys_orm
from . import portfolios_orm
from . import portfolio_analytics_jobs

__all__ = [
    "auth_user_orm",
    "cronjobs_orm",
    "symbols_orm",
    "dip_votes_orm",
    "api_keys_orm",
    "api_usage_orm",
    "dip_history_orm",
    "user_api_keys_orm",
    "portfolios_orm",
    "portfolio_analytics_jobs",
]
