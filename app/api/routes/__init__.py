"""API routes package."""

from . import auth
from . import symbols
from . import dips
from . import cronjobs
from . import health
from . import suggestions
from . import mfa
from . import api_keys
from . import swipe
from . import dip_changes
from . import user_api_keys
from . import dipfinder
from . import admin_settings
from . import metrics
from . import seo

__all__ = [
    "auth",
    "symbols",
    "dips",
    "cronjobs",
    "health",
    "suggestions",
    "mfa",
    "api_keys",
    "swipe",
    "dip_changes",
    "user_api_keys",
    "dipfinder",
    "admin_settings",
    "metrics",
    "seo",
]
