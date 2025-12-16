import os
from dataclasses import dataclass, field
from typing import List


def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    db_path: str = os.getenv("DB_PATH", "dips.sqlite")
    default_symbols: List[str] = field(
        default_factory=lambda: [
            "AAPL",
            "MSFT",
            "GOOG",
            "AMZN",
            "META",
            "TSLA",
            "NVDA",
            "NFLX",
            "AMD",
        ]
    )
    default_min_dip_pct: float = 0.10
    default_min_days: int = 2
    history_days: int = 400
    update_window_days: int = 5
    chart_days: int = 180
    auth_secret: str = os.getenv("AUTH_SECRET", "change-me-secret")
    default_admin_user: str = os.getenv("ADMIN_USER", "admin")
    default_admin_password: str = os.getenv("ADMIN_PASS", "admin")
    domain: str | None = os.getenv("DOMAIN")
    https_enabled: bool = _bool_env("HTTPS", False)
    scheduler_enabled: bool = _bool_env("SCHEDULER_ENABLED", True)


settings = Settings()
