from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Optional, Tuple

from .config import settings


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(password: str, password_hash: str) -> bool:
    return hmac.compare_digest(hash_password(password), password_hash)


def _sign(payload: str) -> str:
    secret = settings.auth_secret.encode("utf-8")
    sig = hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}|{sig}"


def create_session(username: str, ttl_seconds: int = 60 * 60 * 24 * 7) -> str:
    issued = int(time.time())
    expires = issued + ttl_seconds
    raw = f"{username}|{issued}|{expires}"
    return base64.urlsafe_b64encode(_sign(raw).encode("utf-8")).decode("ascii")


def parse_session(token: str) -> Optional[Tuple[str, int, int]]:
    try:
        decoded = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        parts = decoded.split("|")
        if len(parts) != 4:
            return None
        username, issued_str, expires_str, sig = parts
        check = hmac.new(settings.auth_secret.encode("utf-8"), f"{username}|{issued_str}|{expires_str}".encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(check, sig):
            return None
        issued = int(issued_str)
        expires = int(expires_str)
        if expires < int(time.time()):
            return None
        return username, issued, expires
    except Exception:
        return None