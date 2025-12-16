from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import Cookie, Depends, HTTPException, status

from ..db import init_db
from ..auth import parse_session
from ..config import settings
from ..repositories import auth_user as auth_repo


def get_db() -> Iterator[sqlite3.Connection]:
    conn = init_db()
    try:
        yield conn
    finally:
        conn.close()


def _get_user(session: str | None, conn) -> str:
    parsed = parse_session(session) if session else None
    if not parsed:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    username, _, _ = parsed
    record = auth_repo.get_user(conn, username)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return username


def require_user(session: str | None = Cookie(default=None), conn=Depends(get_db)) -> str:
    return _get_user(session, conn)


def require_admin(session: str | None = Cookie(default=None), conn=Depends(get_db)) -> str:
    username = _get_user(session, conn)
    if username != settings.default_admin_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin required")
    return username
