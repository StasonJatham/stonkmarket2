from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel

from ... import auth
from ...api.deps import get_db, require_user
from ...config import settings
from ...repositories import auth_user as auth_repo

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginPayload(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    username: str
    is_admin: bool


class MeResponse(BaseModel):
    username: str
    is_admin: bool


class ChangeCredsPayload(BaseModel):
    current_password: str
    new_username: str
    new_password: str


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginPayload, response: Response, conn=Depends(get_db)):
    record = auth_repo.get_user(conn, payload.username)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    username, password_hash = record
    if not auth.verify_password(payload.password, password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    is_admin = username == settings.default_admin_user
    token = auth.create_session(username)
    response.set_cookie(
        "session",
        token,
        httponly=True,
        samesite="lax",
        secure=settings.https_enabled,
        domain=settings.domain,
    )
    return LoginResponse(username=username, is_admin=is_admin)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response):
    response.delete_cookie("session", domain=settings.domain)
    return None


@router.get("/me", response_model=MeResponse)
def me(user=Depends(require_user)):
    return MeResponse(username=user, is_admin=user == settings.default_admin_user)


@router.post("/change", response_model=MeResponse)
def change_creds(payload: ChangeCredsPayload, response: Response, user=Depends(require_user), conn=Depends(get_db)):
    record = auth_repo.get_user(conn, user)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session invalid")
    _, password_hash = record
    if not auth.verify_password(payload.current_password, password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    new_hash = auth.hash_password(payload.new_password)
    auth_repo.upsert_user(conn, payload.new_username, new_hash)
    if payload.new_username != user:
        conn.execute("DELETE FROM auth_user WHERE username = ?", (user,))
        conn.commit()
    token = auth.create_session(payload.new_username)
    response.set_cookie(
        "session",
        token,
        httponly=True,
        samesite="lax",
        secure=settings.https_enabled,
        domain=settings.domain,
    )
    return MeResponse(username=payload.new_username, is_admin=payload.new_username == settings.default_admin_user)
