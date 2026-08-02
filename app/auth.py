"""Small local user store and JWT authentication for the workbench."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import get_settings
from app.request_context import RequestIdentity


Role = Literal["analyst", "reviewer", "admin"]
_bearer = HTTPBearer(auto_error=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return base64.urlsafe_b64encode(salt + digest).decode("ascii")


def _verify_password(password: str, encoded: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        salt, expected = raw[:16], raw[16:]
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=256)


class RegisterRequest(LoginRequest):
    tenant_id: str = Field(default="default", min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")


class UserStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS app_users (
                    user_id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                    tenant_id TEXT NOT NULL, role TEXT NOT NULL,
                    password_hash TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def create(self, username: str, password: str, tenant_id: str, role: Role = "analyst") -> RequestIdentity:
        identity = RequestIdentity("usr-" + uuid.uuid4().hex[:16], tenant_id, role)
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO app_users VALUES (?, ?, ?, ?, ?, 1, ?)",
                    (identity.user_id, username, tenant_id, role, _hash_password(password), _now()),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username already exists") from exc
        return identity

    def authenticate(self, username: str, password: str) -> RequestIdentity | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM app_users WHERE username = ? AND is_active = 1", (username,)).fetchone()
        if row is None or not _verify_password(password, str(row["password_hash"])):
            return None
        return RequestIdentity(str(row["user_id"]), str(row["tenant_id"]), str(row["role"]))

    def ensure_demo_users(self) -> None:
        settings = get_settings()
        if not settings.AUTH_BOOTSTRAP_DEMO_USERS:
            return
        for username, role in (("analyst_a", "analyst"), ("analyst_b", "analyst"), ("reviewer", "reviewer"), ("admin", "admin")):
            try:
                self.create(username, settings.AUTH_DEMO_PASSWORD, "demo", role)  # type: ignore[arg-type]
            except ValueError:
                pass


def get_user_store() -> UserStore:
    store = UserStore(get_settings().AGENT_AUTH_DB_PATH)
    store.ensure_demo_users()
    return store


def issue_token(identity: RequestIdentity) -> str:
    settings = get_settings()
    expires = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRE_MINUTES)
    return jwt.encode({"sub": identity.user_id, "tenant": identity.tenant_id, "role": identity.role, "exp": expires}, settings.JWT_SECRET, algorithm="HS256")


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> RequestIdentity:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    try:
        payload = jwt.decode(credentials.credentials, get_settings().JWT_SECRET, algorithms=["HS256"])
        identity = RequestIdentity(str(payload["sub"]), str(payload["tenant"]), str(payload["role"]))
    except (jwt.InvalidTokenError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc
    if identity.role not in {"analyst", "reviewer", "admin"}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unknown role")
    return identity


def require_roles(*roles: Role):
    def dependency(identity: RequestIdentity = Depends(current_user)) -> RequestIdentity:
        if identity.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return identity
    return dependency
