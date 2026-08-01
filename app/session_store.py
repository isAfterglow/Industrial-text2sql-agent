"""Durable structured conversation state with Redis-first, SQLite fallback."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionMemoryStore:
    """A small state store; Redis is an optimization, never a dependency."""

    def __init__(self) -> None:
        settings = get_settings()
        self.mode = settings.SESSION_STORE_MODE.lower()
        self.ttl_seconds = max(60, settings.SESSION_MEMORY_TTL_SECONDS)
        self.db_path = Path(settings.SESSION_MEMORY_DB_PATH)
        self.redis_client: Any | None = None
        if self.mode in {"auto", "redis"} and settings.REDIS_URL:
            try:
                import redis  # type: ignore[import-not-found]

                # Conda's Redis server is 5.x in this environment; RESP2 keeps
                # the client compatible with both Redis 5 and newer servers.
                client = redis.Redis.from_url(
                    settings.REDIS_URL, decode_responses=True, protocol=2
                )
                client.ping()
                self.redis_client = client
            except Exception:
                if self.mode == "redis":
                    raise RuntimeError("SESSION_STORE_MODE=redis but Redis is unavailable")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def backend(self) -> str:
        return "redis" if self.redis_client is not None else "sqlite"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS session_memories (
                    session_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    memory_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, profile)
                )"""
            )

    def _key(self, session_id: str, profile: str) -> str:
        return f"text2sql:session:{profile}:{session_id}"

    def load(self, session_id: str, profile: str) -> dict[str, Any] | None:
        if not session_id:
            return None
        if self.redis_client is not None:
            raw = self.redis_client.get(self._key(session_id, profile))
            return json.loads(raw) if raw else None
        now = _now().isoformat()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT memory_json FROM session_memories WHERE session_id = ? AND profile = ? AND expires_at > ?",
                (session_id, profile, now),
            ).fetchone()
            connection.execute("DELETE FROM session_memories WHERE expires_at <= ?", (now,))
        return json.loads(row["memory_json"]) if row else None

    def save(self, memory: dict[str, Any], profile: str) -> dict[str, Any]:
        session_id = str(memory.get("session_id", ""))
        if not session_id:
            return {"saved": False, "reason": "missing_session_id", "backend": self.backend}
        payload = json.dumps(memory, ensure_ascii=False, sort_keys=True)
        if self.redis_client is not None:
            self.redis_client.setex(self._key(session_id, profile), self.ttl_seconds, payload)
        else:
            expires_at = (_now() + timedelta(seconds=self.ttl_seconds)).isoformat()
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO session_memories(session_id, profile, memory_json, expires_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, profile) DO UPDATE SET
                      memory_json=excluded.memory_json, expires_at=excluded.expires_at, updated_at=excluded.updated_at""",
                    (session_id, profile, payload, expires_at, _now().isoformat()),
                )
        return {"saved": True, "session_id": session_id, "profile": profile, "backend": self.backend, "ttl_seconds": self.ttl_seconds}


@lru_cache(maxsize=1)
def get_session_memory_store() -> SessionMemoryStore:
    return SessionMemoryStore()
