"""TTL result-set storage for large conversational Anchors."""

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


class ResultAnchorStore:
    """Redis-first, SQLite fallback store for deferred Anchor members."""

    def __init__(self) -> None:
        settings = get_settings()
        self.ttl_seconds = max(60, settings.SESSION_MEMORY_TTL_SECONDS)
        self.db_path = Path(settings.SESSION_MEMORY_DB_PATH).with_name("result_anchors.sqlite3")
        self.redis = None
        if settings.REDIS_URL:
            try:
                import redis
                client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True, protocol=2)
                client.ping()
                self.redis = client
            except Exception:
                self.redis = None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS result_anchors (
                anchor_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL,
                expires_at TEXT NOT NULL, created_at TEXT NOT NULL
            )""")

    @property
    def backend(self) -> str:
        return "redis" if self.redis is not None else "sqlite"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _key(self, anchor_id: str) -> str:
        return f"text2sql:result-anchor:{anchor_id}"

    def save(self, scope: dict[str, Any]) -> dict[str, Any]:
        anchor_id = str(scope.get("anchor_id", ""))
        if not anchor_id:
            return {"saved": False, "reason": "missing_anchor_id", "backend": self.backend}
        expires_at = scope.get("expires_at") or (_now() + timedelta(seconds=self.ttl_seconds)).isoformat(timespec="seconds")
        payload = json.dumps({"anchor_id": anchor_id, "sample_ids": list(scope.get("sample_ids", [])),
                              "entity_count": int(scope.get("entity_count", scope.get("row_count", 0))),
                              "profile": scope.get("profile", ""), "schema_hash": scope.get("schema_hash", ""),
                              "status": scope.get("status", "truncated")}, ensure_ascii=False, sort_keys=True)
        if self.redis is not None:
            self.redis.setex(self._key(anchor_id), self.ttl_seconds, payload)
        else:
            with self._connect() as connection:
                connection.execute("INSERT OR REPLACE INTO result_anchors(anchor_id, payload_json, expires_at, created_at) VALUES (?, ?, ?, ?)",
                                   (anchor_id, payload, expires_at, _now().isoformat(timespec="seconds")))
        return {"saved": True, "anchor_id": anchor_id, "backend": self.backend, "expires_at": expires_at}

    def load(self, anchor_id: str) -> dict[str, Any] | None:
        if self.redis is not None:
            raw = self.redis.get(self._key(anchor_id))
            return json.loads(raw) if raw else None
        now = _now().isoformat(timespec="seconds")
        with self._connect() as connection:
            row = connection.execute("SELECT payload_json FROM result_anchors WHERE anchor_id = ? AND expires_at > ?", (anchor_id, now)).fetchone()
            connection.execute("DELETE FROM result_anchors WHERE expires_at <= ?", (now,))
        return json.loads(row["payload_json"]) if row else None


@lru_cache(maxsize=1)
def get_result_anchor_store() -> ResultAnchorStore:
    return ResultAnchorStore()

