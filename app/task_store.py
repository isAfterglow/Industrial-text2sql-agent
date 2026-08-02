"""Durable task and event storage for the Agent API."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AgentTaskStore:
    """SQLite-backed API task state; safe for API process restarts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    question TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    approval_id TEXT NOT NULL DEFAULT '',
                    input_json TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT NOT NULL DEFAULT '',
                    finished_at TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_created
                ON agent_tasks(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_agent_tasks_status
                ON agent_tasks(status, updated_at DESC);

                CREATE TABLE IF NOT EXISTS agent_task_events (
                    task_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, sequence),
                    FOREIGN KEY(task_id) REFERENCES agent_tasks(task_id)
                );
                """
            )

    def create_task(self, *, profile: str, question: str, session_id: str, trace_id: str,
                    payload: dict[str, Any], approval_id: str = "") -> dict[str, Any]:
        task_id = "task-" + uuid.uuid4().hex[:16]
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO agent_tasks (
                    task_id, status, profile, question, session_id, trace_id, approval_id,
                    input_json, created_at, updated_at
                ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_id, profile, question, session_id, trace_id, approval_id,
                 json.dumps(payload, ensure_ascii=False), now, now),
            )
        self.append_event(task_id, {"type": "task", "status": "queued", "at": now})
        return self.get_task(task_id) or {}

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM agent_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._task_row(row) if row else None

    def list_tasks(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_tasks ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
        return [self._task_row(row) for row in rows]

    def update_status(self, task_id: str, status: str, *, result: dict[str, Any] | None = None,
                      error_message: str = "") -> None:
        now = utc_now()
        fields = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, now]
        if status == "running":
            fields.append("started_at = ?")
            values.append(now)
        if status in {"completed", "approval_required", "failed"}:
            fields.append("finished_at = ?")
            values.append(now)
        if result is not None:
            fields.append("result_json = ?")
            values.append(json.dumps(result, ensure_ascii=False, default=str))
        if error_message:
            fields.append("error_message = ?")
            values.append(error_message)
        values.append(task_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE agent_tasks SET {', '.join(fields)} WHERE task_id = ?", values)
        self.append_event(task_id, {"type": "task", "status": status, "at": now, "error": error_message})

    def append_event(self, task_id: str, payload: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as connection:
            sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM agent_task_events WHERE task_id = ?",
                (task_id,),
            ).fetchone()["next_sequence"]
            connection.execute(
                "INSERT INTO agent_task_events (task_id, sequence, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (task_id, sequence, json.dumps(payload, ensure_ascii=False, default=str), now),
            )

    def events_after(self, task_id: str, sequence: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT sequence, payload_json, created_at FROM agent_task_events WHERE task_id = ? AND sequence > ? ORDER BY sequence",
                (task_id, max(0, sequence)),
            ).fetchall()
        return [
            {"sequence": row["sequence"], "created_at": row["created_at"], "payload": json.loads(row["payload_json"])}
            for row in rows
        ]

    @staticmethod
    def _task_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "task_id": row["task_id"], "status": row["status"], "profile": row["profile"],
            "question": row["question"], "session_id": row["session_id"], "trace_id": row["trace_id"],
            "approval_id": row["approval_id"], "input": json.loads(row["input_json"]),
            "result": json.loads(row["result_json"]), "error_message": row["error_message"],
            "created_at": row["created_at"], "started_at": row["started_at"],
            "finished_at": row["finished_at"], "updated_at": row["updated_at"],
        }
