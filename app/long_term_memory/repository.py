from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import MemoryRecord, MemoryWriteResult


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _vector_to_blob(vector: list[float] | None) -> tuple[bytes | None, int]:
    if not vector:
        return None, 0
    array = np.asarray(vector, dtype=np.float32)
    return array.tobytes(), int(array.shape[0])


def _blob_to_vector(blob: bytes | None, dim: int) -> list[float] | None:
    if blob is None or dim <= 0:
        return None
    array = np.frombuffer(blob, dtype=np.float32, count=dim)
    return array.tolist()


class SQLiteMemoryRepository:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    memory_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    schema_hash TEXT NOT NULL DEFAULT '',
                    embedding BLOB,
                    embedding_dim INTEGER NOT NULL DEFAULT 0,
                    embedding_model TEXT NOT NULL DEFAULT '',
                    dedupe_key TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(namespace, memory_type, dedupe_key)
                );

                CREATE INDEX IF NOT EXISTS idx_ltm_type_active
                ON long_term_memories(namespace, memory_type, is_active);

                CREATE INDEX IF NOT EXISTS idx_ltm_schema
                ON long_term_memories(namespace, schema_hash, is_active);

                CREATE TABLE IF NOT EXISTS approval_requests (
                    approval_id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    decided_at TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_approval_status
                ON approval_requests(namespace, status, created_at);
                """
            )

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            namespace=row["namespace"],
            memory_type=row["memory_type"],
            title=row["title"],
            content=row["content"],
            metadata=json.loads(row["metadata_json"] or "{}"),
            source=row["source"],
            is_active=bool(row["is_active"]),
            schema_hash=row["schema_hash"],
            embedding=_blob_to_vector(row["embedding"], row["embedding_dim"]),
            embedding_model=row["embedding_model"],
            dedupe_key=row["dedupe_key"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def upsert(self, record: MemoryRecord) -> MemoryWriteResult:
        now = _utc_now_iso()
        memory_id = record.memory_id or "mem-" + uuid.uuid4().hex[:16]
        blob, dim = _vector_to_blob(record.embedding)

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT memory_id, created_at
                FROM long_term_memories
                WHERE namespace = ? AND memory_type = ? AND dedupe_key = ?
                """,
                (record.namespace, record.memory_type, record.dedupe_key),
            ).fetchone()

            created = existing is None
            if existing is not None:
                memory_id = existing["memory_id"]
                created_at = existing["created_at"]
            else:
                created_at = record.created_at or now

            connection.execute(
                """
                INSERT INTO long_term_memories (
                    memory_id, namespace, memory_type, title, content,
                    metadata_json, source, is_active, schema_hash,
                    embedding, embedding_dim, embedding_model,
                    dedupe_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, memory_type, dedupe_key)
                DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    source = excluded.source,
                    is_active = excluded.is_active,
                    schema_hash = excluded.schema_hash,
                    embedding = excluded.embedding,
                    embedding_dim = excluded.embedding_dim,
                    embedding_model = excluded.embedding_model,
                    updated_at = excluded.updated_at
                """,
                (
                    memory_id,
                    record.namespace,
                    record.memory_type,
                    record.title,
                    record.content,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                    record.source,
                    1 if record.is_active else 0,
                    record.schema_hash,
                    blob,
                    dim,
                    record.embedding_model,
                    record.dedupe_key,
                    created_at,
                    now,
                ),
            )

        stored = self.get(memory_id)
        if stored is None:  # pragma: no cover
            raise RuntimeError("长期记忆写入后无法读取。")
        return MemoryWriteResult(record=stored, created=created)

    def get(self, memory_id: str) -> MemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM long_term_memories WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list(
        self,
        *,
        namespace: str,
        memory_type: str | None = None,
        active_only: bool = True,
        limit: int = 100,
    ) -> list[MemoryRecord]:
        clauses = ["namespace = ?"]
        parameters: list[object] = [namespace]
        if memory_type:
            clauses.append("memory_type = ?")
            parameters.append(memory_type)
        if active_only:
            clauses.append("is_active = 1")
        parameters.append(int(limit))

        sql = (
            "SELECT * FROM long_term_memories WHERE "
            + " AND ".join(clauses)
            + " ORDER BY updated_at DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._row_to_record(row) for row in rows]

    def candidates(
        self,
        *,
        namespace: str,
        memory_types: Iterable[str],
        schema_hash: str,
        limit: int = 1000,
    ) -> list[MemoryRecord]:
        types = [value for value in memory_types if value]
        if not types:
            return []
        placeholders = ",".join("?" for _ in types)
        sql = f"""
            SELECT * FROM long_term_memories
            WHERE namespace = ?
              AND memory_type IN ({placeholders})
              AND is_active = 1
              AND (schema_hash = '' OR schema_hash = ?)
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params: list[object] = [namespace, *types, schema_hash, int(limit)]
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def deactivate_by_prefix(self, namespace: str, prefix: str) -> tuple[bool, str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_id FROM long_term_memories
                WHERE namespace = ? AND memory_id LIKE ?
                """,
                (namespace, prefix + "%"),
            ).fetchall()
            if not rows:
                return False, "没有找到匹配的长期记忆。"
            if len(rows) > 1:
                return False, "该前缀匹配多条记忆，请输入更完整的memory_id。"
            memory_id = rows[0]["memory_id"]
            connection.execute(
                """
                UPDATE long_term_memories
                SET is_active = 0, updated_at = ?
                WHERE memory_id = ?
                """,
                (_utc_now_iso(), memory_id),
            )
        return True, memory_id

    def count(self, namespace: str) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_type, COUNT(*) AS n
                FROM long_term_memories
                WHERE namespace = ? AND is_active = 1
                GROUP BY memory_type
                """,
                (namespace,),
            ).fetchall()
        return {row["memory_type"]: int(row["n"]) for row in rows}

    def create_approval_request(
        self, *, namespace: str, profile: str, payload: dict[str, object]
    ) -> dict[str, object]:
        approval_id = "approval-" + uuid.uuid4().hex[:16]
        now = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO approval_requests(approval_id, namespace, profile, status, payload_json, created_at) VALUES (?, ?, ?, 'pending', ?, ?)",
                (approval_id, namespace, profile, json.dumps(payload, ensure_ascii=False, sort_keys=True), now),
            )
        return {"approval_id": approval_id, "status": "pending", "created_at": now, "payload": payload}

    def decide_approval_request(
        self, approval_id: str, decision: dict[str, object]
    ) -> dict[str, object] | None:
        now = _utc_now_iso()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM approval_requests WHERE approval_id = ?", (approval_id,)).fetchone()
            if row is None:
                return None
            if str(row["status"]) != "pending":
                # Decisions are append-only audit events. Repeating a command
                # is idempotent, but it cannot silently replace an earlier one.
                return self._approval_row(row)
            status = str(decision.get("action", "pending"))
            connection.execute(
                "UPDATE approval_requests SET status = ?, decision_json = ?, decided_at = ? WHERE approval_id = ?",
                (status, json.dumps(decision, ensure_ascii=False, sort_keys=True), now, approval_id),
            )
            row = connection.execute("SELECT * FROM approval_requests WHERE approval_id = ?", (approval_id,)).fetchone()
        return self._approval_row(row) if row else None

    def get_approval_request(self, approval_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM approval_requests WHERE approval_id = ?", (approval_id,)).fetchone()
        return self._approval_row(row) if row else None

    def list_approval_requests(
        self, *, namespace: str, status: str | None = None, limit: int = 50
    ) -> list[dict[str, object]]:
        clauses = ["namespace = ?"]
        parameters: list[object] = [namespace]
        if status:
            clauses.append("status = ?")
            parameters.append(status)
        parameters.append(int(limit))
        sql = (
            "SELECT * FROM approval_requests WHERE " + " AND ".join(clauses)
            + " ORDER BY created_at DESC LIMIT ?"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._approval_row(row) for row in rows]

    def update_memory_metadata(self, memory_id: str, metadata: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE long_term_memories SET metadata_json = ?, updated_at = ? WHERE memory_id = ?",
                (json.dumps(metadata, ensure_ascii=False, sort_keys=True), _utc_now_iso(), memory_id),
            )

    @staticmethod
    def _approval_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "approval_id": row["approval_id"], "namespace": row["namespace"], "profile": row["profile"],
            "status": row["status"], "payload": json.loads(row["payload_json"] or "{}"),
            "decision": json.loads(row["decision_json"] or "{}"), "created_at": row["created_at"], "decided_at": row["decided_at"],
        }
