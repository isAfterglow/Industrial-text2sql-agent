from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryRecord:
    memory_id: str
    namespace: str
    memory_type: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "manual"
    is_active: bool = True
    schema_hash: str = ""
    embedding: list[float] | None = None
    embedding_model: str = ""
    dedupe_key: str = ""
    created_at: str = ""
    updated_at: str = ""
    score: float = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "source": self.source,
            "schema_hash": self.schema_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "score": round(float(self.score), 4),
        }


@dataclass
class MemoryWriteResult:
    record: MemoryRecord
    created: bool


@dataclass
class EmbeddingStatus:
    available: bool
    model_name: str
    reason: str = ""