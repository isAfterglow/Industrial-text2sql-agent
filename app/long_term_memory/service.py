from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from app.schema import get_schema_catalog

from .config import LongTermMemorySettings, get_long_term_memory_settings
from .embeddings import EmbeddingProvider
from .models import MemoryRecord, MemoryWriteResult
from .repository import SQLiteMemoryRepository


VALID_MEMORY_TYPES = {"semantic", "episodic", "procedural"}


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text).lower())


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    compact = _compact_text(text)
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def _lexical_score(query: str, candidate: str) -> float:
    query_compact = _compact_text(query)
    candidate_compact = _compact_text(candidate)
    if not query_compact or not candidate_compact:
        return 0.0
    if query_compact in candidate_compact or candidate_compact in query_compact:
        containment = min(len(query_compact), len(candidate_compact)) / max(
            len(query_compact), len(candidate_compact)
        )
        return 0.65 + 0.35 * containment

    qgrams = _char_ngrams(query_compact)
    cgrams = _char_ngrams(candidate_compact)
    jaccard = len(qgrams & cgrams) / max(1, len(qgrams | cgrams))
    sequence = SequenceMatcher(None, query_compact, candidate_compact).ratio()
    return 0.55 * jaccard + 0.45 * sequence


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _schema_hash() -> str:
    payload = json.dumps(
        get_schema_catalog(),
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _column_table_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for table_name, table in get_schema_catalog().get("tables", {}).items():
        for column in table.get("columns", {}):
            result.setdefault(column, table_name)
    return result


def _canonical_label(column: str) -> str:
    terms = get_schema_catalog().get("semantic_terms", {}).get(column, [])
    return str(terms[0]) if terms else column


def _resolve_target_column(target: str) -> tuple[str, str, str]:
    cleaned = target.strip().strip("。.;；")
    catalog = get_schema_catalog()
    owner_map = _column_table_map()
    if cleaned in owner_map:
        return cleaned, owner_map[cleaned], _canonical_label(cleaned)

    for column, terms in catalog.get("semantic_terms", {}).items():
        if cleaned == column or cleaned in terms:
            return column, owner_map.get(column, ""), _canonical_label(column)
        for term in terms:
            if term and term in cleaned:
                return column, owner_map.get(column, ""), _canonical_label(column)
    return "", "", cleaned


def _parse_semantic_relation(text: str) -> dict[str, str]:
    cleaned = text.strip()
    patterns = [
        r"^(.+?)\s*(?:->|→|=>|=)\s*(.+?)$",
        r"^(.+?)\s*(?:对应|表示|指的是|指|等同于)\s*(.+?)$",
        r"^(.+?)\s*[：:]\s*(.+?)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned)
        if not match:
            continue
        term = match.group(1).strip(" ：:")
        target = match.group(2).strip()
        column, table, canonical = _resolve_target_column(target)
        return {
            "term": term,
            "target": target,
            "column": column,
            "table": table,
            "canonical_term": canonical,
        }
    raise ValueError(
        "语义记忆请使用“术语 -> 字段/标准术语”，例如："
        "/remember 生料热导率 -> kv_list"
    )


def _parameterize_sql(sql: str) -> str:
    value = re.sub(r"sample_\d{6}", "<SAMPLE_ID>", sql, flags=re.IGNORECASE)
    value = re.sub(r"\bLIMIT\s+\d+\b", "LIMIT <N>", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\bBETWEEN\s+[-+]?\d+(?:\.\d+)?\s+AND\s+[-+]?\d+(?:\.\d+)?",
        "BETWEEN <LOWER> AND <UPPER>",
        value,
        flags=re.IGNORECASE,
    )
    return value


def _episodic_dedupe_key(query_spec: dict[str, Any]) -> str:
    order_by = query_spec.get("order_by") or {}
    filters = query_spec.get("filters") or query_spec.get("where_filters") or []
    temporal_metrics = query_spec.get("temporal_metrics") or []
    payload = {
        "query_type": query_spec.get("query_type", ""),
        "select_columns": sorted(query_spec.get("select_columns", [])),
        "filter_roles": sorted(
            (item.get("column", ""), item.get("operator", ""))
            for item in filters
            if isinstance(item, dict)
        ),
        "order_by": {
            "kind": order_by.get("kind", ""),
            "column": order_by.get("column", ""),
            "direction": order_by.get("direction", ""),
        },
        "temporal_metrics": sorted(
            (
                item.get("column", ""),
                item.get("aggregation", ""),
            )
            for item in temporal_metrics
            if isinstance(item, dict)
        ),
        "has_limit": query_spec.get("limit") is not None,
        "scalar_tables": sorted(query_spec.get("scalar_tables", [])),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return "episodic:" + digest


def _memory_search_text(record: MemoryRecord) -> str:
    metadata = json.dumps(record.metadata, ensure_ascii=False, sort_keys=True)
    return f"{record.title}\n{record.content}\n{metadata}"


class LongTermMemoryService:
    def __init__(self, settings: LongTermMemorySettings | None = None) -> None:
        self.settings = settings or get_long_term_memory_settings()
        self.repository = SQLiteMemoryRepository(self.settings.db_path)
        self.embedding = EmbeddingProvider(self.settings)
        self.schema_hash = _schema_hash()

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    def _embed_text(self, text: str) -> list[float] | None:
        if not self.enabled:
            return None
        return self.embedding.encode_one(text)

    def _upsert(
        self,
        *,
        memory_type: str,
        title: str,
        content: str,
        metadata: dict[str, Any],
        source: str,
        dedupe_key: str,
        schema_specific: bool = True,
    ) -> MemoryWriteResult:
        if memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(f"不支持的memory_type: {memory_type}")
        embedding_text = f"{title}\n{content}\n{json.dumps(metadata, ensure_ascii=False)}"
        vector = self._embed_text(embedding_text)
        return self.repository.upsert(
            MemoryRecord(
                memory_id="",
                namespace=self.settings.namespace,
                memory_type=memory_type,
                title=title,
                content=content,
                metadata=metadata,
                source=source,
                schema_hash=self.schema_hash if schema_specific else "",
                embedding=vector,
                embedding_model=(
                    self.settings.embedding_model if vector is not None else ""
                ),
                dedupe_key=dedupe_key,
            )
        )

    def remember_semantic(self, text: str, source: str = "manual") -> MemoryWriteResult:
        relation = _parse_semantic_relation(text)
        term = relation["term"]
        canonical = relation["canonical_term"]
        column = relation["column"]
        table = relation["table"]
        if column:
            content = (
                f"用户术语“{term}”对应标准术语“{canonical}”，"
                f"数据库字段为{table}.{column}。"
            )
        else:
            content = f"用户术语“{term}”表示：{relation['target']}。"

        return self._upsert(
            memory_type="semantic",
            title=f"语义：{term}",
            content=content,
            metadata={
                **relation,
                "replacement": canonical if column else "",
            },
            source=source,
            dedupe_key="semantic:" + _compact_text(term),
        )

    def remember_case(
        self,
        *,
        question: str,
        resolved_question: str,
        query_spec: dict[str, Any],
        sql: str,
        source: str,
        repaired: bool = False,
    ) -> MemoryWriteResult:
        query_type = str(query_spec.get("query_type", "complex_or_uncertain"))
        title = f"案例：{query_type}"
        content = (
            f"用户问题：{question}\n"
            f"消解问题：{resolved_question}\n"
            f"QuerySpec：{json.dumps(query_spec, ensure_ascii=False, sort_keys=True)}\n"
            f"已验证SQL：{sql}"
        )
        return self._upsert(
            memory_type="episodic",
            title=title,
            content=content,
            metadata={
                "question": question,
                "resolved_question": resolved_question,
                "query_type": query_type,
                "query_spec": query_spec,
                "sql": sql,
                "sql_template": _parameterize_sql(sql),
                "repaired": repaired,
            },
            source=source,
            dedupe_key=_episodic_dedupe_key(query_spec),
        )

    def remember_case_from_short_memory(
        self,
        memory: dict[str, Any],
        source: str = "manual_case",
    ) -> MemoryWriteResult:
        query_spec = dict(
            memory.get("last_successful_query_state")
            or memory.get("last_query_spec")
            or {}
        )
        sql = str(memory.get("last_validated_sql", "")).strip()
        if not query_spec or not sql:
            raise ValueError("当前没有可以保存的成功查询案例。")
        return self.remember_case(
            question=str(memory.get("last_successful_question", "")),
            resolved_question=str(memory.get("last_resolved_question", "")),
            query_spec=query_spec,
            sql=sql,
            source=source,
            repaired=False,
        )

    def remember_procedural(
        self,
        *,
        error_signature: str,
        description: str,
        bad_pattern: str,
        preferred_pattern: str,
        source: str,
        schema_specific: bool = True,
    ) -> MemoryWriteResult:
        normalized_signature = _compact_text(error_signature)[:120]
        content = (
            f"错误：{description}\n"
            f"不推荐结构：{bad_pattern}\n"
            f"推荐处理：{preferred_pattern}"
        )
        return self._upsert(
            memory_type="procedural",
            title=f"修复经验：{error_signature}",
            content=content,
            metadata={
                "error_signature": error_signature,
                "description": description,
                "bad_pattern": bad_pattern,
                "preferred_pattern": preferred_pattern,
            },
            source=source,
            dedupe_key="procedural:" + hashlib.sha256(
                normalized_signature.encode("utf-8")
            ).hexdigest()[:24],
            schema_specific=schema_specific,
        )

    def ensure_default_memories(self) -> None:
        if not self.enabled:
            return
        defaults = [
            (
                "limit_in_in_subquery",
                "MySQL不支持或不应使用IN子查询中的LIMIT结构。",
                "WHERE sample_id IN (SELECT sample_id ... LIMIT N)",
                "改为最少必要表直接JOIN，在顶层ORDER BY并LIMIT。",
            ),
            (
                "unnecessary_thermal_response_join",
                "不需要时序字段时连接thermal_response会把每个样本展开成多行。",
                "静态属性查询JOIN thermal_response",
                "仅连接目标字段所属的一对一表。",
            ),
            (
                "historical_scope_leak",
                "完整独立查询不能继承上一轮sample_ids。",
                "独立QuerySpec仍携带历史sample_ids",
                "无显式代词时清除历史scope，以当前QuerySpec为准。",
            ),
            (
                "field_ownership",
                "跨表字段必须使用真实所属表。",
                "把rhov_i、rhoc_i写到material_thermal_property",
                "根据Schema字段归属选择最少必要表并使用正确别名。",
            ),
        ]
        for signature, description, bad, preferred in defaults:
            self.remember_procedural(
                error_signature=signature,
                description=description,
                bad_pattern=bad,
                preferred_pattern=preferred,
                source="system_rule",
                schema_specific=False,
            )

    def list_memories(
        self,
        memory_type: str | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        if memory_type and memory_type not in VALID_MEMORY_TYPES:
            raise ValueError("memory_type只支持semantic、episodic、procedural。")
        return self.repository.list(
            namespace=self.settings.namespace,
            memory_type=memory_type,
            active_only=True,
            limit=limit,
        )

    def forget(self, memory_id_prefix: str) -> tuple[bool, str]:
        return self.repository.deactivate_by_prefix(
            self.settings.namespace,
            memory_id_prefix.strip(),
        )

    def exact_semantic_matches(self, question: str) -> list[MemoryRecord]:
        matches: list[MemoryRecord] = []
        for record in self.list_memories("semantic", limit=500):
            term = str(record.metadata.get("term", "")).strip()
            if term and term in question:
                record.score = 1.0
                matches.append(record)
        matches.sort(key=lambda item: len(str(item.metadata.get("term", ""))), reverse=True)
        return matches

    def search(
        self,
        query: str,
        *,
        memory_types: Iterable[str],
        top_k: int,
        min_score: float,
    ) -> list[MemoryRecord]:
        if not self.enabled or not query.strip():
            return []
        types = [value for value in memory_types if value in VALID_MEMORY_TYPES]
        if not types:
            return []

        candidates = self.repository.candidates(
            namespace=self.settings.namespace,
            memory_types=types,
            schema_hash=self.schema_hash,
        )
        if not candidates:
            return []

        query_vector = self._embed_text(query)
        scored: list[MemoryRecord] = []
        for record in candidates:
            candidate_text = _memory_search_text(record)
            metadata = record.metadata
            lexical_parts = [
                record.title,
                record.content,
                str(metadata.get("question", "")),
                str(metadata.get("resolved_question", "")),
                str(metadata.get("query_type", "")),
                str(metadata.get("term", "")),
                str(metadata.get("error_signature", "")),
                str(metadata.get("description", "")),
                candidate_text,
            ]
            lexical = max(
                (_lexical_score(query, part) for part in lexical_parts if part),
                default=0.0,
            )
            vector = _cosine_similarity(query_vector, record.embedding)
            score = (
                0.75 * vector + 0.25 * lexical
                if query_vector is not None and record.embedding is not None
                else lexical
            )
            if score < min_score:
                continue
            record.score = float(score)
            scored.append(record)

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(1, int(top_k))]

    def retrieve_semantic(self, question: str, force_vector: bool = False) -> list[MemoryRecord]:
        exact = self.exact_semantic_matches(question)
        if exact:
            return exact[: self.settings.semantic_top_k]
        if not force_vector:
            return []
        return self.search(
            question,
            memory_types=["semantic"],
            top_k=self.settings.semantic_top_k,
            min_score=self.settings.semantic_min_score,
        )

    def apply_semantic_memories(
        self,
        question: str,
        memories: list[MemoryRecord],
    ) -> tuple[str, list[str], list[str]]:
        augmented = question
        hints: list[str] = []
        applied_ids: list[str] = []
        for record in memories:
            term = str(record.metadata.get("term", "")).strip()
            replacement = str(record.metadata.get("replacement", "")).strip()
            if term and replacement and term in augmented:
                augmented = augmented.replace(term, replacement)
                applied_ids.append(record.memory_id)
            hints.append(record.content)
        return augmented, hints, applied_ids

    def retrieve_episodic(
        self,
        question: str,
        query_spec: dict[str, Any],
    ) -> list[MemoryRecord]:
        query_type = str(query_spec.get("query_type", ""))
        retrieval_text = (
            f"问题：{question}\n"
            f"查询类型：{query_type}\n"
            f"返回字段：{query_spec.get('select_columns', [])}\n"
            f"排序：{query_spec.get('order_by')}\n"
            f"聚合：{query_spec.get('temporal_metrics', [])}"
        )
        return self.search(
            retrieval_text,
            memory_types=["episodic"],
            top_k=self.settings.episodic_top_k,
            min_score=self.settings.episodic_min_score,
        )

    def retrieve_procedural(self, error_text: str) -> list[MemoryRecord]:
        return self.search(
            error_text,
            memory_types=["procedural"],
            top_k=self.settings.procedural_top_k,
            min_score=self.settings.procedural_min_score,
        )

    def build_few_shot_context(self, memories: list[MemoryRecord]) -> str:
        if not memories:
            return ""
        blocks = [
            "以下是长期记忆中已验证成功的相似案例，仅参考查询结构。",
            "必须以当前问题、当前Schema和当前QuerySpec为准，",
            "不得复制历史sample_id、过滤值或LIMIT。",
        ]
        for index, record in enumerate(memories, start=1):
            metadata = record.metadata
            blocks.append(
                "\n".join(
                    [
                        f"案例{index}（相似度{record.score:.3f}）：",
                        f"问题：{metadata.get('resolved_question') or metadata.get('question', '')}",
                        f"查询类型：{metadata.get('query_type', '')}",
                        "QuerySpec："
                        + json.dumps(
                            metadata.get("query_spec", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        f"SQL结构模板：{metadata.get('sql_template', '')}",
                    ]
                )
            )
        return "\n\n".join(blocks)[: self.settings.max_prompt_chars]

    def build_procedural_context(self, memories: list[MemoryRecord]) -> str:
        if not memories:
            return ""
        lines = ["长期记忆中的相关修复经验："]
        for index, record in enumerate(memories, start=1):
            lines.append(f"{index}. {record.content}")
        return "\n".join(lines)[: self.settings.max_prompt_chars]

    def should_retrieve_episodic(self, query_spec: dict[str, Any]) -> bool:
        query_type = str(query_spec.get("query_type", ""))
        if not query_spec:
            return False
        if not query_spec.get("eligible"):
            return True
        return any(
            marker in query_type
            for marker in ("multi_table", "temporal", "aggregate", "final")
        )

    def should_auto_save_case(self, state: dict[str, Any]) -> bool:
        if not self.settings.auto_save:
            return False
        query_spec = state.get("query_spec", {})
        query_type = str(query_spec.get("query_type", ""))
        if state.get("retry_count", 0) > 0:
            return True
        return any(
            marker in query_type
            for marker in ("multi_table", "temporal", "aggregate", "final")
        )

    def auto_save_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "saved": []}

        saved: list[dict[str, Any]] = []
        if self.should_auto_save_case(state):
            result = self.remember_case(
                question=str(state.get("question", "")),
                resolved_question=str(
                    state.get("resolved_question")
                    or state.get("normalized_question")
                    or state.get("question", "")
                ),
                query_spec=dict(state.get("query_spec", {})),
                sql=str(state.get("validated_sql", "")),
                source=(
                    "repaired_query"
                    if state.get("retry_count", 0) > 0
                    else "successful_query"
                ),
                repaired=bool(state.get("retry_count", 0) > 0),
            )
            saved.append(
                {
                    "memory_id": result.record.memory_id,
                    "memory_type": "episodic",
                    "created": result.created,
                }
            )

        if state.get("retry_count", 0) > 0 and state.get("last_repair_reason"):
            repair_reason = str(state.get("last_repair_reason", "")).strip()
            signature_seed = re.sub(r"\s+", " ", repair_reason)[:160]
            signature = "repair:" + (signature_seed or "sql_repair")
            result = self.remember_procedural(
                error_signature=signature,
                description=str(state.get("last_repair_reason", "")),
                bad_pattern=str(state.get("repair_bad_sql", "")),
                preferred_pattern=str(
                    state.get("repair_action")
                    or state.get("validated_sql", "")
                ),
                source="repaired_query",
            )
            saved.append(
                {
                    "memory_id": result.record.memory_id,
                    "memory_type": "procedural",
                    "created": result.created,
                }
            )

        return {"enabled": True, "saved": saved}

    def format_list(self, records: list[MemoryRecord]) -> str:
        if not records:
            return "当前没有匹配的长期记忆。"
        lines: list[str] = []
        for record in records:
            lines.extend(
                [
                    f"[{record.memory_id}] {record.title}",
                    f"类型: {record.memory_type}; 来源: {record.source}",
                    record.content,
                    "-" * 60,
                ]
            )
        return "\n".join(lines).rstrip("-\n")

    def status_summary(self) -> str:
        counts = self.repository.count(self.settings.namespace)
        embedding_status = self.embedding.status()
        return "\n".join(
            [
                f"长期记忆启用: {'是' if self.enabled else '否'}",
                f"SQLite: {self.settings.db_path}",
                f"namespace: {self.settings.namespace}",
                f"schema_hash: {self.schema_hash}",
                "记忆数量: "
                + ", ".join(
                    f"{key}={value}" for key, value in sorted(counts.items())
                )
                if counts
                else "记忆数量: 0",
                f"Embedding模型: {embedding_status.model_name}",
                (
                    "Embedding状态: 已加载"
                    if embedding_status.available
                    else f"Embedding状态: 未加载/词法回退（{embedding_status.reason}）"
                ),
            ]
        )


@lru_cache(maxsize=1)
def get_long_term_memory_service() -> LongTermMemoryService:
    service = LongTermMemoryService()
    service.ensure_default_memories()
    return service