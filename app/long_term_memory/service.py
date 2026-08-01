from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from app.schema import active_profile_name, get_schema_catalog
from app.query_enhancement import (
    build_query_signature,
    hard_signature_compatible,
    query_signature_similarity,
    signature_summary,
)

from .config import LongTermMemorySettings, get_long_term_memory_settings
from .embeddings import EmbeddingProvider
from .models import MemoryRecord, MemoryWriteResult
from .repository import SQLiteMemoryRepository


VALID_MEMORY_TYPES = {
    "semantic", "episodic", "procedural", "candidate_episodic",
    "candidate_procedural", "failure",
}


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
        "query_signature": build_query_signature(query_spec),
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
        "advanced_plan": query_spec.get("advanced_plan", {}),
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
    @property
    def namespace(self) -> str:
        return f"{self.settings.namespace}:{active_profile_name()}"

    @property
    def schema_hash(self) -> str:
        return _schema_hash()

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
                namespace=self.namespace,
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
        case_context: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        query_type = str(query_spec.get("query_type", "complex_or_uncertain"))
        title = f"案例：{query_type}"
        context = dict(case_context or {})
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
                "query_signature": build_query_signature(query_spec, resolved_question or question),
                "sql": sql,
                "sql_template": _parameterize_sql(sql),
                "repaired": repaired,
                "dependency": context.get("dependency", ""),
                "memory_used": bool(context.get("memory_used", False)),
                "independent_case": bool(
                    context.get("independent_case", source == "manual_case")
                ),
            },
            source=source,
            dedupe_key=_episodic_dedupe_key(query_spec),
        )

    def remember_candidate_case(
        self,
        *,
        question: str,
        resolved_question: str,
        query_spec: dict[str, Any],
        sql: str,
        approval_id: str = "",
        source: str = "approved_query",
        approval_reason: str = "",
    ) -> MemoryWriteResult:
        """Store a validated case outside the retrieval pool until promotion."""

        query_type = str(query_spec.get("query_type", "advanced_analysis"))
        metadata = {
            "question": question, "resolved_question": resolved_question,
            "query_spec": query_spec,
            "query_signature": build_query_signature(query_spec, resolved_question or question),
            "sql": sql, "sql_template": _parameterize_sql(sql),
            "approval_id": approval_id, "approval_reason": approval_reason,
            "promotion_status": "candidate", "validation": "guard_execution_result_contract",
        }
        return self._upsert(
            memory_type="candidate_episodic", title=f"候选案例：{query_type}",
            content=f"用户问题：{question}\n消解问题：{resolved_question}\n已验证SQL：{sql}",
            metadata=metadata, source=source,
            dedupe_key="candidate:" + _episodic_dedupe_key(query_spec),
        )

    def promote_candidate(self, memory_id: str, *, evidence: str) -> MemoryWriteResult:
        candidate = self.repository.get(memory_id)
        if candidate is None or candidate.memory_type != "candidate_episodic":
            raise ValueError("只能晋升候选情景记忆。")
        metadata = dict(candidate.metadata)
        validations = metadata.get("independent_validations", [])
        if not isinstance(validations, list) or len(validations) < 3:
            raise ValueError("候选案例至少需要3条独立成功验证后才能晋升。")
        plan = metadata.get("advanced_plan") or dict(metadata.get("query_spec", {})).get("advanced_plan", {})
        if isinstance(plan, dict) and plan:
            metadata.update({
                "memory_role": "advanced_plan_example",
                "advanced_plan": plan,
                "schema_tables": sorted(get_schema_catalog().get("tables", {})),
                "quality": {"retrieval_count": 0, "success_count": 0, "failure_count": 0},
            })
        metadata.update({"promotion_status": "promoted", "promotion_evidence": evidence, "candidate_memory_id": memory_id})
        return self._upsert(
            memory_type="episodic", title=candidate.title.replace("候选", "正式"),
            content=candidate.content, metadata=metadata, source="candidate_promoted",
            dedupe_key=_episodic_dedupe_key(dict(metadata.get("query_spec", {}))),
        )

    def create_approval_request(self, *, profile: str, payload: dict[str, object]) -> dict[str, object]:
        return self.repository.create_approval_request(namespace=self.namespace, profile=profile, payload=payload)

    def decide_approval_request(self, approval_id: str, decision: dict[str, object]) -> dict[str, object] | None:
        return self.repository.decide_approval_request(approval_id, decision)

    def list_approval_requests(self, status: str | None = None, limit: int = 50) -> list[dict[str, object]]:
        return self.repository.list_approval_requests(
            namespace=self.namespace, status=status, limit=limit
        )

    def record_candidate_validation(
        self, memory_id: str, *, question: str, plan: dict[str, Any], evidence: str
    ) -> None:
        """Attach independent successful validation evidence without making it retrievable."""

        record = self.repository.get(memory_id)
        if record is None or record.memory_type != "candidate_episodic":
            raise ValueError("只能为候选情景记忆记录验证。")
        metadata = dict(record.metadata)
        validations = list(metadata.get("independent_validations", []))
        fingerprint = hashlib.sha256(
            json.dumps({"question": question, "plan": plan}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        if not any(item.get("fingerprint") == fingerprint for item in validations if isinstance(item, dict)):
            validations.append({"fingerprint": fingerprint, "question": question, "plan": plan, "evidence": evidence})
        metadata["independent_validations"] = validations
        metadata["validated_variant_count"] = len(validations)
        self.repository.update_memory_metadata(memory_id, metadata)

    def retrieve_advanced_plan_examples(
        self, question: str, family: str, *, max_examples: int = 2
    ) -> tuple[list[MemoryRecord], dict[str, Any]]:
        """Retrieve only curated, promoted AdvancedPlan examples for completion."""

        candidates = self.search(question, memory_types=["episodic"], top_k=20, min_score=0.0)
        selected: list[MemoryRecord] = []
        rejected: dict[str, int] = {}
        current_tables = set(get_schema_catalog().get("tables", {}))
        for record in candidates:
            metadata = dict(record.metadata or {})
            plan = metadata.get("advanced_plan") or metadata.get("query_spec", {}).get("advanced_plan")
            if metadata.get("memory_role") != "advanced_plan_example":
                rejected["not_advanced_example"] = rejected.get("not_advanced_example", 0) + 1
                continue
            if not isinstance(plan, dict) or plan.get("family") != family:
                rejected["family_mismatch"] = rejected.get("family_mismatch", 0) + 1
                continue
            tables = set(metadata.get("schema_tables", []))
            if tables and not tables.issubset(current_tables):
                rejected["schema_mismatch"] = rejected.get("schema_mismatch", 0) + 1
                continue
            quality = metadata.get("quality", {}) if isinstance(metadata.get("quality"), dict) else {}
            success_rate = float(quality.get("success_count", 0)) / max(1, int(quality.get("retrieval_count", 0)))
            record.score = 0.8 * float(record.score) + 0.2 * success_rate
            selected.append(record)
        selected.sort(key=lambda item: item.score, reverse=True)
        selected = selected[:max(1, max_examples)]
        return selected, {
            "candidate_count": len(candidates), "selected_count": len(selected),
            "family": family, "rejected_reasons": rejected,
        }

    def build_advanced_plan_few_shot_context(self, memories: list[MemoryRecord]) -> str:
        if not memories:
            return ""
        blocks = [
            "Approved structural examples for the selected family follow.",
            "Adapt only the structure to the current question and schema; never copy values blindly.",
        ]
        for index, record in enumerate(memories, start=1):
            metadata = record.metadata
            blocks.append(
                "Example %d:\nQuestion: %s\nAdvancedPlan: %s\nSchema tables: %s" % (
                    index,
                    metadata.get("resolved_question") or metadata.get("question", ""),
                    json.dumps(metadata.get("advanced_plan", {}), ensure_ascii=False, sort_keys=True),
                    ", ".join(metadata.get("schema_tables", [])),
                )
            )
        return "\n\n".join(blocks)[: self.settings.max_prompt_chars]

    def record_advanced_plan_usage(self, memory_ids: list[str], *, success: bool) -> None:
        for memory_id in memory_ids:
            record = self.repository.get(memory_id)
            if record is None or record.memory_type != "episodic":
                continue
            metadata = dict(record.metadata)
            quality = dict(metadata.get("quality", {}))
            quality["retrieval_count"] = int(quality.get("retrieval_count", 0)) + 1
            quality["success_count"] = int(quality.get("success_count", 0)) + int(success)
            quality["failure_count"] = int(quality.get("failure_count", 0)) + int(not success)
            metadata["quality"] = quality
            self.repository.update_memory_metadata(memory_id, metadata)

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
            namespace=self.namespace,
            memory_type=memory_type,
            active_only=True,
            limit=limit,
        )

    def forget(self, memory_id_prefix: str) -> tuple[bool, str]:
        return self.repository.deactivate_by_prefix(
            self.namespace,
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
            namespace=self.namespace,
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

    def _record_is_safe_few_shot(self, record: MemoryRecord) -> tuple[bool, str]:
        """拒绝历史范围污染、具体样本集合和不支持结构产生的案例。"""

        metadata = dict(record.metadata or {})
        query_spec = metadata.get("query_spec")
        if not isinstance(query_spec, dict):
            return False, "missing_query_spec"
        if query_spec.get("sample_ids"):
            return False, "history_bound_sample_scope"
        if query_spec.get("capability_check", {}).get("unsupported"):
            return False, "unsupported_case"
        if str(query_spec.get("query_type", "")) == "unsupported_nested_topk":
            return False, "unsupported_case"
        if query_spec.get("memory_resolved") or query_spec.get("structured_context_complete"):
            return False, "memory_resolved_case"
        resolved = str(metadata.get("resolved_question", ""))
        if "样本编号限定为" in resolved:
            return False, "history_bound_sample_scope"
        if "independent_case" in metadata and not bool(metadata.get("independent_case")):
            return False, "non_independent_case"
        return True, ""


    def _record_signature(self, record: MemoryRecord) -> dict[str, Any]:
        signature = record.metadata.get("query_signature")
        if isinstance(signature, dict) and signature:
            return signature
        query_spec = record.metadata.get("query_spec")
        if isinstance(query_spec, dict):
            return build_query_signature(
                query_spec,
                str(
                    record.metadata.get("resolved_question")
                    or record.metadata.get("question", "")
                ),
            )
        return {}

    def _candidate_similarity(
        self,
        left: MemoryRecord,
        right: MemoryRecord,
    ) -> float:
        vector_score = _cosine_similarity(left.embedding, right.embedding)
        if vector_score > 0.0:
            return vector_score
        left_question = str(
            left.metadata.get("resolved_question")
            or left.metadata.get("question", "")
        )
        right_question = str(
            right.metadata.get("resolved_question")
            or right.metadata.get("question", "")
        )
        structure_score = query_signature_similarity(
            self._record_signature(left),
            self._record_signature(right),
        )
        return 0.55 * _lexical_score(left_question, right_question) + 0.45 * structure_score

    def _mmr_select(
        self,
        candidates: list[MemoryRecord],
        max_examples: int,
    ) -> list[MemoryRecord]:
        if not candidates or max_examples <= 0:
            return []
        selected: list[MemoryRecord] = []
        remaining = list(candidates)
        lambda_value = min(1.0, max(0.0, self.settings.episodic_mmr_lambda))

        while remaining and len(selected) < max_examples:
            if not selected:
                chosen = max(remaining, key=lambda item: item.score)
            else:
                def mmr_score(item: MemoryRecord) -> float:
                    redundancy = max(
                        self._candidate_similarity(item, selected_item)
                        for selected_item in selected
                    )
                    return lambda_value * item.score - (1.0 - lambda_value) * redundancy

                chosen = max(remaining, key=mmr_score)
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    def retrieve_episodic_with_diagnostics(
        self,
        question: str,
        query_spec: dict[str, Any],
    ) -> tuple[list[MemoryRecord], dict[str, Any]]:
        current_signature = build_query_signature(query_spec, question)
        if current_signature.get("has_nested_topk"):
            return [], {
                "candidate_count": 0,
                "hard_compatible_count": 0,
                "selected_count": 0,
                "few_shot_used": False,
                "skip_reason": "unsupported_nested_topk",
                "query_signature": current_signature,
            }

        query_type = str(query_spec.get("query_type", ""))
        retrieval_text = (
            f"问题：{question}\n"
            f"查询类型：{query_type}\n"
            f"返回字段：{query_spec.get('output_columns') or query_spec.get('select_columns', [])}\n"
            f"排序：{query_spec.get('order_by')}\n"
            f"聚合：{query_spec.get('all_temporal_metrics') or query_spec.get('temporal_metrics', [])}\n"
            f"派生指标：{query_spec.get('derived_metrics', [])}\n"
            f"结构签名：{signature_summary(current_signature)}"
        )

        semantic_candidates = self.search(
            retrieval_text,
            memory_types=["episodic"],
            top_k=max(1, self.settings.episodic_candidate_k),
            min_score=self.settings.episodic_min_score,
        )

        compatible: list[MemoryRecord] = []
        rejected_reasons: dict[str, int] = {}
        for record in semantic_candidates:
            safe_case, unsafe_reason = self._record_is_safe_few_shot(record)
            if not safe_case:
                rejected_reasons[unsafe_reason] = rejected_reasons.get(unsafe_reason, 0) + 1
                continue
            candidate_signature = self._record_signature(record)
            is_compatible, reason = hard_signature_compatible(
                current_signature, candidate_signature
            )
            if not is_compatible:
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                continue

            structural_score = query_signature_similarity(
                current_signature, candidate_signature
            )
            if structural_score < self.settings.episodic_structural_min_score:
                rejected_reasons["low_structural_score"] = (
                    rejected_reasons.get("low_structural_score", 0) + 1
                )
                continue

            semantic_score = float(record.score)
            current_tables = set(current_signature.get("tables", []))
            candidate_tables = set(candidate_signature.get("tables", []))
            schema_score = (
                len(current_tables & candidate_tables)
                / max(1, len(current_tables | candidate_tables))
                if current_tables or candidate_tables
                else 1.0
            )
            final_score = (
                0.35 * semantic_score
                + 0.50 * structural_score
                + 0.15 * schema_score
            )
            if final_score < self.settings.episodic_final_min_score:
                rejected_reasons["low_final_score"] = (
                    rejected_reasons.get("low_final_score", 0) + 1
                )
                continue

            record.metadata = dict(record.metadata)
            record.metadata["retrieval_scores"] = {
                "semantic": round(semantic_score, 4),
                "structural": round(structural_score, 4),
                "schema": round(schema_score, 4),
                "final": round(final_score, 4),
            }
            record.metadata["query_signature"] = candidate_signature
            record.score = float(final_score)
            compatible.append(record)

        compatible.sort(key=lambda item: item.score, reverse=True)
        selected = self._mmr_select(
            compatible,
            max_examples=max(1, self.settings.episodic_max_examples),
        )
        diagnostics = {
            "candidate_count": len(semantic_candidates),
            "hard_compatible_count": len(compatible),
            "selected_count": len(selected),
            "few_shot_used": bool(selected),
            "skip_reason": "" if selected else "no_structurally_compatible_example",
            "rejected_reasons": rejected_reasons,
            "query_signature": current_signature,
        }
        return selected, diagnostics

    def retrieve_episodic(
        self,
        question: str,
        query_spec: dict[str, Any],
    ) -> list[MemoryRecord]:
        memories, _ = self.retrieve_episodic_with_diagnostics(question, query_spec)
        return memories

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
            "以下案例经过Schema与SQL Guard验证，只用于参考结构。",
            "必须以当前问题、当前Schema和当前QuerySpec为准。",
            "禁止复制历史sample_id、过滤值、LIMIT或不匹配的派生指标。",
            "没有完全匹配的信息时不得猜测；应触发澄清或能力边界提示。",
        ]
        for index, record in enumerate(memories, start=1):
            metadata = record.metadata
            query_spec = metadata.get("query_spec", {})
            signature = metadata.get("query_signature") or self._record_signature(record)
            scores = metadata.get("retrieval_scores", {})
            blocks.append(
                "\n".join(
                    [
                        f"案例{index}（混合匹配分{record.score:.3f}）：",
                        f"问题：{metadata.get('resolved_question') or metadata.get('question', '')}",
                        f"结构：{signature_summary(signature)}",
                        f"过滤：{query_spec.get('filters') or query_spec.get('where_filters', [])}",
                        f"排序：{query_spec.get('order_by')}",
                        f"时序指标：{query_spec.get('all_temporal_metrics') or query_spec.get('temporal_metrics', [])}",
                        f"派生指标：{query_spec.get('derived_metrics', [])}",
                        "参考结构：QuerySpec字段、过滤角色、排序和时序口径；"
                        "不得复制历史SQL、样本值或返回结果。",
                        f"检索分解：{json.dumps(scores, ensure_ascii=False, sort_keys=True)}",
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
        if not query_spec or query_type == "unsupported_nested_topk":
            return False
        if query_spec.get("capability_check", {}).get("unsupported"):
            return False
        if not query_spec.get("eligible"):
            return True
        return bool(
            query_spec.get("temporal_metrics")
            or query_spec.get("derived_metrics")
            or len(query_spec.get("scalar_tables", [])) > 1
            or any(
                marker in query_type
                for marker in ("multi_table", "temporal", "aggregate", "final", "extended")
            )
        )

    def should_auto_save_case(self, state: dict[str, Any]) -> bool:
        """只自动保存自包含、无历史范围绑定且完整校验成功的案例。"""

        if not self.settings.auto_save:
            return False
        if state.get("execution_error") or state.get("validation_error"):
            return False
        if not state.get("context_resolution_valid", True):
            return False
        coverage = state.get("current_turn_coverage") or {}
        if coverage and not coverage.get("passed", False):
            return False
        if coverage.get("historical_scope_leak"):
            return False

        query_spec = state.get("query_spec", {})
        if not isinstance(query_spec, dict) or not query_spec:
            return False
        if query_spec.get("sample_ids"):
            return False
        if query_spec.get("capability_check", {}).get("unsupported"):
            return False
        query_type = str(query_spec.get("query_type", ""))
        if query_type == "unsupported_nested_topk":
            return False

        delta = state.get("query_delta") or {}
        if str(delta.get("dependency", "independent")) != "independent":
            return False
        if state.get("memory_used"):
            return False
        if delta.get("independent_complete") is False and delta.get("explicit_reference"):
            return False

        if state.get("retry_count", 0) > 0:
            return True
        return any(
            marker in query_type
            for marker in ("multi_table", "temporal", "aggregate", "final", "extended")
        )

    def auto_save_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "saved": []}

        saved: list[dict[str, Any]] = []
        if self.should_auto_save_case(state) or state.get("approval_approved"):
            query_spec = dict(state.get("query_spec", {}))
            if state.get("advanced_plan"):
                query_spec["advanced_plan"] = dict(state["advanced_plan"])
                query_spec["query_type"] = "advanced_" + str(state["advanced_plan"].get("family", "analysis"))
            result = self.remember_candidate_case(
                question=str(state.get("question", "")),
                resolved_question=str(
                    state.get("resolved_question")
                    or state.get("normalized_question")
                    or state.get("question", "")
                ),
                query_spec=query_spec,
                sql=str(state.get("validated_sql", "")),
                approval_id=str((state.get("approval_request") or {}).get("approval_id", "")),
                source="approved_query" if state.get("approval_approved") else "validated_query",
            )
            saved.append(
                {
                    "memory_id": result.record.memory_id,
                    "memory_type": "candidate_episodic",
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
        counts = self.repository.count(self.namespace)
        embedding_status = self.embedding.status()
        return "\n".join(
            [
                f"长期记忆启用: {'是' if self.enabled else '否'}",
                f"SQLite: {self.settings.db_path}",
                f"namespace: {self.namespace}",
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
def _cached_long_term_memory_service() -> LongTermMemoryService:
    return LongTermMemoryService()


def get_long_term_memory_service() -> LongTermMemoryService:
    service = _cached_long_term_memory_service()
    # The service is cached for embeddings/SQLite handles, while the current
    # Profile is part of its namespace. Seed generic procedural safeguards for
    # each Profile on first use without cross-domain retrieval.
    service.ensure_default_memories()
    return service
