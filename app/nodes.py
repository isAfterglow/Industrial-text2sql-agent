import re
import json
from typing import Any, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from tabulate import tabulate

from app.config import get_settings
from app.db import execute_readonly_query
from app.llm import invoke_model, model_call_log, reset_model_call_log
from app.advanced_plan import (
    advanced_plan_completion_prompt,
    advanced_plan_family_prompt,
    advanced_plan_prompt,
    compile_advanced_analysis_plan,
    parse_advanced_plan_family,
    parse_advanced_plan,
)
from app.material_plan import (
    MATERIAL_PLAN_FAMILY,
    compile_material_plan,
    is_material_plan_candidate,
    material_plan_prompt,
    parse_material_plan,
)
from app.capabilities import capability_family
from app.delivery import build_delivery_policy
from app.query_expectations import assert_query_expectation, build_query_expectation
from app.long_term_memory.service import get_long_term_memory_service
from app.result_assertions import assert_advanced_result
from app.query_enhancement import (
    augment_common_query_spec,
    compile_extended_query_sql,
    detect_unsupported_nested_topk,
    validate_compiled_extended_sql,
)
from app.memory import (
    build_deterministic_query_delta,
    build_query_delta_prompts,
    mark_current_turn_status,
    new_short_term_memory,
    parse_query_delta_response,
    record_user_turn,
    resolve_conversation_context as resolve_memory_context,
    update_short_term_memory,
)
from app.session_store import get_session_memory_store
from app.schema import (
    build_compact_sql_context,
    build_query_spec,
    compile_query_spec_sql,
    build_question_field_hint,
    build_robust_schema_linking,
    build_schema_context,
    extract_requested_sample_ids,
    extract_sql_schema_elements,
    get_schema_catalog,
    get_column_owner_map,
    infer_requested_output_columns,
    match_question_semantic_columns,
    infer_relevant_tables,
    normalize_question_sample_ids,
    resolve_profile,
    requires_llm_query_planning,
    set_active_profile,
)
from app.sql_guard import (
    assess_deterministic_semantic_coverage,
    build_sql_review_summary,
    clean_llm_sql,
    normalize_sample_id_literals,
    validate_and_normalize_sql,
    validate_question_policy,
)
from app.state import Text2SQLState




def effective_question(state: Text2SQLState) -> str:
    """返回已经结合短期记忆消解后的本轮完整问题。"""

    return (
        state.get("resolved_question")
        or state.get("memory_augmented_question")
        or state.get("normalized_question")
        or state.get("question", "")
    )


def resolved_query_spec(state: Text2SQLState, question: str) -> dict[str, Any]:
    """Return the context-resolved QuerySpec, building it only as a fallback."""

    if state.get("domain_profile") != "resin":
        return build_query_spec(question)

    spec = state.get("resolved_query_spec")
    if isinstance(spec, dict) and spec:
        return spec
    return augment_common_query_spec(
        question,
        build_query_spec(question),
        state.get("query_delta", {}),
    )

def message_content_to_text(content: Any) -> str:
    """兼容常见OpenAI-compatible接口的content格式。"""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)

        if parts:
            return "\n".join(parts)

    return str(content)


def invoke_text(
    system_prompt: str,
    user_prompt: str,
    *,
    purpose: str = "planning",
    repair_attempt: int = 0,
) -> str:
    return invoke_model(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        purpose=purpose,
        repair_attempt=repair_attempt,
    )


def failure_event(
    stage: str,
    error: str,
    error_type: str,
    repairable: bool,
) -> dict[str, Any]:
    """Emit a stable failure taxonomy for traces and evaluation reports."""

    normalized_type = error_type or "unknown"
    if stage == "plan":
        category = "plan_contract"
    elif normalized_type == "policy":
        category = "policy"
    elif stage == "review":
        category = "semantic_review"
    elif stage == "execution":
        category = "database_execution"
    elif normalized_type in {"syntax", "schema", "generation"}:
        category = f"guard_{normalized_type}"
    else:
        category = f"validation_{normalized_type}"
    return {
        "stage": stage,
        "category": category,
        "error_type": normalized_type,
        "repairable": repairable,
        "message": error[:500],
    }


def parse_review_line(
    text: str,
) -> tuple[bool | None, str]:
    """只接受一条PASS:原因或FAIL:原因。"""

    cleaned = text.strip()
    code_block = re.search(
        r"```(?:text)?\s*(.*?)```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if code_block:
        cleaned = code_block.group(1).strip()

    status_matches = list(
        re.finditer(
            r"(?:^|\n)\s*(PASS|FAIL)\s*(?::|：)\s*([^\n]+)",
            cleaned,
            flags=re.IGNORECASE,
        )
    )

    statuses = {
        match.group(1).upper()
        for match in status_matches
    }

    if len(status_matches) != 1 or len(statuses) != 1:
        return None, "审查器没有返回唯一可信的PASS/FAIL结论。"

    match = status_matches[0]
    reason = match.group(2).strip()

    if not reason:
        reason = "未提供具体原因。"

    return (
        match.group(1).upper() == "PASS",
        reason,
    )


def review_complex_sql(
    question: str,
    schema_context: str,
    sql: str,
) -> tuple[bool | None, str]:
    """仅审查确定性规则无法充分覆盖的复杂SQL。"""

    summary = build_sql_review_summary(sql)

    system_prompt = """
你只负责审查一条复杂SQL是否满足用户问题。

优先核对：
1. SELECT实际返回字段；
2. WHERE实际过滤字段和值；
3. ORDER BY实际排序字段和方向；
4. 聚合函数与GROUP BY；
5. 是否因多余的一对多JOIN产生重复行。

不要因为SQL写法不够简洁而判错。
不要把ORDER BY字段误认为SELECT返回字段。
不得要求增加用户问题中没有出现的数值、过滤条件或业务约束。
如果SQL已经完整包含用户明确给出的条件，不得自行补充“>0”等条件。

只能输出一行：
PASS: 简短原因
或
FAIL: 简短且具体的原因
""".strip()

    user_prompt = f"""
数据库Schema：
{schema_context}

用户问题：
{question}

候选SQL：
{sql}

{summary}
""".strip()

    result = parse_review_line(
        invoke_text(
            system_prompt,
            user_prompt,
        )
    )

    if result[0] is not None:
        return result

    retry = invoke_text(
        "只能输出一行PASS:原因或FAIL:原因。",
        f"用户问题：{question}\nSQL：{sql}",
    )
    return parse_review_line(retry)


def load_schema(
    state: Text2SQLState,
) -> dict[str, Any]:
    reset_model_call_log()
    profile = resolve_profile(state["question"])
    set_active_profile(profile)
    return {
        "domain_profile": profile,
        "normalized_question": (
            normalize_question_sample_ids(
                state["question"]
            )
        ),
        "schema_context": build_schema_context(),
        "memory_augmented_question": "",
        "long_term_memory_enabled": False,
        "semantic_memory_matches": [],
        "semantic_memory_applied_ids": [],
        "semantic_memory_hint": "",
        "episodic_memory_matches": [],
        "few_shot_context": "",
        "few_shot_retrieval_diagnostics": {},
        "query_signature": {},
        "unsupported_query": False,
        "unsupported_query_reason": "",
        "unsupported_query_suggestions": [],
        "procedural_memory_matches": [],
        "procedural_memory_context": "",
        "long_term_memory_retrieval_summary": {},
        "long_term_memory_write_summary": {},
        "resolved_question": "",
        "resolved_query_spec": {},
        "session_id": state.get("session_id", ""),
        "conversation_memory": state.get("conversation_memory", {}),
        "query_delta": {},
        "query_delta_source": "",
        "query_delta_llm_called": False,
        "query_delta_llm_raw_output": "",
        "turn_type": "new_query",
        "memory_used": False,
        "context_resolution": {},
        "context_resolution_valid": True,
        "clarification_required": False,
        "clarification_cancelled": False,
        "clarification_question": "",
        "pending_clarification": {},
        "policy_precheck_failed": False,
        "current_turn_coverage": {},
        "inherited_fields": [],
        "overridden_fields": [],
        "memory_update_summary": {},
        "session_store_summary": {},
        "query_spec": {},
        "query_plan_mode": "",
        "query_plan_reason": "",
        "capability_family": "",
        "delivery_policy": {},
        "advanced_plan": {},
        "advanced_plan_raw": "",
        "advanced_plan_error": "",
        "query_expectation": {},
        "deterministic_sql": "",
        "full_schema_context": "",
        "full_generator_raw_output": "",
        "full_sql": "",
        "forward_schema_tables": [],
        "forward_schema_columns": [],
        "backward_schema_tables": [],
        "backward_schema_columns": [],
        "accepted_backward_tables": [],
        "rejected_backward_tables": [],
        "robust_schema_context": "",
        "robust_schema_tables": [],
        "robust_schema_columns": [],
        "pruned_generator_raw_output": "",
        "pruned_sql": "",
        "candidate_full_valid": False,
        "candidate_full_normalized_sql": "",
        "candidate_full_error": "",
        "candidate_full_error_type": "",
        "candidate_full_score": 0.0,
        "candidate_pruned_valid": False,
        "candidate_pruned_normalized_sql": "",
        "candidate_pruned_error": "",
        "candidate_pruned_error_type": "",
        "candidate_pruned_score": 0.0,
        "selected_candidate": "",
        "candidate_selection_reason": "",
        "generation_schema_context": "",
        "generation_relevant_tables": [],
        "field_hint": "",
        "generator_raw_output": "",
        "initial_sql": "",
        "raw_sql": "",
        "validated_sql": "",
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
        "result_assertion": {},
        "result_assertion_passed": True,
        "approval_required": False,
        "approval_request": state.get("approval_request", {}),
        "approval_decision": state.get("approval_decision", {}),
        "approval_approved": False,
        "approval_summary": {},
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "retry_count": 0,
        "last_repair_reason": "",
        "repair_source": "",
        "repair_action": "",
        "repair_bad_sql": "",
        "repair_raw_output": "",
        "repair_model_role": "",
        "repair_plan_mode": "",
        "failure_events": [],
        "model_calls": [],
        "final_status": "",
        "final_answer": "",
    }


def hydrate_session_memory(state: Text2SQLState) -> dict[str, Any]:
    """Hydrate only when the caller did not provide authoritative memory."""

    supplied = state.get("conversation_memory", {})
    session_id = str(state.get("session_id", ""))
    if supplied:
        return {"session_store_summary": {"loaded": False, "reason": "caller_memory"}}
    if not session_id:
        return {"session_store_summary": {"loaded": False, "reason": "no_session_id"}}
    try:
        stored = get_session_memory_store().load(session_id, state.get("domain_profile", ""))
        return {
            "conversation_memory": stored or new_short_term_memory(session_id),
            "session_store_summary": {
                "loaded": bool(stored),
                "backend": get_session_memory_store().backend,
                "profile": state.get("domain_profile", ""),
            },
        }
    except Exception as exc:
        return {"session_store_summary": {"loaded": False, "error": f"{type(exc).__name__}: {exc}"}}


def identify_query_intent(state: Text2SQLState) -> dict[str, Any]:
    """Classify the request without invoking an LLM or changing the SQL plan.

    The classification is an observable decision signal for routing, evaluation,
    and a future approval workflow. SQL planning remains the source of truth.
    """
    question = state.get("normalized_question") or state.get("question", "")
    policy_error = validate_question_policy(question)
    matches = match_question_semantic_columns(question)
    requested = infer_requested_output_columns(question)
    owners = get_column_owner_map()
    related_tables = set()
    for column in set(matches) | requested:
        related_tables.update(owners.get(column, set()))
    has_aggregation = bool(re.search(r"统计|总(?:计|量)|合计|平均|均值|数量|多少|计数|COUNT", question, re.I))
    has_ranking = bool(re.search(r"最高|最低|最大|最小|top\s*\d+|bottom\s*\d+|前\s*\d+", question, re.I))
    has_grouping = bool(re.search(r"按|每(?:个|月|年|天|小时|种)|不同(?:负荷|类型)|各(?:负荷|类型)|比较.+和|(?:平日|工作日).*(?:周末)|每月.*各", question))
    has_explicit_time_filter = bool(
        re.search(r"(?:20\d{2}年|\d{1,2}月|\d{1,2}点).*(?:的|到|至|之间)|(?:第一|第[一二三四])季度", question)
    )

    if policy_error:
        intent, confidence, evidence = "unsafe_request", 1.0, ["policy_precheck"]
    elif re.search(r"^(再|这些|它们|其中|同样|取消)", question.strip()):
        intent, confidence, evidence = "follow_up", 0.95, ["conversation_reference"]
    elif has_ranking:
        intent, confidence, evidence = "topk", 0.98, ["ranking_cue"]
    elif has_explicit_time_filter and not has_grouping:
        intent, confidence, evidence = "time_filter", 0.92, ["time_constraint"]
    elif has_aggregation:
        if has_grouping:
            intent, confidence, evidence = "group_by", 0.96, ["aggregation", "grouping_cue"]
        else:
            intent, confidence, evidence = "aggregate", 0.96, ["aggregation"]
    elif has_grouping:
        intent, confidence, evidence = "group_by", 0.78, ["grouping_cue"]
    elif len(related_tables) > 1:
        intent, confidence, evidence = "cross_table", 0.88, ["multiple_field_owners"]
    elif re.search(r"年|月|日|小时|工作日|周末|时间|日期|recorded_at", question, re.I):
        intent, confidence, evidence = "time_filter", 0.80, ["time_cue"]
    elif matches or re.search(r"查询|查看|列出|显示|给我", question):
        intent, confidence, evidence = "lookup", 0.82, ["lookup_cue"]
    else:
        intent, confidence, evidence = "ambiguous", 0.60, ["no_reliable_schema_signal"]

    return {
        "query_intent": intent,
        "intent_confidence": confidence,
        "intent_evidence": evidence,
        "intent_related_tables": sorted(related_tables),
    }



def policy_precheck(state: Text2SQLState) -> dict[str, Any]:
    """在记忆解析前拒绝写入/破坏性请求，避免数字被误作LIMIT。"""

    question = state.get("normalized_question") or state.get("question", "")
    error = validate_question_policy(question)
    if error is None:
        return {"policy_precheck_failed": False}

    memory = record_user_turn(state.get("conversation_memory", {}), state.get("question", question))
    memory = mark_current_turn_status(
        memory,
        question=state.get("question", question),
        status="policy_rejected",
        resolved_question=question,
    )
    return {
        "policy_precheck_failed": True,
        "validation_error": error,
        "validation_error_type": "policy",
        "validation_repairable": False,
        "conversation_memory": memory,
    }


def route_after_policy_precheck(state: Text2SQLState) -> Literal["continue", "error"]:
    return "error" if state.get("policy_precheck_failed") else "continue"


def retrieve_semantic_memory(
    state: Text2SQLState,
) -> dict[str, Any]:
    """按需检索语义记忆，并在QueryDelta提取前做可审计术语改写。"""

    question = state.get("normalized_question") or state.get("question", "")
    try:
        service = get_long_term_memory_service()
        if not service.enabled:
            return {
                "long_term_memory_enabled": False,
                "memory_augmented_question": question,
            }

        preliminary_spec = build_query_spec(question)
        memories = service.retrieve_semantic(
            question,
            force_vector=not bool(preliminary_spec.get("eligible")),
        )
        augmented, hints, applied_ids = service.apply_semantic_memories(
            question,
            memories,
        )
        public_matches = [record.to_public_dict() for record in memories]
        return {
            "long_term_memory_enabled": True,
            "memory_augmented_question": augmented,
            "semantic_memory_matches": public_matches,
            "semantic_memory_applied_ids": applied_ids,
            "semantic_memory_hint": "\n".join(hints),
            "long_term_memory_retrieval_summary": {
                "semantic_count": len(memories),
                "semantic_applied_count": len(applied_ids),
                "question_rewritten": augmented != question,
            },
        }
    except Exception as exc:
        # 长期记忆是增强层，故障不能阻断主Text2SQL链路。
        return {
            "long_term_memory_enabled": False,
            "memory_augmented_question": question,
            "long_term_memory_retrieval_summary": {
                "semantic_error": f"{type(exc).__name__}: {exc}",
            },
        }


def extract_query_delta(
    state: Text2SQLState,
) -> dict[str, Any]:
    """提取当前轮相对上一成功QuerySpec的变化。

    常见承接表达走确定性提取；只有无法判断“继续什么/怎么改”的
    模糊短句才调用一次轻量LLM，且LLM只输出QueryDelta，不生成SQL。
    """

    question = (
        state.get("memory_augmented_question")
        or state.get("normalized_question")
        or state.get("question", "")
    )
    memory = state.get("conversation_memory", {})
    delta = build_deterministic_query_delta(question, memory)

    llm_called = False
    llm_raw_output = ""
    if delta.get("needs_llm"):
        system_prompt, user_prompt = build_query_delta_prompts(
            question,
            memory,
            delta,
        )
        try:
            llm_called = True
            llm_raw_output = invoke_text(system_prompt, user_prompt)
            delta = parse_query_delta_response(llm_raw_output, delta)
        except Exception as exc:
            delta = dict(delta)
            delta["source"] = "deterministic_fallback"
            delta["needs_llm"] = False
            delta["llm_error"] = f"{type(exc).__name__}: {exc}"

    # 原始对话窗口与最后一次成功状态分离。即使后续SQL失败，
    # 当前用户输入也会保留在最近两轮文本中，用于下一轮指代解析。
    turn_memory = record_user_turn(memory, state.get("question", question))
    return {
        "conversation_memory": turn_memory,
        "query_delta": delta,
        "query_delta_source": str(delta.get("source", "deterministic")),
        "query_delta_llm_called": llm_called,
        "query_delta_llm_raw_output": llm_raw_output,
    }


def resolve_conversation_context(
    state: Text2SQLState,
) -> dict[str, Any]:
    """消解多轮指代，并生成可供现有规划器使用的完整问题与QuerySpec。"""

    original_question = (
        state.get("memory_augmented_question")
        or state.get("normalized_question")
        or state.get("question", "")
    )
    # 安全策略永远优先于记忆继承。禁止把“删除这些样本”等
    # 越权承接表达重写成上一轮的只读SELECT。
    if validate_question_policy(original_question) is not None:
        return {
            "resolved_question": original_question,
            "resolved_query_spec": build_query_spec(original_question),
            "turn_type": "new_query",
            "memory_used": False,
            "context_resolution": {
                "reason": "当前问题触发安全策略，不使用历史记忆改写。"
            },
            "context_resolution_valid": True,
            "current_turn_coverage": {"passed": True, "mode": "policy"},
            "inherited_fields": [],
            "overridden_fields": [],
        }

    # Requests such as "the best samples" lack a measurable business target.
    # They must be clarified before memory, planning, or repair can invent one.
    if re.search(r"(?:最好|最佳|最优|合适|优秀).{0,8}(?:样本|材料|记录)?", original_question):
        return {
            "resolved_question": original_question,
            "resolved_query_spec": {},
            "turn_type": "clarification_required",
            "memory_used": False,
            "context_resolution": {"reason": "missing_rank_metric"},
            "context_resolution_valid": False,
            "clarification_required": True,
            "clarification_cancelled": False,
            "clarification_question": "“最好”缺少可计算指标。请说明按原始密度、热解热、导热率、温度响应或其它字段排序。",
            "pending_clarification": {"reason": "missing_rank_metric", "original_question": original_question},
            "current_turn_coverage": {"passed": False, "mode": "missing_rank_metric"},
            "inherited_fields": [],
            "overridden_fields": [],
        }

    query_delta = state.get("query_delta", {})

    # The existing deterministic resolver encodes resin-specific field and
    # sample conventions. Applying its "unknown field" clarification to a
    # newly onboarded Profile blocks legitimate advanced requests before the
    # Profile-aware planner can inspect them. Independent non-resin questions
    # therefore enter their own schema/planning path directly.
    if (
        state.get("domain_profile") != "resin"
        and not query_delta.get("explicit_reference")
        and query_delta.get("dependency", "independent") == "independent"
    ):
        direct_spec = build_query_spec(original_question)
        return {
            "conversation_memory": state.get("conversation_memory", {}),
            "resolved_question": original_question,
            "resolved_query_spec": direct_spec,
            "turn_type": "new_query",
            "memory_used": False,
            "context_resolution": {"reason": "non_resin_independent_profile_guard"},
            "context_resolution_valid": True,
            "clarification_required": False,
            "clarification_cancelled": False,
            "clarification_question": "",
            "pending_clarification": {},
            "current_turn_coverage": {"passed": True, "mode": "non_resin_independent"},
            "inherited_fields": [],
            "overridden_fields": [],
        }

    # 能力边界应基于原始问题、且早于历史状态合并判断。
    # 一个复杂查询可能暂时不支持执行，但仍然是完整独立的新查询；
    # 不能因为QuerySpec暂时为complex_or_uncertain就继承上一轮过滤和范围。
    raw_capability = detect_unsupported_nested_topk(original_question)
    if raw_capability.get("unsupported"):
        authoritative_spec = augment_common_query_spec(
            original_question,
            query_delta.get("current_spec") or build_query_spec(original_question),
            query_delta,
        )
        memory = state.get("conversation_memory", {})
        return {
            "conversation_memory": memory,
            "resolved_question": original_question,
            "resolved_query_spec": authoritative_spec,
            "turn_type": "new_query",
            "memory_used": False,
            "context_resolution": {
                "reason": (
                    "原始问题包含完整的多阶段Top-K结构，按独立新查询处理；"
                    "当前版本将在规划阶段给出多轮拆分建议。"
                ),
                "capability_check": raw_capability,
            },
            "context_resolution_valid": True,
            "clarification_required": False,
            "clarification_cancelled": False,
            "clarification_question": "",
            "pending_clarification": {},
            "current_turn_coverage": {
                "passed": True,
                "mode": "independent_unsupported_guard",
                "historical_scope_leak": False,
                "sample_ids": [],
            },
            "inherited_fields": [],
            "overridden_fields": [],
        }

    resolved = resolve_memory_context(
        original_question,
        state.get("conversation_memory", {}),
        query_delta,
    )

    # A complete, high-confidence independent QuerySpec is stronger evidence
    # than generic conversational wording such as "同时返回它们".  Do not ask
    # for clarification when the Profile can already compile the request.
    direct_spec = query_delta.get("current_spec") or build_query_spec(original_question)
    if (
        resolved.get("clarification_required")
        and direct_spec.get("eligible")
        and not query_delta.get("explicit_reference")
        and (
            query_delta.get("independent_complete")
            or not state.get("conversation_memory", {}).get("last_successful_query_state")
        )
    ):
        resolved = {
            **resolved,
            "resolved_question": original_question,
            "resolved_query_spec": direct_spec,
            "turn_type": "new_query",
            "memory_used": False,
            "context_resolution_valid": True,
            "clarification_required": False,
            "clarification_cancelled": False,
            "clarification_question": "",
            "pending_clarification": {},
            "current_turn_coverage": {"passed": True, "mode": "independent_high_confidence"},
            "context_resolution": {"reason": "complete_independent_queryspec_overrides_generic_clarification"},
        }

    # 历史范围防泄漏：完整且无显式指代的新查询以当前QuerySpec为准。
    if (
        query_delta.get("independent_complete")
        and not query_delta.get("explicit_reference")
    ):
        authoritative_spec = (
            query_delta.get("current_spec")
            or build_query_spec(original_question)
        )
        resolved = {
            **resolved,
            "resolved_question": original_question,
            "resolved_query_spec": authoritative_spec,
            "turn_type": "new_query",
            "memory_used": False,
            "context_resolution_valid": True,
            "current_turn_coverage": {
                "passed": True,
                "mode": "independent_scope_guard",
                "historical_scope_leak": False,
                "sample_ids": list(authoritative_spec.get("sample_ids", [])),
            },
            "context_resolution": {
                **resolved.get("context_resolution", {}),
                "reason": (
                    "当前轮可独立形成完整QuerySpec且没有显式指代，"
                    "强制清空历史样本范围。"
                ),
            },
            "inherited_fields": [],
            "overridden_fields": [],
            "clarification_required": False,
            "clarification_cancelled": False,
            "clarification_question": "",
            "pending_clarification": {},
        }

    memory = state.get("conversation_memory", {})
    if resolved.get("clarification_required"):
        memory = dict(memory)
        memory["pending_clarification"] = resolved.get("pending_clarification", {})
        memory = mark_current_turn_status(
            memory,
            question=state.get("question", original_question),
            status=(
                "clarification_cancelled"
                if resolved.get("clarification_cancelled")
                else "clarification_required"
            ),
            resolved_question=original_question,
        )
    elif (
        state.get("query_delta", {}).get("clarification_resolved")
        or state.get("query_delta", {}).get("clear_pending_clarification")
    ):
        memory = dict(memory)
        memory["pending_clarification"] = {}

    resolved_question = resolved.get("resolved_question", "") or original_question
    if state.get("domain_profile") == "resin":
        resolved_spec = augment_common_query_spec(
            original_question,
            resolved.get("resolved_query_spec", {}),
            query_delta,
        )
    else:
        # The resin enhancement layer encodes temporal/sample conventions that
        # do not apply to normalized fact/dimension Profiles.
        resolved_spec = build_query_spec(resolved_question)

    return {
        "conversation_memory": memory,
        "resolved_question": resolved_question,
        "resolved_query_spec": resolved_spec,
        "turn_type": resolved.get("turn_type", "new_query"),
        "memory_used": bool(resolved.get("memory_used", False)),
        "context_resolution": resolved.get("context_resolution", {}),
        "context_resolution_valid": bool(
            resolved.get("context_resolution_valid", True)
        ),
        "clarification_required": bool(resolved.get("clarification_required", False)),
        "clarification_cancelled": bool(resolved.get("clarification_cancelled", False)),
        "clarification_question": resolved.get("clarification_question", ""),
        "pending_clarification": resolved.get("pending_clarification", {}),
        "current_turn_coverage": resolved.get("current_turn_coverage", {}),
        "inherited_fields": resolved.get("inherited_fields", []),
        "overridden_fields": resolved.get("overridden_fields", []),
    }



def route_after_context_resolution(state: Text2SQLState) -> Literal["clarify", "continue"]:
    return "clarify" if state.get("clarification_required") else "continue"


def request_clarification(state: Text2SQLState) -> dict[str, Any]:
    question = state.get("clarification_question") or "请补充必要信息后再继续查询。"
    cancelled = bool(state.get("clarification_cancelled"))
    return {
        "final_status": "clarification_cancelled" if cancelled else "clarification_required",
        "final_answer": (
            question
            if cancelled
            else "需要补充确认后才能安全执行。\n\n" + question
        ),
    }


def retrieve_few_shot_memory(
    state: Text2SQLState,
) -> dict[str, Any]:
    """BGE-M3粗召回后使用QuerySignature硬过滤、结构重排和MMR选例。"""

    try:
        service = get_long_term_memory_service()
        question = effective_question(state)
        query_spec = resolved_query_spec(state, question)
        capability = query_spec.get("capability_check", {})
        if capability.get("unsupported"):
            diagnostics = {
                "candidate_count": 0,
                "hard_compatible_count": 0,
                "selected_count": 0,
                "few_shot_used": False,
                "skip_reason": "unsupported_nested_topk",
            }
            summary = dict(state.get("long_term_memory_retrieval_summary", {}))
            summary.update({"episodic_count": 0, **diagnostics})
            return {
                "episodic_memory_matches": [],
                "few_shot_context": "",
                "few_shot_retrieval_diagnostics": diagnostics,
                "long_term_memory_retrieval_summary": summary,
            }

        if not service.enabled or not service.should_retrieve_episodic(query_spec):
            diagnostics = {
                "candidate_count": 0,
                "hard_compatible_count": 0,
                "selected_count": 0,
                "few_shot_used": False,
                "skip_reason": "simple_or_deterministic_query",
            }
            return {
                "episodic_memory_matches": [],
                "few_shot_context": "",
                "few_shot_retrieval_diagnostics": diagnostics,
            }

        memories, diagnostics = service.retrieve_episodic_with_diagnostics(
            question,
            query_spec,
        )
        context = service.build_few_shot_context(memories)
        summary = dict(state.get("long_term_memory_retrieval_summary", {}))
        summary.update(
            {
                "episodic_count": len(memories),
                "few_shot_used": bool(memories),
                "few_shot_skip_reason": diagnostics.get("skip_reason", ""),
                "episodic_candidate_count": diagnostics.get("candidate_count", 0),
                "episodic_structural_count": diagnostics.get("hard_compatible_count", 0),
            }
        )
        return {
            "episodic_memory_matches": [record.to_public_dict() for record in memories],
            "few_shot_context": context,
            "few_shot_retrieval_diagnostics": diagnostics,
            "query_signature": diagnostics.get("query_signature", {}),
            "long_term_memory_retrieval_summary": summary,
        }
    except Exception as exc:
        summary = dict(state.get("long_term_memory_retrieval_summary", {}))
        summary["episodic_error"] = f"{type(exc).__name__}: {exc}"
        return {
            "episodic_memory_matches": [],
            "few_shot_context": "",
            "few_shot_retrieval_diagnostics": {
                "few_shot_used": False,
                "skip_reason": "retrieval_error",
            },
            "long_term_memory_retrieval_summary": summary,
        }


def build_query_plan(
    state: Text2SQLState,
) -> dict[str, Any]:
    """构建查询计划；常见时序派生指标走确定性扩展路径。"""

    question = effective_question(state)
    spec = resolved_query_spec(state, question)
    capability = spec.get("capability_check", {})
    if capability.get("unsupported"):
        return {
            "query_spec": spec,
            "query_plan_mode": "unsupported",
            "query_plan_reason": capability.get("reason", "当前问题超出单轮能力边界。"),
            "deterministic_sql": "",
            "unsupported_query": True,
            "unsupported_query_reason": capability.get("reason", ""),
            "unsupported_query_suggestions": capability.get("suggested_turns", []),
            "capability_family": "unsupported",
        }

    if spec.get("mode") == "deterministic_extended":
        deterministic_sql = compile_extended_query_sql(spec)
        return {
            "query_spec": spec,
            "query_plan_mode": "deterministic_extended",
            "query_plan_reason": spec.get("reason", "确定性扩展查询快路径。"),
            "deterministic_sql": deterministic_sql,
            "unsupported_query": False,
            "capability_family": capability_family(state.get("domain_profile", "resin"), str(spec.get("query_type", ""))),
        }

    deterministic_sql = compile_query_spec_sql(spec) if spec.get("eligible") else ""
    return {
        "query_spec": spec,
        "query_plan_mode": spec.get("mode", "rsl"),
        "query_plan_reason": spec.get("reason", ""),
        "deterministic_sql": deterministic_sql,
        "unsupported_query": False,
        "capability_family": capability_family(state.get("domain_profile", "resin"), str(spec.get("query_type", ""))),
    }


def route_after_query_plan(
    state: Text2SQLState,
) -> Literal["simple", "rsl", "unsupported"]:
    if state.get("query_plan_mode") == "unsupported":
        return "unsupported"
    if (
        state.get("query_plan_mode") in {"deterministic", "deterministic_extended"}
        and state.get("deterministic_sql")
    ):
        return "simple"
    return "rsl"


def generate_structured_query_spec(state: Text2SQLState) -> dict[str, Any]:
    """Ask the LLM for a constrained QuerySpec before allowing free-form SQL."""

    question = effective_question(state)
    if state.get("domain_profile") == "resin" and is_material_plan_candidate(question):
        service = get_long_term_memory_service()
        try:
            examples, diagnostics = service.retrieve_advanced_plan_examples(
                question, MATERIAL_PLAN_FAMILY
            )
            context = service.build_advanced_plan_few_shot_context(examples)
            raw = invoke_text(
                "Complete one constrained MaterialAnalysisPlan JSON object, never SQL.",
                material_plan_prompt(state["schema_context"], question, context),
                purpose="planning",
            )
            plan = parse_material_plan(clean_llm_sql(raw))
            sql = compile_material_plan(plan)
            memory_summary = dict(state.get("long_term_memory_retrieval_summary", {}))
            memory_summary["advanced_plan"] = {
                "family": MATERIAL_PLAN_FAMILY,
                "selected_count": len(examples),
                "memory_ids": [item.memory_id for item in examples],
                "stage": "3b_material_completion",
            }
            return {
                "advanced_plan_raw": raw,
                "advanced_plan": plan,
                "advanced_plan_error": "",
                "query_expectation": build_query_expectation(question, plan),
                "query_plan_mode": "advanced_analysis_plan",
                "query_plan_reason": "3B结合材料Profile生成受限时序聚合计划。",
                "deterministic_sql": sql,
                "advanced_plan_family": MATERIAL_PLAN_FAMILY,
                "advanced_plan_memory_matches": [item.to_public_dict() for item in examples],
                "advanced_plan_memory_diagnostics": diagnostics,
                "long_term_memory_retrieval_summary": memory_summary,
            }
        except Exception as exc:
            return {
                "advanced_plan_raw": locals().get("raw", ""),
                "advanced_plan_family": MATERIAL_PLAN_FAMILY,
                "advanced_plan_error": f"{type(exc).__name__}: {exc}",
                "failure_events": [failure_event("plan", f"{type(exc).__name__}: {exc}", "contract", True)],
                "deterministic_sql": "",
            }

    if requires_llm_query_planning(question):
        raw = invoke_text(
            "You classify one analytical query family. Return JSON only.",
            advanced_plan_family_prompt(state["schema_context"], question),
        )
        try:
            family = parse_advanced_plan_family(clean_llm_sql(raw))
            service = get_long_term_memory_service()
            examples, diagnostics = service.retrieve_advanced_plan_examples(question, family)
            example_context = service.build_advanced_plan_few_shot_context(examples)
            completed_raw = invoke_text(
                "Complete one constrained AdvancedAnalysisPlan JSON object, never SQL.",
                advanced_plan_completion_prompt(
                    state["schema_context"], question, family, example_context
                ),
                # 3B first: a validated plan executes immediately; expensive
                # completion is an escalation path, not the default.
                purpose="planning",
            )
            plan = parse_advanced_plan(clean_llm_sql(completed_raw))
            if plan["family"] != family:
                raise ValueError("completed plan family differs from the 3B classification")
            sql = compile_advanced_analysis_plan(plan)
            memory_summary = dict(state.get("long_term_memory_retrieval_summary", {}))
            memory_summary["advanced_plan"] = {
                "family": family, "selected_count": len(examples),
                "memory_ids": [item.memory_id for item in examples], "stage": "3b_completion",
            }
            return {
                "advanced_plan_raw": completed_raw,
                "advanced_plan": plan,
                "advanced_plan_error": "",
                "query_expectation": build_query_expectation(question, plan),
                "query_plan_mode": "advanced_analysis_plan",
                "query_plan_reason": "3B选择分析族并结合正式案例补全受限计划。",
                "deterministic_sql": sql,
                "advanced_plan_family": family,
                "advanced_plan_memory_matches": [item.to_public_dict() for item in examples],
                "advanced_plan_memory_diagnostics": diagnostics,
                "long_term_memory_retrieval_summary": memory_summary,
            }
        except Exception as exc:
            return {
                "advanced_plan_raw": raw,
                "advanced_plan_family": locals().get("family", ""),
                "advanced_plan_memory_matches": [item.to_public_dict() for item in locals().get("examples", [])],
                "advanced_plan_memory_diagnostics": locals().get("diagnostics", {}),
                "advanced_plan_error": f"{type(exc).__name__}: {exc}",
                "failure_events": [
                    failure_event(
                        "plan",
                        f"{type(exc).__name__}: {exc}",
                        "contract",
                        True,
                    )
                ],
                "deterministic_sql": "",
            }

    prompt = (
        "Return JSON only: {\"query_spec\": {...}}. Use only schema fields. "
        "Supported query_type: single_table_filter, single_table_topk, exact_sample, "
        "response_detail, one_to_one_join, per_sample_temporal_aggregate. "
        "Include eligible=true, table, select_columns, filters, order_by, limit, sample_ids as applicable.\n"
        f"Schema:\n{state['schema_context']}\n"
        f"{state.get('few_shot_context', '')}\n"
        f"Question:\n{question}"
    )
    raw = invoke_text("You produce validated Text2SQL QuerySpec JSON, never SQL.", prompt)
    try:
        payload = json.loads(clean_llm_sql(raw).strip().removeprefix("```json").removesuffix("```").strip())
        spec = payload.get("query_spec", payload)
        if not isinstance(spec, dict) or not spec.get("eligible"):
            raise ValueError("missing eligible QuerySpec")
        allowed = {"single_table_filter", "single_table_topk", "exact_sample", "response_detail", "one_to_one_join", "per_sample_temporal_aggregate"}
        if spec.get("query_type") not in allowed:
            raise ValueError("unsupported QuerySpec type")
        sql = compile_query_spec_sql(spec)
        if not sql:
            raise ValueError("QuerySpec cannot compile")
        return {"query_spec_json_raw": raw, "query_spec": spec, "query_plan_mode": "llm_query_spec", "query_plan_reason": "LLM constrained QuerySpec compiled successfully.", "deterministic_sql": sql}
    except Exception as exc:
        return {"query_spec_json_raw": raw, "query_spec_json_error": f"{type(exc).__name__}: {exc}", "deterministic_sql": ""}


def route_after_structured_query_spec(state: Text2SQLState) -> Literal["structured", "sql", "regenerate"]:
    if state.get("advanced_plan_family") and state.get("advanced_plan_error"):
        return "regenerate"
    return "structured" if state.get("deterministic_sql") else "sql"


def regenerate_advanced_plan(state: Text2SQLState) -> dict[str, Any]:
    """Escalate an invalid 3B plan to a strong model before free SQL fallback."""

    question = effective_question(state)
    prior = state.get("advanced_plan_raw", "")
    error = state.get("advanced_plan_error", "invalid advanced plan")
    family = str(state.get("advanced_plan_family", ""))
    service = get_long_term_memory_service()
    examples, diagnostics = (
        service.retrieve_advanced_plan_examples(question, family)
        if family else ([], {"family": "", "selected_count": 0, "skip_reason": "family_unknown"})
    )
    context = service.build_advanced_plan_few_shot_context(examples)
    is_material = family == MATERIAL_PLAN_FAMILY
    base_prompt = (
        material_plan_prompt(state["schema_context"], question, context)
        if is_material
        else (
            advanced_plan_completion_prompt(state["schema_context"], question, family, context)
            if family else advanced_plan_prompt(state["schema_context"], question)
        )
    )
    prompt = f"""{base_prompt}

The prior local-model output below failed validation. Regenerate from the current question and schema.
Keep the selected family when one is supplied. Do not preserve wrong fields and do not output SQL.
Contract error: {error}
Bad output:
{prior}
"""
    failures: list[str] = []
    for repair_attempt in (2, 3):
        try:
            raw = invoke_text(
                "Return only one valid AdvancedAnalysisPlan JSON object.",
                prompt,
                purpose="repair",
                repair_attempt=repair_attempt,
            )
            plan = parse_material_plan(clean_llm_sql(raw)) if is_material else parse_advanced_plan(clean_llm_sql(raw))
            if family and plan["family"] != family:
                raise ValueError("regenerated plan family differs from selected family")
            memory_summary = dict(state.get("long_term_memory_retrieval_summary", {}))
            memory_summary["advanced_plan"] = {
                "family": plan["family"], "selected_count": len(examples),
                "memory_ids": [item.memory_id for item in examples], "stage": "escalated_regeneration",
            }
            return {
                "advanced_plan": plan,
                "advanced_plan_raw": raw,
                "advanced_plan_error": "",
                "query_expectation": build_query_expectation(question, plan),
                "query_plan_mode": "advanced_analysis_plan",
                "query_plan_reason": "3B计划失败后由升级模型结合正式案例重生成并编译。",
                "deterministic_sql": compile_material_plan(plan) if is_material else compile_advanced_analysis_plan(plan),
                "advanced_plan_family": plan["family"],
                "advanced_plan_memory_matches": [item.to_public_dict() for item in examples],
                "advanced_plan_memory_diagnostics": diagnostics,
                "long_term_memory_retrieval_summary": memory_summary,
                "retry_count": 1,
                "repair_source": "计划契约升级重生成",
                "repair_plan_mode": "regenerate_plan",
            }
        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
    reason = " | ".join(failures)
    return {
        "advanced_plan_error": reason,
        "validation_error": f"高级计划重生成失败：{reason}",
        "validation_error_type": "plan_contract",
        "validation_repairable": False,
        "deterministic_sql": "",
        "failure_events": [failure_event("plan", reason, "contract", False)],
    }


def route_after_regenerated_plan(state: Text2SQLState) -> Literal["structured", "error"]:
    return "structured" if state.get("deterministic_sql") else "error"


def generate_simple_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """将可信QuerySpec编译结果送入统一Guard。"""

    sql = state.get("deterministic_sql", "")
    selected_label = (
        "deterministic_extended"
        if state.get("query_plan_mode") == "deterministic_extended"
        else (
            "advanced_analysis_plan"
            if state.get("query_plan_mode") == "advanced_analysis_plan"
            else "deterministic"
        )
    )
    return {
        "selected_candidate": selected_label,
        "candidate_selection_reason": state.get(
            "query_plan_reason", "基础查询确定性快路径。"
        ),
        "generator_raw_output": "",
        "initial_sql": sql,
        "raw_sql": sql,
        "validated_sql": "",
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
    }


def _generation_system_prompt(
    candidate_mode: str,
) -> str:
    mode_instruction = (
        "当前提供的是完整Schema。先保证字段和表选择正确，"
        "仍然只使用回答问题所需的最少数据表。"
        if candidate_mode == "full"
        else
        "当前提供的是正向与反向Schema Linking合并后的稳健裁剪Schema。"
        "严格限制在该Schema中生成更聚焦的候选SQL。"
    )

    return f"""
你是受限Profile的Text2SQL生成器。

{mode_instruction}

生成一条MySQL只读SELECT查询。

要求：
1. 只输出SQL；
2. 使用Schema中的真实表名和真实字段；
3. 只能使用当前Schema声明的真实表名、字段和别名；
4. 严格使用提供的业务字段对应关系；
5. 只返回用户要求的字段，只使用必要数据表；
6. 不增加用户未要求的LIKE、IS NOT NULL或其他过滤；
7. 普通记录级Top-K使用ORDER BY目标字段加LIMIT，不使用无意义MAX、GROUP BY或IN子查询；
8. 仅在Schema确有样本时序粒度时，才使用sample_id和point_index规则；
9. 用户明确请求某个白名单表全部数据时，只查询该表并显式列出全部字段；
15. 禁止SELECT *、写操作和跨库查询；
16. 科学计数法是一个完整数值，例如2e-12不得拆成2和12；
17. 用户没有要求数量时，不得自行添加LIMIT 1或其他限制性LIMIT，系统会统一添加资源上限；
13. Profile未声明的领域指标、时序口径或字段不得猜测；
14. Few-shot结构与当前Schema不一致时不得复制；没有可靠结构时以Schema为准；
15. 若问题包含“前N个中再取前M个”等多阶段Top-K，不得猜测或拼接多个ORDER BY/LIMIT。
""".strip()


def _generate_candidate_sql(
    question: str,
    schema_context: str,
    field_hint: str,
    candidate_mode: str,
    few_shot_context: str = "",
) -> tuple[str, str]:
    user_prompt = f"""
数据库Schema：
{schema_context}

用户问题：
{question}

{field_hint}

{few_shot_context}

只输出一条完整SQL。
""".strip()

    raw_output = invoke_text(
        _generation_system_prompt(
            candidate_mode
        ),
        user_prompt,
    )
    cleaned_sql = normalize_sample_id_literals(
        clean_llm_sql(raw_output)
    )
    return raw_output, cleaned_sql


def generate_full_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """使用完整Schema生成第一条候选SQL。"""

    question = effective_question(state)
    if validate_question_policy(question) is not None:
        return {
            "full_schema_context": state.get(
                "schema_context", ""
            ),
            "full_generator_raw_output": "",
            "full_sql": "",
        }

    field_hint = build_question_field_hint(
        question
    )
    semantic_hint = state.get("semantic_memory_hint", "").strip()
    if semantic_hint:
        field_hint = (
            field_hint
            + "\n长期语义记忆提示：\n"
            + semantic_hint
        ).strip()
    full_context = state["schema_context"]
    raw_output, sql = _generate_candidate_sql(
        question=question,
        schema_context=full_context,
        field_hint=field_hint,
        candidate_mode="full",
        few_shot_context=state.get("few_shot_context", ""),
    )

    return {
        "field_hint": field_hint,
        "full_schema_context": full_context,
        "full_generator_raw_output": raw_output,
        "full_sql": sql,
    }


def build_robust_schema(
    state: Text2SQLState,
) -> dict[str, Any]:
    """将问题正向链接与SQL1反向链接合并为稳健裁剪Schema。"""

    linking = build_robust_schema_linking(
        question=effective_question(state),
        preliminary_sql=state.get(
            "full_sql", ""
        ),
    )

    return {
        "forward_schema_tables": linking[
            "forward_tables"
        ],
        "forward_schema_columns": linking[
            "forward_columns"
        ],
        "backward_schema_tables": linking[
            "backward_tables"
        ],
        "backward_schema_columns": linking[
            "backward_columns"
        ],
        "accepted_backward_tables": linking[
            "accepted_backward_tables"
        ],
        "rejected_backward_tables": linking[
            "rejected_backward_tables"
        ],
        "robust_schema_context": linking[
            "context"
        ],
        "robust_schema_tables": linking[
            "robust_tables"
        ],
        "robust_schema_columns": linking[
            "robust_columns"
        ],
        # 保留V0.5可观测字段兼容性
        "generation_schema_context": linking[
            "context"
        ],
        "generation_relevant_tables": linking[
            "robust_tables"
        ],
    }


def generate_pruned_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """使用稳健裁剪Schema生成第二条候选SQL。"""

    question = effective_question(state)
    if validate_question_policy(question) is not None:
        return {
            "pruned_generator_raw_output": "",
            "pruned_sql": "",
        }

    raw_output, sql = _generate_candidate_sql(
        question=question,
        schema_context=state[
            "robust_schema_context"
        ],
        field_hint=state.get(
            "field_hint", ""
        ),
        candidate_mode="pruned",
        few_shot_context=state.get("few_shot_context", ""),
    )

    return {
        "pruned_generator_raw_output": raw_output,
        "pruned_sql": sql,
    }


def _sql_complexity_metrics(
    sql: str,
) -> dict[str, int]:
    try:
        tree = sqlglot.parse_one(
            sql,
            read="mysql",
        )
    except ParseError:
        return {
            "joins": 99,
            "subqueries": 99,
            "groups": 99,
            "aggregates": 99,
            "tables": 99,
        }

    return {
        "joins": len(
            list(tree.find_all(exp.Join))
        ),
        "subqueries": len(
            list(tree.find_all(exp.Subquery))
        ),
        "groups": len(
            list(tree.find_all(exp.Group))
        ),
        "aggregates": len(
            list(tree.find_all(exp.AggFunc))
        ),
        "tables": len(
            list(tree.find_all(exp.Table))
        ),
    }


def _canonical_sql(sql: str) -> str:
    try:
        return sqlglot.parse_one(
            sql,
            read="mysql",
        ).sql(dialect="mysql")
    except ParseError:
        return sql.strip()


def _guard_error_penalty(error: str, error_type: str) -> float:
    """候选失败时按错误严重度评分，避免未知别名压过轻微冗余字段。"""

    base = {
        "policy": 1000.0,
        "generation": 260.0,
        "syntax": 240.0,
        "schema": 120.0,
        "semantic": 100.0,
        "resource": 90.0,
    }.get(error_type, 120.0)
    rules = (
        (r"未知表或派生表别名|不存在的表|未知表", 90.0),
        (r"字段归属错误|未知字段|多个来源中存在", 65.0),
        (r"排序字段错误|排序方向错误|Top-K数量不一致", 55.0),
        (r"没有返回该真实字段|缺少返回字段|没有使用该字段", 45.0),
        (r"IN子查询中使用LIMIT|multiple 'ORDER BY'", 50.0),
        (r"无关字段|冗余字段", 12.0),
    )
    penalty = base
    for pattern, weight in rules:
        penalty += len(re.findall(pattern, error, flags=re.IGNORECASE)) * weight
    return penalty


def _evaluate_candidate(
    label: str,
    sql: str,
    question: str,
    query_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    result = validate_and_normalize_sql(
        sql=sql,
        allowed_tables=set(get_schema_catalog()["tables"]),
        max_rows=settings.SQL_MAX_ROWS,
        question=question,
        query_spec=query_spec,
    )

    if not result.valid:
        severity = _guard_error_penalty(result.error, result.error_type)
        error_lines = max(1, result.error.count("\n") + 1)
        return {
            "label": label,
            "valid": False,
            "normalized_sql": "",
            "error": result.error,
            "error_type": result.error_type,
            "repairable": result.repairable,
            "score": -severity - error_lines,
            "covered": False,
            "coverage_reason": "",
            "tables": [],
            "columns": [],
            "metrics": _sql_complexity_metrics(sql),
        }

    normalized_sql = result.sql
    covered, coverage_reason = (
        assess_deterministic_semantic_coverage(
            question=question,
            sql=normalized_sql,
        )
    )
    used_tables, used_columns = (
        extract_sql_schema_elements(
            normalized_sql
        )
    )
    expected_tables = infer_relevant_tables(
        question
    )
    metrics = _sql_complexity_metrics(
        normalized_sql
    )

    score = 100.0
    if covered:
        score += 15.0

    if expected_tables:
        if used_tables == expected_tables:
            score += 10.0
        elif expected_tables.issubset(
            used_tables
        ):
            score += 4.0
        else:
            score -= 12.0 * len(
                expected_tables - used_tables
            )
        score -= 5.0 * len(
            used_tables - expected_tables
        )

    score -= 1.5 * metrics["joins"]
    score -= 3.0 * metrics["subqueries"]
    score -= 0.5 * metrics["groups"]
    score -= 0.25 * metrics["aggregates"]

    # 分数完全相同时，稳健裁剪候选作为轻微、可解释的tie-break。
    if label == "pruned":
        score += 0.1

    return {
        "label": label,
        "valid": True,
        "normalized_sql": normalized_sql,
        "error": "",
        "error_type": "",
        "repairable": True,
        "score": round(score, 3),
        "covered": covered,
        "coverage_reason": coverage_reason,
        "tables": sorted(used_tables),
        "columns": sorted(used_columns),
        "metrics": metrics,
    }


def select_sql_candidate(
    state: Text2SQLState,
) -> dict[str, Any]:
    """先用Guard淘汰，再用确定性覆盖与复杂度选择候选。"""

    question = effective_question(state)
    full_eval = _evaluate_candidate(
        label="full",
        sql=state.get("full_sql", ""),
        question=question,
        query_spec=state.get("query_spec"),
    )
    pruned_eval = _evaluate_candidate(
        label="pruned",
        sql=state.get("pruned_sql", ""),
        question=question,
        query_spec=state.get("query_spec"),
    )

    if full_eval["valid"] and not pruned_eval["valid"]:
        selected = full_eval
        reason = "仅完整Schema候选通过确定性Guard。"
    elif pruned_eval["valid"] and not full_eval["valid"]:
        selected = pruned_eval
        reason = "仅稳健裁剪Schema候选通过确定性Guard。"
    elif full_eval["valid"] and pruned_eval["valid"]:
        if _canonical_sql(
            full_eval["normalized_sql"]
        ) == _canonical_sql(
            pruned_eval["normalized_sql"]
        ):
            selected = pruned_eval
            reason = (
                "两个候选规范化后等价，选择稳健裁剪Schema候选。"
            )
        elif (
            full_eval["score"]
            > pruned_eval["score"]
        ):
            selected = full_eval
            reason = (
                "两个候选均通过Guard，完整Schema候选的"
                "确定性覆盖与结构评分更高。"
            )
        else:
            selected = pruned_eval
            reason = (
                "两个候选均通过Guard，稳健裁剪Schema候选的"
                "确定性覆盖与结构评分更高或并列。"
            )
    else:
        # 两条都失败时，只选择错误风险较低的一条进入既有一次修复。
        if full_eval["score"] > pruned_eval["score"]:
            selected = full_eval
        else:
            selected = pruned_eval
        reason = (
            "两个候选均未通过Guard，选择错误严重度较低的候选"
            "进入现有一次修复链路。"
        )

    selected_label = selected["label"]
    selected_original_sql = (
        state.get("full_sql", "")
        if selected_label == "full"
        else state.get("pruned_sql", "")
    )
    selected_sql = (
        selected["normalized_sql"]
        if selected["valid"]
        else selected_original_sql
    )

    return {
        "candidate_full_valid": full_eval["valid"],
        "candidate_full_normalized_sql": full_eval[
            "normalized_sql"
        ],
        "candidate_full_error": full_eval["error"],
        "candidate_full_error_type": full_eval[
            "error_type"
        ],
        "candidate_full_score": full_eval["score"],
        "candidate_pruned_valid": pruned_eval["valid"],
        "candidate_pruned_normalized_sql": pruned_eval[
            "normalized_sql"
        ],
        "candidate_pruned_error": pruned_eval["error"],
        "candidate_pruned_error_type": pruned_eval[
            "error_type"
        ],
        "candidate_pruned_score": pruned_eval["score"],
        "selected_candidate": selected_label,
        "candidate_selection_reason": reason,
        "generator_raw_output": (
            state.get("full_generator_raw_output", "")
            if selected_label == "full"
            else state.get("pruned_generator_raw_output", "")
        ),
        "initial_sql": selected_sql,
        "raw_sql": selected_sql,
        "validated_sql": "",
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
    }



def validate_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    settings = get_settings()

    if state.get("memory_used") and not state.get(
        "context_resolution_valid", True
    ):
        coverage = state.get("current_turn_coverage", {})
        error = "短期记忆合并未完整覆盖当前轮明确语义：" + str(coverage)
        return {
            "validation_error": error,
            "validation_repairable": False,
            "validation_error_type": "context",
            "validated_sql": "",
            "execution_error": "",
            "failure_events": [
                failure_event("validation", error, "context", False)
            ],
        }

    if state.get("query_plan_mode") == "deterministic_extended":
        valid, normalized_sql, error = validate_compiled_extended_sql(
            state.get("raw_sql", ""),
            state.get("query_spec", {}),
            set(get_schema_catalog()["tables"]),
            settings.SQL_MAX_ROWS,
        )
        if not valid:
            return {
                "validation_error": error,
                "validation_repairable": False,
                "validation_error_type": "deterministic_extended",
                "validated_sql": "",
                "execution_error": "",
                "failure_events": [
                    failure_event("validation", error, "deterministic_extended", False)
                ],
            }
        return {
            "validation_error": "",
            "validation_repairable": True,
            "validation_error_type": "",
            "validated_sql": normalized_sql,
            "execution_error": "",
        }

    # The Guard needs the compiled-plan family to preserve semantics such as
    # "top K within every group".  QuerySpec itself remains the generic
    # contract, so carry the advanced plan in an evaluation-only copy.
    guard_query_spec = dict(state.get("query_spec") or {})
    if state.get("advanced_plan"):
        guard_query_spec["advanced_plan"] = state["advanced_plan"]
        if guard_query_spec.get("limit") is None and state["advanced_plan"].get("limit") is not None:
            guard_query_spec["limit"] = state["advanced_plan"]["limit"]

    result = validate_and_normalize_sql(
        sql=state.get("raw_sql", ""),
        allowed_tables=set(get_schema_catalog()["tables"]),
        max_rows=settings.SQL_MAX_ROWS,
        question=effective_question(state),
        query_spec=guard_query_spec,
    )

    if not result.valid:
        return {
            "validation_error": result.error,
            "validation_repairable": result.repairable,
            "validation_error_type": result.error_type,
            "validated_sql": "",
            "execution_error": "",
            "failure_events": [
                failure_event(
                    "validation",
                    result.error,
                    result.error_type,
                    result.repairable,
                )
            ],
        }

    # A confirmed conversational entity scope is a hard execution constraint,
    # not a hint for the repair model.  This prevents repaired SQL from
    # widening "that sample" into every sample at the requested time point.
    inherited_scope = list(
        (state.get("resolved_query_spec") or {}).get("sample_ids", [])
        or state.get("query_delta", {}).get("sample_ids", [])
    )
    if state.get("memory_used") and inherited_scope:
        normalized_sql = result.sql.lower()
        missing_scope = [
            sample_id for sample_id in inherited_scope
            if str(sample_id).lower() not in normalized_sql
        ]
        if missing_scope:
            error = "修复后的SQL丢失已确认的实体范围：" + ", ".join(missing_scope)
            return {
                "validation_error": error,
                "validation_repairable": True,
                "validation_error_type": "context_scope",
                "validated_sql": "",
                "execution_error": "",
                "failure_events": [failure_event("validation", error, "context_scope", True)],
            }

    return {
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "validated_sql": result.sql,
        "execution_error": "",
    }


def approval_gate(state: Text2SQLState) -> dict[str, Any]:
    """Pause risky execution for an auditable plan-level human decision."""

    mode = str(state.get("approval_mode") or get_settings().APPROVAL_MODE).lower()
    decision = dict(state.get("approval_decision") or {})
    risky = state.get("query_plan_mode") in {"advanced_analysis_plan", "rsl", "llm_query_spec"}
    required = bool(state.get("force_approval", False) or mode == "always" or (mode == "risk" and risky))
    if not required and not decision:
        return {"approval_required": False, "approval_approved": False, "approval_summary": {"required": False, "reason": "approval_mode_off_or_low_risk"}}

    payload = {
        "question": effective_question(state), "profile": state.get("domain_profile", ""),
        "intent": state.get("query_intent", ""), "query_plan_mode": state.get("query_plan_mode", ""),
        "query_spec": state.get("query_spec", {}), "advanced_plan": state.get("advanced_plan", {}),
        "compiled_sql": state.get("validated_sql", ""), "schema_tables": state.get("intent_related_tables", []),
        "model_calls": state.get("model_calls", []), "failure_events": state.get("failure_events", []),
        "risk_reason": "forced" if state.get("force_approval") else "advanced_or_freeform_plan",
    }
    service = get_long_term_memory_service()
    request = dict(state.get("approval_request") or {})
    if not request:
        request = service.create_approval_request(profile=str(state.get("domain_profile", "")), payload=payload)

    action = str(decision.get("action", "")).lower()
    if action in {"approve", "approved"}:
        decided = service.decide_approval_request(str(request["approval_id"]), {**decision, "action": "approved"})
        return {"approval_required": False, "approval_request": decided or request, "approval_approved": True, "approval_summary": {"required": True, "action": "approved", "approval_id": request["approval_id"]}}
    if action in {"reject", "rejected"}:
        decided = service.decide_approval_request(str(request["approval_id"]), {**decision, "action": "rejected"})
        return {"approval_required": False, "approval_request": decided or request, "approval_approved": False, "validation_error": "人工审批拒绝执行该查询计划。", "validation_error_type": "approval", "validation_repairable": False, "approval_summary": {"required": True, "action": "rejected", "approval_id": request["approval_id"]}}
    if action == "edit_plan":
        edited = decision.get("advanced_plan")
        if not isinstance(edited, dict):
            return {"approval_required": True, "approval_request": request, "approval_summary": {"required": True, "error": "edit_plan requires advanced_plan"}}
        try:
            plan = parse_advanced_plan(json.dumps(edited, ensure_ascii=False))
            sql = compile_advanced_analysis_plan(plan)
            decided = service.decide_approval_request(str(request["approval_id"]), {**decision, "action": "edited_plan"})
            return {"approval_required": False, "approval_request": decided or request, "approval_approved": True, "advanced_plan": plan, "query_expectation": build_query_expectation(effective_question(state), plan), "raw_sql": sql, "validated_sql": "", "approval_summary": {"required": True, "action": "edited_plan", "approval_id": request["approval_id"]}}
        except Exception as exc:
            return {"approval_required": True, "approval_request": request, "approval_summary": {"required": True, "error": f"invalid edited plan: {type(exc).__name__}: {exc}"}}
    return {"approval_required": True, "approval_request": request, "approval_approved": False, "approval_summary": {"required": True, "action": "pending", "approval_id": request["approval_id"]}}


def route_after_approval(state: Text2SQLState) -> Literal["review", "revalidate", "pending", "error"]:
    if state.get("approval_required"):
        return "pending"
    if state.get("validation_error"):
        return "error"
    if state.get("approval_summary", {}).get("action") == "edited_plan":
        return "revalidate"
    return "review"


def format_approval_required(state: Text2SQLState) -> dict[str, Any]:
    request = state.get("approval_request", {})
    approval_id = request.get("approval_id", "")
    return {
        "final_status": "approval_required",
        "final_answer": "查询已通过 SQL Guard，正在等待人工审批 QueryPlan。审批编号：" + str(approval_id),
        "model_calls": model_call_log(),
    }


def review_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """优先使用结构化QuerySpec和Guard；仅审查真正模糊的独立复杂SQL。"""

    question = effective_question(state)
    if state.get("query_plan_mode") == "deterministic_extended":
        return {
            "review_called": False,
            "review_passed": True,
            "review_reason": "确定性扩展QuerySpec语义检查通过。",
            "review_note": (
                "INITIAL/FINAL、白名单派生指标、字段间比较、排序和LIMIT均由确定性编译器生成并校验。"
            ),
            "review_input_summary": "",
        }

    if state.get("query_plan_mode") == "advanced_analysis_plan":
        return {
            "review_called": False,
            "review_passed": True,
            "review_reason": "受限高级AnalysisPlan编译并通过SQL Guard。",
            "review_note": "查询族、字段、Join和高级算子已由Plan校验后编译，无需自由LLM复审。",
            "review_input_summary": "",
        }

    covered, coverage_reason = assess_deterministic_semantic_coverage(
        question=question,
        sql=state["validated_sql"],
        query_spec=state.get("query_spec"),
    )

    if covered:
        return {
            "review_called": False,
            "review_passed": True,
            "review_reason": "确定性语义检查通过。",
            "review_note": coverage_reason,
            "review_input_summary": "",
        }

    query_spec = state.get("query_spec", {})
    if (
        state.get("memory_used", False)
        and state.get("context_resolution_valid", True)
        and query_spec.get("memory_resolved")
        and query_spec.get("structured_context_complete")
    ):
        # SQL已经通过统一Guard，当前轮字段、范围、过滤、排序、聚合和数量
        # 又通过短期记忆Coverage；此时再次让LLM审查容易把正确跨表SQL误杀。
        return {
            "review_called": False,
            "review_passed": True,
            "review_reason": "结构化记忆语义检查通过。",
            "review_note": (
                "当前轮QueryDelta覆盖、陈旧状态清理和SQL Guard均已通过，"
                "无需再次调用LLM语义审查。"
            ),
            "review_input_summary": "",
        }

    review_input_summary = build_sql_review_summary(state["validated_sql"])
    passed, reason = review_complex_sql(
        question=question,
        schema_context=state["schema_context"],
        sql=state["validated_sql"],
    )

    if passed is None:
        return {
            "review_called": True,
            "review_passed": False,
            "review_reason": "复杂SQL语义审查未能返回可信结论。",
            "review_note": "review_unavailable",
            "review_input_summary": review_input_summary,
            "failure_events": [
                failure_event(
                    "review",
                    "复杂SQL语义审查未能返回可信结论。",
                    "unavailable",
                    False,
                )
            ],
        }

    return {
        "review_called": True,
        "review_passed": passed,
        "review_reason": reason,
        "review_note": "llm_review",
        "review_input_summary": review_input_summary,
        "failure_events": (
            []
            if passed
            else [failure_event("review", reason, "semantic", True)]
        ),
    }


def build_explicit_repair_action(
    question: str,
    reason: str,
) -> str:
    """把高可信Guard错误转换为简短、明确的修复动作。"""

    requested_samples = (
        extract_requested_sample_ids(
            question
        )
    )

    if (
        "用户未指定的固定sample_id过滤" in reason
        or (
            "用户没有指定任何样本" in reason
            and "sample_id" in reason
        )
    ):
        return (
            "用户没有指定任何样本。"
            "删除整个固定sample_id过滤谓词；"
            "不要改成IN、LIKE、其他样本编号或其他固定样本条件。"
        )

    if (
        "固定样本过滤不能使用LIKE" in reason
        and requested_samples
    ):
        return (
            "删除sample_id LIKE谓词，"
            "改为用户指定sample_id的等值或IN条件："
            + ", ".join(
                sorted(requested_samples)
            )
            + "。"
        )

    if (
        "用户明确指定了样本" in reason
        and "没有使用对应的sample_id" in reason
        and requested_samples
    ):
        return (
            "加入用户指定sample_id的等值或IN过滤："
            + ", ".join(
                sorted(requested_samples)
            )
            + "。不得使用LIKE。"
        )

    if (
        "IN子查询中使用LIMIT" in reason
        or "Top-K查询缺少顶层ORDER BY" in reason
    ):
        return (
            "这是普通Top-K结构错误。"
            "丢弃IN子查询、派生表包装和无意义GROUP BY，"
            "从正确字段所属的最少表重新生成："
            "顶层SELECT返回sample_id及用户要求字段，"
            "顶层ORDER BY目标字段并使用正确LIMIT。"
        )

    if (
        "字段归属错误" in reason
        or "未知字段" in reason
    ):
        return (
            "根据确定性字段提示重新选择真实字段所属表。"
            "不要沿用原SQL中的错误表或嵌套子查询；"
            "使用包含目标字段的最少必要表重新生成。"
        )

    if "多个来源中存在" in reason:
        return (
            "为所有可能歧义的字段添加正确表别名；"
            "如果这是普通Top-K，优先改写成单层ORDER BY加LIMIT，"
            "不要保留不必要的派生表JOIN。"
        )

    if (
        "不能限制point_index" in reason
        or "固定point_index代替峰值" in reason
    ):
        return (
            "删除WHERE、JOIN ON和HAVING中的固定point_index条件，"
            "在完整thermal_response序列上按sample_id分组并使用MAX。"
        )

    if "全部数据" in reason:
        return (
            "用户明确请求白名单表全部数据。"
            "只查询该目标表，显式列出该表全部字段，"
            "删除WHERE、GROUP BY、聚合和其他表JOIN。"
        )

    if (
        "普通样本级字段Top-K" in reason
        or "样本级Top-K必须返回sample_id" in reason
    ):
        return (
            "这是每个样本一行字段的普通Top-K。"
            "返回sample_id及用户要求字段，"
            "直接按目标字段ORDER BY并LIMIT；"
            "删除MAX、GROUP BY和子查询。"
        )

    return (
        "只修复错误原因中明确指出的问题，"
        "使用正确字段所属的最少必要表，"
        "不要增加用户未要求的过滤、字段或数据表。"
        "科学计数法必须作为一个完整数值保留，"
        "不得把指数部分解释成LIMIT或其他条件。"
    )


def _repair_advanced_plan(
    state: Text2SQLState,
    *,
    question: str,
    reason: str,
    source: str,
    repair_attempt: int,
) -> tuple[dict[str, Any] | None, str, str]:
    """Repair an advanced query through its bounded JSON contract first."""

    prior_plan = state.get("advanced_plan", {})
    if not isinstance(prior_plan, dict) or not prior_plan:
        return None, "", "no prior advanced plan"
    is_material = str(prior_plan.get("family", "")) == MATERIAL_PLAN_FAMILY
    prompt = material_plan_prompt(state["schema_context"], question) if is_material else advanced_plan_prompt(state["schema_context"], question)
    repair_prompt = f"""{prompt}

This is repair attempt {repair_attempt}. Return a replacement advanced_plan JSON only.
Failure source: {source}
Failure reason: {reason}
Previous validated-plan candidate:
{json.dumps(prior_plan, ensure_ascii=False)}

Only change fields needed to resolve the stated failure. Do not output SQL."""
    try:
        raw = invoke_text(
            "You repair only the constrained AdvancedAnalysisPlan JSON contract, never SQL.",
            repair_prompt,
            purpose="repair",
            repair_attempt=repair_attempt,
        )
        plan = parse_material_plan(clean_llm_sql(raw)) if is_material else parse_advanced_plan(clean_llm_sql(raw))
        sql = compile_material_plan(plan) if is_material else compile_advanced_analysis_plan(plan)
        return plan, sql, raw
    except Exception as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def repair_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """只根据可信的Guard、数据库或一次语义审查错误重写SQL。"""

    if not state.get("result_assertion_passed", True):
        source = "结果级断言"
        reason = str(
            state.get("result_assertion", {}).get(
                "reason", "结果性质不满足计划约束"
            )
        )
        bad_sql = state.get("validated_sql") or state.get("raw_sql", "")
    elif state.get("validation_error"):
        source = "确定性Guard"
        reason = state["validation_error"]
        bad_sql = state.get("raw_sql", "")
    elif state.get("execution_error"):
        source = "数据库执行"
        reason = state["execution_error"]
        bad_sql = (
            state.get("validated_sql")
            or state.get("raw_sql", "")
        )
    else:
        source = "复杂SQL语义审查"
        reason = state.get(
            "review_reason",
            "语义不一致",
        )
        bad_sql = (
            state.get("validated_sql")
            or state.get("raw_sql", "")
        )

    next_retry_count = (
        state.get("retry_count", 0) + 1
    )
    question = effective_question(state)
    compact_context = (
        state.get("robust_schema_context")
        or build_compact_sql_context(
            question
        )
    )
    repair_action = (
        build_explicit_repair_action(
            question=question,
            reason=reason,
        )
    )

    # Advanced queries keep their SQL syntax out of the repair model. A valid
    # replacement plan is recompiled and still goes through the common Guard.
    repaired_plan, compiled_plan_sql, plan_output = _repair_advanced_plan(
        state,
        question=question,
        reason=reason,
        source=source,
        repair_attempt=next_retry_count,
    )
    if repaired_plan is not None:
        calls = model_call_log()
        role = str(calls[-1].get("role", "")) if calls else ""
        return {
            "raw_sql": compiled_plan_sql,
            "validated_sql": "",
            "advanced_plan": repaired_plan,
            "advanced_plan_raw": plan_output,
            "advanced_plan_error": "",
            "query_expectation": build_query_expectation(question, repaired_plan),
            "selected_candidate": "repair_plan",
            "candidate_selection_reason": "受限AdvancedAnalysisPlan修复后重新编译SQL。",
            "retry_count": next_retry_count,
            "last_repair_reason": f"{source}: {reason}",
            "repair_source": source,
            "repair_action": repair_action,
            "repair_bad_sql": bad_sql,
            "repair_raw_output": plan_output,
            "repair_model_role": role,
            "repair_plan_mode": "advanced_analysis_plan",
            "validation_error": "",
            "validation_repairable": True,
            "validation_error_type": "",
            "review_called": False,
            "review_passed": False,
            "review_reason": "",
            "review_note": "",
            "review_input_summary": "",
            "execution_error": "",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
        }

    # The first constrained repair is intentionally not followed by another
    # free-SQL attempt on the same 3B model. Its failure advances the graph to
    # the configured DeepSeek repair route on the next retry.
    if state.get("advanced_plan") and next_retry_count == 1:
        return {
            "raw_sql": "",
            "validated_sql": "",
            "advanced_plan_error": plan_output,
            "selected_candidate": "repair_plan_failed",
            "candidate_selection_reason": "3B受限计划修复失败，升级到DeepSeek API。",
            "retry_count": next_retry_count,
            "last_repair_reason": f"{source}: {reason}",
            "repair_source": source,
            "repair_action": repair_action,
            "repair_bad_sql": bad_sql,
            "repair_raw_output": plan_output,
            "repair_model_role": "primary_3b",
            "repair_plan_mode": "advanced_plan_failed_escalate_api",
            "validation_error": "",
            "validation_repairable": True,
            "validation_error_type": "",
            "review_called": False,
            "review_passed": False,
            "review_reason": "",
            "review_note": "",
            "review_input_summary": "",
            "execution_error": "",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "failure_events": [
                failure_event("plan", plan_output, "contract", True)
            ],
        }

    procedural_memories = []
    procedural_context = ""
    try:
        memory_service = get_long_term_memory_service()
        procedural_memories = memory_service.retrieve_procedural(reason)
        procedural_context = memory_service.build_procedural_context(
            procedural_memories
        )
    except Exception:
        procedural_memories = []
        procedural_context = ""

    system_prompt = """
你是MySQL SQL修复器。

只输出一条完整SELECT SQL。
根据明确错误修复原SQL，不要解释，不要复制错误结构。
必须使用真实表名、真实字段和最少必要表。
不得用AS别名伪装错误字段。
普通样本级Top-K返回sample_id，直接使用目标字段ORDER BY和LIMIT，不使用MAX、GROUP BY或IN子查询。
标准时序查询必须区分“样本内聚合”和“聚合结果排名”：峰值使用MAX，平均值使用AVG，均按sample_id分组；“峰值最低”仍是MAX后ASC排序。
最终值表示每个样本最大point_index对应的记录，不能用MIN或MAX(目标值)替代。
指定样本的point_index明细查询直接使用sample_id和point_index条件，不使用聚合。
用户明确指定样本编号时，必须使用对应sample_id等值或IN过滤，禁止LIKE。
用户没有指定样本编号时，禁止保留或新增任何固定sample_id过滤。
""".strip()

    user_prompt = f"""
{compact_context}

用户问题：
{question}

错误来源：{source}
错误原因：
{reason}

必须执行的修复动作：
{repair_action}

{procedural_context}

错误SQL：
{bad_sql}

请从零输出修复后的完整SQL。
""".strip()

    repair_raw_output = invoke_text(
        system_prompt,
        user_prompt + (
            "\n受限高级计划修复未通过，现允许最后的自由SQL兜底。原因："
            + plan_output
            if plan_output and state.get("advanced_plan")
            else ""
        ),
        purpose="repair",
        repair_attempt=next_retry_count,
    )
    repaired_sql = normalize_sample_id_literals(
        clean_llm_sql(
            repair_raw_output
        )
    )

    calls = model_call_log()
    role = str(calls[-1].get("role", "")) if calls else ""
    return {
        "raw_sql": repaired_sql,
        "validated_sql": "",
        "selected_candidate": "repair",
        "candidate_selection_reason": (
            state.get("candidate_selection_reason", "")
            + " 修复器基于已选候选重写SQL。"
        ).strip(),
        "retry_count": next_retry_count,
        "last_repair_reason": (
            f"{source}: {reason}"
        ),
        "repair_source": source,
        "repair_action": repair_action,
        "repair_bad_sql": bad_sql,
        "repair_raw_output": repair_raw_output,
        "repair_model_role": role,
        "repair_plan_mode": "free_sql_fallback" if state.get("advanced_plan") else "free_sql",
        "procedural_memory_matches": [
            record.to_public_dict() for record in procedural_memories
        ],
        "procedural_memory_context": procedural_context,
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
    }


def execute_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    settings = get_settings()

    try:
        result = execute_readonly_query(
            sql=state["validated_sql"],
            max_rows=settings.SQL_MAX_ROWS,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {
            "execution_error": error,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "failure_events": [
                failure_event("execution", error, "database", True)
            ],
        }

    order_ids = list(state.get("query_spec", {}).get("result_order_sample_ids", []))
    if order_ids and "sample_id" in result.get("columns", []):
        index = result["columns"].index("sample_id")
        rank = {sample_id: position for position, sample_id in enumerate(order_ids)}
        result["rows"] = sorted(
            result.get("rows", []),
            key=lambda row: rank.get(row[index] if index < len(row) else None, len(rank)),
        )

    return {
        "execution_error": "",
        "delivery_policy": build_delivery_policy(
            effective_question(state), state.get("query_spec", {}), settings.SQL_MAX_ROWS
        ),
        **result,
    }


def validate_result_assertions(
    state: Text2SQLState,
) -> dict[str, Any]:
    """Validate result invariants for a constrained advanced plan."""

    plan_assertion = assert_advanced_result(
        state.get("advanced_plan", {}),
        list(state.get("columns", [])),
        list(state.get("rows", [])),
    )
    expectation = state.get("query_expectation") or build_query_expectation(
        effective_question(state), state.get("advanced_plan", {}),
    )
    question_assertion = assert_query_expectation(
        expectation, list(state.get("columns", [])),
    )
    passed = bool(plan_assertion["passed"] and question_assertion["passed"])
    reasons = [
        value.get("reason", "")
        for value in (plan_assertion, question_assertion)
        if value.get("reason")
    ]
    assertion = {
        "checked": bool(plan_assertion.get("checked") or question_assertion.get("checked")),
        "passed": passed,
        "family": plan_assertion.get("family", ""),
        "plan_invariants": plan_assertion,
        "question_expectation": question_assertion,
        "reason": "；".join(reasons),
    }
    if passed:
        return {
            "result_assertion": assertion,
            "result_assertion_passed": True,
        }
    return {
        "result_assertion": assertion,
        "result_assertion_passed": False,
        "failure_events": [
            failure_event(
                "result_assertion",
                assertion["reason"],
                "result_invariant",
                True,
            )
        ],
    }


def update_session_memory(
    state: Text2SQLState,
) -> dict[str, Any]:
    """仅在执行成功且当前轮语义覆盖通过后提交短期记忆。"""

    if not state.get("context_resolution_valid", True):
        return {
            "conversation_memory": state.get("conversation_memory", {}),
            "memory_update_summary": {
                "updated": False,
                "reason": "当前轮语义覆盖未通过，保留上一成功记忆。",
            },
        }

    final_status = (
        "repaired_success"
        if state.get("retry_count", 0) > 0
        else "first_pass_success"
    )
    memory = update_short_term_memory(
        state.get("conversation_memory", {}),
        question=state.get("question", ""),
        resolved_question=effective_question(state),
        query_spec=state.get("query_spec", {}),
        validated_sql=state.get("validated_sql", ""),
        columns=list(state.get("columns", [])),
        rows=list(state.get("rows", [])),
        row_count=int(state.get("row_count", 0)),
        truncated=bool(state.get("truncated", False)),
        final_status=final_status,
        turn_type=state.get("turn_type", "new_query"),
        query_delta=state.get("query_delta", {}),
    )
    return {
        "conversation_memory": memory,
        "memory_update_summary": {
            "updated": True,
            "session_id": memory.get("session_id", ""),
            "turn_type": state.get("turn_type", "new_query"),
            "active_sample_count": len(memory.get("active_sample_ids", [])),
            "parent_sample_count": len(
                memory.get("parent_result_scope", {}).get("sample_ids", [])
            ),
            "recent_user_turn_count": len(memory.get("recent_user_turns", [])),
            "recent_turn_count": len(memory.get("recent_turns", [])),
        },
    }


def persist_session_memory(state: Text2SQLState) -> dict[str, Any]:
    """Persist only the structured successful-session state, never raw traces."""

    memory = state.get("conversation_memory", {})
    if not memory:
        return {"session_store_summary": {"saved": False, "reason": "empty_memory"}}
    try:
        return {"session_store_summary": get_session_memory_store().save(memory, state.get("domain_profile", ""))}
    except Exception as exc:
        # Session persistence is an availability enhancement, not a query blocker.
        return {"session_store_summary": {"saved": False, "error": f"{type(exc).__name__}: {exc}"}}


def update_long_term_memory(
    state: Text2SQLState,
) -> dict[str, Any]:
    """成功执行后按规则保存高价值情节记忆和修复经验。"""

    try:
        service = get_long_term_memory_service()
        summary = service.auto_save_from_state(dict(state))
        used_examples = state.get("advanced_plan_memory_matches", [])
        example_ids = [item.get("memory_id", "") for item in used_examples if isinstance(item, dict)]
        if example_ids:
            service.record_advanced_plan_usage(example_ids, success=True)
            summary["advanced_plan_memory_usage"] = {"used": example_ids, "success": True}
        return {"long_term_memory_write_summary": summary}
    except Exception as exc:
        return {
            "long_term_memory_write_summary": {
                "enabled": False,
                "error": f"{type(exc).__name__}: {exc}",
                "saved": [],
            }
        }


def shorten_cell(
    value: Any,
    max_length: int = 100,
) -> Any:
    if value is None:
        return None

    text = str(value)
    if len(text) <= max_length:
        return value

    return text[:max_length] + "..."


def format_result(
    state: Text2SQLState,
) -> dict[str, Any]:
    rows = state.get("rows", [])
    columns = state.get("columns", [])

    if rows:
        result_text = tabulate(
            [
                [
                    shorten_cell(value)
                    for value in row
                ]
                for row in rows
            ],
            headers=columns,
            tablefmt="github",
            stralign="left",
            numalign="right",
        )
    else:
        result_text = (
            "查询执行成功，但没有返回符合条件的数据。"
        )

    retry_count = state.get("retry_count", 0)
    execution_notice = (
        f"本次SQL经过{retry_count}次自动修复后执行成功。"
        if retry_count > 0
        else "本次SQL首次生成后执行成功。"
    )

    review_section = ""
    if state.get("review_reason"):
        review_section = (
            "\n\n语义校验："
            + state["review_reason"]
        )

    if (
        state.get("review_note")
        and state.get("review_note")
        not in {
            "llm_review",
            "review_unavailable",
        }
    ):
        review_section += (
            "\n校验说明："
            + state["review_note"]
        )

    truncate_notice = ""
    if state.get("truncated", False):
        truncate_notice = (
            "\n\n注意：结果超过最大返回行数，"
            f"当前只展示前{get_settings().SQL_MAX_ROWS}行。"
        )

    return {
        "final_status": (
            "repaired_success"
            if retry_count > 0
            else "first_pass_success"
        ),
        "final_answer": f"""
查询执行成功。

{execution_notice}

实际执行 SQL：

```sql
{state["validated_sql"]}
```

查询结果：

{result_text}{truncate_notice}{review_section}
""".strip()
        ,
        "model_calls": model_call_log(),
    }


def format_unsupported_query(
    state: Text2SQLState,
) -> dict[str, Any]:
    suggestions = list(state.get("unsupported_query_suggestions", []))
    suggestion_text = "\n".join(
        f"{index}. {item}" for index, item in enumerate(suggestions, start=1)
    ) or "请把多阶段筛选拆成两到三轮查询。"
    memory = mark_current_turn_status(
        state.get("conversation_memory", {}),
        question=state.get("question", ""),
        status="unsupported",
        resolved_question=effective_question(state),
    )
    return {
        "conversation_memory": memory,
        "final_status": "unsupported",
        "final_answer": (
            "当前问题包含多阶段候选筛选或二次Top-K，当前版本不在单条SQL中自动猜测其执行顺序。\n\n"
            + str(state.get("unsupported_query_reason", ""))
            + "\n\n建议拆成多轮查询：\n"
            + suggestion_text
            + "\n\n拆分后系统会通过短期记忆自动保留上一轮样本集合。"
        ).strip(),
        "model_calls": model_call_log(),
    }


def format_error(
    state: Text2SQLState,
) -> dict[str, Any]:
    error = (
        state.get("validation_error")
        or state.get("execution_error")
        or state.get("review_reason")
        or "未知错误"
    )

    if state.get("validation_error_type") == "policy":
        description = "该请求违反只读、白名单或跨库安全策略，系统不会尝试改写。"
    elif state.get("retry_count", 0) > 0:
        description = (
            "系统已经自动修复一次，但仍未通过确定性校验、"
            "复杂语义审查或数据库执行。"
        )
    else:
        description = "本次查询未能生成可执行结果。"

    final_status = (
        "policy_rejected"
        if state.get("validation_error_type") == "policy"
        else "failed"
    )
    memory = mark_current_turn_status(
        state.get("conversation_memory", {}),
        question=state.get("question", ""),
        status=final_status,
        resolved_question=effective_question(state),
    )
    used_examples = state.get("advanced_plan_memory_matches", [])
    example_ids = [item.get("memory_id", "") for item in used_examples if isinstance(item, dict)]
    if example_ids:
        try:
            get_long_term_memory_service().record_advanced_plan_usage(example_ids, success=False)
        except Exception:
            # Memory telemetry must never hide the primary query error.
            pass

    return {
        "conversation_memory": memory,
        "final_status": final_status,
        "final_answer": f"""
本次查询没有成功执行。

{description}

错误信息：

{error}

首次SQL：

```sql
{state.get("initial_sql") or "未生成SQL"}
```

最后SQL：

```sql
{state.get("raw_sql") or "未生成SQL"}
```
""".strip(),
        "model_calls": model_call_log(),
    }


def route_after_validation(
    state: Text2SQLState,
) -> Literal["review", "repair", "error"]:
    if not state.get("validation_error"):
        return "review"

    if not state.get(
        "validation_repairable",
        True,
    ):
        return "error"

    if (
        state.get("retry_count", 0)
        < get_settings().SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"


def route_after_review(
    state: Text2SQLState,
) -> Literal["execute", "repair", "error"]:
    if state.get("review_passed", False):
        return "execute"

    if (
        state.get("review_note")
        == "review_unavailable"
    ):
        return "error"

    if (
        state.get("retry_count", 0)
        < get_settings().SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"


def route_after_execution(
    state: Text2SQLState,
) -> Literal["success", "repair", "error"]:
    if not state.get("execution_error"):
        return "success"

    # 确定性扩展SQL由代码编译。若数据库仍拒绝执行，应直接暴露编译器问题，
    # 而不是调用通用LLM修复并等待长时间超时。
    if state.get("query_plan_mode") == "deterministic_extended":
        return "error"

    if (
        state.get("retry_count", 0)
        < get_settings().SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"


def route_after_result_assertions(
    state: Text2SQLState,
) -> Literal["success", "repair", "error"]:
    if state.get("result_assertion_passed", True):
        return "success"
    if state.get("retry_count", 0) < get_settings().SQL_MAX_REPAIR_ATTEMPTS:
        return "repair"
    return "error"
