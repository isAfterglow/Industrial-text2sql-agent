from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import re
import uuid
from typing import Any

from app.schema import (
    NUMERIC_LITERAL_PATTERN,
    build_query_spec,
    canonical_metric_alias,
    extract_requested_limit_from_question,
    extract_requested_sample_ids,
    get_schema_catalog,
    infer_question_ranking_column,
    infer_requested_output_columns,
    match_question_semantic_columns,
    normalize_question_sample_ids,
)


# 文本窗口与可执行状态分离：最近两轮原始输入用于指代理解；
# 最后一次成功QuerySpec和结果范围用于可靠状态继承。
MAX_RECENT_USER_TURNS = 2
MAX_RESULT_SAMPLE_IDS = 100
MAX_REFERENT_HISTORY = 6
MAX_CLARIFICATION_ATTEMPTS = 2

_DEPENDENCIES = {
    "independent",
    "same_sample",
    "previous_result_set",
    "previous_query",
}
_PROJECTION_ACTIONS = {"keep", "replace", "add", "remove"}
_STATE_ACTIONS = {"keep", "replace", "add", "remove", "clear"}
_SCOPE_ACTIONS = {"replace", "keep", "same_sample", "previous_result", "parent_result"}

_CONTINUATION_PREFIX_RE = re.compile(
    r"^\s*(?:再|继续|然后|接着|改成|改为|数量改为|数量改成|换成|只看|只返回|"
    r"只显示|仅返回|仅显示|只保留|不要|去掉|移除|取消|加上|增加|补充|顺便|"
    r"另外|再要求|这些|上述|上面|刚才|前面|其中|它们|它的|它呢|这个样本|"
    r"该样本|同一个样本|那|那么)",
    flags=re.IGNORECASE,
)
_MODIFICATION_RE = re.compile(
    r"数量\s*(?:改成|改为)|改成|改为|换成|只保留|再要求|再加上|加上|增加|"
    r"补充|不要|去掉|移除|取消|清除|只返回|只显示|仅返回|仅显示|只看",
    flags=re.IGNORECASE,
)
# “最高的7个/最低的3个”既可能是完整新查询的一部分，也可能是修改片段，
# 不能仅凭这类Top-K表达判定为承接上一查询。
_TOPK_FRAGMENT_RE = re.compile(
    r"^(?:最高|最低)的?\s*\d+\s*(?:个|条)?(?:样本)?[。！!?？]*$|"
    r"^前\s*\d+\s*(?:个|条)?(?:样本)?[。！!?？]*$",
    flags=re.IGNORECASE,
)
_EXPLICIT_QUERY_START_RE = re.compile(
    r"^\s*(?:请|帮我|麻烦)?\s*(?:(?:重新|从头|另行|另查)\s*)?"
    r"(?:查询|查找|查一下|找出|筛选|统计|列出|获取|查看|检索)",
    flags=re.IGNORECASE,
)
_RESET_QUERY_RE = re.compile(
    r"^\s*(?:请|帮我|麻烦)?\s*(?:重新(?:查询|查找|找出|筛选|统计|检索)|"
    r"从头(?:查询|查找|找出|筛选|统计|检索)|另行(?:查询|查找)|另查)",
    flags=re.IGNORECASE,
)
_RESULT_REFERENCE_RE = re.compile(
    r"这些样本|上述样本|上面(?:这些|几个)?样本|刚才(?:返回|查到)的样本|"
    r"前面(?:返回|查到)的样本|这些结果|上述结果|其中|它们",
    flags=re.IGNORECASE,
)
_PARENT_RESULT_RE = re.compile(
    r"原来(?:那批|那些)?样本|上一批样本|前一批样本|最初(?:那批|那些)?样本",
    flags=re.IGNORECASE,
)
_SAME_SAMPLE_RE = re.compile(
    r"它的|它呢|这个样本|该样本|同一个样本|继续看它|再看它|这个材料|该材料",
    flags=re.IGNORECASE,
)
_PLURAL_SAMPLE_RE = re.compile(
    r"它们|这些样本|上述样本|上面(?:这些|几个)?样本|刚才(?:返回|查到)的样本|"
    r"前面(?:返回|查到)的样本|这些结果|上述结果|其中",
    flags=re.IGNORECASE,
)
_CANCEL_CLARIFICATION_RE = re.compile(
    r"^\s*(?:取消|算了|不用了|结束澄清|取消澄清|重新提问|重新开始)\s*[。！!?？]*\s*$",
    flags=re.IGNORECASE,
)
_ADD_PROJECTION_RE = re.compile(
    r"再(?:返回|显示|看)|还(?:要|需要)?(?:返回|显示|看)?|顺便|另外|"
    r"加上|增加|补充|同时(?:返回|显示|看)",
    flags=re.IGNORECASE,
)
_REPLACE_PROJECTION_RE = re.compile(
    r"只返回|只显示|仅返回|仅显示|只看|换成|改查|不要原来的|是多少|呢\s*[？?]?$",
    flags=re.IGNORECASE,
)
_REMOVE_PROJECTION_RE = re.compile(
    r"不要(?:返回|显示|看)?|去掉|移除|删除.*字段",
    flags=re.IGNORECASE,
)
_CLEAR_ALL_FILTER_RE = re.compile(
    r"取消所有条件|去掉所有条件|不要任何条件|清除筛选|取消筛选|清空条件",
    flags=re.IGNORECASE,
)
_REMOVE_FILTER_RE = re.compile(
    r"(?:不要|去掉|移除|取消).{0,12}(?:条件|筛选)|"
    r"(?:条件|筛选).{0,8}(?:不要|去掉|移除|取消)",
    flags=re.IGNORECASE,
)
_REPLACE_FILTER_RE = re.compile(
    r"(?:把|将).{0,20}(?:条件|筛选).{0,8}(?:改成|改为)|"
    r"(?:条件|筛选).{0,8}(?:改成|改为)",
    flags=re.IGNORECASE,
)
_CLEAR_RANKING_RE = re.compile(r"取消排序|不要排序|不再排序", flags=re.IGNORECASE)
_CLEAR_TEMPORAL_RE = re.compile(r"不要聚合|取消聚合|不看峰值|不看平均", flags=re.IGNORECASE)


_RESPONSE_COLUMNS = {"surface_temperature", "back_temperature", "mass"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return "session-" + uuid.uuid4().hex[:10]


def _empty_scope() -> dict[str, Any]:
    return {
        "sample_ids": [],
        "ordered_sample_ids": [],
        "selection_order_by": None,
        "row_count": 0,
        "truncated": False,
        "source_question": "",
    }


def new_short_term_memory(session_id: str | None = None) -> dict[str, Any]:
    return {
        "session_id": session_id or new_session_id(),
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "last_successful_question": "",
        "last_resolved_question": "",
        "last_successful_query_state": {},
        # 保留旧键，兼容CLI和已有调用。
        "last_query_spec": {},
        "last_validated_sql": "",
        "last_result": {
            "columns": [],
            "row_count": 0,
            "sample_ids": [],
            "truncated": False,
        },
        "last_result_scope": _empty_scope(),
        "parent_result_scope": _empty_scope(),
        # 单数和复数代词使用独立锚点：它 -> 最近单样本；它们/这些样本 -> 最近样本集合。
        "last_single_sample_scope": _empty_scope(),
        "last_multi_sample_scope": _empty_scope(),
        "referent_history": [],
        "active_sample_ids": [],
        "pending_clarification": {},
        "recent_user_turns": [],
        # 保留旧键；成功提交时同步维护。
        "recent_turns": [],
    }


def reset_short_term_memory(
    memory: dict[str, Any] | None,
    *,
    keep_session_id: bool = True,
) -> dict[str, Any]:
    session_id = ""
    if keep_session_id and memory:
        session_id = str(memory.get("session_id", ""))
    return new_short_term_memory(session_id or None)


def record_user_turn(
    memory: dict[str, Any] | None,
    question: str,
) -> dict[str, Any]:
    """在查询开始时记录原始输入，成功和失败轮次都保留。"""

    updated = deepcopy(memory or new_short_term_memory())
    turns = list(updated.get("recent_user_turns", []))
    normalized = str(question).strip()
    turns.append(
        {
            "user_question": normalized,
            "status": "pending",
            "resolved_question": "",
            "created_at": _utc_now_iso(),
        }
    )
    updated["recent_user_turns"] = turns[-MAX_RECENT_USER_TURNS:]
    updated["updated_at"] = _utc_now_iso()
    return updated


def mark_current_turn_status(
    memory: dict[str, Any] | None,
    *,
    question: str,
    status: str,
    resolved_question: str = "",
) -> dict[str, Any]:
    updated = deepcopy(memory or new_short_term_memory())
    turns = list(updated.get("recent_user_turns", []))
    for item in reversed(turns):
        if str(item.get("user_question", "")).strip() == str(question).strip():
            item["status"] = status
            if resolved_question:
                item["resolved_question"] = resolved_question
            break
    updated["recent_user_turns"] = turns[-MAX_RECENT_USER_TURNS:]
    updated["updated_at"] = _utc_now_iso()
    return updated


def _previous_spec(memory: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(
        memory.get("last_successful_query_state")
        or memory.get("last_query_spec")
        or {}
    )


def _known_columns() -> set[str]:
    columns: set[str] = set()
    for table in get_schema_catalog().get("tables", {}).values():
        columns.update(table.get("columns", {}).keys())
    return columns


def _column_label(column: str) -> str:
    terms = get_schema_catalog().get("semantic_terms", {}).get(column, [])
    return str(terms[0]) if terms else column


def _ordered_projection(columns: set[str] | list[str]) -> list[str]:
    values = list(dict.fromkeys(str(value) for value in columns if value))
    priority = [column for column in ("sample_id", "point_index") if column in values]
    rest = sorted(column for column in values if column not in {"sample_id", "point_index"})
    return priority + rest


def _is_meaningful_spec(spec: dict[str, Any]) -> bool:
    return bool(
        spec.get("eligible")
        or spec.get("select_columns")
        or spec.get("filters")
        or spec.get("sample_ids")
        or spec.get("order_by")
        or spec.get("temporal_metrics")
        or spec.get("limit") is not None
    )


def _aggregation_from_question(question: str) -> str | None:
    if re.search(r"峰值|最大值", question):
        return "MAX"
    if re.search(r"平均|均值", question):
        return "AVG"
    if re.search(r"最小值", question):
        return "MIN"
    if re.search(r"最终|最后一个点|末时刻", question):
        return "FINAL"
    return None


def _direction_from_question(question: str) -> str | None:
    if re.search(r"最高|最大|降序|从高到低", question):
        return "DESC"
    if re.search(r"最低|最小|升序|从低到高", question):
        return "ASC"
    return None


def _extract_followup_point_range(question: str) -> dict[str, Any] | None:
    between = re.search(
        rf"({NUMERIC_LITERAL_PATTERN})\s*(?:到|至|~|～)\s*({NUMERIC_LITERAL_PATTERN})",
        question,
        flags=re.IGNORECASE,
    )
    if between and re.search(r"point_index|序列点|点位|从", question):
        return {
            "column": "point_index",
            "operator": "BETWEEN",
            "value": between.group(1),
            "value2": between.group(2),
        }
    exact = re.search(
        rf"point_index\s*(?:等于|=|为)?\s*({NUMERIC_LITERAL_PATTERN})",
        question,
        flags=re.IGNORECASE,
    )
    if exact:
        return {"column": "point_index", "operator": "=", "value": exact.group(1)}
    return None


def _replace_filters_by_column(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    current_columns = {str(item.get("column", "")) for item in current}
    retained = [
        deepcopy(item)
        for item in previous
        if str(item.get("column", "")) not in current_columns
    ]
    return retained + deepcopy(current)


def _remove_filters_by_column(
    previous: list[dict[str, Any]],
    columns: set[str],
) -> list[dict[str, Any]]:
    return [
        deepcopy(item)
        for item in previous
        if str(item.get("column", "")) not in columns
    ]


def _has_explicit_reference(question: str) -> bool:
    """当前轮是否明确引用历史中的单样本或样本集合。"""

    return bool(
        _RESULT_REFERENCE_RE.search(question)
        or _SAME_SAMPLE_RE.search(question)
        or _PARENT_RESULT_RE.search(question)
    )


def _question_is_complete(current_spec: dict[str, Any], question: str) -> bool:
    """判断当前轮是否是一个自包含查询，而不是依赖历史的修改片段。

    “是否独立”与“当前版本是否能够执行”必须分开：复杂或暂不支持的查询，
    只要自身包含明确对象、指标、条件/排序/数量，也仍然属于独立新查询。
    """

    normalized = str(question).strip()
    if not normalized or _has_explicit_reference(normalized):
        return False

    # 纯修改片段仍然承接上一轮；但完整句中出现“最高/最低”不属于修改信号。
    if _TOPK_FRAGMENT_RE.fullmatch(normalized):
        return False
    if _CONTINUATION_PREFIX_RE.search(normalized) or _MODIFICATION_RE.search(normalized):
        # “重新查询/从头查询”是显式重置，不受修改词影响。
        if not _RESET_QUERY_RE.search(normalized):
            return False

    explicit_columns = set(match_question_semantic_columns(normalized))
    sample_ids = set(extract_requested_sample_ids(normalized))
    has_limit = extract_requested_limit_from_question(normalized) is not None
    has_query_verb = bool(
        re.search(
            r"查询|查找|查一下|找出|筛选|统计|列出|获取|查看|检索|返回",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    has_scope_and_action = bool(
        re.search(
            r"(?:在|从).{0,80}(?:样本|结果)(?:中|里|范围内).{0,80}"
            r"(?:找出|查询|筛选|排列|排序|取前)",
            normalized,
            flags=re.IGNORECASE,
        )
    )
    has_structured_signal = bool(
        explicit_columns
        or sample_ids
        or has_limit
        or current_spec.get("filters")
        or current_spec.get("order_by")
        or current_spec.get("temporal_metrics")
        or current_spec.get("select_columns")
    )

    if _RESET_QUERY_RE.search(normalized):
        return has_structured_signal
    if _EXPLICIT_QUERY_START_RE.search(normalized):
        return has_structured_signal
    if has_query_verb and has_structured_signal:
        return True
    if has_scope_and_action and has_structured_signal:
        return True

    # 没有显式查询动词时，仍允许“原始密度最低的3个样本”这类完整短句。
    meaningful_spec = bool(
        current_spec.get("eligible")
        or current_spec.get("order_by")
        or current_spec.get("filters")
        or current_spec.get("sample_ids")
    )
    return bool(meaningful_spec and has_structured_signal)

def _extract_explicit_order_by(question: str) -> dict[str, Any] | None:
    """优先解析“按X从低到高排列”等明确排序短语。

    明确排序短语的优先级高于后文出现的返回字段，避免把“返回峰值表温”
    错当成排序字段。
    """

    normalized = str(question)
    match = re.search(
        r"(?:按|根据)\s*(?P<target>.{1,28}?)\s*"
        r"(?P<direction>从高到低|从低到高|由高到低|由低到高|升序|降序)"
        r"(?:排列|排序)?",
        normalized,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    target = match.group("target").strip("，,、。；; ")
    direction_word = match.group("direction")
    direction = "DESC" if direction_word in {"从高到低", "由高到低", "降序"} else "ASC"

    derived_map = {
        "质量损失率": "mass_loss_rate",
        "背温抬升": "back_temperature_rise",
        "背面温度抬升": "back_temperature_rise",
        "背温升高量": "back_temperature_rise",
    }
    for phrase, alias in derived_map.items():
        if phrase in target:
            return {
                "kind": "derived",
                "column": alias,
                "alias": alias,
                "direction": direction,
            }

    ranking = infer_question_ranking_column(target)
    if ranking:
        column = ranking[0]
        return {"kind": "column", "column": column, "direction": direction}

    columns = sorted(match_question_semantic_columns(target))
    if columns:
        return {"kind": "column", "column": columns[0], "direction": direction}
    return None


def _looks_like_follow_up(
    question: str,
    current_spec: dict[str, Any],
    memory: dict[str, Any],
    explicit_columns: set[str],
) -> bool:
    previous_spec = _previous_spec(memory)
    if not previous_spec:
        return False

    # 显式代词永远优先触发对应记忆锚点。
    if _RESULT_REFERENCE_RE.search(question) or _SAME_SAMPLE_RE.search(question):
        return True

    # 当前轮已经能独立形成完整QuerySpec时，不允许历史锚点主动吞掉它。
    # 这一步必须早于短句和“最高/最低”启发式。
    if _question_is_complete(current_spec, question):
        return False

    if _CONTINUATION_PREFIX_RE.search(question) or _MODIFICATION_RE.search(question):
        return True
    if _TOPK_FRAGMENT_RE.search(question):
        return True

    previous_ids = list(previous_spec.get("sample_ids", []))
    active_ids = list(memory.get("active_sample_ids", []))
    # 支持“碳化密度呢？”这类正常省略问法，但仅限不能独立成句的短片段。
    if (
        len(question.strip()) <= 20
        and explicit_columns
        and (len(previous_ids) == 1 or len(active_ids) == 1)
        and not current_spec.get("sample_ids")
        and not current_spec.get("order_by")
        and not current_spec.get("filters")
    ):
        return True

    # 无法独立形成查询、但包含数量/方向/过滤等变化时，默认承接上一成功状态。
    return bool(
        len(question.strip()) <= 32
        and (
            _is_meaningful_spec(current_spec)
            or re.search(r"数量|最高|最低|条件|筛选|保留|取消|不要", question)
        )
    )


def _scope_ids(scope: dict[str, Any] | None) -> list[str]:
    scope = scope or {}
    return list(scope.get("ordered_sample_ids") or scope.get("sample_ids", []))


def _scopes_equal(
    first: dict[str, Any] | None,
    second: dict[str, Any] | None,
) -> bool:
    return _scope_ids(first) == _scope_ids(second) and bool(_scope_ids(first))


def _make_scope(
    sample_ids: list[str],
    *,
    selection_order_by: dict[str, Any] | None = None,
    row_count: int | None = None,
    truncated: bool = False,
    source_question: str = "",
) -> dict[str, Any]:
    ordered = list(dict.fromkeys(sample_ids))[:MAX_RESULT_SAMPLE_IDS]
    return {
        "sample_ids": ordered,
        "ordered_sample_ids": ordered,
        "selection_order_by": deepcopy(selection_order_by),
        "row_count": int(len(ordered) if row_count is None else row_count),
        "truncated": bool(truncated or len(sample_ids) > MAX_RESULT_SAMPLE_IDS),
        "source_question": source_question,
    }


def _append_referent(
    memory: dict[str, Any],
    scope: dict[str, Any],
    *,
    kind: str,
) -> None:
    ids = _scope_ids(scope)
    if not ids:
        return
    history = list(memory.get("referent_history", []))
    entry = {
        "kind": kind,
        "scope": deepcopy(scope),
        "updated_at": _utc_now_iso(),
    }
    # 相同类型且相同集合只刷新，不重复堆叠。
    history = [
        item
        for item in history
        if not (
            item.get("kind") == kind
            and _scope_ids(item.get("scope")) == ids
        )
    ]
    history.append(entry)
    memory["referent_history"] = history[-MAX_REFERENT_HISTORY:]


def _latest_scope_by_kind(
    memory: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    direct_key = (
        "last_single_sample_scope"
        if kind == "single"
        else "last_multi_sample_scope"
    )
    direct = deepcopy(memory.get(direct_key) or _empty_scope())
    if _scope_ids(direct):
        return direct
    for item in reversed(list(memory.get("referent_history", []))):
        if item.get("kind") == kind and _scope_ids(item.get("scope")):
            return deepcopy(item["scope"])
    return _empty_scope()


def cancel_pending_clarification(
    memory: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = deepcopy(memory or new_short_term_memory())
    updated["pending_clarification"] = {}
    updated["updated_at"] = _utc_now_iso()
    return updated


def _select_result_scope(
    memory: dict[str, Any],
    *,
    requested_limit: int | None,
    prefer_parent: bool,
) -> tuple[list[str], str]:
    """选择复数指代范围。

    复数代词优先最近一次多样本集合，而不是最近一次任意结果；
    这样即使中间查询了单个样本，“它们”仍能回指最近那批样本。
    """

    last_ids = _scope_ids(memory.get("last_result_scope"))
    parent_ids = _scope_ids(memory.get("parent_result_scope"))
    multi_ids = _scope_ids(_latest_scope_by_kind(memory, "multi"))

    if prefer_parent and parent_ids:
        return parent_ids, "parent_result_scope"

    # 当当前结果不足以满足数量要求时，优先选择足够大的父集合或最近多样本集合。
    if requested_limit is not None:
        for ids, source in (
            (parent_ids, "parent_result_scope_by_cardinality"),
            (multi_ids, "last_multi_sample_scope_by_cardinality"),
            (last_ids, "last_result_scope_by_cardinality"),
        ):
            if len(ids) >= requested_limit:
                return ids, source

    # 普通“它们/这些样本/其中”默认回指最近多样本集合。
    if multi_ids:
        return multi_ids, "last_multi_sample_scope"
    if len(last_ids) > 1:
        return last_ids, "last_result_scope"
    if parent_ids:
        return parent_ids, "parent_result_scope_fallback"
    return [], "missing_result_scope"


def _contains_write_intent(question: str) -> bool:
    return bool(re.search(r"删除|删掉|清空|修改|更新|写入|插入|改成\s*[-+]?\d+(?:\.\d+)?\s*(?:$|。|，)", question))


def _clarification_choice_delta(
    answer: str,
    pending: dict[str, Any],
    memory: dict[str, Any],
) -> dict[str, Any] | None:
    """将用户对澄清问题的回答转换为QueryDelta。"""

    normalized_answer = answer.strip()
    choice_match = re.fullmatch(
        r"\s*(?:选择|选)?\s*([A-Ea-e])\s*[。.!！?？]*\s*",
        normalized_answer,
    )
    choice = choice_match.group(1).upper() if choice_match else ""
    candidates = pending.get("candidate_deltas", {})
    if choice in candidates:
        delta = deepcopy(candidates[choice])
        delta.update({
            "source": "clarification_choice",
            "needs_llm": False,
            "clarification_resolved": True,
            "clarification_choice": choice,
            "clear_pending_clarification": True,
        })
        return delta

    if _CANCEL_CLARIFICATION_RE.fullmatch(normalized_answer):
        return {
            "dependency": "independent",
            "scope_action": "replace",
            "projection_action": "keep",
            "filter_action": "keep",
            "ranking_action": "keep",
            "temporal_action": "keep",
            "limit_action": "keep",
            "result_order_action": "none",
            "columns": [],
            "remove_columns": [],
            "filters": [],
            "remove_filter_columns": [],
            "order_by": None,
            "limit": None,
            "sample_ids": [],
            "temporal_metrics": [],
            "strict_projection": False,
            "explicit_columns": [],
            "needs_llm": False,
            "source": "clarification_cancel",
            "confidence": "high",
            "clarification_cancelled": True,
            "clear_pending_clarification": True,
        }

    reason = pending.get("reason")
    recognized_columns = sorted(match_question_semantic_columns(answer))
    if reason == "unknown_field" and recognized_columns:
        dependency = pending.get("dependency", "previous_query")
        return {
            "dependency": dependency,
            "scope_action": pending.get("scope_action", "keep"),
            "projection_action": "replace",
            "filter_action": "keep",
            "ranking_action": "clear" if dependency == "previous_result_set" else "keep",
            "temporal_action": "clear" if dependency == "previous_result_set" else "keep",
            "limit_action": "clear" if dependency == "previous_result_set" else "keep",
            "result_order_action": "preserve" if dependency == "previous_result_set" else "none",
            "columns": recognized_columns,
            "remove_columns": [],
            "filters": [],
            "remove_filter_columns": [],
            "order_by": None,
            "limit": None,
            "sample_ids": [],
            "temporal_metrics": [],
            "strict_projection": True,
            "explicit_columns": recognized_columns,
            "needs_llm": False,
            "source": "clarification_text",
            "confidence": "high",
            "clarification_resolved": True,
            "clear_pending_clarification": True,
        }

    if reason == "missing_reference":
        sample_ids = extract_requested_sample_ids(normalize_question_sample_ids(answer))
        if sample_ids:
            columns = list(pending.get("columns", []))
            return {
                "dependency": "independent",
                "scope_action": "replace",
                "projection_action": "keep",
                "filter_action": "keep",
                "ranking_action": "keep",
                "temporal_action": "keep",
                "limit_action": "keep",
                "result_order_action": "none",
                "columns": columns,
                "remove_columns": [],
                "filters": [],
                "remove_filter_columns": [],
                "order_by": None,
                "limit": None,
                "sample_ids": sample_ids,
                "temporal_metrics": [],
                "strict_projection": True,
                "explicit_columns": columns,
                "needs_llm": False,
                "source": "clarification_sample",
                "confidence": "high",
                "clarification_resolved": True,
                "clear_pending_clarification": True,
            }
    return None


def detect_clarification_need(
    question: str,
    memory: dict[str, Any],
    delta: dict[str, Any],
) -> dict[str, Any]:
    """只在无法安全确定语义时请求澄清。"""

    normalized = normalize_question_sample_ids(question)
    previous_spec = _previous_spec(memory)
    last_scope = _scope_ids(memory.get("last_result_scope"))
    parent_scope = _scope_ids(memory.get("parent_result_scope"))
    single_scope = _scope_ids(_latest_scope_by_kind(memory, "single"))
    multi_scope = _scope_ids(_latest_scope_by_kind(memory, "multi"))
    explicit_columns = list(delta.get("explicit_columns") or delta.get("columns") or [])

    if _contains_write_intent(normalized):
        return {"required": False}

    # 单数与复数代词分别查找最近锚点。中间穿插单样本查询不会覆盖最近样本集合。
    singular_reference = bool(_SAME_SAMPLE_RE.search(normalized))
    plural_reference = bool(_PLURAL_SAMPLE_RE.search(normalized))
    if singular_reference and not single_scope:
        return {
            "required": True,
            "reason": "missing_reference",
            "question": "当前会话中没有可供“它/这个样本”指代的单个样本。请提供样本编号，或重新给出完整查询。",
            "columns": explicit_columns,
            "dependency": "same_sample",
            "scope_action": "same_sample",
            "candidate_deltas": {},
        }
    if plural_reference and not multi_scope:
        return {
            "required": True,
            "reason": "missing_reference",
            "question": "当前会话中没有可供“它们/这些样本”指代的样本集合。请先给出一批样本，或重新给出完整查询。",
            "columns": explicit_columns,
            "dependency": "previous_result_set",
            "scope_action": "previous_result",
            "candidate_deltas": {},
        }

    # 当前词无法映射到Schema时停止猜测。
    current_spec = delta.get("current_spec", {})
    query_cue = re.search(r"查询|查看|返回|显示|多少|哪个|哪一个|呢|韧性|强度|性能", normalized)
    if query_cue and not explicit_columns and not _is_meaningful_spec(current_spec) and not _MODIFICATION_RE.search(normalized):
        return {
            "required": True,
            "reason": "unknown_field",
            "question": (
                "当前数据库没有识别出你要查询的字段。请改用明确字段，例如：原始/碳化密度、孔隙率、"
                "渗透率、热导率、比热容、热解热、表面发射率、表面温度、背面温度或质量。"
            ),
            "dependency": delta.get("dependency", "previous_query"),
            "scope_action": delta.get("scope_action", "keep"),
            "candidate_deltas": {},
        }

    # “换成X / X呢”在上一轮含排名时，既可能只换展示，也可能换排名目标。
    ambiguous_switch = bool(
        delta.get("dependency") in {"previous_query", "previous_result_set"}
        and explicit_columns
        and previous_spec.get("order_by")
        and re.search(r"换成|改成|改为|^\s*(?:那)?[^，。！？?]{1,16}呢[？?]?\s*$", normalized)
        and not re.search(r"返回|显示|列出|这些样本中|其中.*(?:最高|最低)|最高|最低|升序|降序", normalized)
    )
    if ambiguous_switch:
        column = explicit_columns[0]
        base = deepcopy(delta)
        base.update({"columns": explicit_columns, "projection_action": "replace", "strict_projection": True})
        preserve = deepcopy(base)
        preserve.update({
            "dependency": "previous_result_set" if last_scope else "previous_query",
            "scope_action": "previous_result" if last_scope else "keep",
            "ranking_action": "clear",
            "temporal_action": "clear" if last_scope else "keep",
            "limit_action": "clear" if last_scope else "keep",
            "result_order_action": "preserve" if last_scope else "none",
            "order_by": None,
        })
        current_high = deepcopy(base)
        current_high.update({
            "dependency": "previous_result_set",
            "scope_action": "previous_result",
            "ranking_action": "replace",
            "temporal_action": "clear",
            "limit_action": "keep",
            "result_order_action": "rerank",
            "order_by": {"kind": "column", "column": column, "direction": "DESC"},
        })
        current_low = deepcopy(current_high)
        current_low["order_by"] = {"kind": "column", "column": column, "direction": "ASC"}
        global_high = deepcopy(base)
        global_high.update({
            "dependency": "previous_query",
            "scope_action": "keep",
            "ranking_action": "replace",
            "temporal_action": "clear",
            "result_order_action": "rerank",
            "order_by": {"kind": "column", "column": column, "direction": "DESC"},
        })
        global_low = deepcopy(global_high)
        global_low["order_by"] = {"kind": "column", "column": column, "direction": "ASC"}
        return {
            "required": True,
            "reason": "ranking_intent_ambiguous",
            "question": (
                f"“换成{_column_label(column)}”有多种含义，请选择：\n"
                f"A. 保留当前样本及原顺序，只把展示字段换成{_column_label(column)}；\n"
                f"B. 在当前结果集合中按{_column_label(column)}从高到低排序；\n"
                f"C. 在当前结果集合中按{_column_label(column)}从低到高排序；\n"
                f"D. 将原查询改为按{_column_label(column)}从高到低重新选样本；\n"
                f"E. 将原查询改为按{_column_label(column)}从低到高重新选样本。"
            ),
            "candidate_deltas": {"A": preserve, "B": current_high, "C": current_low, "D": global_high, "E": global_low},
        }

    # 单数/复数锚点已经按最近使用原则唯一解析，不再因为存在父集合而重复追问。

    # 完全没有Schema字段、样本、条件或承接动作时，不进入昂贵RSL链路猜SQL。
    if (
        delta.get("dependency") == "independent"
        and not explicit_columns
        and not _is_meaningful_spec(current_spec)
        and not extract_requested_sample_ids(normalized)
        and not _CONTINUATION_PREFIX_RE.search(normalized)
    ):
        return {
            "required": True,
            "reason": "not_database_query",
            "question": "没有识别到可执行的材料数据库查询。请重新输入完整问题，并包含要查询的材料字段、样本编号、条件或排序要求。",
            "candidate_deltas": {},
        }

    return {"required": False}


def build_deterministic_query_delta(
    question: str,
    memory: dict[str, Any] | None,
) -> dict[str, Any]:
    """将当前轮拆成范围、投影、过滤、排序、聚合和数量六类变化。"""

    normalized = normalize_question_sample_ids(question)
    memory = deepcopy(memory or new_short_term_memory())
    pending = deepcopy(memory.get("pending_clarification") or {})
    pending_abandoned = False
    if pending:
        resolved_delta = _clarification_choice_delta(normalized, pending, memory)
        if resolved_delta is not None:
            return resolved_delta

        provisional_spec = build_query_spec(normalized)
        # 用户在澄清期间重新给出一个完整的新查询时，自动放弃旧澄清，避免被旧状态锁住。
        if _question_is_complete(provisional_spec, normalized):
            pending_abandoned = True
        else:
            attempts = int(pending.get("attempts", 0)) + 1
            return {
                "dependency": "independent",
                "scope_action": "replace",
                "projection_action": "keep",
                "filter_action": "keep",
                "ranking_action": "keep",
                "temporal_action": "keep",
                "limit_action": "keep",
                "result_order_action": "none",
                "columns": [],
                "remove_columns": [],
                "filters": [],
                "remove_filter_columns": [],
                "order_by": None,
                "limit": None,
                "sample_ids": [],
                "temporal_metrics": [],
                "strict_projection": False,
                "current_spec": provisional_spec,
                "explicit_columns": sorted(match_question_semantic_columns(normalized)),
                "needs_llm": False,
                "confidence": "low",
                "source": "clarification_unresolved",
                "clarification_unresolved": True,
                "clarification_attempts": attempts,
                "pending_snapshot": pending,
                "reason": "当前输入没有匹配澄清选项，也不能独立形成完整查询。",
            }
    current_spec = build_query_spec(normalized)
    previous_spec = _previous_spec(memory)

    explicit_columns = set(match_question_semantic_columns(normalized))
    requested_outputs = set(infer_requested_output_columns(normalized))
    filter_columns = {
        str(item.get("column", "")) for item in current_spec.get("filters", [])
    }
    projection_columns = set(requested_outputs)
    if not projection_columns:
        projection_columns = explicit_columns - {"point_index"}
        projection_cues = re.search(
            r"返回|显示|查看|看看|是多少|多少|哪个|哪一个|谁|只看|换成|"
            r"再看|加上|增加|补充|同时|顺便|呢",
            normalized,
        )
        if filter_columns and not projection_cues:
            projection_columns.difference_update(filter_columns)

    explicit_reference = _has_explicit_reference(normalized)
    independent_complete = _question_is_complete(current_spec, normalized)
    reset_query = bool(_RESET_QUERY_RE.search(normalized) and not explicit_reference)

    # 显式“重新查询”或完整QuerySpec在没有代词时优先作为新查询。
    # 只有明确代词或不能独立成句的修改片段才允许读取历史状态。
    is_follow_up = False
    if not reset_query and not (independent_complete and not explicit_reference):
        is_follow_up = _looks_like_follow_up(
            normalized, current_spec, memory, explicit_columns
        )

    dependency = "independent"
    if is_follow_up:
        if _RESULT_REFERENCE_RE.search(normalized):
            dependency = "previous_result_set"
        elif _SAME_SAMPLE_RE.search(normalized):
            dependency = "same_sample"
        else:
            previous_ids = list(previous_spec.get("sample_ids", []))
            active_ids = list(memory.get("active_sample_ids", []))
            if (
                projection_columns
                and not current_spec.get("sample_ids")
                and (len(previous_ids) == 1 or len(active_ids) == 1)
                and len(normalized.strip()) <= 20
                and not independent_complete
            ):
                dependency = "same_sample"
            else:
                dependency = "previous_query"

    current_limit = extract_requested_limit_from_question(normalized)
    if current_limit is None:
        followup_limit = re.search(
            r"(?:改成|改为|数量\s*(?:改成|改为)?|取)\s*(\d+)\s*(?:个|条)?",
            normalized,
            flags=re.IGNORECASE,
        )
        if followup_limit:
            current_limit = int(followup_limit.group(1))
        elif dependency != "independent" and re.search(r"哪个|哪一个|谁", normalized):
            current_limit = 1

    projection_action = "keep"
    remove_columns: set[str] = set()
    if dependency != "independent":
        if _REMOVE_PROJECTION_RE.search(normalized) and explicit_columns and not re.search(r"条件|筛选", normalized):
            projection_action = "remove"
            remove_columns = explicit_columns - {"point_index"}
        elif _ADD_PROJECTION_RE.search(normalized) and projection_columns:
            projection_action = "add"
        elif projection_columns:
            projection_action = "replace"
        elif _REPLACE_PROJECTION_RE.search(normalized):
            projection_action = "replace"

    filters = list(current_spec.get("filters", []))
    point_filter = _extract_followup_point_range(normalized)
    if point_filter and not any(item.get("column") == "point_index" for item in filters):
        filters.append(point_filter)

    filter_action = "keep"
    remove_filter_columns: set[str] = set()
    if dependency == "independent":
        filter_action = "replace" if filters else "keep"
    elif _CLEAR_ALL_FILTER_RE.search(normalized):
        filter_action = "clear"
    elif _REMOVE_FILTER_RE.search(normalized):
        filter_action = "remove"
        remove_filter_columns = explicit_columns | filter_columns
    elif filters:
        filter_action = "replace" if _REPLACE_FILTER_RE.search(normalized) else "add"

    explicit_order_by = _extract_explicit_order_by(normalized)
    order_by = explicit_order_by or deepcopy(current_spec.get("order_by"))
    ranking = infer_question_ranking_column(normalized)
    direction = _direction_from_question(normalized)
    if not order_by and ranking and direction:
        order_by = {"kind": "column", "column": ranking[0], "direction": direction}
    elif (
        not order_by
        and direction
        and dependency != "independent"
        and isinstance(previous_spec.get("order_by"), dict)
    ):
        order_by = deepcopy(previous_spec["order_by"])
        order_by["direction"] = direction

    ranking_action = "keep"
    if _CLEAR_RANKING_RE.search(normalized):
        ranking_action = "clear"
    elif order_by:
        ranking_action = "replace"
    elif dependency == "previous_result_set":
        ranking_action = "clear"

    temporal_metrics = deepcopy(current_spec.get("temporal_metrics", []))
    aggregation = _aggregation_from_question(normalized)
    response_columns = explicit_columns & _RESPONSE_COLUMNS
    if not temporal_metrics and aggregation and response_columns:
        temporal_metrics = [
            {
                "column": column,
                "aggregation": aggregation,
                "alias": canonical_metric_alias(column, aggregation),
            }
            for column in sorted(response_columns)
        ]

    temporal_action = "keep"
    if _CLEAR_TEMPORAL_RE.search(normalized):
        temporal_action = "clear"
    elif temporal_metrics:
        temporal_action = "replace"
    elif dependency == "previous_result_set":
        temporal_action = "clear"

    limit_action = "replace" if current_limit is not None else "keep"
    if dependency == "previous_result_set" and current_limit is None:
        limit_action = "clear"

    scope_action = "replace"
    if dependency == "previous_query":
        scope_action = "keep"
    elif dependency == "same_sample":
        scope_action = "same_sample"
    elif dependency == "previous_result_set":
        scope_action = "parent_result" if _PARENT_RESULT_RE.search(normalized) else "previous_result"

    strict_projection = bool(
        re.search(r"只返回|只显示|仅返回|仅显示|只看|换成|是多少|呢\s*[？?]?$", normalized)
    )
    if projection_action == "add":
        strict_projection = False

    actionable = bool(
        projection_columns
        or remove_columns
        or filters
        or remove_filter_columns
        or order_by
        or current_limit is not None
        or current_spec.get("sample_ids")
        or temporal_metrics
        or filter_action in {"clear", "remove"}
        or ranking_action == "clear"
        or temporal_action == "clear"
    )
    needs_llm = bool(dependency != "independent" and not actionable)

    return {
        "dependency": dependency,
        "scope_action": scope_action,
        "projection_action": projection_action,
        "filter_action": filter_action,
        "ranking_action": ranking_action,
        "temporal_action": temporal_action,
        "limit_action": limit_action,
        "result_order_action": (
            "preserve"
            if dependency == "previous_result_set"
            and projection_action == "replace"
            and not order_by
            and bool(re.search(r"返回|显示|列出", normalized))
            else "rerank" if order_by else "none"
        ),
        "columns": sorted(projection_columns & _known_columns()),
        "remove_columns": sorted(remove_columns & _known_columns()),
        "filters": filters,
        "remove_filter_columns": sorted(remove_filter_columns & _known_columns()),
        "order_by": order_by,
        "limit": current_limit,
        "sample_ids": sorted(extract_requested_sample_ids(normalized)),
        "temporal_metrics": temporal_metrics,
        "strict_projection": strict_projection,
        "current_spec": current_spec,
        "explicit_columns": sorted(explicit_columns & _known_columns()),
        "needs_llm": needs_llm,
        "confidence": "medium" if needs_llm else "high",
        "source": "new_query_after_clarification" if pending_abandoned else "deterministic",
        "clear_pending_clarification": pending_abandoned,
        "explicit_reference": explicit_reference,
        "independent_complete": bool(dependency == "independent" and independent_complete),
        "reset_query": reset_query,
        "reason": (
            "当前轮包含可确定的范围、字段、条件、排序或数量变化。"
            if dependency != "independent"
            else "当前问题可独立理解，不使用历史查询状态。"
        ),
    }


def build_query_delta_prompts(
    question: str,
    memory: dict[str, Any],
    base_delta: dict[str, Any],
) -> tuple[str, str]:
    allowed_columns = sorted(_known_columns())
    recent_turns = list(memory.get("recent_user_turns", []))[-2:]
    previous_spec = _previous_spec(memory)
    system_prompt = """
你只负责判断当前短句如何修改上一轮Text2SQL状态，不生成SQL。
只能输出一个JSON对象，不要输出Markdown。
dependency只能是independent、same_sample、previous_result_set、previous_query；
projection_action只能是keep、replace、add、remove；
columns和remove_columns只能使用给定真实字段；
“这些样本/它们/其中”优先使用previous_result_set；
“它/该样本/这个样本”使用same_sample；
当前轮明确字段且没有“再加上/同时/另外”时使用replace。
输出：{"dependency":"...","projection_action":"...","columns":[],"remove_columns":[],"reason":"..."}
""".strip()
    user_prompt = json.dumps(
        {
            "current_question": question,
            "allowed_columns": allowed_columns,
            "previous_query_spec": previous_spec,
            "recent_user_turns": recent_turns,
            "base_delta": {
                key: base_delta.get(key)
                for key in ("dependency", "projection_action", "columns", "remove_columns")
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return system_prompt, user_prompt


def parse_query_delta_response(
    text: str,
    base_delta: dict[str, Any],
) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()
    object_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if object_match:
        cleaned = object_match.group(0)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        fallback = deepcopy(base_delta)
        fallback["source"] = "deterministic_fallback"
        fallback["needs_llm"] = False
        fallback["llm_parse_error"] = "LLM没有返回有效JSON。"
        return fallback

    merged = deepcopy(base_delta)
    dependency = str(payload.get("dependency", ""))
    if dependency in _DEPENDENCIES:
        merged["dependency"] = dependency
    action = str(payload.get("projection_action", ""))
    if action in _PROJECTION_ACTIONS:
        merged["projection_action"] = action
    known = _known_columns()
    for key in ("columns", "remove_columns"):
        values = payload.get(key, [])
        if isinstance(values, list):
            cleaned_values = sorted({str(value) for value in values if str(value) in known})
            if cleaned_values:
                merged[key] = cleaned_values
    merged["source"] = "llm_fallback"
    merged["needs_llm"] = False
    merged["confidence"] = "medium"
    merged["reason"] = str(payload.get("reason") or merged.get("reason", ""))
    return merged


def _stable_previous_spec(previous_spec: dict[str, Any]) -> dict[str, Any]:
    stable_keys = (
        "select_columns",
        "filters",
        "where_filters",
        "having_filters",
        "order_by",
        "limit",
        "sample_ids",
        "strict_projection",
        "temporal_metrics",
        "all_temporal_metrics",
        "scalar_columns",
        "scalar_tables",
    )
    return {key: deepcopy(previous_spec[key]) for key in stable_keys if key in previous_spec}


def _apply_query_delta(
    previous_spec: dict[str, Any],
    delta: dict[str, Any],
    memory: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str], dict[str, Any]]:
    dependency = str(delta.get("dependency", "previous_query"))
    # 结果集合引用只继承样本范围，不继承生成该集合的过滤、排序和聚合。
    merged: dict[str, Any] = {} if dependency == "previous_result_set" else _stable_previous_spec(previous_spec)
    inherited_fields = list(merged)
    overridden_fields: list[str] = []
    resolution_meta: dict[str, Any] = {}

    explicit_ids = list(delta.get("sample_ids", []))
    if explicit_ids:
        merged["sample_ids"] = explicit_ids
        overridden_fields.append("sample_ids")
    elif dependency == "previous_result_set":
        scope_ids, scope_source = _select_result_scope(
            memory,
            requested_limit=delta.get("limit"),
            prefer_parent=delta.get("scope_action") == "parent_result",
        )
        if scope_ids:
            merged["sample_ids"] = scope_ids
            overridden_fields.append("sample_ids")
            if delta.get("result_order_action") == "preserve":
                merged["result_order_sample_ids"] = list(scope_ids)
                overridden_fields.append("result_order_sample_ids")
            else:
                merged.pop("result_order_sample_ids", None)
        resolution_meta["result_scope_source"] = scope_source
        resolution_meta["result_scope_size"] = len(scope_ids)
    elif dependency == "same_sample":
        # 单数代词始终回指最近一次单样本锚点，而不是最近任意结果。
        ids = _scope_ids(_latest_scope_by_kind(memory, "single"))
        if not ids:
            previous_ids = list(previous_spec.get("sample_ids", []))
            if len(previous_ids) == 1:
                ids = previous_ids
        if not ids:
            active_ids = list(memory.get("active_sample_ids", []))
            if len(active_ids) == 1:
                ids = active_ids
        if ids:
            merged["sample_ids"] = ids[:1]
            overridden_fields.append("sample_ids")
            resolution_meta["single_referent_source"] = "last_single_sample_scope"

    action = str(delta.get("projection_action", "keep"))
    columns = set(delta.get("columns", []))
    remove_columns = set(delta.get("remove_columns", []))
    previous_columns = set(merged.get("select_columns", [])) | set(merged.get("scalar_columns", []))
    if action == "replace" and columns:
        selected = set(columns)
        if not delta.get("strict_projection"):
            selected.add("sample_id")
        else:
            selected.add("sample_id")
        merged["select_columns"] = _ordered_projection(selected)
        merged["scalar_columns"] = sorted(columns - {"sample_id"})
        merged["strict_projection"] = bool(delta.get("strict_projection", False))
        overridden_fields.extend(["select_columns", "scalar_columns", "strict_projection"])
    elif action == "add" and columns:
        selected = previous_columns | columns | {"sample_id"}
        merged["select_columns"] = _ordered_projection(selected)
        merged["scalar_columns"] = sorted(selected - {"sample_id"})
        merged["strict_projection"] = False
        overridden_fields.extend(["select_columns", "scalar_columns", "strict_projection"])
    elif action == "remove" and remove_columns:
        selected = previous_columns - remove_columns
        selected.add("sample_id")
        merged["select_columns"] = _ordered_projection(selected)
        merged["scalar_columns"] = sorted(selected - {"sample_id"})
        overridden_fields.extend(["select_columns", "scalar_columns"])

    filter_action = str(delta.get("filter_action", "keep"))
    previous_filters = list(merged.get("filters", []))
    if filter_action == "clear":
        merged["filters"] = []
        merged["where_filters"] = []
        merged["having_filters"] = []
        overridden_fields.extend(["filters", "where_filters", "having_filters"])
    elif filter_action == "remove":
        removed = set(delta.get("remove_filter_columns", []))
        current_filters = _remove_filters_by_column(previous_filters, removed)
        merged["filters"] = current_filters
        merged["where_filters"] = deepcopy(current_filters)
        merged["having_filters"] = []
        overridden_fields.extend(["filters", "where_filters", "having_filters"])
    elif filter_action in {"add", "replace"} and delta.get("filters"):
        current = list(delta["filters"])
        if filter_action == "replace":
            filters = _replace_filters_by_column(previous_filters, current)
        else:
            filters = _replace_filters_by_column(previous_filters, current)
        merged["filters"] = filters
        merged["where_filters"] = deepcopy(filters)
        merged["having_filters"] = []
        overridden_fields.extend(["filters", "where_filters", "having_filters"])

    ranking_action = str(delta.get("ranking_action", "keep"))
    if ranking_action == "clear":
        merged["order_by"] = None
        overridden_fields.append("order_by")
    elif ranking_action == "replace" and delta.get("order_by"):
        merged["order_by"] = deepcopy(delta["order_by"])
        overridden_fields.append("order_by")

    temporal_action = str(delta.get("temporal_action", "keep"))
    if temporal_action == "clear":
        merged["temporal_metrics"] = []
        merged["all_temporal_metrics"] = []
        overridden_fields.extend(["temporal_metrics", "all_temporal_metrics"])
    elif temporal_action == "replace" and delta.get("temporal_metrics"):
        metrics = deepcopy(delta["temporal_metrics"])
        merged["temporal_metrics"] = metrics
        merged["all_temporal_metrics"] = deepcopy(metrics)
        overridden_fields.extend(["temporal_metrics", "all_temporal_metrics"])
        if isinstance(delta.get("order_by"), dict) and metrics:
            metric = metrics[0]
            merged["order_by"] = {
                "kind": "metric",
                "column": metric.get("column"),
                "alias": metric.get("alias"),
                "direction": delta["order_by"].get("direction", "DESC"),
            }
            overridden_fields.append("order_by")

    limit_action = str(delta.get("limit_action", "keep"))
    if limit_action == "clear":
        merged["limit"] = None
        overridden_fields.append("limit")
    elif limit_action == "replace" and delta.get("limit") is not None:
        merged["limit"] = int(delta["limit"])
        overridden_fields.append("limit")

    overridden_fields = list(dict.fromkeys(overridden_fields))
    inherited_fields = [
        key for key in dict.fromkeys(inherited_fields) if key not in overridden_fields
    ]
    return merged, inherited_fields, overridden_fields, resolution_meta


def render_query_spec_as_question(spec: dict[str, Any]) -> str:
    pieces: list[str] = []
    sample_ids = list(spec.get("sample_ids", []))
    if len(sample_ids) == 1:
        pieces.append(f"查询样本{sample_ids[0]}")
    elif sample_ids:
        pieces.append("查询指定样本集合")
    elif spec.get("temporal_metrics"):
        pieces.append("查询每个样本")
    else:
        pieces.append("查询样本")

    metrics = list(spec.get("temporal_metrics", []))
    metric_phrases: list[str] = []
    for metric in metrics:
        aggregation = str(metric.get("aggregation", ""))
        label = _column_label(str(metric.get("column", "")))
        suffix = {
            "MAX": "峰值",
            "MIN": "最小值",
            "AVG": "平均值",
            "SUM": "总和",
            "FINAL": "最终值",
        }.get(aggregation, aggregation)
        metric_phrases.append(f"{label}{suffix}")
    if metric_phrases:
        pieces.append("计算" + "、".join(metric_phrases))

    filter_phrases: list[str] = []
    for item in spec.get("filters", []):
        label = _column_label(str(item.get("column", "")))
        if item.get("operator") == "BETWEEN":
            filter_phrases.append(f"{label}在{item.get('value')}到{item.get('value2')}之间")
        else:
            operator_text = {
                ">": "大于",
                ">=": "大于等于",
                "<": "小于",
                "<=": "小于等于",
                "=": "等于",
            }.get(str(item.get("operator")), str(item.get("operator")))
            filter_phrases.append(f"{label}{operator_text}{item.get('value')}")
    if filter_phrases:
        pieces.append("条件为" + "且".join(filter_phrases))

    selected = list(dict.fromkeys(spec.get("select_columns", [])))
    output_labels = [_column_label(column) for column in selected]
    output_labels.extend(metric_phrases)
    output_labels = list(dict.fromkeys(output_labels))
    if output_labels:
        marker = "只返回" if spec.get("strict_projection") else "返回"
        pieces.append(marker + "、".join(output_labels))

    order_by = spec.get("order_by")
    if isinstance(order_by, dict) and order_by.get("column"):
        label = _column_label(str(order_by["column"]))
        if order_by.get("kind") == "metric":
            metric = next(
                (item for item in metrics if item.get("column") == order_by.get("column")),
                None,
            )
            if metric:
                label += {
                    "MAX": "峰值",
                    "MIN": "最小值",
                    "AVG": "平均值",
                    "FINAL": "最终值",
                }.get(str(metric.get("aggregation")), "")
        direction_text = "降序" if order_by.get("direction") == "DESC" else "升序"
        pieces.append(f"按{label}{direction_text}排列")

    limit = spec.get("limit")
    if isinstance(limit, int) and limit > 0:
        pieces.append(f"取前{limit}个样本")
    if len(sample_ids) > 1:
        pieces.append("样本编号限定为" + "、".join(sample_ids))
    return "，".join(pieces) + "。"


def _spec_role_columns(spec: dict[str, Any]) -> set[str]:
    columns = set(spec.get("select_columns", []))
    columns.update(spec.get("scalar_columns", []))
    columns.update(str(item.get("column", "")) for item in spec.get("filters", []))
    columns.update(str(item.get("column", "")) for item in spec.get("temporal_metrics", []))
    order_by = spec.get("order_by")
    if isinstance(order_by, dict) and order_by.get("column"):
        columns.add(str(order_by["column"]))
    return {column for column in columns if column}


def validate_current_turn_coverage(
    query_delta: dict[str, Any],
    final_spec: dict[str, Any],
    previous_spec: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    actual_columns = _spec_role_columns(final_spec)
    action = str(query_delta.get("projection_action", "keep"))
    expected_columns = set(query_delta.get("columns", []))
    removed_columns = set(query_delta.get("remove_columns", []))
    selected = set(final_spec.get("select_columns", []))

    missing_columns = sorted(expected_columns - actual_columns) if action in {"replace", "add"} else []
    removed_columns_still_present = sorted(removed_columns & selected) if action == "remove" else []

    stale_projection_columns: list[str] = []
    if action == "replace" and query_delta.get("strict_projection"):
        allowed = expected_columns | {"sample_id"}
        stale_projection_columns = sorted(selected - allowed)

    stale_ranking = bool(
        query_delta.get("ranking_action") == "clear" and final_spec.get("order_by")
    )
    stale_temporal_metrics = bool(
        query_delta.get("temporal_action") == "clear" and final_spec.get("temporal_metrics")
    )
    stale_filters = bool(
        query_delta.get("filter_action") == "clear" and final_spec.get("filters")
    )
    removed_filter_columns = set(query_delta.get("remove_filter_columns", []))
    removed_filters_still_present = sorted(
        removed_filter_columns
        & {str(item.get("column", "")) for item in final_spec.get("filters", [])}
    )

    limit_mismatch = False
    if query_delta.get("limit_action") == "replace":
        limit_mismatch = final_spec.get("limit") != query_delta.get("limit")
    elif query_delta.get("limit_action") == "clear":
        limit_mismatch = final_spec.get("limit") is not None

    sample_scope_missing = bool(
        query_delta.get("dependency") in {"same_sample", "previous_result_set"}
        and not final_spec.get("sample_ids")
    )

    passed = not any(
        [
            missing_columns,
            removed_columns_still_present,
            stale_projection_columns,
            stale_ranking,
            stale_temporal_metrics,
            stale_filters,
            removed_filters_still_present,
            limit_mismatch,
            sample_scope_missing,
        ]
    )
    return passed, {
        "passed": passed,
        "missing_columns": missing_columns,
        "removed_columns_still_present": removed_columns_still_present,
        "stale_projection_columns": stale_projection_columns,
        "stale_ranking": stale_ranking,
        "stale_temporal_metrics": stale_temporal_metrics,
        "stale_filters": stale_filters,
        "removed_filters_still_present": removed_filters_still_present,
        "limit_mismatch": limit_mismatch,
        "sample_scope_missing": sample_scope_missing,
        "actual_role_columns": sorted(actual_columns),
        "previous_role_columns": sorted(_spec_role_columns(previous_spec)),
    }


def _overlay_merged_roles(
    rebuilt: dict[str, Any],
    merged: dict[str, Any],
    delta: dict[str, Any],
) -> dict[str, Any]:
    final_spec = deepcopy(rebuilt)
    for key in (
        "sample_ids",
        "filters",
        "where_filters",
        "having_filters",
        "order_by",
        "limit",
        "temporal_metrics",
        "all_temporal_metrics",
        "result_order_sample_ids",
    ):
        if key in merged:
            final_spec[key] = deepcopy(merged[key])

    if delta.get("projection_action") in {"replace", "add", "remove"} or delta.get("dependency") == "previous_result_set":
        final_spec["select_columns"] = _ordered_projection(merged.get("select_columns", []))
        final_spec["strict_projection"] = bool(merged.get("strict_projection", False))

    role_columns = set(final_spec.get("select_columns", []))
    role_columns.update(
        str(item.get("column", "")) for item in final_spec.get("filters", [])
    )
    order_by = final_spec.get("order_by")
    if isinstance(order_by, dict) and order_by.get("column"):
        role_columns.add(str(order_by["column"]))
    role_columns.update(
        str(item.get("column", "")) for item in final_spec.get("temporal_metrics", [])
    )
    final_spec["scalar_columns"] = sorted(
        column for column in role_columns if column not in {"sample_id", "point_index"} and column not in _RESPONSE_COLUMNS
    )
    final_spec["memory_resolved"] = True
    final_spec["structured_context_complete"] = True
    return final_spec


def resolve_conversation_context(
    question: str,
    memory: dict[str, Any] | None,
    query_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_question_sample_ids(question)
    memory = deepcopy(memory or new_short_term_memory())
    current_spec = build_query_spec(normalized)
    previous_spec = _previous_spec(memory)
    delta = deepcopy(query_delta or build_deterministic_query_delta(normalized, memory))

    # 有效选项、新完整查询或显式取消都会先清空旧澄清，避免旧澄清再次触发自身。
    if delta.get("clear_pending_clarification"):
        memory["pending_clarification"] = {}

    if delta.get("clarification_cancelled"):
        return {
            "turn_type": "clarification_cancelled",
            "memory_used": bool(previous_spec),
            "resolved_question": normalized,
            "resolved_query_spec": {},
            "context_resolution_valid": False,
            "current_turn_coverage": {"passed": False, "mode": "clarification_cancelled"},
            "context_resolution": {"reason": "用户取消了待澄清问题。"},
            "inherited_fields": [],
            "overridden_fields": [],
            "clarification_required": True,
            "clarification_cancelled": True,
            "clarification_question": "已取消本次澄清。你可以直接输入一个新的完整数据库问题。",
            "pending_clarification": {},
        }

    if delta.get("clarification_unresolved"):
        pending = deepcopy(delta.get("pending_snapshot") or memory.get("pending_clarification") or {})
        attempts = int(delta.get("clarification_attempts", 1))
        if attempts >= int(pending.get("max_attempts", MAX_CLARIFICATION_ATTEMPTS)):
            memory["pending_clarification"] = {}
            return {
                "turn_type": "clarification_cancelled",
                "memory_used": bool(previous_spec),
                "resolved_question": normalized,
                "resolved_query_spec": {},
                "context_resolution_valid": False,
                "current_turn_coverage": {"passed": False, "mode": "clarification_cancelled"},
                "context_resolution": {"reason": "澄清回答连续无法解析，已自动退出。"},
                "inherited_fields": [],
                "overridden_fields": [],
                "clarification_required": True,
                "clarification_cancelled": True,
                "clarification_question": "连续两次没有识别到有效澄清回答，本次澄清已自动取消。请重新输入一个完整数据库问题。",
                "pending_clarification": {},
            }
        pending["attempts"] = attempts
        pending["max_attempts"] = int(pending.get("max_attempts", MAX_CLARIFICATION_ATTEMPTS))
        retry_question = str(pending.get("question") or "请按给出的选项回答，或输入“取消”。")
        return {
            "turn_type": "clarification_required",
            "memory_used": bool(previous_spec),
            "resolved_question": normalized,
            "resolved_query_spec": {},
            "context_resolution_valid": False,
            "current_turn_coverage": {"passed": False, "mode": "clarification_retry"},
            "context_resolution": {"reason": "澄清回答无法解析。"},
            "inherited_fields": [],
            "overridden_fields": [],
            "clarification_required": True,
            "clarification_cancelled": False,
            "clarification_question": "没有识别到有效选项。" + retry_question + "\n也可以输入“取消”结束本次澄清。",
            "pending_clarification": pending,
        }

    # 已解析的澄清候选是确定动作，不再进入同一轮歧义检测。
    clarification = {"required": False}
    if not delta.get("clarification_resolved"):
        clarification = detect_clarification_need(normalized, memory, delta)
    if clarification.get("required"):
        return {
            "turn_type": "clarification_required",
            "memory_used": bool(previous_spec),
            "resolved_question": normalized,
            "resolved_query_spec": {},
            "context_resolution_valid": False,
            "current_turn_coverage": {"passed": False, "mode": "clarification"},
            "context_resolution": {"reason": clarification.get("reason", "ambiguous")},
            "inherited_fields": [],
            "overridden_fields": [],
            "clarification_required": True,
            "clarification_cancelled": False,
            "clarification_question": clarification.get("question", "请补充必要信息。"),
            "pending_clarification": {
                **clarification,
                "original_question": normalized,
                "created_at": _utc_now_iso(),
                "attempts": 0,
                "max_attempts": MAX_CLARIFICATION_ATTEMPTS,
            },
        }

    if delta.get("clarification_resolved"):
        memory["pending_clarification"] = {}

    if not previous_spec or delta.get("dependency") == "independent":
        if delta.get("clarification_resolved") and (delta.get("sample_ids") or delta.get("columns")):
            clarified_seed = {
                "sample_ids": list(delta.get("sample_ids", [])),
                "select_columns": _ordered_projection({"sample_id", *delta.get("columns", [])}),
                "scalar_columns": sorted(set(delta.get("columns", [])) - {"sample_id"}),
                "filters": list(delta.get("filters", [])),
                "where_filters": list(delta.get("filters", [])),
                "having_filters": [],
                "order_by": deepcopy(delta.get("order_by")),
                "limit": delta.get("limit"),
                "temporal_metrics": list(delta.get("temporal_metrics", [])),
                "strict_projection": bool(delta.get("strict_projection", True)),
            }
            clarified_question = render_query_spec_as_question(clarified_seed)
            clarified_built = build_query_spec(clarified_question)
            if _is_meaningful_spec(clarified_built):
                current_spec = _overlay_merged_roles(clarified_built, clarified_seed, delta)
                normalized = clarified_question
        return {
            "turn_type": "new_query",
            "memory_used": False,
            "resolved_question": normalized,
            "resolved_query_spec": current_spec,
            "context_resolution_valid": True,
            "current_turn_coverage": {"passed": True, "mode": "independent"},
            "context_resolution": {
                "reason": "当前问题可独立理解或当前会话没有可继承的成功查询。",
                "previous_question": memory.get("last_resolved_question", ""),
                "query_delta_source": delta.get("source", "deterministic"),
            },
            "inherited_fields": [],
            "overridden_fields": [],
            "clarification_required": False,
            "clarification_cancelled": False,
            "clarification_question": "",
            "pending_clarification": {},
        }

    merged, inherited_fields, overridden_fields, resolution_meta = _apply_query_delta(
        previous_spec, delta, memory
    )
    resolved_question = render_query_spec_as_question(merged)
    rebuilt_spec = build_query_spec(resolved_question)
    if _is_meaningful_spec(rebuilt_spec):
        final_spec = _overlay_merged_roles(rebuilt_spec, merged, delta)
    else:
        final_spec = deepcopy(merged)
        final_spec.update(
            {
                "eligible": False,
                "mode": "rsl",
                "query_type": "complex_or_uncertain",
                "reason": "结构化记忆合并结果无法确定性编译，进入RSL路径。",
                "memory_resolved": True,
                "structured_context_complete": True,
            }
        )

    coverage_passed, coverage = validate_current_turn_coverage(
        delta, final_spec, previous_spec
    )
    final_spec["structured_context_complete"] = coverage_passed

    dependency = str(delta.get("dependency"))
    turn_type = "modify_previous"
    if dependency == "previous_result_set":
        turn_type = "reference_previous_results"
    elif dependency == "same_sample":
        turn_type = "follow_same_sample"
    elif delta.get("projection_action") == "add":
        turn_type = "add_projection"
    elif delta.get("projection_action") == "replace":
        turn_type = "replace_projection"
    elif delta.get("projection_action") == "remove":
        turn_type = "remove_projection"

    return {
        "turn_type": turn_type,
        "memory_used": True,
        "resolved_question": resolved_question,
        "resolved_query_spec": final_spec,
        "context_resolution_valid": coverage_passed,
        "current_turn_coverage": coverage,
        "context_resolution": {
            "reason": "按范围、投影、过滤、排序、聚合和数量六类动作合并最后一次成功状态。",
            "previous_question": memory.get("last_resolved_question", ""),
            "current_fragment": normalized,
            "query_delta_source": delta.get("source", "deterministic"),
            "query_delta_reason": delta.get("reason", ""),
            **resolution_meta,
        },
        "inherited_fields": inherited_fields,
        "overridden_fields": overridden_fields,
        "clarification_required": False,
        "clarification_cancelled": False,
        "clarification_question": "",
        "pending_clarification": {},
    }


def _extract_result_sample_ids(
    columns: list[str],
    rows: list[list[Any]],
) -> list[str]:
    indexes = [index for index, column in enumerate(columns) if str(column).lower() == "sample_id"]
    if not indexes:
        return []
    index = indexes[0]
    values: list[str] = []
    for row in rows:
        if index < len(row) and isinstance(row[index], str) and row[index].startswith("sample_"):
            values.append(row[index])
    return list(dict.fromkeys(values))


def update_short_term_memory(
    memory: dict[str, Any] | None,
    *,
    question: str,
    resolved_question: str,
    query_spec: dict[str, Any],
    validated_sql: str,
    columns: list[str],
    rows: list[list[Any]],
    row_count: int,
    truncated: bool,
    final_status: str,
    turn_type: str,
    query_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = deepcopy(memory or new_short_term_memory())
    delta = deepcopy(query_delta or {})
    sample_ids = _extract_result_sample_ids(columns, rows)
    sample_ids_truncated = len(sample_ids) > MAX_RESULT_SAMPLE_IDS
    stored_sample_ids = sample_ids[:MAX_RESULT_SAMPLE_IDS]

    previous_last_scope = deepcopy(updated.get("last_result_scope") or _empty_scope())
    previous_parent_scope = deepcopy(updated.get("parent_result_scope") or _empty_scope())
    source_ids = list(dict.fromkeys(query_spec.get("sample_ids", [])))

    if delta.get("dependency") == "previous_result_set":
        # 只有真正从较大候选集缩小结果时才建立父层级；仅换展示字段且集合不变时不制造重复父集合。
        source_members = set(source_ids)
        result_members = set(stored_sample_ids)
        previous_parent_members = set(_scope_ids(previous_parent_scope))
        # 仅成员发生变化时建立父集合；同一批样本只改变顺序或返回字段时，
        # 不制造 active=50 / parent=50 的重复层级。
        if source_ids and source_members != result_members:
            updated["parent_result_scope"] = _make_scope(
                source_ids,
                selection_order_by=updated.get("last_successful_query_state", {}).get("order_by"),
                row_count=len(source_ids),
                source_question=updated.get("last_resolved_question", ""),
            )
        elif previous_parent_members and previous_parent_members != result_members:
            updated["parent_result_scope"] = previous_parent_scope
        else:
            updated["parent_result_scope"] = _empty_scope()
    elif delta.get("dependency") == "independent":
        updated["parent_result_scope"] = _empty_scope()
    elif previous_last_scope.get("sample_ids"):
        # 同一查询链中的普通修改保留已有父层级，不额外扩张。
        updated.setdefault("parent_result_scope", _empty_scope())

    current_scope = _make_scope(
        stored_sample_ids,
        selection_order_by=query_spec.get("order_by"),
        row_count=int(row_count),
        truncated=bool(truncated or sample_ids_truncated),
        source_question=resolved_question,
    )
    updated["last_result_scope"] = current_scope

    # 单数与复数锚点独立成长。单样本查询不会覆盖“它们”的最近样本集合。
    if len(stored_sample_ids) == 1:
        updated["last_single_sample_scope"] = deepcopy(current_scope)
        _append_referent(updated, current_scope, kind="single")
    elif len(stored_sample_ids) > 1:
        updated["last_multi_sample_scope"] = deepcopy(current_scope)
        _append_referent(updated, current_scope, kind="multi")
    updated["updated_at"] = _utc_now_iso()
    updated["last_successful_question"] = question
    updated["last_resolved_question"] = resolved_question
    updated["last_successful_query_state"] = deepcopy(query_spec)
    updated["last_query_spec"] = deepcopy(query_spec)
    updated["last_validated_sql"] = validated_sql
    updated["last_result"] = {
        "columns": list(columns),
        "row_count": int(row_count),
        "sample_ids": stored_sample_ids,
        "truncated": bool(truncated or sample_ids_truncated),
    }
    updated["active_sample_ids"] = stored_sample_ids
    updated["pending_clarification"] = {}
    updated = mark_current_turn_status(
        updated,
        question=question,
        status=final_status,
        resolved_question=resolved_question,
    )

    recent_successful = list(updated.get("recent_turns", []))
    recent_successful.append(
        {
            "user_question": question,
            "resolved_question": resolved_question,
            "turn_type": turn_type,
            "status": final_status,
            "query_delta": {
                key: delta.get(key)
                for key in (
                    "dependency",
                    "scope_action",
                    "projection_action",
                    "filter_action",
                    "ranking_action",
                    "temporal_action",
                    "limit_action",
                    "columns",
                    "source",
                )
            },
        }
    )
    updated["recent_turns"] = recent_successful[-MAX_RECENT_USER_TURNS:]
    return updated


def format_memory_summary(memory: dict[str, Any] | None) -> str:
    if not memory:
        return "当前没有短期记忆。"
    spec = _previous_spec(memory)
    result = memory.get("last_result", {})
    last_scope = memory.get("last_result_scope", {})
    parent_scope = memory.get("parent_result_scope", {})
    single_scope = memory.get("last_single_sample_scope", {})
    multi_scope = memory.get("last_multi_sample_scope", {})
    raw_turns = memory.get("recent_user_turns", [])
    raw_summary = " | ".join(
        f"{item.get('user_question', '')}[{item.get('status', '')}]" for item in raw_turns
    ) or "无"
    lines = [
        f"session_id: {memory.get('session_id', '')}",
        f"最后成功问题: {memory.get('last_resolved_question') or '无'}",
        f"查询类型: {spec.get('query_type') or '无'}",
        f"返回字段: {', '.join(spec.get('select_columns', [])) or '无'}",
        f"样本约束: {', '.join(spec.get('sample_ids', [])) or '无'}",
        f"排序: {spec.get('order_by') or '无'}",
        f"数量限制: {spec.get('limit') if spec.get('limit') is not None else '无'}",
        f"上次结果行数: {result.get('row_count', 0)}",
        f"当前结果集合: {len(last_scope.get('sample_ids', []))}个样本",
        f"父结果集合: {len(parent_scope.get('sample_ids', []))}个样本",
        f"最近单样本锚点: {', '.join(single_scope.get('sample_ids', [])) or '无'}",
        f"最近多样本锚点: {len(multi_scope.get('sample_ids', []))}个样本",
        f"结果顺序: {'已保存' if last_scope.get('ordered_sample_ids') else '无'}",
        f"待澄清: {(memory.get('pending_clarification') or {}).get('reason', '无')}",
        f"澄清尝试: {(memory.get('pending_clarification') or {}).get('attempts', 0)}/{(memory.get('pending_clarification') or {}).get('max_attempts', MAX_CLARIFICATION_ATTEMPTS)}",
        f"最近原始输入: {raw_summary}",
    ]
    return "\n".join(lines)
