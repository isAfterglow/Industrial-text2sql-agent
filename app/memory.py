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
    extract_requested_limit_from_question,
    extract_requested_sample_ids,
    get_schema_catalog,
    infer_question_ranking_column,
    infer_requested_output_columns,
    match_question_semantic_columns,
    normalize_question_sample_ids,
)


# 第一版只保留最近两轮成功对话。结构化状态仍保存最后一次成功QuerySpec。
MAX_RECENT_TURNS = 2
MAX_RESULT_SAMPLE_IDS = 100

_DEPENDENCIES = {
    "independent",
    "same_sample",
    "previous_result_set",
    "previous_query",
}
_PROJECTION_ACTIONS = {"keep", "replace", "add", "remove"}

_CONTINUATION_PREFIX_RE = re.compile(
    r"^\s*(?:再|继续|然后|接着|改成|改为|换成|只看|只返回|只显示|"
    r"仅返回|仅显示|不要|去掉|移除|加上|增加|补充|顺便|另外|这些|"
    r"上述|上面|刚才|其中|它们|它的|它呢|这个样本|该样本|同一个样本|"
    r"那|那么)",
    flags=re.IGNORECASE,
)
_RESULT_REFERENCE_RE = re.compile(
    r"这些样本|上述样本|上面(?:这些|几个)?样本|刚才(?:返回|查到)的样本|"
    r"前面(?:返回|查到)的样本|这些结果|上述结果|其中",
    flags=re.IGNORECASE,
)
_SAME_SAMPLE_RE = re.compile(
    r"它的|它呢|这个样本|该样本|同一个样本|继续看它|再看它|"
    r"这个材料|该材料",
    flags=re.IGNORECASE,
)
_ADD_PROJECTION_RE = re.compile(
    r"再(?:返回|显示|看)|还(?:要|需要)?(?:返回|显示|看)?|顺便|另外|"
    r"加上|增加|补充|同时(?:返回|显示|看)",
    flags=re.IGNORECASE,
)
_REPLACE_PROJECTION_RE = re.compile(
    r"只返回|只显示|仅返回|仅显示|只看|换成|改查|不要原来的",
    flags=re.IGNORECASE,
)
_REMOVE_PROJECTION_RE = re.compile(
    r"不要(?:返回|显示|看)?|去掉|移除|删除.*字段",
    flags=re.IGNORECASE,
)
_RESET_FILTER_RE = re.compile(
    r"取消条件|去掉条件|不要条件|清除筛选|取消筛选",
    flags=re.IGNORECASE,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return "session-" + uuid.uuid4().hex[:10]


def new_short_term_memory(
    session_id: str | None = None,
) -> dict[str, Any]:
    return {
        "session_id": session_id or new_session_id(),
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "last_successful_question": "",
        "last_resolved_question": "",
        "last_query_spec": {},
        "last_validated_sql": "",
        "last_result": {
            "columns": [],
            "row_count": 0,
            "sample_ids": [],
            "truncated": False,
        },
        "active_sample_ids": [],
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


def _known_columns() -> set[str]:
    catalog = get_schema_catalog()
    columns: set[str] = set()
    for table in catalog.get("tables", {}).values():
        columns.update(table.get("columns", {}).keys())
    return columns


def _column_label(column: str) -> str:
    catalog = get_schema_catalog()
    terms = catalog.get("semantic_terms", {}).get(column, [])
    if terms:
        return str(terms[0])
    return column


def _metric_alias(column: str, aggregation: str) -> str:
    prefix = {
        "MAX": "peak",
        "MIN": "minimum",
        "AVG": "average",
        "SUM": "sum",
        "FINAL": "final",
    }.get(aggregation, aggregation.lower())
    return f"{prefix}_{column}"


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
        rf"({NUMERIC_LITERAL_PATTERN})\s*(?:到|至|~|～)\s*"
        rf"({NUMERIC_LITERAL_PATTERN})",
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
        return {
            "column": "point_index",
            "operator": "=",
            "value": exact.group(1),
        }
    return None


def _replace_filters_by_column(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not current:
        return deepcopy(previous)
    current_columns = {str(item.get("column", "")) for item in current}
    retained = [
        deepcopy(item)
        for item in previous
        if str(item.get("column", "")) not in current_columns
    ]
    return retained + deepcopy(current)


def _looks_like_follow_up(
    question: str,
    current_spec: dict[str, Any],
    memory: dict[str, Any],
    explicit_columns: set[str],
) -> bool:
    previous_spec = memory.get("last_query_spec", {})
    if not previous_spec:
        return False

    if _CONTINUATION_PREFIX_RE.search(question):
        return True
    if _RESULT_REFERENCE_RE.search(question) or _SAME_SAMPLE_RE.search(question):
        return True

    # “碳化密度呢”“原始孔隙率是多少”一类短句，在上一轮只有一个活跃样本时，
    # 默认视为继续查询同一样本。
    previous_ids = list(previous_spec.get("sample_ids", []))
    active_ids = list(memory.get("active_sample_ids", []))
    if (
        len(question.strip()) <= 20
        and explicit_columns
        and (len(previous_ids) == 1 or len(active_ids) == 1)
        and not current_spec.get("sample_ids")
        and not current_spec.get("order_by")
        and not current_spec.get("filters")
    ):
        return True

    # 已经能够独立形成完整查询时，不继承历史，避免条件污染。
    if current_spec.get("eligible"):
        return False
    if (
        current_spec.get("query_type")
        in {"multi_table_topk", "multi_table_filter", "multi_table_projection"}
        and (
            current_spec.get("select_columns")
            or current_spec.get("filters")
            or current_spec.get("sample_ids")
        )
    ):
        return False

    return len(question.strip()) <= 24 and _is_meaningful_spec(current_spec)


def build_deterministic_query_delta(
    question: str,
    memory: dict[str, Any] | None,
) -> dict[str, Any]:
    """提取当前轮相对上一成功状态的结构化变化。

    规则只负责高确定性的词法和Schema字段识别；自然语言特别模糊时，
    nodes.py会调用一次轻量LLM兜底，LLM只输出QueryDelta，不生成SQL。
    """

    normalized = normalize_question_sample_ids(question)
    memory = deepcopy(memory or new_short_term_memory())
    current_spec = build_query_spec(normalized)
    previous_spec = deepcopy(memory.get("last_query_spec", {}))

    explicit_columns = set(match_question_semantic_columns(normalized))
    requested_outputs = set(infer_requested_output_columns(normalized))
    projection_columns = set(requested_outputs)
    if not projection_columns:
        projection_columns = explicit_columns - {"point_index"}
        filter_columns = {
            str(item.get("column", ""))
            for item in current_spec.get("filters", [])
        }
        projection_cues = re.search(
            r"返回|显示|查看|看看|是多少|多少|哪个|哪一个|谁|最高|最低|"
            r"最大|最小|只看|换成|再看|加上|增加|补充|同时|顺便|呢",
            normalized,
        )
        # “再筛选原始密度大于400”中的rhov_i是过滤字段，不应覆盖上一轮返回字段。
        if filter_columns and not projection_cues:
            projection_columns.difference_update(filter_columns)

    is_follow_up = _looks_like_follow_up(
        normalized,
        current_spec,
        memory,
        explicit_columns,
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
            ):
                dependency = "same_sample"
            else:
                dependency = "previous_query"

    projection_action = "keep"
    remove_columns: set[str] = set()
    if dependency != "independent":
        if _REMOVE_PROJECTION_RE.search(normalized) and explicit_columns:
            projection_action = "remove"
            remove_columns = explicit_columns - {"point_index"}
        elif _ADD_PROJECTION_RE.search(normalized) and projection_columns:
            projection_action = "add"
        elif projection_columns:
            # 承接问题中出现明确字段时，默认以本轮字段替换旧返回字段。
            # “再返回/加上/同时”才执行字段并集。
            projection_action = "replace"
        elif _REPLACE_PROJECTION_RE.search(normalized):
            projection_action = "replace"

    current_limit = extract_requested_limit_from_question(normalized)
    if current_limit is None:
        followup_limit = re.search(
            r"(?:改成|改为|数量(?:改成|改为)?|取)\s*(\d+)\s*(?:个|条)?",
            normalized,
            flags=re.IGNORECASE,
        )
        if followup_limit:
            current_limit = int(followup_limit.group(1))
        elif dependency != "independent" and re.search(r"哪个|哪一个|谁", normalized):
            current_limit = 1

    filters = list(current_spec.get("filters", []))
    point_filter = _extract_followup_point_range(normalized)
    if point_filter and not any(
        item.get("column") == "point_index" for item in filters
    ):
        filters.append(point_filter)

    order_by = deepcopy(current_spec.get("order_by"))
    ranking = infer_question_ranking_column(normalized)
    direction = _direction_from_question(normalized)
    if not order_by and ranking and direction:
        order_by = {
            "kind": "column",
            "column": ranking[0],
            "direction": direction,
        }
    elif (
        not order_by
        and direction
        and dependency != "independent"
        and isinstance(previous_spec.get("order_by"), dict)
    ):
        order_by = deepcopy(previous_spec["order_by"])
        order_by["direction"] = direction

    temporal_metrics = deepcopy(current_spec.get("temporal_metrics", []))
    aggregation = _aggregation_from_question(normalized)
    response_columns = explicit_columns & {
        "surface_temperature",
        "back_temperature",
        "mass",
    }
    if not temporal_metrics and aggregation and response_columns:
        temporal_metrics = [
            {
                "column": column,
                "aggregation": aggregation,
                "alias": _metric_alias(column, aggregation),
            }
            for column in sorted(response_columns)
        ]

    actionable = bool(
        projection_columns
        or remove_columns
        or filters
        or order_by
        or current_limit is not None
        or current_spec.get("sample_ids")
        or temporal_metrics
        or _RESET_FILTER_RE.search(normalized)
    )
    needs_llm = bool(dependency != "independent" and not actionable)

    return {
        "dependency": dependency,
        "projection_action": projection_action,
        "columns": sorted(projection_columns & _known_columns()),
        "remove_columns": sorted(remove_columns & _known_columns()),
        "filters": filters,
        "clear_filters": bool(_RESET_FILTER_RE.search(normalized)),
        "order_by": order_by,
        "limit": current_limit,
        "sample_ids": sorted(extract_requested_sample_ids(normalized)),
        "temporal_metrics": temporal_metrics,
        "strict_projection": bool(
            re.search(r"只返回|只显示|仅返回|仅显示|只看", normalized)
        ),
        "current_spec": current_spec,
        "explicit_columns": sorted(explicit_columns & _known_columns()),
        "needs_llm": needs_llm,
        "confidence": "medium" if needs_llm else "high",
        "source": "deterministic",
        "reason": (
            "当前轮包含可确定的指代、字段或查询修改。"
            if dependency != "independent"
            else "当前问题可独立理解，不使用历史状态。"
        ),
    }


def build_query_delta_prompts(
    question: str,
    memory: dict[str, Any],
    base_delta: dict[str, Any],
) -> tuple[str, str]:
    """构造仅用于模糊承接问题的轻量QueryDelta提取Prompt。"""

    allowed_columns = sorted(_known_columns())
    recent_turns = list(memory.get("recent_turns", []))[-2:]
    previous_spec = memory.get("last_query_spec", {})

    system_prompt = """
你只负责判断当前用户短句如何修改上一轮Text2SQL查询状态，不生成SQL。
只能输出一个JSON对象，不要输出Markdown或解释。

字段说明：
- dependency只能是independent、same_sample、previous_result_set、previous_query；
- projection_action只能是keep、replace、add、remove；
- columns和remove_columns只能使用给定数据库真实字段；
- 当前轮明确说了新查询字段但没有“再加上/同时/另外/顺便”时，使用replace；
- “这些样本/上面几个”使用previous_result_set；
- “它/该样本/这个样本”使用same_sample；
- 无法确定时保持base_delta，不要猜SQL结构。

输出格式：
{
  "dependency": "...",
  "projection_action": "...",
  "columns": [],
  "remove_columns": [],
  "reason": "一句话"
}
""".strip()

    user_prompt = json.dumps(
        {
            "current_question": question,
            "allowed_columns": allowed_columns,
            "previous_query_spec": previous_spec,
            "recent_turns": recent_turns,
            "base_delta": {
                key: base_delta.get(key)
                for key in (
                    "dependency",
                    "projection_action",
                    "columns",
                    "remove_columns",
                )
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
    """解析并白名单化LLM返回；失败时原样回退确定性Delta。"""

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
            cleaned_values = sorted(
                {str(value) for value in values if str(value) in known}
            )
            if cleaned_values:
                merged[key] = cleaned_values

    merged["source"] = "llm_fallback"
    merged["needs_llm"] = False
    merged["confidence"] = "medium"
    merged["reason"] = str(payload.get("reason") or merged.get("reason", ""))
    return merged


def _stable_previous_spec(previous_spec: dict[str, Any]) -> dict[str, Any]:
    """只继承用户语义状态，不继承table/query_type/mode等派生字段。"""

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
    return {
        key: deepcopy(previous_spec[key])
        for key in stable_keys
        if key in previous_spec
    }


def _apply_query_delta(
    previous_spec: dict[str, Any],
    delta: dict[str, Any],
    memory: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    merged = _stable_previous_spec(previous_spec)
    inherited_fields = list(merged)
    overridden_fields: list[str] = []

    dependency = delta.get("dependency", "previous_query")
    explicit_ids = list(delta.get("sample_ids", []))
    if explicit_ids:
        merged["sample_ids"] = explicit_ids
        overridden_fields.append("sample_ids")
    elif dependency == "previous_result_set":
        active_ids = list(memory.get("active_sample_ids", []))
        if active_ids:
            merged["sample_ids"] = active_ids
            overridden_fields.append("sample_ids")
    elif dependency == "same_sample":
        previous_ids = list(previous_spec.get("sample_ids", []))
        if not previous_ids:
            active_ids = list(memory.get("active_sample_ids", []))
            if len(active_ids) == 1:
                previous_ids = active_ids
        if previous_ids:
            merged["sample_ids"] = previous_ids
            overridden_fields.append("sample_ids")

    action = delta.get("projection_action", "keep")
    columns = set(delta.get("columns", []))
    remove_columns = set(delta.get("remove_columns", []))
    previous_columns = set(merged.get("select_columns", []))
    previous_columns.update(merged.get("scalar_columns", []))

    if action == "replace" and columns:
        selected = set(columns)
        if not delta.get("strict_projection"):
            selected.add("sample_id")
        merged["select_columns"] = sorted(selected)
        merged["scalar_columns"] = sorted(columns - {"sample_id"})
        merged["strict_projection"] = bool(delta.get("strict_projection", False))
        overridden_fields.extend(["select_columns", "scalar_columns", "strict_projection"])
    elif action == "add" and columns:
        selected = previous_columns | columns
        selected.add("sample_id")
        merged["select_columns"] = sorted(selected)
        merged["scalar_columns"] = sorted(selected - {"sample_id"})
        merged["strict_projection"] = False
        overridden_fields.extend(["select_columns", "scalar_columns", "strict_projection"])
    elif action == "remove" and remove_columns:
        selected = previous_columns - remove_columns
        if not merged.get("strict_projection"):
            selected.add("sample_id")
        merged["select_columns"] = sorted(selected)
        merged["scalar_columns"] = sorted(selected - {"sample_id"})
        overridden_fields.extend(["select_columns", "scalar_columns"])

    if delta.get("clear_filters"):
        merged["filters"] = []
        merged["where_filters"] = []
        merged["having_filters"] = []
        overridden_fields.extend(["filters", "where_filters", "having_filters"])
    elif delta.get("filters"):
        current_filters = list(delta["filters"])
        merged["filters"] = _replace_filters_by_column(
            list(previous_spec.get("filters", [])),
            current_filters,
        )
        merged["where_filters"] = deepcopy(merged["filters"])
        merged["having_filters"] = deepcopy(previous_spec.get("having_filters", []))
        overridden_fields.extend(["filters", "where_filters"])

    if delta.get("order_by"):
        merged["order_by"] = deepcopy(delta["order_by"])
        overridden_fields.append("order_by")
    if delta.get("limit") is not None:
        merged["limit"] = int(delta["limit"])
        overridden_fields.append("limit")

    if delta.get("temporal_metrics"):
        metrics = deepcopy(delta["temporal_metrics"])
        if action == "add":
            indexed = {
                (item.get("column"), item.get("aggregation")): deepcopy(item)
                for item in merged.get("temporal_metrics", [])
            }
            for item in metrics:
                indexed[(item.get("column"), item.get("aggregation"))] = item
            metrics = list(indexed.values())
        merged["temporal_metrics"] = metrics
        merged["all_temporal_metrics"] = deepcopy(metrics)
        overridden_fields.extend(["temporal_metrics", "all_temporal_metrics"])

        direction = None
        if isinstance(delta.get("order_by"), dict):
            direction = delta["order_by"].get("direction")
        if direction and metrics:
            metric = metrics[0]
            merged["order_by"] = {
                "kind": "metric",
                "column": metric.get("column"),
                "alias": metric.get("alias"),
                "direction": direction,
            }
            overridden_fields.append("order_by")

    overridden_fields = list(dict.fromkeys(overridden_fields))
    inherited_fields = [
        key for key in dict.fromkeys(inherited_fields)
        if key not in overridden_fields
    ]
    return merged, inherited_fields, overridden_fields


def render_query_spec_as_question(spec: dict[str, Any]) -> str:
    """将结构化状态渲染成稳定的规范问题，再交给现有QuerySpec构建器。"""

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
            filter_phrases.append(
                f"{label}在{item.get('value')}到{item.get('value2')}之间"
            )
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

    selected = list(spec.get("select_columns", []))
    selected.extend(spec.get("scalar_columns", []))
    selected = list(dict.fromkeys(selected))
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
                (
                    item
                    for item in metrics
                    if item.get("column") == order_by.get("column")
                ),
                None,
            )
            if metric:
                suffix = {
                    "MAX": "峰值",
                    "MIN": "最小值",
                    "AVG": "平均值",
                    "FINAL": "最终值",
                }.get(str(metric.get("aggregation")), "")
                label += suffix
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
    columns.update(
        str(item.get("column", "")) for item in spec.get("filters", [])
    )
    columns.update(
        str(item.get("column", "")) for item in spec.get("temporal_metrics", [])
    )
    order_by = spec.get("order_by")
    if isinstance(order_by, dict) and order_by.get("column"):
        columns.add(str(order_by["column"]))
    return {column for column in columns if column}


def validate_current_turn_coverage(
    query_delta: dict[str, Any],
    final_spec: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """防止错误QuerySpec与错误SQL彼此验证通过。"""

    actual_columns = _spec_role_columns(final_spec)
    action = query_delta.get("projection_action", "keep")
    expected_columns = set(query_delta.get("columns", []))
    removed_columns = set(query_delta.get("remove_columns", []))

    missing_columns: list[str] = []
    unexpected_removed_columns: list[str] = []
    if action in {"replace", "add"}:
        missing_columns = sorted(expected_columns - actual_columns)
    if action == "remove":
        selected = set(final_spec.get("select_columns", []))
        selected.update(final_spec.get("scalar_columns", []))
        unexpected_removed_columns = sorted(removed_columns & selected)

    limit_mismatch = False
    if query_delta.get("limit") is not None:
        limit_mismatch = final_spec.get("limit") != query_delta.get("limit")

    sample_scope_missing = False
    if query_delta.get("dependency") in {"same_sample", "previous_result_set"}:
        sample_scope_missing = not bool(final_spec.get("sample_ids"))

    passed = not (
        missing_columns
        or unexpected_removed_columns
        or limit_mismatch
        or sample_scope_missing
    )
    return passed, {
        "passed": passed,
        "missing_columns": missing_columns,
        "removed_columns_still_present": unexpected_removed_columns,
        "limit_mismatch": limit_mismatch,
        "sample_scope_missing": sample_scope_missing,
        "actual_role_columns": sorted(actual_columns),
    }


def resolve_conversation_context(
    question: str,
    memory: dict[str, Any] | None,
    query_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = normalize_question_sample_ids(question)
    memory = deepcopy(memory or new_short_term_memory())
    current_spec = build_query_spec(normalized)
    previous_spec = deepcopy(memory.get("last_query_spec", {}))
    delta = deepcopy(query_delta or build_deterministic_query_delta(normalized, memory))

    if not previous_spec or delta.get("dependency") == "independent":
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
        }

    merged, inherited_fields, overridden_fields = _apply_query_delta(
        previous_spec,
        delta,
        memory,
    )
    resolved_question = render_query_spec_as_question(merged)
    rebuilt_spec = build_query_spec(resolved_question)

    if _is_meaningful_spec(rebuilt_spec):
        final_spec = rebuilt_spec
        if merged.get("strict_projection"):
            final_spec["strict_projection"] = True
            final_spec["select_columns"] = deepcopy(merged.get("select_columns", []))
        # 规范问题重建后如果丢失样本集合，显式恢复记忆范围。
        if merged.get("sample_ids") and not final_spec.get("sample_ids"):
            final_spec["sample_ids"] = deepcopy(merged["sample_ids"])
    else:
        final_spec = merged
        final_spec["eligible"] = False
        final_spec["mode"] = "rsl"
        final_spec["reason"] = "结构化记忆合并结果无法确定性编译，进入RSL路径。"

    coverage_passed, coverage = validate_current_turn_coverage(delta, final_spec)

    dependency = delta.get("dependency")
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
            "reason": "使用QueryDelta合并最后一次成功QuerySpec。",
            "previous_question": memory.get("last_resolved_question", ""),
            "current_fragment": normalized,
            "query_delta_source": delta.get("source", "deterministic"),
            "query_delta_reason": delta.get("reason", ""),
        },
        "inherited_fields": inherited_fields,
        "overridden_fields": overridden_fields,
    }


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

    sample_ids: list[str] = []
    sample_indexes = [
        index
        for index, column in enumerate(columns)
        if str(column).lower() == "sample_id"
    ]
    if sample_indexes:
        index = sample_indexes[0]
        for row in rows:
            if index >= len(row):
                continue
            value = row[index]
            if isinstance(value, str) and value.startswith("sample_"):
                sample_ids.append(value)
    sample_ids = list(dict.fromkeys(sample_ids))
    sample_ids_truncated = len(sample_ids) > MAX_RESULT_SAMPLE_IDS
    stored_sample_ids = sample_ids[:MAX_RESULT_SAMPLE_IDS]

    updated["updated_at"] = _utc_now_iso()
    updated["last_successful_question"] = question
    updated["last_resolved_question"] = resolved_question
    updated["last_query_spec"] = deepcopy(query_spec)
    updated["last_validated_sql"] = validated_sql
    updated["last_result"] = {
        "columns": list(columns),
        "row_count": int(row_count),
        "sample_ids": stored_sample_ids,
        "truncated": bool(truncated or sample_ids_truncated),
    }
    updated["active_sample_ids"] = stored_sample_ids

    recent_turns = list(updated.get("recent_turns", []))
    recent_turns.append(
        {
            "user_question": question,
            "resolved_question": resolved_question,
            "turn_type": turn_type,
            "status": final_status,
            "query_delta": {
                key: (query_delta or {}).get(key)
                for key in (
                    "dependency",
                    "projection_action",
                    "columns",
                    "source",
                )
            },
        }
    )
    updated["recent_turns"] = recent_turns[-MAX_RECENT_TURNS:]
    return updated


def format_memory_summary(memory: dict[str, Any] | None) -> str:
    if not memory:
        return "当前没有短期记忆。"

    spec = memory.get("last_query_spec", {})
    result = memory.get("last_result", {})
    lines = [
        f"session_id: {memory.get('session_id', '')}",
        f"最后成功问题: {memory.get('last_resolved_question') or '无'}",
        f"查询类型: {spec.get('query_type') or '无'}",
        f"返回字段: {', '.join(spec.get('select_columns', [])) or '无'}",
        f"样本约束: {', '.join(spec.get('sample_ids', [])) or '无'}",
        f"排序: {spec.get('order_by') or '无'}",
        f"数量限制: {spec.get('limit') if spec.get('limit') is not None else '无'}",
        f"上次结果行数: {result.get('row_count', 0)}",
        f"可引用样本数: {len(memory.get('active_sample_ids', []))}",
        f"工作记忆轮数: {len(memory.get('recent_turns', []))}/2",
    ]
    return "\n".join(lines)