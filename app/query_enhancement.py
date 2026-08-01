from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.schema import (
    canonical_metric_alias,
    get_schema_catalog,
    infer_requested_output_columns,
    match_question_semantic_columns,
)


_RESPONSE_COLUMNS = {"surface_temperature", "back_temperature", "mass"}
_SUPPORTED_TEMPORAL_AGGREGATIONS = {"MAX", "MIN", "AVG", "SUM", "INITIAL", "FINAL"}
_DERIVED_ALIASES = {"mass_loss_rate", "back_temperature_rise"}

_TABLE_ALIASES = {
    "material_static": "ms",
    "material_thermal_property": "mtp",
    "thermal_response": "tr",
}

_METRIC_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"峰值表面温度|表面温度峰值|峰值表温", "surface_temperature", "MAX"),
    (r"峰值背面温度|背面温度峰值|峰值背温", "back_temperature", "MAX"),
    (r"平均表面温度|表面温度平均值|平均表温", "surface_temperature", "AVG"),
    (r"平均背面温度|背面温度平均值|平均背温", "back_temperature", "AVG"),
    (r"初始质量", "mass", "INITIAL"),
    (r"最终质量|末时刻质量", "mass", "FINAL"),
    (r"初始背面温度|初始背温", "back_temperature", "INITIAL"),
    (r"最终背面温度|最终背温|末时刻背面温度|末时刻背温", "back_temperature", "FINAL"),
    (r"初始表面温度|初始表温", "surface_temperature", "INITIAL"),
    (r"最终表面温度|最终表温|末时刻表面温度|末时刻表温", "surface_temperature", "FINAL"),
)

_RANKING_METRICS: tuple[tuple[str, str], ...] = (
    (r"质量损失率", "mass_loss_rate"),
    (r"背温抬升|背面温度抬升|背温升高量", "back_temperature_rise"),
    (r"峰值表面温度|表面温度峰值|峰值表温", "peak_surface_temperature"),
    (r"峰值背面温度|背面温度峰值|峰值背温", "peak_back_temperature"),
    (r"最终背面温度|最终背温", "final_back_temperature"),
    (r"最终质量", "final_mass"),
    (r"初始质量", "initial_mass"),
)

_STAGE_METRICS: tuple[tuple[str, str], ...] = (
    (r"原始热导率|原始材料热导率", "原始热导率"),
    (r"碳化热导率|碳化材料热导率", "碳化热导率"),
    (r"峰值表面温度|表面温度峰值|峰值表温", "峰值表面温度"),
    (r"峰值背面温度|背面温度峰值|峰值背温", "峰值背面温度"),
    (r"最终背面温度|最终背温", "最终背面温度"),
    (r"原始密度", "原始密度"),
    (r"碳化密度", "碳化密度"),
    (r"表面发射率|发射率", "表面发射率"),
    (r"质量损失率", "质量损失率"),
    (r"背温抬升|背面温度抬升", "背温抬升"),
)


def _owner_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for table_name, table_info in get_schema_catalog().get("tables", {}).items():
        for column in table_info.get("columns", {}):
            result.setdefault(str(column), str(table_name))
    return result


def _field_terms() -> dict[str, tuple[str, ...]]:
    """Read field vocabulary from the schema catalog instead of copying it."""

    semantic_terms = get_schema_catalog().get("semantic_terms", {})
    return {
        str(column): tuple(dict.fromkeys([*terms, str(column)]))
        for column, terms in semantic_terms.items()
        if column not in {"sample_id", "point_index"}
    }


def _dedupe_dicts(items: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        marker = tuple(item.get(key) for key in keys)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(deepcopy(item))
    return result


def _direction(word: str) -> str:
    return "DESC" if word in {"最高", "最大", "最多", "高"} else "ASC"


def _normalize_numeric(value: Any) -> str:
    text = str(value).strip()
    try:
        return format(Decimal(text), "f")
    except (InvalidOperation, ValueError):
        return text


def detect_unsupported_nested_topk(question: str) -> dict[str, Any]:
    """识别单句中的多阶段Top-K。

    当前项目明确不在单轮QuerySpec中表达多个排名阶段。检测到后应立即
    引导用户拆成多轮，避免把两个ORDER BY/LIMIT交给LLM自由拼接。
    """

    normalized = re.sub(r"\s+", "", str(question))
    stages: list[dict[str, Any]] = []

    for metric_pattern, label in _STAGE_METRICS:
        pattern = re.compile(
            rf"(?P<metric>{metric_pattern}).{{0,10}}?(?P<direction>最高|最低|最大|最小)的?"
            rf"\s*(?P<limit>\d+)\s*(?:个|条)?(?:样本)?",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(question):
            stages.append(
                {
                    "metric": label,
                    "direction": _direction(match.group("direction")),
                    "limit": int(match.group("limit")),
                    "start": match.start(),
                }
            )

        rank_pattern = re.compile(
            rf"(?P<metric>{metric_pattern}).{{0,12}}?(?:仍然)?(?:排在)?前\s*"
            rf"(?P<limit>\d+)\s*(?P<direction>低|高)",
            flags=re.IGNORECASE,
        )
        for match in rank_pattern.finditer(question):
            between = question[match.start() : match.start("direction")]
            # 避免“质量损失率最低但背温抬升排在前15低”把15错误绑定到质量损失率。
            if re.search(r"最高|最低|最大|最小", between):
                continue
            stages.append(
                {
                    "metric": label,
                    "direction": _direction(match.group("direction")),
                    "limit": int(match.group("limit")),
                    "start": match.start(),
                }
            )

    stages = sorted(
        _dedupe_dicts(stages, ("metric", "direction", "limit", "start")),
        key=lambda item: int(item.get("start", 0)),
    )
    has_scope_link = bool(
        re.search(r"(?:个|条)?样本(?:中|里|范围内)|这些样本中|其中|再从|然后从", normalized)
    )
    unsupported = len(stages) >= 2 and has_scope_link

    if not unsupported:
        return {
            "unsupported": False,
            "reason": "",
            "stages": stages,
            "suggested_turns": [],
        }

    suggestions: list[str] = []
    first = stages[0]
    first_direction = "最高" if first["direction"] == "DESC" else "最低"
    suggestions.append(
        f"先查询{first['metric']}{first_direction}的{first['limit']}个样本。"
    )
    for stage in stages[1:]:
        direction_text = "从高到低" if stage["direction"] == "DESC" else "从低到高"
        suggestions.append(
            f"再在这些样本中按{stage['metric']}{direction_text}排列，取前{stage['limit']}个。"
        )
    if "质量损失率" in question and all("质量损失率" not in item for item in suggestions[1:]):
        suggestions.append("最后在当前样本集合中按质量损失率排序并返回所需字段。")

    return {
        "unsupported": True,
        "reason": "当前问题包含两个及以上依次执行的Top-K/排名阶段，超出当前单轮QuerySpec能力边界。",
        "stages": stages,
        "suggested_turns": suggestions,
    }


def extract_common_temporal_metrics(question: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    derived: list[dict[str, Any]] = []

    for pattern, column, aggregation in _METRIC_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            metrics.append(
                {
                    "column": column,
                    "aggregation": aggregation,
                    "alias": canonical_metric_alias(column, aggregation),
                }
            )

    if re.search(r"质量损失率", question):
        metrics.extend(
            [
                {"column": "mass", "aggregation": "INITIAL", "alias": "initial_mass"},
                {"column": "mass", "aggregation": "FINAL", "alias": "final_mass"},
            ]
        )
        derived.append(
            {
                "name": "mass_loss_rate",
                "alias": "mass_loss_rate",
                "operation": "RATE_LOSS",
                "dependencies": ["initial_mass", "final_mass"],
                "formula": "(initial_mass - final_mass) / initial_mass",
            }
        )

    if re.search(r"背温抬升|背面温度抬升|背温升高量", question):
        metrics.extend(
            [
                {
                    "column": "back_temperature",
                    "aggregation": "INITIAL",
                    "alias": "initial_back_temperature",
                },
                {
                    "column": "back_temperature",
                    "aggregation": "FINAL",
                    "alias": "final_back_temperature",
                },
            ]
        )
        derived.append(
            {
                "name": "back_temperature_rise",
                "alias": "back_temperature_rise",
                "operation": "DELTA",
                "dependencies": ["initial_back_temperature", "final_back_temperature"],
                "formula": "final_back_temperature - initial_back_temperature",
            }
        )

    return (
        _dedupe_dicts(metrics, ("column", "aggregation", "alias")),
        _dedupe_dicts(derived, ("alias", "operation")),
    )


def extract_common_filters(
    question: str,
    existing_filters: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    for raw_item in existing_filters or []:
        item = deepcopy(raw_item)
        item.setdefault("value_type", "literal")
        filters.append(item)

    comparator_patterns: tuple[tuple[str, str], ...] = (
        (r"不低于|不少于|大于等于|>=", ">="),
        (r"不高于|不超过|小于等于|<=", "<="),
        (r"高于|大于|超过|>", ">"),
        (r"低于|小于|少于|<", "<"),
        (r"等于|为|=", "="),
    )
    number_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

    field_terms = _field_terms()
    for column, terms in field_terms.items():
        term_pattern = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
        for comparator_pattern, operator in comparator_patterns:
            pattern = re.compile(
                rf"(?:{term_pattern})\s*(?:{comparator_pattern})\s*({number_pattern})",
                flags=re.IGNORECASE,
            )
            match = pattern.search(question)
            if not match:
                continue
            filters.append(
                {
                    "column": column,
                    "operator": operator,
                    "value": _normalize_numeric(match.group(1)),
                    "value_type": "literal",
                }
            )
            break

    columns = list(field_terms)
    for left_column in columns:
        for right_column in columns:
            if left_column == right_column:
                continue
            left_terms = "|".join(re.escape(term) for term in field_terms[left_column])
            right_terms = "|".join(re.escape(term) for term in field_terms[right_column])
            for comparator_pattern, operator in comparator_patterns[:4]:
                pattern = re.compile(
                    rf"(?:{left_terms})\s*(?:{comparator_pattern})\s*(?:{right_terms})",
                    flags=re.IGNORECASE,
                )
                if pattern.search(question):
                    filters.append(
                        {
                            "column": left_column,
                            "operator": operator,
                            "value": right_column,
                            "value_type": "column",
                        }
                    )

    return _dedupe_dicts(filters, ("column", "operator", "value", "value_type"))


def _extract_explicit_order(question: str) -> dict[str, Any] | None:
    """解析“按X从低到高排列”等显式排序，优先于其他字段线索。"""

    match = re.search(
        r"(?:按|根据)\s*(?P<target>.{1,32}?)\s*"
        r"(?P<direction>从高到低|从低到高|由高到低|由低到高|升序|降序)"
        r"(?:排列|排序)?",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    target = match.group("target").strip("，,、。；; ")
    direction_word = match.group("direction")
    direction = "DESC" if direction_word in {"从高到低", "由高到低", "降序"} else "ASC"

    for metric_pattern, alias in _RANKING_METRICS:
        if re.search(metric_pattern, target, flags=re.IGNORECASE):
            metric = next(
                (item for item in _METRIC_PATTERNS if canonical_metric_alias(item[1], item[2]) == alias),
                None,
            )
            if metric:
                return {
                    "kind": "metric",
                "column": metric[1],
                    "alias": alias,
                    "direction": direction,
                }
            return {
                "kind": "derived" if alias in _DERIVED_ALIASES else "metric",
                "column": alias,
                "alias": alias,
                "direction": direction,
            }

    for column, terms in _field_terms().items():
        if any(term in target for term in terms):
            return {"kind": "column", "column": column, "direction": direction}
    return None


def _requested_metric_aliases(question: str) -> set[str]:
    aliases: set[str] = set()
    for pattern, column, aggregation in _METRIC_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            aliases.add(canonical_metric_alias(column, aggregation))
    if re.search(r"质量损失率", question):
        aliases.add("mass_loss_rate")
    if re.search(r"背温抬升|背面温度抬升|背温升高量", question):
        aliases.add("back_temperature_rise")
    return aliases


def _extract_metric_order(question: str) -> dict[str, Any] | None:
    for metric_pattern, alias in _RANKING_METRICS:
        match = re.search(
            rf"(?:{metric_pattern}).{{0,10}}?(最高|最低|最大|最小|从高到低|从低到高|降序|升序)",
            question,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        direction_word = match.group(1)
        direction = (
            "DESC"
            if direction_word in {"最高", "最大", "从高到低", "降序"}
            else "ASC"
        )
        return {
            "kind": "derived" if alias in _DERIVED_ALIASES else "metric",
            "alias": alias,
            "column": alias,
            "direction": direction,
        }
    return None


def augment_common_query_spec(
    question: str,
    base_spec: dict[str, Any] | None,
    query_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """补齐常见时序、派生指标、字段比较与明确排序语义。"""

    spec = deepcopy(base_spec or {})
    delta = query_delta or {}
    unsupported = detect_unsupported_nested_topk(question)
    spec["capability_check"] = unsupported
    if unsupported["unsupported"]:
        spec.update(
            {
                "eligible": False,
                "mode": "unsupported",
                "query_type": "unsupported_nested_topk",
                "reason": unsupported["reason"],
            }
        )
        return spec

    extracted_metrics, extracted_derived = extract_common_temporal_metrics(question)
    existing_metrics = list(spec.get("all_temporal_metrics") or spec.get("temporal_metrics") or [])
    delta_metrics = list(delta.get("temporal_metrics") or [])
    # A metric is defined by its source column and aggregation. Different
    # aliases for AVG(back_temperature) must not create duplicate SQL columns.
    metrics = _dedupe_dicts(
        [*extracted_metrics, *existing_metrics, *delta_metrics],
        ("column", "aggregation"),
    )
    metrics = [
        item
        for item in metrics
        if str(item.get("aggregation", "")).upper() in _SUPPORTED_TEMPORAL_AGGREGATIONS
    ]
    derived_metrics = _dedupe_dicts(
        [
            *list(spec.get("derived_metrics") or []),
            *list(delta.get("derived_metrics") or []),
            *extracted_derived,
        ],
        ("alias", "operation"),
    )

    filters = extract_common_filters(
        question,
        spec.get("filters") or spec.get("where_filters") or delta.get("filters") or [],
    )

    requested_outputs = set(infer_requested_output_columns(question))
    explicit_columns = set(match_question_semantic_columns(question))
    has_output_cue = bool(
        re.search(r"返回|显示|列出|同时返回|并返回|只返回|只显示", question)
    )
    if not requested_outputs and has_output_cue:
        requested_outputs = explicit_columns

    metric_aliases = {str(item.get("alias", "")) for item in metrics if item.get("alias")}
    derived_aliases = {str(item.get("alias", "")) for item in derived_metrics if item.get("alias")}
    explicitly_requested_aliases = _requested_metric_aliases(question)

    output_columns: list[str] = ["sample_id"]
    if has_output_cue:
        output_columns.extend(
            sorted(
                column
                for column in requested_outputs
                if column not in _RESPONSE_COLUMNS and column != "point_index"
            )
        )
        # 不依赖上游的原始字段映射，直接按“初始/最终/峰值/变化率”短语补充别名。
        output_columns.extend(sorted(explicitly_requested_aliases))
    else:
        output_columns.extend(
            column
            for column in spec.get("select_columns", [])
            if column not in _RESPONSE_COLUMNS and column != "point_index"
        )
        output_columns.extend(alias for alias in metric_aliases if alias)
        output_columns.extend(alias for alias in derived_aliases if alias)

    # 明确“按X排列”优先级最高；其次才是“X最高/最低”的排名语义；
    # 最后才沿用上游QuerySpec，避免返回字段覆盖排序字段。
    order_by = (
        _extract_explicit_order(question)
        or _extract_metric_order(question)
        or deepcopy(spec.get("order_by"))
    )
    if isinstance(order_by, dict):
        order_alias = str(order_by.get("alias") or order_by.get("column") or "")
        matching_metric = next(
            (item for item in metrics if str(item.get("alias", "")) == order_alias),
            None,
        )
        if matching_metric:
            order_by = {
                "kind": "metric",
                "column": matching_metric.get("column"),
                "alias": matching_metric.get("alias"),
                "direction": order_by.get("direction", "ASC"),
            }
        # 排序所需的聚合/派生指标必须在当前查询层真实定义。
        if order_alias in metric_aliases or order_alias in derived_aliases or order_alias in _DERIVED_ALIASES:
            output_columns.append(order_alias)

    output_columns = list(dict.fromkeys(output_columns))

    owner_map = _owner_map()
    scalar_columns = {
        column
        for column in output_columns
        if column in owner_map and column not in _RESPONSE_COLUMNS and column != "sample_id"
    }
    for item in filters:
        column = str(item.get("column", ""))
        value = str(item.get("value", ""))
        if column in owner_map and column not in _RESPONSE_COLUMNS:
            scalar_columns.add(column)
        if item.get("value_type") == "column" and value in owner_map and value not in _RESPONSE_COLUMNS:
            scalar_columns.add(value)
    if isinstance(order_by, dict):
        order_column = str(order_by.get("column", ""))
        if order_column in owner_map and order_column not in _RESPONSE_COLUMNS:
            scalar_columns.add(order_column)

    scalar_tables = sorted({owner_map[column] for column in scalar_columns if column in owner_map})
    needs_extended = bool(
        metrics
        or derived_metrics
        or any(item.get("value_type") == "column" for item in filters)
    )

    if needs_extended:
        spec.update(
            {
                "eligible": True,
                "mode": "deterministic_extended",
                "query_type": "common_temporal_aggregate" if (metrics or derived_metrics) else "extended_filter",
                "table": "thermal_response" if (metrics or derived_metrics) else (scalar_tables[0] if len(scalar_tables) == 1 else ""),
                "select_columns": [
                    column for column in output_columns if column in owner_map or column == "sample_id"
                ],
                "output_columns": output_columns,
                "filters": filters,
                "where_filters": deepcopy(filters),
                "having_filters": [],
                "order_by": order_by,
                "temporal_metrics": metrics,
                "all_temporal_metrics": deepcopy(metrics),
                "derived_metrics": derived_metrics,
                "scalar_columns": sorted(scalar_columns),
                "scalar_tables": scalar_tables,
                "reason": (
                    "识别到常见INITIAL/FINAL/MAX、字段间比较或白名单派生指标，"
                    "使用确定性扩展查询编译器。"
                ),
                "confidence": 1.0,
            }
        )
    else:
        spec["filters"] = filters
        spec["where_filters"] = deepcopy(filters)
        if metrics:
            spec["temporal_metrics"] = metrics
            spec["all_temporal_metrics"] = deepcopy(metrics)
        if derived_metrics:
            spec["derived_metrics"] = derived_metrics
        if order_by:
            spec["order_by"] = order_by

    return spec

def _metric_alias_expression(alias: str) -> str:
    mapping = {
        "peak_surface_temperature": "agg.peak_surface_temperature",
        "peak_back_temperature": "agg.peak_back_temperature",
        "average_surface_temperature": "agg.average_surface_temperature",
        "average_back_temperature": "agg.average_back_temperature",
        "initial_mass": "ini.initial_mass",
        "final_mass": "fin.final_mass",
        "initial_back_temperature": "ini.initial_back_temperature",
        "final_back_temperature": "fin.final_back_temperature",
        "initial_surface_temperature": "ini.initial_surface_temperature",
        "final_surface_temperature": "fin.final_surface_temperature",
        "mass_loss_rate": "(ini.initial_mass - fin.final_mass) / NULLIF(ini.initial_mass, 0)",
        "back_temperature_rise": "fin.final_back_temperature - ini.initial_back_temperature",
    }
    if alias in mapping:
        return mapping[alias]
    if alias.startswith(("peak_", "minimum_", "average_", "sum_")):
        return f"agg.{alias}"
    if alias.startswith("initial_"):
        return f"ini.{alias}"
    if alias.startswith("final_"):
        return f"fin.{alias}"
    return alias


def compile_extended_query_sql(spec: dict[str, Any]) -> str:
    """确定性编译常见时序聚合、初末值、变化量与字段间比较。"""

    if spec.get("mode") != "deterministic_extended":
        return ""

    owner_map = _owner_map()
    output_columns = list(spec.get("output_columns") or spec.get("select_columns") or ["sample_id"])
    metrics = list(spec.get("temporal_metrics") or [])
    derived_metrics = list(spec.get("derived_metrics") or [])
    filters = list(spec.get("filters") or [])
    order_by = spec.get("order_by") or {}

    metric_by_alias = {str(item.get("alias", "")): item for item in metrics if item.get("alias")}
    derived_aliases = {str(item.get("alias", "")) for item in derived_metrics if item.get("alias")}
    order_alias = str(order_by.get("alias") or order_by.get("column") or "")
    if order_alias in metric_by_alias or order_alias in derived_aliases or order_alias in _DERIVED_ALIASES:
        output_columns.append(order_alias)
    output_columns = list(dict.fromkeys(output_columns))

    needed_scalar_columns: set[str] = set()
    for column in output_columns:
        if column in owner_map and column not in _RESPONSE_COLUMNS and column != "sample_id":
            needed_scalar_columns.add(column)
    for item in filters:
        column = str(item.get("column", ""))
        value = str(item.get("value", ""))
        if column in owner_map and column not in _RESPONSE_COLUMNS:
            needed_scalar_columns.add(column)
        if item.get("value_type") == "column" and value in owner_map and value not in _RESPONSE_COLUMNS:
            needed_scalar_columns.add(value)
    if order_alias in owner_map and order_alias not in _RESPONSE_COLUMNS:
        needed_scalar_columns.add(order_alias)

    needs_mtp = any(owner_map.get(column) == "material_thermal_property" for column in needed_scalar_columns)
    aggregate_metrics = [
        item for item in metrics
        if str(item.get("aggregation", "")).upper() in {"MAX", "MIN", "AVG", "SUM"}
    ]
    initial_metrics = [
        item for item in metrics
        if str(item.get("aggregation", "")).upper() == "INITIAL"
    ]
    final_metrics = [
        item for item in metrics
        if str(item.get("aggregation", "")).upper() == "FINAL"
    ]

    joins: list[str] = []
    if needs_mtp:
        joins.append("JOIN material_thermal_property AS mtp ON ms.sample_id = mtp.sample_id")

    if aggregate_metrics:
        aggregate_selects = ["tr.sample_id"]
        for metric in aggregate_metrics:
            aggregation = str(metric.get("aggregation", "")).upper()
            column = str(metric.get("column", ""))
            alias = str(metric.get("alias", ""))
            aggregate_selects.append(f"{aggregation}(tr.{column}) AS {alias}")
        joins.append(
            "JOIN (SELECT " + ", ".join(aggregate_selects)
            + " FROM thermal_response AS tr GROUP BY tr.sample_id) AS agg "
            + "ON ms.sample_id = agg.sample_id"
        )

    aggregate_alias_by_column = {
        str(metric.get("column", "")): str(metric.get("alias", ""))
        for metric in aggregate_metrics
    }
    initial_alias_by_column = {
        str(metric.get("column", "")): str(metric.get("alias", ""))
        for metric in initial_metrics
    }
    final_alias_by_column = {
        str(metric.get("column", "")): str(metric.get("alias", ""))
        for metric in final_metrics
    }

    if initial_metrics:
        select_items = ["initial_row.sample_id"]
        for metric in initial_metrics:
            select_items.append(
                f"initial_row.{metric.get('column')} AS {metric.get('alias')}"
            )
        joins.append(
            "JOIN (SELECT " + ", ".join(dict.fromkeys(select_items))
            + " FROM thermal_response AS initial_row "
            + "JOIN (SELECT sample_id, MIN(point_index) AS min_point_index "
            + "FROM thermal_response GROUP BY sample_id) AS initial_idx "
            + "ON initial_row.sample_id = initial_idx.sample_id "
            + "AND initial_row.point_index = initial_idx.min_point_index) AS ini "
            + "ON ms.sample_id = ini.sample_id"
        )

    if final_metrics:
        select_items = ["final_row.sample_id"]
        for metric in final_metrics:
            select_items.append(
                f"final_row.{metric.get('column')} AS {metric.get('alias')}"
            )
        joins.append(
            "JOIN (SELECT " + ", ".join(dict.fromkeys(select_items))
            + " FROM thermal_response AS final_row "
            + "JOIN (SELECT sample_id, MAX(point_index) AS max_point_index "
            + "FROM thermal_response GROUP BY sample_id) AS final_idx "
            + "ON final_row.sample_id = final_idx.sample_id "
            + "AND final_row.point_index = final_idx.max_point_index) AS fin "
            + "ON ms.sample_id = fin.sample_id"
        )

    select_items: list[str] = []
    for column in output_columns:
        if column == "sample_id":
            select_items.append("ms.sample_id")
        elif column in owner_map and column not in _RESPONSE_COLUMNS:
            table = owner_map[column]
            alias = _TABLE_ALIASES.get(table, table)
            select_items.append(f"{alias}.{column}")
        elif column in metric_by_alias or column in derived_aliases or column in _DERIVED_ALIASES:
            select_items.append(f"{_metric_alias_expression(column)} AS {column}")
    if not select_items:
        select_items = ["ms.sample_id"]

    where_parts: list[str] = []
    sample_ids = [str(value) for value in spec.get("sample_ids", []) if str(value)]
    if len(sample_ids) == 1:
        where_parts.append(f"ms.sample_id = '{sample_ids[0]}'")
    elif sample_ids:
        where_parts.append(
            "ms.sample_id IN (" + ", ".join(f"'{value}'" for value in sample_ids) + ")"
        )

    for item in filters:
        column = str(item.get("column", ""))
        operator = str(item.get("operator", "="))
        value = item.get("value")
        if column not in owner_map:
            continue
        if column in _RESPONSE_COLUMNS:
            # Response rows are materialized as aggregated/initial/final
            # subqueries, so the original `tr` alias is not visible here.
            if column in aggregate_alias_by_column:
                left = f"agg.{aggregate_alias_by_column[column]}"
            elif column in initial_alias_by_column:
                left = f"ini.{initial_alias_by_column[column]}"
            elif column in final_alias_by_column:
                left = f"fin.{final_alias_by_column[column]}"
            else:
                continue
        else:
            left_alias = _TABLE_ALIASES.get(owner_map[column], owner_map[column])
            left = f"{left_alias}.{column}"
        if item.get("value_type") == "column" and str(value) in owner_map:
            right_column = str(value)
            right_alias = _TABLE_ALIASES.get(owner_map[right_column], owner_map[right_column])
            right = f"{right_alias}.{right_column}"
        else:
            right = _normalize_numeric(value)
        where_parts.append(f"{left} {operator} {right}")

    sql_parts = ["SELECT " + ", ".join(select_items), "FROM material_static AS ms", *joins]
    if where_parts:
        sql_parts.append("WHERE " + " AND ".join(where_parts))

    if order_alias:
        if order_alias in metric_by_alias or order_alias in derived_aliases or order_alias in _DERIVED_ALIASES:
            # 上面已确保该别名进入SELECT，ORDER BY别名在MySQL中可见。
            order_expression = order_alias
        elif order_alias in owner_map:
            table_alias = _TABLE_ALIASES.get(owner_map[order_alias], owner_map[order_alias])
            order_expression = f"{table_alias}.{order_alias}"
        else:
            order_expression = order_alias
        direction = "DESC" if str(order_by.get("direction", "ASC")).upper() == "DESC" else "ASC"
        sql_parts.append(f"ORDER BY {order_expression} {direction}")

    limit = spec.get("limit")
    if isinstance(limit, int) and limit > 0:
        sql_parts.append(f"LIMIT {limit}")
    return " ".join(sql_parts)

def validate_compiled_extended_sql(
    sql: str,
    spec: dict[str, Any],
    allowed_tables: set[str],
    max_rows: int,
) -> tuple[bool, str, str]:
    """验证确定性扩展SQL，包括派生排序别名是否真实定义。"""

    try:
        tree = sqlglot.parse_one(sql, read="mysql")
    except ParseError as exc:
        return False, "", f"确定性扩展SQL解析失败：{exc}"
    if not isinstance(tree, exp.Select):
        return False, "", "确定性扩展查询必须是SELECT。"

    used_tables = {table.name for table in tree.find_all(exp.Table)}
    unknown_tables = used_tables - set(allowed_tables)
    if unknown_tables:
        return False, "", "确定性扩展SQL使用未知表：" + ", ".join(sorted(unknown_tables))
    if tuple(tree.find_all(exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Create)):
        return False, "", "确定性扩展SQL包含写操作。"

    top_level_names: set[str] = set()
    for expression in tree.expressions:
        alias = expression.alias_or_name
        if alias:
            top_level_names.add(alias)
        elif isinstance(expression, exp.Column):
            top_level_names.add(expression.name)

    expected_outputs = set(spec.get("output_columns") or spec.get("select_columns") or [])
    missing = expected_outputs - top_level_names
    if missing:
        return False, "", "确定性扩展SQL缺少返回字段：" + ", ".join(sorted(missing))

    expected_limit = spec.get("limit")
    limit_node = tree.args.get("limit")
    actual_limit: int | None = None
    if limit_node is not None and isinstance(limit_node.expression, exp.Literal):
        try:
            actual_limit = int(limit_node.expression.this)
        except (TypeError, ValueError):
            actual_limit = None
    if isinstance(expected_limit, int) and actual_limit != expected_limit:
        return False, "", f"确定性扩展SQL的LIMIT应为{expected_limit}，实际为{actual_limit}。"
    if actual_limit is not None and actual_limit > max_rows:
        return False, "", f"LIMIT {actual_limit}超过资源上限{max_rows}。"

    order_by = spec.get("order_by") or {}
    expected_order = str(order_by.get("alias") or order_by.get("column") or "")
    if expected_order:
        order_node = tree.args.get("order")
        if order_node is None or not order_node.expressions:
            return False, "", "确定性扩展SQL缺少ORDER BY。"
        ordered = order_node.expressions[0]
        ordered_sql = ordered.this.sql(dialect="mysql") if isinstance(ordered, exp.Ordered) else ordered.sql(dialect="mysql")
        if expected_order not in ordered_sql:
            return False, "", f"确定性扩展SQL排序字段应为{expected_order}。"
        if (
            expected_order in _DERIVED_ALIASES
            or str(order_by.get("kind", "")) in {"metric", "derived"}
        ) and expected_order not in top_level_names:
            return False, "", f"排序别名{expected_order}未在顶层SELECT中定义。"
        actual_direction = "DESC" if isinstance(ordered, exp.Ordered) and bool(ordered.args.get("desc")) else "ASC"
        expected_direction = str(order_by.get("direction", "ASC")).upper()
        if actual_direction != expected_direction:
            return False, "", f"确定性扩展SQL排序方向应为{expected_direction}。"

    return True, tree.sql(dialect="mysql"), ""

def _set_jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {str(item) for item in left if str(item)}
    right_set = {str(item) for item in right if str(item)}
    if not left_set and not right_set:
        return 1.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def build_query_signature(query_spec: dict[str, Any], question: str = "") -> dict[str, Any]:
    filters = list(query_spec.get("filters") or query_spec.get("where_filters") or [])
    temporal_metrics = list(query_spec.get("all_temporal_metrics") or query_spec.get("temporal_metrics") or [])
    derived_metrics = list(query_spec.get("derived_metrics") or [])
    capability = query_spec.get("capability_check") or detect_unsupported_nested_topk(question)
    order_by = query_spec.get("order_by") or {}

    temporal_ops = {
        str(item.get("aggregation", "")).upper()
        for item in temporal_metrics
        if item.get("aggregation")
    }
    for item in derived_metrics:
        operation = str(item.get("operation", "")).upper()
        if operation == "RATE_LOSS":
            temporal_ops.add("RATE")
        elif operation == "DELTA":
            temporal_ops.add("DELTA")
    order_alias = str(order_by.get("alias") or order_by.get("column") or "")
    if order_alias == "mass_loss_rate":
        temporal_ops.update({"INITIAL", "FINAL", "RATE"})
    elif order_alias == "back_temperature_rise":
        temporal_ops.update({"INITIAL", "FINAL", "DELTA"})

    filter_types = {
        "column_column" if item.get("value_type") == "column" else "column_literal"
        for item in filters
    }
    tables = set(query_spec.get("scalar_tables") or [])
    if temporal_metrics or derived_metrics or str(order_by.get("kind", "")) in {"metric", "derived"}:
        tables.add("thermal_response")

    query_type = str(query_spec.get("query_type", "complex_or_uncertain"))
    family = "temporal" if temporal_metrics or derived_metrics or "temporal" in query_type else "scalar"
    if "multi_table" in query_type or len(tables) > 1:
        family += "_multi_table"

    stage_count = max(1, len(capability.get("stages", []))) if capability.get("unsupported") else 1
    has_derived = bool(
        derived_metrics
        or str(order_by.get("kind", "")) == "derived"
        or order_alias in _DERIVED_ALIASES
    )
    return {
        "query_type": query_type,
        "query_family": family,
        "tables": sorted(tables),
        "temporal_ops": sorted(temporal_ops),
        "filter_types": sorted(filter_types),
        "ranking_kind": str(order_by.get("kind", "none")),
        "ranking_direction": str(order_by.get("direction", "")),
        "ranking_stage_count": stage_count,
        "has_nested_topk": bool(capability.get("unsupported", False)),
        "has_derived_metric": has_derived,
        "select_role_count": len(query_spec.get("output_columns") or query_spec.get("select_columns") or []),
        "has_limit": query_spec.get("limit") is not None,
    }

def hard_signature_compatible(
    current: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, str]:
    if bool(current.get("has_nested_topk")) != bool(candidate.get("has_nested_topk")):
        return False, "nested_topk_mismatch"
    if int(current.get("ranking_stage_count", 1)) != int(candidate.get("ranking_stage_count", 1)):
        return False, "ranking_stage_count_mismatch"

    current_ops = set(current.get("temporal_ops") or [])
    candidate_ops = set(candidate.get("temporal_ops") or [])
    for required in current_ops:
        if required not in candidate_ops:
            return False, f"missing_temporal_op:{required}"

    current_filters = set(current.get("filter_types") or [])
    candidate_filters = set(candidate.get("filter_types") or [])
    if "column_column" in current_filters and "column_column" not in candidate_filters:
        return False, "missing_column_column_filter"

    return True, ""


def query_signature_similarity(current: dict[str, Any], candidate: dict[str, Any]) -> float:
    score = 0.0
    score += 0.20 if current.get("query_family") == candidate.get("query_family") else 0.0
    score += 0.12 if current.get("query_type") == candidate.get("query_type") else 0.0
    score += 0.20 * _set_jaccard(current.get("temporal_ops", []), candidate.get("temporal_ops", []))
    score += 0.14 * _set_jaccard(current.get("filter_types", []), candidate.get("filter_types", []))
    score += 0.12 if current.get("ranking_kind") == candidate.get("ranking_kind") else 0.0
    score += 0.08 if current.get("ranking_direction") == candidate.get("ranking_direction") else 0.0
    score += 0.08 if current.get("has_derived_metric") == candidate.get("has_derived_metric") else 0.0
    score += 0.06 if current.get("has_limit") == candidate.get("has_limit") else 0.0
    return min(1.0, score)


def signature_summary(signature: dict[str, Any]) -> str:
    return json.dumps(
        {
            "family": signature.get("query_family"),
            "temporal_ops": signature.get("temporal_ops", []),
            "filter_types": signature.get("filter_types", []),
            "ranking_kind": signature.get("ranking_kind"),
            "ranking_stage_count": signature.get("ranking_stage_count"),
            "has_derived_metric": signature.get("has_derived_metric"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
