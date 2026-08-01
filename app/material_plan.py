"""Constrained analytical plans for the resin material Profile."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.query_enhancement import compile_extended_query_sql
from app.schema import active_profile_name, get_schema_catalog


MATERIAL_PLAN_FAMILY = "static_filter_temporal_aggregate"
_RESPONSE_COLUMNS = {"surface_temperature", "back_temperature", "mass"}
_AGGREGATIONS = {"MAX", "MIN", "AVG", "SUM", "INITIAL", "FINAL"}


class _PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Filter(_PlanModel):
    column: str
    operator: Literal["=", "!=", ">", ">=", "<", "<="]
    value: float


class TemporalMetric(_PlanModel):
    column: str
    aggregation: Literal["MAX", "MIN", "AVG", "SUM", "INITIAL", "FINAL"]
    alias: str = Field(min_length=1, max_length=80)


class TemporalFilter(_PlanModel):
    alias: str = Field(min_length=1, max_length=80)
    operator: Literal["=", "!=", ">", ">=", "<", "<="]
    value: float


class MaterialTemporalAggregatePlan(_PlanModel):
    family: Literal["static_filter_temporal_aggregate"]
    static_filters: list[Filter] = Field(default_factory=list, max_length=6)
    temporal_metrics: list[TemporalMetric] = Field(min_length=1, max_length=3)
    temporal_filters: list[TemporalFilter] = Field(default_factory=list, max_length=4)
    output_columns: list[str] = Field(min_length=1, max_length=8)
    order_by: dict[str, str] | None = None
    limit: int | None = Field(default=None, ge=1, le=100)


def is_material_plan_candidate(question: str) -> bool:
    """Detect analytical material phrasing outside the deterministic DSL."""

    if active_profile_name() != "resin":
        return False
    has_temporal_operation = bool(re.search(r"峰值|平均|均值|最终|初始|最大值|最小值", question))
    has_response = bool(re.search(r"表面温度|表温|背面温度|背温|质量", question))
    has_composition = bool(re.search(r"同时|并且|且|满足|超过|高于|低于|筛选", question))
    return has_temporal_operation and has_response and has_composition


def material_plan_prompt(schema_context: str, question: str, few_shot_context: str = "") -> str:
    return f"""Return JSON only: {{"advanced_plan": {{...}}}}. Do not output SQL.
Use exactly this family: `{MATERIAL_PLAN_FAMILY}`.
The plan expresses static material filters plus per-sample temporal aggregates.

Contract keys: family, static_filters, temporal_metrics, temporal_filters, output_columns, order_by, limit.
- static_filters: filters on material_static or material_thermal_property scalar columns.
- temporal_metrics: only surface_temperature, back_temperature, or mass; aggregation is MAX/MIN/AVG/SUM/INITIAL/FINAL.
- temporal_filters reference a temporal metric alias, never raw time-series rows.
- output_columns contain sample_id, requested scalar columns, and requested temporal aliases.
- order_by is null or {{"alias":"...", "direction":"ASC|DESC"}}.
- Do not invent fields, aliases, filters, or a limit.

Required JSON shape example (field names are literal):
{{"advanced_plan":{{"family":"static_filter_temporal_aggregate","static_filters":[{{"column":"rhoc_i","operator":">","value":330}}],"temporal_metrics":[{{"column":"surface_temperature","aggregation":"MAX","alias":"peak_surface_temperature"}}],"temporal_filters":[{{"alias":"peak_surface_temperature","operator":">","value":1300}}],"output_columns":["sample_id","rhoc_i","peak_surface_temperature"],"order_by":{{"alias":"peak_surface_temperature","direction":"DESC"}},"limit":5}}}}

Schema:
{schema_context}
{few_shot_context}
Question:
{question}"""


def parse_material_plan(raw: str) -> dict[str, Any]:
    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    decoder = json.JSONDecoder()
    payload: dict[str, Any] | None = None
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        raise ValueError("no JSON object found in material plan output")
    plan = payload.get("advanced_plan", payload)
    if not isinstance(plan, dict):
        raise ValueError("advanced_plan must be an object")
    plan = _normalize_common_plan_variants(plan)
    try:
        parsed = MaterialTemporalAggregatePlan.model_validate(plan).model_dump()
    except ValidationError as exc:
        raise ValueError(f"invalid material plan contract: {exc}") from exc

    catalog = get_schema_catalog()
    columns = {
        column
        for table in catalog["tables"].values()
        for column in table["columns"]
    }
    static_columns = columns - _RESPONSE_COLUMNS - {"sample_id", "point_index"}
    if any(item["column"] not in static_columns for item in parsed["static_filters"]):
        raise ValueError("material plan has an invalid static filter column")
    if any(item["column"] not in _RESPONSE_COLUMNS or item["aggregation"] not in _AGGREGATIONS for item in parsed["temporal_metrics"]):
        raise ValueError("material plan has an invalid temporal metric")
    aliases = {item["alias"] for item in parsed["temporal_metrics"]}
    if len(aliases) != len(parsed["temporal_metrics"]):
        raise ValueError("material temporal metric aliases must be unique")
    if any(item["alias"] not in aliases for item in parsed["temporal_filters"]):
        raise ValueError("material temporal filter must reference a declared alias")
    allowed_outputs = columns | aliases
    if any(column not in allowed_outputs for column in parsed["output_columns"]):
        raise ValueError("material plan has an invalid output column")
    order_by = parsed.get("order_by")
    if order_by:
        if set(order_by) != {"alias", "direction"} or order_by["alias"] not in allowed_outputs or order_by["direction"] not in {"ASC", "DESC"}:
            raise ValueError("material plan has an invalid order_by")
    return parsed


def _normalize_common_plan_variants(plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize predictable 3B JSON vocabulary without accepting new semantics."""

    normalized = dict(plan)
    operator_names = {
        "gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "=", "ne": "!=",
    }

    def normalize_filter(item: Any, *, temporal: bool) -> Any:
        if not isinstance(item, dict):
            return item
        value = dict(item)
        nested = value.pop("filter", None)
        if isinstance(nested, dict) and len(nested) == 1 and "operator" not in value:
            raw_operator, raw_value = next(iter(nested.items()))
            value["operator"] = operator_names.get(str(raw_operator).lower(), raw_operator)
            value["value"] = raw_value
        if temporal and "alias" not in value and "column" in value:
            value["alias"] = value.pop("column")
        return value

    normalized["static_filters"] = [
        normalize_filter(item, temporal=False)
        for item in normalized.get("static_filters", [])
    ]
    normalized["temporal_filters"] = [
        normalize_filter(item, temporal=True)
        for item in normalized.get("temporal_filters", [])
    ]
    metrics: list[Any] = []
    for item in normalized.get("temporal_metrics", []):
        if isinstance(item, dict):
            value = dict(item)
            if "aggregation" not in value and "metric" in value:
                value["aggregation"] = value.pop("metric")
            metrics.append(value)
        else:
            metrics.append(item)
    normalized["temporal_metrics"] = metrics
    aliases_by_column = {
        str(item.get("column")): str(item.get("alias"))
        for item in metrics
        if isinstance(item, dict) and item.get("column") and item.get("alias")
    }
    for item in normalized["temporal_filters"]:
        if isinstance(item, dict) and item.get("alias") in aliases_by_column:
            item["alias"] = aliases_by_column[item["alias"]]
    return normalized


def compile_material_plan(plan: dict[str, Any]) -> str:
    """Translate a validated material plan into the shared safe compiler DSL."""

    if plan.get("family") != MATERIAL_PLAN_FAMILY:
        raise ValueError("unsupported material plan family")
    metric_by_alias = {item["alias"]: item for item in plan["temporal_metrics"]}
    filters = [*plan.get("static_filters", [])]
    for item in plan.get("temporal_filters", []):
        metric = metric_by_alias[item["alias"]]
        filters.append({"column": metric["column"], "operator": item["operator"], "value": item["value"]})
    order_by = plan.get("order_by")
    spec = {
        "mode": "deterministic_extended",
        "query_type": "material_plan_temporal_aggregate",
        "output_columns": plan["output_columns"],
        "temporal_metrics": plan["temporal_metrics"],
        "filters": filters,
        "order_by": (
            {"kind": "metric", "alias": order_by["alias"], "direction": order_by["direction"]}
            if order_by and order_by["alias"] in metric_by_alias
            else ({"kind": "scalar", "column": order_by["alias"], "direction": order_by["direction"]} if order_by else {})
        ),
        "limit": plan.get("limit"),
    }
    sql = compile_extended_query_sql(spec)
    if not sql:
        raise ValueError("material plan cannot compile")
    return sql
