"""Constrained compiler for advanced fact/dimension analytical queries."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schema import active_profile_name, get_column_owner_map, get_schema_catalog, _load_profile


ADVANCED_FAMILIES = {
    "group_topk",
    "period_change",
    "group_outlier",
    "cumulative_share",
    "conditional_comparison",
    "group_threshold",
    "correlation",
    "group_share",
    "rising_sequence",
}


def _json_object_from_text(raw: str) -> dict[str, Any]:
    """Extract one JSON object from otherwise chatty local-model output."""

    cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("no JSON object found in model output")


class _PlanModel(BaseModel):
    """Strict contract for an LLM-produced advanced plan."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    family: str


class GroupTopKPlan(_PlanModel):
    family: str = "group_topk"
    metric: str
    group_columns: list[str] = Field(min_length=1, max_length=3)
    limit: int = Field(ge=1, le=100)
    output_columns: list[str] = Field(min_length=1, max_length=8)


class PeriodChangePlan(_PlanModel):
    family: str = "period_change"
    period_column: str
    derived_metric: str
    limit: int = Field(ge=1, le=100)


class GroupOutlierPlan(_PlanModel):
    family: str = "group_outlier"
    metric: str
    group_column: str
    output_columns: list[str] = Field(min_length=1, max_length=8)


class CumulativeSharePlan(_PlanModel):
    family: str = "cumulative_share"
    metric: str
    threshold: float = Field(gt=0, le=1)
    output_columns: list[str] = Field(min_length=1, max_length=4)


class ConditionalComparisonPlan(_PlanModel):
    family: str = "conditional_comparison"
    metric: str
    group_column: str
    condition_column: str
    left_value: str = Field(min_length=1, max_length=120)
    right_value: str = Field(min_length=1, max_length=120)
    difference_alias: str = Field(min_length=1, max_length=80)
    percentage_alias: str = Field(min_length=1, max_length=80)


class GroupThresholdPlan(_PlanModel):
    family: str = "group_threshold"
    metric: str
    baseline_group_column: str
    threshold_multiplier: float = Field(gt=0, le=10)
    output_columns: list[str] = Field(min_length=1, max_length=8)


class CorrelationPlan(_PlanModel):
    family: str = "correlation"
    group_column: str
    x_metric: str
    y_metric: str
    output_alias: str = Field(min_length=1, max_length=80)


class GroupSharePlan(_PlanModel):
    family: str = "group_share"
    period_column: str
    group_column: str
    metric: str
    output_alias: str = Field(min_length=1, max_length=80)


class RisingSequencePlan(_PlanModel):
    family: str = "rising_sequence"
    period_column: str
    metric: str
    window_size: int = Field(ge=3, le=5)
    output_alias: str = Field(min_length=1, max_length=80)


PLAN_MODELS = {
    "group_topk": GroupTopKPlan,
    "period_change": PeriodChangePlan,
    "group_outlier": GroupOutlierPlan,
    "cumulative_share": CumulativeSharePlan,
    "conditional_comparison": ConditionalComparisonPlan,
    "group_threshold": GroupThresholdPlan,
    "correlation": CorrelationPlan,
    "group_share": GroupSharePlan,
    "rising_sequence": RisingSequencePlan,
}


def advanced_plan_prompt(schema_context: str, question: str) -> str:
    return f"""Return JSON only: {{\"advanced_plan\": {{...}}}}.
Supported family values: group_topk, period_change, group_outlier, cumulative_share, conditional_comparison, group_threshold, correlation, group_share, rising_sequence.
Use real field names from the schema. Do not output SQL.
Do not add keys outside the selected family contract. String fields must be real schema fields except condition values.

Family contracts:
- group_topk: metric, group_columns, limit, output_columns
- period_change: period_column, derived_metric, limit
- group_outlier: metric, group_column, output_columns
- cumulative_share: metric, threshold, output_columns
- conditional_comparison: metric, group_column, condition_column, left_value, right_value, difference_alias, percentage_alias
- group_threshold: metric, baseline_group_column, threshold_multiplier, output_columns
- correlation: group_column, x_metric, y_metric, output_alias
- group_share: period_column, group_column, metric, output_alias
- rising_sequence: period_column, metric, window_size, output_alias

Schema:
{schema_context}
Question:
{question}"""


def advanced_plan_family_prompt(schema_context: str, question: str) -> str:
    """The cheap first stage only classifies the analytical operator.

    Asking a 3B model to emit every family-specific field was the dominant
    cause of contract failures.  This deliberately tiny protocol gives it one
    unambiguous task; schema fields are completed in a second constrained step.
    """

    return f"""Return JSON only: {{\"family\": \"...\"}}.
Choose exactly one family:
- group_topk: top K records inside every group;
- period_change: period-over-period carbon-intensity change;
- group_outlier: values below a group mean minus one standard deviation;
- cumulative_share: minimum records reaching a total metric share;
- conditional_comparison: compare two named condition values inside each group;
- group_threshold: records above a multiplier of their group's mean;
- correlation: Pearson correlation inside each group;
- group_share: each group's metric share within each period;
- rising_sequence: a consecutive increasing metric sequence.
Do not output SQL or any other keys.

Schema:
{schema_context}
Question:
{question}"""


def parse_advanced_plan_family(raw: str) -> str:
    payload = _json_object_from_text(raw)
    family = str(payload.get("family", ""))
    if family not in ADVANCED_FAMILIES:
        raise ValueError("unsupported advanced plan family")
    return family


def advanced_plan_completion_prompt(
    schema_context: str, question: str, family: str, few_shot_context: str = "",
) -> str:
    """Give the stronger model a single-family contract, not nine schemas."""

    contracts = {
        "group_topk": "metric, group_columns, limit, output_columns",
        "period_change": "period_column, derived_metric, limit",
        "group_outlier": "metric, group_column, output_columns",
        "cumulative_share": "metric, threshold, output_columns",
        "conditional_comparison": "metric, group_column, condition_column, left_value, right_value, difference_alias, percentage_alias",
        "group_threshold": "metric, baseline_group_column, threshold_multiplier, output_columns",
        "correlation": "group_column, x_metric, y_metric, output_alias",
        "group_share": "period_column, group_column, metric, output_alias",
        "rising_sequence": "period_column, metric, window_size, output_alias",
    }
    return f"""Return JSON only: {{\"advanced_plan\": {{...}}}}. Do not output SQL.
The family has already been selected as `{family}`. Include `family` plus exactly
these keys: {contracts[family]}. Use only real field names from the schema.

Schema:
{schema_context}
{few_shot_context}
Question:
{question}"""


def parse_advanced_plan(raw: str) -> dict[str, Any]:
    payload = _json_object_from_text(raw)
    plan = payload.get("advanced_plan", payload)
    if not isinstance(plan, dict):
        raise ValueError("advanced_plan must be an object")
    # Small local models often use `table.column` despite the contract asking
    # for field names. Normalize only known schema suffixes before validation.
    owners = get_column_owner_map()
    normalized_plan: dict[str, Any] = {}
    for key, value in plan.items():
        if isinstance(value, str) and "." in value:
            suffix = value.rsplit(".", 1)[-1]
            normalized_plan[key] = suffix if suffix in owners else value
        elif isinstance(value, list):
            normalized_plan[key] = [
                item.rsplit(".", 1)[-1]
                if isinstance(item, str) and "." in item and item.rsplit(".", 1)[-1] in owners
                else item
                for item in value
            ]
        else:
            normalized_plan[key] = value
    family = str(normalized_plan.get("family", ""))
    model = PLAN_MODELS.get(family)
    if model is None:
        raise ValueError("unsupported advanced plan family")
    try:
        parsed = model.model_validate(normalized_plan).model_dump()
        # Calculated columns are a compiler API, not a model preference. Stable
        # aliases make downstream result contracts and client integrations safe.
        if family == "correlation":
            parsed["output_alias"] = "correlation_coefficient"
        elif family == "group_share":
            parsed["output_alias"] = f"{parsed['metric']}_share"
        elif family == "rising_sequence":
            parsed["output_alias"] = f"start_{parsed['period_column']}"
        elif family == "conditional_comparison":
            metric = str(parsed["metric"])
            stem, _, unit = metric.partition("_")
            parsed["difference_alias"] = f"{stem}_difference_{unit}" if unit else f"{metric}_difference"
            parsed["percentage_alias"] = f"{stem}_difference_percentage"
        return parsed
    except ValidationError as exc:
        raise ValueError(f"invalid {family} plan contract: {exc}") from exc


def _fact_and_aliases() -> tuple[str, str, dict[str, str], dict[str, set[str]]]:
    catalog = get_schema_catalog()
    facts = [name for name, info in catalog["tables"].items() if info.get("grain") == "fact"]
    if len(facts) != 1:
        raise ValueError("advanced compiler requires exactly one fact table")
    fact = facts[0]
    aliases = {name: str(info["alias"]) for name, info in catalog["tables"].items()}
    return fact, aliases[fact], aliases, get_column_owner_map()


def _joins_for_columns(columns: list[str], fact: str, aliases: dict[str, str], owners: dict[str, set[str]]) -> list[str]:
    profile = _load_profile(active_profile_name())
    tables = {owner for column in columns for owner in owners.get(column, set()) if owner != fact}
    joins: list[str] = []
    for table in sorted(tables):
        relation = next((item for item in profile.get("relationships", []) if {str(item["left"]).split(".")[0], str(item["right"]).split(".")[0]} == {fact, table}), None)
        if relation is None:
            raise ValueError(f"no declared fact relationship for {table}")
        left_table, left_column = str(relation["left"]).split(".", 1)
        right_table, right_column = str(relation["right"]).split(".", 1)
        joins.append(f"JOIN {table} AS {aliases[table]} ON {aliases[left_table]}.{left_column} = {aliases[right_table]}.{right_column}")
    return joins


def _qualified(column: str, aliases: dict[str, str], owners: dict[str, set[str]]) -> str:
    owner = next(iter(owners.get(column, set())), "")
    if not owner:
        raise ValueError(f"unknown plan field: {column}")
    return f"{aliases[owner]}.{column}"


def _literal(value: str) -> str:
    return value.replace("'", "''")


def compile_advanced_analysis_plan(plan: dict[str, Any]) -> str:
    """Validate and compile one bounded analytical family into MySQL SQL."""

    family = str(plan.get("family", ""))
    if family not in ADVANCED_FAMILIES:
        raise ValueError("unsupported advanced plan family")
    fact, fact_alias, aliases, owners = _fact_and_aliases()
    catalog = get_schema_catalog()
    aggregations = catalog.get("aggregations", {})

    def field(name: str) -> str:
        value = str(plan.get(name, ""))
        if value not in owners:
            raise ValueError(f"missing or unknown {name}")
        return value

    if family == "group_topk":
        metric = field("metric")
        groups = [str(value) for value in plan.get("group_columns", [])]
        limit = int(plan.get("limit", 0))
        outputs = [str(value) for value in plan.get("output_columns", [])]
        if metric not in aggregations or not groups or limit < 1 or not outputs:
            raise ValueError("invalid group_topk plan")
        if any(column not in owners for column in groups + outputs):
            raise ValueError("unknown group_topk field")
        joins = _joins_for_columns(groups + outputs, fact, aliases, owners)
        partition = ", ".join(_qualified(column, aliases, owners) for column in groups)
        inner_columns = list(dict.fromkeys(groups + outputs))
        select = ", ".join(_qualified(column, aliases, owners) for column in inner_columns)
        final_order = ", ".join(outputs)
        return (
            f"WITH ranked AS (SELECT {select}, ROW_NUMBER() OVER (PARTITION BY {partition} "
            f"ORDER BY {fact_alias}.{metric} DESC, {fact_alias}.{catalog['tables'][fact]['key']}) AS row_num "
            f"FROM {fact} AS {fact_alias} {' '.join(joins)}) "
            f"SELECT {', '.join(outputs)} FROM ranked WHERE row_num <= {limit} ORDER BY {final_order}"
        )

    if family == "period_change":
        period = field("period_column")
        derived = str(plan.get("derived_metric", ""))
        limit = int(plan.get("limit", 0))
        if derived != "carbon_intensity" or limit < 1:
            raise ValueError("invalid period_change plan")
        joins = _joins_for_columns([period], fact, aliases, owners)
        period_sql = _qualified(period, aliases, owners)
        metric_sql = f"SUM({fact_alias}.co2_tco2) / NULLIF(SUM({fact_alias}.usage_kwh), 0)"
        return (
            f"WITH periods AS (SELECT {period_sql} AS {period}, {metric_sql} AS carbon_intensity "
            f"FROM {fact} AS {fact_alias} {' '.join(joins)} GROUP BY {period_sql}), "
            f"changes AS (SELECT {period}, (carbon_intensity - LAG(carbon_intensity) OVER (ORDER BY {period})) / "
            f"NULLIF(LAG(carbon_intensity) OVER (ORDER BY {period}), 0) AS period_change FROM periods) "
            f"SELECT {period}, period_change AS month_over_month_change FROM changes WHERE period_change IS NOT NULL "
            f"ORDER BY month_over_month_change DESC LIMIT {limit}"
        )

    if family == "group_outlier":
        metric, group = field("metric"), field("group_column")
        outputs = [str(value) for value in plan.get("output_columns", [])]
        if (
            metric not in catalog["tables"][fact]["columns"]
            or not outputs
            or any(column not in owners for column in outputs)
        ):
            raise ValueError("invalid group_outlier plan")
        joins = _joins_for_columns([group] + outputs, fact, aliases, owners)
        group_sql = _qualified(group, aliases, owners)
        return (
            f"WITH stats AS (SELECT {group_sql} AS group_key, AVG({fact_alias}.{metric}) AS average_metric, "
            f"STDDEV_SAMP({fact_alias}.{metric}) AS stddev_metric FROM {fact} AS {fact_alias} {' '.join(joins)} "
            f"GROUP BY {group_sql}) SELECT {', '.join(_qualified(column, aliases, owners) for column in outputs)} "
            f"FROM {fact} AS {fact_alias} {' '.join(joins)} JOIN stats ON stats.group_key = {group_sql} "
            f"WHERE {fact_alias}.{metric} < stats.average_metric - stats.stddev_metric "
            f"ORDER BY {fact_alias}.{catalog['tables'][fact]['key']}"
        )

    if family == "cumulative_share":
        metric = field("metric")
        threshold = float(plan.get("threshold", 0))
        outputs = [str(value) for value in plan.get("output_columns", [])]
        key = catalog["tables"][fact]["key"]
        allowed_outputs = {key, metric, f"cumulative_{metric}"}
        if (
            metric not in aggregations
            or not 0 < threshold <= 1
            or not outputs
            or any(column not in allowed_outputs for column in outputs)
        ):
            raise ValueError("invalid cumulative_share plan")
        return (
            f"WITH ordered AS (SELECT {fact_alias}.{key}, {fact_alias}.{metric}, "
            f"SUM({fact_alias}.{metric}) OVER (ORDER BY {fact_alias}.{metric} DESC, {fact_alias}.{key}) AS cumulative_{metric}, "
            f"SUM({fact_alias}.{metric}) OVER () AS total_{metric} FROM {fact} AS {fact_alias}) "
            f"SELECT {', '.join(outputs)} FROM ordered WHERE cumulative_{metric} - {metric} < total_{metric} * {threshold} "
            f"ORDER BY {metric} DESC, {key}"
        )

    if family == "group_threshold":
        metric, baseline = field("metric"), field("baseline_group_column")
        multiplier = float(plan.get("threshold_multiplier", 0))
        outputs = [str(value) for value in plan.get("output_columns", [])]
        if (
            metric not in catalog["tables"][fact]["columns"]
            or baseline not in owners
            or multiplier <= 0
            or not outputs
            or any(column not in owners for column in outputs)
        ):
            raise ValueError("invalid group_threshold plan")
        joins = _joins_for_columns([baseline] + outputs, fact, aliases, owners)
        select = ", ".join(_qualified(column, aliases, owners) for column in outputs)
        key = catalog["tables"][fact]["key"]
        baseline_sql = _qualified(baseline, aliases, owners)
        return (
            f"WITH baseline AS (SELECT {baseline_sql} AS baseline_key, AVG({fact_alias}.{metric}) AS average_metric "
            f"FROM {fact} AS {fact_alias} {' '.join(joins)} GROUP BY {baseline_sql}) "
            f"SELECT {select} FROM {fact} AS {fact_alias} {' '.join(joins)} "
            f"JOIN baseline ON baseline.baseline_key = {baseline_sql} "
            f"WHERE {fact_alias}.{metric} > baseline.average_metric * {multiplier} "
            f"ORDER BY {fact_alias}.{key}"
        )

    if family == "correlation":
        group, x_metric, y_metric = field("group_column"), field("x_metric"), field("y_metric")
        alias = str(plan.get("output_alias", ""))
        if x_metric not in aggregations or y_metric not in aggregations or not alias:
            raise ValueError("invalid correlation plan")
        joins = _joins_for_columns([group], fact, aliases, owners)
        group_sql = _qualified(group, aliases, owners)
        numerator = f"COUNT(*) * SUM({fact_alias}.{x_metric} * {fact_alias}.{y_metric}) - SUM({fact_alias}.{x_metric}) * SUM({fact_alias}.{y_metric})"
        denominator = f"SQRT((COUNT(*) * SUM({fact_alias}.{x_metric} * {fact_alias}.{x_metric}) - POW(SUM({fact_alias}.{x_metric}), 2)) * (COUNT(*) * SUM({fact_alias}.{y_metric} * {fact_alias}.{y_metric}) - POW(SUM({fact_alias}.{y_metric}), 2)))"
        return (
            f"SELECT {group_sql} AS {group}, ({numerator}) / NULLIF({denominator}, 0) AS {alias} "
            f"FROM {fact} AS {fact_alias} {' '.join(joins)} GROUP BY {group_sql} ORDER BY {group_sql}"
        )

    if family == "group_share":
        period, group, metric = field("period_column"), field("group_column"), field("metric")
        alias = str(plan.get("output_alias", ""))
        if metric not in aggregations or not alias:
            raise ValueError("invalid group_share plan")
        joins = _joins_for_columns([period, group], fact, aliases, owners)
        period_sql, group_sql = _qualified(period, aliases, owners), _qualified(group, aliases, owners)
        return (
            f"WITH grouped AS (SELECT {period_sql} AS {period}, {group_sql} AS {group}, "
            f"SUM({fact_alias}.{metric}) AS grouped_metric FROM {fact} AS {fact_alias} {' '.join(joins)} "
            f"GROUP BY {period_sql}, {group_sql}), totals AS (SELECT {period}, SUM(grouped_metric) AS total_metric "
            f"FROM grouped GROUP BY {period}) SELECT grouped.{period}, grouped.{group}, "
            f"grouped.grouped_metric / NULLIF(totals.total_metric, 0) AS {alias} "
            f"FROM grouped JOIN totals ON totals.{period} = grouped.{period} "
            f"ORDER BY grouped.{period}, grouped.{group}"
        )

    if family == "rising_sequence":
        period, metric = field("period_column"), field("metric")
        window_size = int(plan.get("window_size", 0))
        alias = str(plan.get("output_alias", ""))
        if metric not in aggregations or window_size < 3 or not alias:
            raise ValueError("invalid rising_sequence plan")
        joins = _joins_for_columns([period], fact, aliases, owners)
        period_sql = _qualified(period, aliases, owners)
        lag_columns = ", ".join(
            f"LAG(average_metric, {offset}) OVER (ORDER BY {period}) AS prior_{offset}"
            for offset in range(1, window_size)
        )
        comparisons = " AND ".join(
            "average_metric > prior_1" if offset == 1 else f"prior_{offset - 1} > prior_{offset}"
            for offset in range(1, window_size)
        )
        return (
            f"WITH periods AS (SELECT {period_sql} AS {period}, AVG({fact_alias}.{metric}) AS average_metric "
            f"FROM {fact} AS {fact_alias} {' '.join(joins)} GROUP BY {period_sql}), sequenced AS "
            f"(SELECT {period}, average_metric, {lag_columns} FROM periods) "
            f"SELECT {period} - {window_size - 1} AS {alias} FROM sequenced WHERE {comparisons} ORDER BY {alias}"
        )

    metric, group, condition = field("metric"), field("group_column"), field("condition_column")
    left, right = str(plan.get("left_value", "")), str(plan.get("right_value", ""))
    difference_alias = str(plan.get("difference_alias", ""))
    percentage_alias = str(plan.get("percentage_alias", ""))
    if metric not in aggregations or not left or not right or not difference_alias or not percentage_alias:
        raise ValueError("invalid conditional_comparison plan")
    joins = _joins_for_columns([group, condition], fact, aliases, owners)
    group_sql, condition_sql = _qualified(group, aliases, owners), _qualified(condition, aliases, owners)
    return (
        f"SELECT {group_sql} AS {group}, AVG(CASE WHEN {condition_sql} = '{_literal(left)}' THEN {fact_alias}.{metric} END) - "
        f"AVG(CASE WHEN {condition_sql} = '{_literal(right)}' THEN {fact_alias}.{metric} END) AS {difference_alias}, "
        f"(AVG(CASE WHEN {condition_sql} = '{_literal(left)}' THEN {fact_alias}.{metric} END) - AVG(CASE WHEN {condition_sql} = '{_literal(right)}' THEN {fact_alias}.{metric} END)) / "
        f"NULLIF(AVG(CASE WHEN {condition_sql} = '{_literal(right)}' THEN {fact_alias}.{metric} END), 0) AS {percentage_alias} "
        f"FROM {fact} AS {fact_alias} {' '.join(joins)} GROUP BY {group_sql} ORDER BY {group_sql}"
    )
