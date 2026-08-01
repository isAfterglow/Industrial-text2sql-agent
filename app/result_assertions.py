"""Low-cost result invariants for constrained advanced analysis plans."""

from __future__ import annotations

from typing import Any


def _column(rows: list[list[Any]], columns: list[str], name: str) -> list[Any]:
    if name not in columns:
        return []
    index = columns.index(name)
    return [row[index] for row in rows if index < len(row)]


def _numeric(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def _descending(values: list[float]) -> bool:
    return all(left >= right for left, right in zip(values, values[1:]))


def assert_advanced_result(
    plan: dict[str, Any],
    columns: list[str],
    rows: list[list[Any]],
) -> dict[str, Any]:
    """Validate plan-specific result properties without re-running the query."""

    family = str(plan.get("family", ""))
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    if not plan:
        return {"checked": False, "passed": True, "family": "", "checks": [], "reason": "not_advanced_plan"}
    if not rows:
        return {"checked": True, "passed": True, "family": family, "checks": [], "reason": "empty_result"}

    if family == "period_change":
        alias = "month_over_month_change"
        values = _numeric(_column(rows, columns, alias))
        check("change_values_present", len(values) == len(rows), "环比结果必须是非空数值。")
        check("change_descending", _descending(values), "环比结果必须按增幅降序返回。")
    elif family == "cumulative_share":
        metric = str(plan.get("metric", ""))
        metric_values = _numeric(_column(rows, columns, metric))
        cumulative_values = _numeric(_column(rows, columns, f"cumulative_{metric}"))
        check("metric_descending", _descending(metric_values), "累计贡献必须按指标降序。")
        check(
            "cumulative_monotonic",
            all(left <= right for left, right in zip(cumulative_values, cumulative_values[1:])),
            "累计指标必须单调不减。",
        )
    elif family == "correlation":
        values = _numeric(_column(rows, columns, str(plan.get("output_alias", ""))))
        check("correlation_range", bool(values) and all(-1.000001 <= value <= 1.000001 for value in values), "皮尔逊相关系数必须在[-1, 1]内。")
    elif family == "group_share":
        period = str(plan.get("period_column", ""))
        share = str(plan.get("output_alias", ""))
        period_values = _column(rows, columns, period)
        share_values = _numeric(_column(rows, columns, share))
        check("share_range", len(share_values) == len(rows) and all(-1e-9 <= value <= 1.000001 for value in share_values), "份额必须在[0, 1]内。")
        grouped: dict[Any, float] = {}
        for period_value, share_value in zip(period_values, share_values):
            grouped[period_value] = grouped.get(period_value, 0.0) + share_value
        check("share_denominator", bool(grouped) and all(abs(total - 1.0) < 1e-6 for total in grouped.values()), "同一周期各组份额之和必须为1。")
    elif family == "rising_sequence":
        values = _numeric(_column(rows, columns, str(plan.get("output_alias", ""))))
        check("sequence_order", values == sorted(set(values)), "连续上升序列起点必须去重并升序。")
    elif family == "conditional_comparison":
        aliases = [str(plan.get("difference_alias", "")), str(plan.get("percentage_alias", ""))]
        check("comparison_values_present", all(len(_numeric(_column(rows, columns, alias))) == len(rows) for alias in aliases), "条件比较的差值和百分比必须均为数值。")
    elif family == "group_topk":
        limit = int(plan.get("limit", 0))
        check("topk_nonempty", limit > 0 and bool(rows), "分组Top-K必须返回至少一条结果。")
    elif family in {"group_outlier", "group_threshold"}:
        check("filtered_rows", bool(rows), "异常/阈值计划返回的行必须满足已编译过滤条件。")

    failed = [item["detail"] for item in checks if not item["passed"]]
    return {
        "checked": True,
        "passed": not failed,
        "family": family,
        "checks": checks,
        "reason": "；".join(failed),
    }
