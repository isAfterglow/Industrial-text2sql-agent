"""Question-derived output contracts for constrained analytical plans."""

from __future__ import annotations

from typing import Any

from app.schema import infer_requested_output_columns


def build_query_expectation(question: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Build a small contract from the user request, independent of SQL text."""

    # This contract is the second safety net for constrained advanced plans.
    # Conventional QuerySpec paths already have their own semantic-coverage
    # validator; applying this lightweight projection check to them created a
    # duplicate, incompatible gate during regression evaluation.
    if not plan:
        return {
            "checked": False,
            "required_columns": [],
            "family": "",
            "source": "not_advanced_plan",
        }
    expected = set(infer_requested_output_columns(question))
    family = str(plan.get("family", ""))
    calculated: list[str] = []
    if family == "period_change":
        calculated = ["month_over_month_change"]
    elif family == "cumulative_share":
        calculated = [f"cumulative_{plan.get('metric', '')}"]
    elif family in {"correlation", "group_share", "rising_sequence"}:
        calculated = [str(plan.get("output_alias", ""))]
    elif family == "conditional_comparison":
        calculated = [str(plan.get("difference_alias", "")), str(plan.get("percentage_alias", ""))]
    expected.update(value for value in calculated if value)
    return {
        "checked": bool(expected),
        "required_columns": sorted(expected),
        "family": family,
        "source": "question_and_plan_api",
    }


def assert_query_expectation(expectation: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    if not expectation.get("checked"):
        return {"checked": False, "passed": True, "checks": [], "reason": "no_explicit_output_contract"}
    required = set(expectation.get("required_columns", []))
    actual = set(columns)
    missing = sorted(required - actual)
    return {
        "checked": True,
        "passed": not missing,
        "checks": [{
            "name": "question_required_columns",
            "passed": not missing,
            "detail": "问题明确要求的字段必须出现在结果中。",
            "missing": missing,
        }],
        "reason": "缺少问题要求的结果字段：" + ", ".join(missing) if missing else "",
    }
