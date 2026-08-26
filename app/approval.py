"""Risk assessment and immutable snapshots for human approval."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from app.schema import get_schema_catalog


RISKY_PLAN_MODES = {
    "advanced_analysis_plan",
    "llm_query_spec",
    "rsl",
}
DECISION_ACTIONS = {"approved", "rejected", "edited_plan"}


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def plan_fingerprint(snapshot: dict[str, Any]) -> str:
    """Return a stable digest for the executable plan, excluding audit metadata."""

    executable = {
        key: snapshot.get(key)
        for key in (
            "question",
            "profile",
            "query_plan_mode",
            "query_spec",
            "advanced_plan",
            "compiled_sql",
            "delivery_policy",
            "retry_count",
        )
    }
    return hashlib.sha256(_canonical_json(executable).encode("utf-8")).hexdigest()


def _selected_columns(state: dict[str, Any]) -> set[str]:
    spec = state.get("query_spec") or {}
    plan = state.get("advanced_plan") or {}
    values: set[str] = set()
    for key in ("select_columns", "output_columns", "scalar_columns"):
        values.update(str(item) for item in spec.get(key, []) if item)
        values.update(str(item) for item in plan.get(key, []) if item)
    for key in ("metric", "dimension", "group_column", "period_column"):
        value = plan.get(key)
        if value:
            values.add(str(value))
    return values


def assess_approval_risk(state: dict[str, Any]) -> dict[str, Any]:
    """Classify execution risk from structured state, never from LLM prose."""

    spec = state.get("query_spec") or {}
    delivery = state.get("delivery_policy") or {}
    policy = get_schema_catalog().get("policy", {})
    reasons: list[str] = []

    mode = str(state.get("query_plan_mode") or "")
    if mode in RISKY_PLAN_MODES:
        reasons.append("advanced_or_llm_plan")
    if int(state.get("retry_count") or 0) > 0 or state.get("repair_source"):
        reasons.append("repaired_sql")
    if spec.get("query_type") == "full_table" or delivery.get("full_result_requested"):
        reasons.append("full_table_or_export")

    sensitive = {str(value) for value in policy.get("sensitive_columns", [])}
    selected_sensitive = sorted(_selected_columns(state) & sensitive)
    if selected_sensitive:
        reasons.append("sensitive_columns:" + ",".join(selected_sensitive))

    force = bool(state.get("force_approval", False))
    if force:
        reasons.append("forced")
    severity = "high" if selected_sensitive or "full_table_or_export" in reasons else "medium"
    if not reasons:
        severity = "low"
    return {
        "risky": bool(reasons),
        "severity": severity,
        "reasons": reasons,
        "selected_sensitive_columns": selected_sensitive,
    }


def build_approval_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Build the persisted, auditable approval payload before execution."""

    risk = assess_approval_risk(state)
    snapshot: dict[str, Any] = {
        "approval_version": 1,
        "trace_id": str(state.get("trace_id", "")),
        "question": str(state.get("question", "")),
        "resolved_question": str(state.get("resolved_question", "")),
        "profile": str(state.get("domain_profile", "")),
        "intent": str(state.get("query_intent", "")),
        "query_plan_mode": str(state.get("query_plan_mode", "")),
        "query_spec": dict(state.get("query_spec") or {}),
        "advanced_plan": dict(state.get("advanced_plan") or {}),
        "compiled_sql": str(state.get("validated_sql", "")),
        "schema_tables": list(state.get("intent_related_tables", [])),
        "delivery_policy": dict(state.get("delivery_policy") or {}),
        "retry_count": int(state.get("retry_count") or 0),
        "failure_events": list(state.get("failure_events", [])),
        "model_calls": list(state.get("model_calls", [])),
        "risk": risk,
        "evidence": {
            "selected_columns": sorted(_selected_columns(state)),
            "tables": sorted(str(item) for item in state.get("intent_related_tables", [])),
            "validated": bool(state.get("validated_sql")),
            "result_assertion": dict(state.get("result_assertion") or {}),
            "failure_categories": sorted({str(item.get("category", "unknown")) for item in state.get("failure_events", []) if isinstance(item, dict)}),
        },
    }
    snapshot["plan_fingerprint"] = plan_fingerprint(snapshot)
    return snapshot


def normalize_approval_decision(decision: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize user input into an auditable, restricted decision object."""

    value = dict(decision or {})
    action = str(value.get("action", "")).strip().lower()
    aliases = {"approve": "approved", "reject": "rejected"}
    action = aliases.get(action, action)
    if action and action not in DECISION_ACTIONS:
        raise ValueError("approval action must be approved, rejected, or edited_plan")
    if not action:
        return {}
    value["action"] = action
    value["actor"] = str(value.get("actor") or "graph_caller").strip()
    value["comment"] = re.sub(r"\s+", " ", str(value.get("comment") or "")).strip()
    return value
