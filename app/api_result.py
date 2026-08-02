"""API result projection shared by API and background workers."""
from __future__ import annotations
from typing import Any
from app.trace import safe_json_value

def public_result(result: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    return safe_json_value({key: result.get(key, default) for key, default in {
        "final_status":"", "final_answer":"", "domain_profile":"", "query_intent":"", "query_plan_mode":"", "query_spec":{}, "advanced_plan":{}, "columns":[], "rows":[], "row_count":0, "truncated":False, "retry_count":0, "repair_source":"", "model_calls":[], "failure_events":[], "few_shot_retrieval_diagnostics":{}, "approval_required":False, "approval_request":{}, "approval_summary":{}, "validation_error":"", "execution_error":""}.items()} | {"profile": result.get("domain_profile", ""), "sql": result.get("validated_sql") or result.get("raw_sql", ""), "few_shot": result.get("few_shot_retrieval_diagnostics", {}), "elapsed_ms": round(elapsed_ms, 3)})
