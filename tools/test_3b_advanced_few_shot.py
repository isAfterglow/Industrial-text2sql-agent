"""Measure local 3B AdvancedPlan completion with promoted Few-shot examples."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage

from app.advanced_plan import (
    advanced_plan_completion_prompt,
    advanced_plan_family_prompt,
    compile_advanced_analysis_plan,
    parse_advanced_plan,
    parse_advanced_plan_family,
)
from app.db import execute_readonly_query
from app.llm import invoke_model, model_call_log, reset_model_call_log
from app.long_term_memory import get_long_term_memory_service
from app.result_assertions import assert_advanced_result
from app.schema import build_schema_context, get_schema_catalog, set_active_profile
from app.sql_guard import clean_llm_sql, validate_and_normalize_sql


QUESTIONS = [
    ("找出耗电量高于所属负荷类型平均耗电量20%的记录，返回读数编号、负荷类型和耗电量。", "group_threshold"),
    ("找出累计二氧化碳排放贡献达到总排放80%所需的最少读数，返回读数编号、排放量和累计排放。", "cumulative_share"),
]


def ask_3b(system: str, prompt: str) -> str:
    return invoke_model([SystemMessage(content=system), HumanMessage(content=prompt)], purpose="planning")


def run(question: str, forced_family: str = "") -> dict[str, object]:
    schema = build_schema_context()
    raw_family = ask_3b("Return JSON only.", advanced_plan_family_prompt(schema, question))
    family = forced_family or parse_advanced_plan_family(clean_llm_sql(raw_family))
    service = get_long_term_memory_service()
    memories, diagnostics = service.retrieve_advanced_plan_examples(question, family)
    context = service.build_advanced_plan_few_shot_context(memories)
    started = time.perf_counter()
    raw_plan = ask_3b(
        "Complete one constrained AdvancedAnalysisPlan JSON object, never SQL.",
        advanced_plan_completion_prompt(schema, question, family, context),
    )
    plan = parse_advanced_plan(clean_llm_sql(raw_plan))
    sql = compile_advanced_analysis_plan(plan)
    guarded = validate_and_normalize_sql(
        sql=sql, allowed_tables=set(get_schema_catalog()["tables"]), max_rows=200,
        question=question, query_spec={"query_type": "advanced_" + family, "advanced_plan": plan},
    )
    if not guarded.valid:
        raise RuntimeError(guarded.error)
    executed = execute_readonly_query(guarded.sql, 200)
    assertion = assert_advanced_result(plan, executed["columns"], executed["rows"])
    return {
        "question": question, "family": family, "forced_family": bool(forced_family), "memory_ids": [item.memory_id for item in memories],
        "retrieval": diagnostics, "plan": plan, "row_count": executed["row_count"],
        "assertion_passed": assertion.get("passed"), "completion_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def main() -> None:
    set_active_profile("steel_industry")
    reset_model_call_log()
    outcomes = []
    for question, expected_family in QUESTIONS:
        try:
            outcomes.append({"status": "success", **run(question)})
        except Exception as exc:
            outcomes.append({"status": "failed", "question": question, "error": f"{type(exc).__name__}: {exc}"})
        try:
            outcomes.append({"status": "success", **run(question, forced_family=expected_family)})
        except Exception as exc:
            outcomes.append({"status": "failed", "question": question, "forced_family": True, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"outcomes": outcomes, "model_calls": model_call_log()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
