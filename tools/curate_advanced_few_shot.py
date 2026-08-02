"""Curate non-benchmark AdvancedPlan memories through real database validation.

This is deliberately a small, reviewable corpus.  It creates candidates first,
validates three independently worded variants per analytical family, and only
then promotes the base example.  Re-running it is idempotent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.advanced_plan import compile_advanced_analysis_plan, parse_advanced_plan
from app.db import execute_readonly_query
from app.long_term_memory import get_long_term_memory_service
from app.schema import get_schema_catalog, set_active_profile
from app.sql_guard import validate_and_normalize_sql


CASES = [
    {
        "question": "筛选碳排放超过各自负荷类型平均值25%的读数，给出读数、负荷类型与碳排放。",
        "plan": {
            "family": "group_threshold", "metric": "co2_tco2",
            "baseline_group_column": "load_type_name", "threshold_multiplier": 1.25,
            "output_columns": ["reading_id", "load_type_name", "co2_tco2"],
        },
        "variants": [
            ("找出耗电量比所属负荷类型平均耗电高20%的记录。", {"family": "group_threshold", "metric": "usage_kwh", "baseline_group_column": "load_type_name", "threshold_multiplier": 1.2, "output_columns": ["reading_id", "load_type_name", "usage_kwh"]}),
            ("列出碳排放高于本负荷类别均值10%的读数及排放量。", {"family": "group_threshold", "metric": "co2_tco2", "baseline_group_column": "load_type_name", "threshold_multiplier": 1.1, "output_columns": ["reading_id", "co2_tco2"]}),
            ("哪些读数的能耗达到所属负荷类型平均能耗的1.5倍以上？", {"family": "group_threshold", "metric": "usage_kwh", "baseline_group_column": "load_type_name", "threshold_multiplier": 1.5, "output_columns": ["reading_id", "load_type_name", "usage_kwh"]}),
        ],
    },
    {
        "question": "找出累计碳排放达到总排放75%所需的最少读数，返回读数、排放量和累计排放。",
        "plan": {
            "family": "cumulative_share", "metric": "co2_tco2", "threshold": 0.75,
            "output_columns": ["reading_id", "co2_tco2", "cumulative_co2_tco2"],
        },
        "variants": [
            ("按碳排放从高到低累计，覆盖总碳排放60%的记录有哪些？", {"family": "cumulative_share", "metric": "co2_tco2", "threshold": 0.6, "output_columns": ["reading_id", "co2_tco2", "cumulative_co2_tco2"]}),
            ("为了覆盖八成总碳排放，需要保留哪些最高排放读数？", {"family": "cumulative_share", "metric": "co2_tco2", "threshold": 0.8, "output_columns": ["reading_id", "co2_tco2", "cumulative_co2_tco2"]}),
            ("输出累计耗电量达到总耗电90%的最少读数。", {"family": "cumulative_share", "metric": "usage_kwh", "threshold": 0.9, "output_columns": ["reading_id", "usage_kwh", "cumulative_usage_kwh"]}),
        ],
    },
]


def validate(question: str, raw_plan: dict[str, object]) -> dict[str, object]:
    plan = parse_advanced_plan(json.dumps(raw_plan))
    sql = compile_advanced_analysis_plan(plan)
    result = validate_and_normalize_sql(
        sql=sql,
        allowed_tables=set(get_schema_catalog()["tables"]),
        max_rows=200,
        question=question,
        query_spec={"query_type": "advanced_" + plan["family"], "advanced_plan": plan},
    )
    if not result.valid:
        raise RuntimeError(f"Guard failed: {result.error}")
    executed = execute_readonly_query(result.sql, max_rows=200)
    return {"plan": plan, "sql": result.sql, "row_count": executed["row_count"]}


def main() -> None:
    set_active_profile("steel_industry")
    service = get_long_term_memory_service()
    for case in CASES:
        base = validate(case["question"], case["plan"])
        candidate = service.remember_candidate_case(
            question=case["question"], resolved_question=case["question"],
            query_spec={"query_type": "advanced_" + base["plan"]["family"], "advanced_plan": base["plan"]},
            sql=base["sql"], source="curated_non_benchmark", approval_reason="人工复核结构、Guard与真实数据库执行。",
        )
        for question, raw_plan in case["variants"]:
            outcome = validate(question, raw_plan)
            service.record_candidate_validation(
                candidate.record.memory_id, question=question, plan=outcome["plan"],
                evidence=f"non_benchmark; guard+execution; rows={outcome['row_count']}",
                validator="curation_runner",
            )
        promoted = service.promote_candidate(
            candidate.record.memory_id,
            evidence="Three independent non-benchmark variants passed compile, Guard and database execution.",
            approver="curation_reviewer",
            approval_reason="Three independent non-benchmark variants passed compile, Guard and database execution.",
        )
        print(f"{promoted.record.memory_id} {promoted.record.metadata['advanced_plan']['family']}")


if __name__ == "__main__":
    main()
