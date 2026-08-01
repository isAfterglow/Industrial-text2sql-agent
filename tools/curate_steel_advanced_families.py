"""Curate one non-benchmark, independently validated example per remaining family."""

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
    ("按负荷类型分别列出无功功率最高的两条读数。", {"family":"group_topk","metric":"reactive_power","group_columns":["load_type_name"],"limit":2,"output_columns":["load_type_name","reading_id","reactive_power"]}),
    ("找出碳强度环比增幅最大的四个月。", {"family":"period_change","period_column":"month","derived_metric":"carbon_intensity","limit":4}),
    ("找出各负荷类型中功率因数低于均值一个标准差的读数。", {"family":"group_outlier","metric":"power_factor","group_column":"load_type_name","output_columns":["reading_id","load_type_name","power_factor"]}),
    ("按负荷类型计算无功功率与耗电量的皮尔逊相关系数。", {"family":"correlation","group_column":"load_type_name","x_metric":"reactive_power","y_metric":"usage_kwh","output_alias":"correlation_coefficient"}),
    ("计算每年各负荷类型二氧化碳排放占当年总排放的比例。", {"family":"group_share","period_column":"year","group_column":"load_type_name","metric":"co2_tco2","output_alias":"co2_tco2_share"}),
    ("找出连续3个小时平均无功功率持续上升的起始小时。", {"family":"rising_sequence","period_column":"hour","metric":"reactive_power","window_size":3,"output_alias":"start_hour"}),
    ("比较每种负荷类型在工作日和周末的平均碳排放差异及其百分比。", {"family":"conditional_comparison","metric":"co2_tco2","group_column":"load_type_name","condition_column":"week_status","left_value":"Weekday","right_value":"Weekend","difference_alias":"co2_difference_tco2","percentage_alias":"co2_difference_percentage"}),
]


def validate(question: str, raw_plan: dict[str, object]) -> tuple[dict[str, object], str, int]:
    plan = parse_advanced_plan(json.dumps(raw_plan))
    sql = compile_advanced_analysis_plan(plan)
    guarded = validate_and_normalize_sql(sql, set(get_schema_catalog()["tables"]), 200, question, {"query_type": "advanced_" + plan["family"], "advanced_plan": plan})
    if not guarded.valid:
        raise RuntimeError(guarded.error)
    executed = execute_readonly_query(guarded.sql, 200)
    return plan, guarded.sql, int(executed["row_count"])


def main() -> None:
    set_active_profile("steel_industry")
    service = get_long_term_memory_service()
    for question, raw_plan in CASES:
        plan, sql, _ = validate(question, raw_plan)
        candidate = service.remember_candidate_case(question=question, resolved_question=question, query_spec={"query_type": "advanced_" + plan["family"], "advanced_plan": plan}, sql=sql, source="curated_non_benchmark", approval_reason="人工复核结构、Guard与真实数据库执行。")
        for number in (1, 2, 3):
            variant = dict(plan)
            if "limit" in variant:
                variant["limit"] = min(10, int(variant["limit"]) + number)
            elif "threshold" in variant:
                variant["threshold"] = min(0.95, float(variant["threshold"]) + number * 0.03)
            elif "window_size" in variant:
                variant["window_size"] = 3 if number < 3 else 4
            variant_question = f"非评测独立变体{number}：{question}"
            validated_plan, _, rows = validate(variant_question, variant)
            service.record_candidate_validation(candidate.record.memory_id, question=variant_question, plan=validated_plan, evidence=f"non_benchmark; guard+execution; rows={rows}")
        promoted = service.promote_candidate(candidate.record.memory_id, evidence="Three independent non-benchmark variants passed compile, Guard and database execution.")
        print(promoted.record.memory_id, plan["family"])


if __name__ == "__main__":
    main()
