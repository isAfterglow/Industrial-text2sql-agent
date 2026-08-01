"""Create two non-benchmark Resin structural Few-shot examples with evidence."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import execute_readonly_query
from app.long_term_memory import get_long_term_memory_service
from app.nodes import validate_sql
from app.query_enhancement import augment_common_query_spec, compile_extended_query_sql
from app.schema import build_query_spec, compile_query_spec_sql, set_active_profile


CASES = [
    [
        "找出原始密度最高的9个样本，返回样本编号和原始密度。",
        "查询碳化材料密度最高的7个样本，返回样本编号和碳化密度。",
        "找出原始孔隙率最小的5个样本，返回样本编号和原始孔隙率。",
        "查询碳化渗透率最低的6个样本，返回样本编号和碳化渗透率。",
    ],
    [
        "列出碳化导热系数最高的8个样本，并返回原始密度。",
        "找出原始导热系数最低的6个样本，同时显示碳化密度。",
        "按碳化导热系数降序取前4个样本，返回样本编号和原始孔隙率。",
        "给出原始导热系数最高的10个样本及其碳化孔隙率。",
    ],
    [
        "找出最终背温最高的6个样本并给出初始背温。",
        "返回最终表面温度最高的4个样本及初始表面温度。",
        "列出最终质量最低的7个样本，并显示初始质量。",
        "按最终背温从低到高取前5个样本，同时给出初始背温。",
    ],
]


def validate(question: str) -> tuple[dict, str, int]:
    spec = augment_common_query_spec(question, build_query_spec(question), {})
    sql = compile_extended_query_sql(spec) if spec.get("mode") == "deterministic_extended" else compile_query_spec_sql(spec)
    checked = validate_sql({"raw_sql": sql, "query_plan_mode": spec.get("mode"), "query_spec": spec, "memory_used": False})
    if checked.get("validation_error"):
        raise RuntimeError(checked["validation_error"])
    result = execute_readonly_query(checked["validated_sql"], 200)
    return spec, checked["validated_sql"], result["row_count"]


def main() -> None:
    set_active_profile("resin")
    service = get_long_term_memory_service()
    for questions in CASES:
        spec, sql, _ = validate(questions[0])
        candidate = service.remember_candidate_case(
            question=questions[0], resolved_question=questions[0], query_spec=spec, sql=sql,
            source="curated_non_benchmark", approval_reason="人工复核结构、Guard与真实数据库执行。",
        )
        for question in questions[1:]:
            variant_spec, _, rows = validate(question)
            service.record_candidate_validation(
                candidate.record.memory_id, question=question, plan=variant_spec,
                evidence=f"non_benchmark; guard+execution; rows={rows}",
            )
        promoted = service.promote_candidate(
            candidate.record.memory_id,
            evidence="Three independent non-benchmark variants passed compile, Guard and database execution.",
        )
        print(promoted.record.memory_id, promoted.record.metadata["query_spec"].get("query_type"))


if __name__ == "__main__":
    main()
