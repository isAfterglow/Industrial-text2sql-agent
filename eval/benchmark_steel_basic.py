"""Steel deterministic/basic benchmark, excluding safety and LLM challenge cases."""

from eval.steel_benchmark_v1 import SUITE as _SOURCE

_BASIC_CATEGORIES = {"topk", "aggregate", "lookup", "time_analysis", "cross_table", "time_filter", "derived_metric", "complex_cross_filter", "complex_ranking", "repair_candidate", "robustness"}

SUITE = {
    "name": "steel_basic_deterministic",
    "version": "2.0.0",
    "profile": "steel_industry",
    "description": "钢铁事实表/维表基础查询与可配置聚合",
    "cases": [case for case in _SOURCE["cases"] if case.get("category") in _BASIC_CATEGORIES and "llm_fallback" not in case.get("tags", [])],
}
