"""Steel safety and policy benchmark."""

from eval.steel_benchmark_v1 import SUITE as _SOURCE

SUITE = {
    "name": "steel_safety_policy",
    "version": "2.0.0",
    "profile": "steel_industry",
    "description": "钢铁写操作、越权和系统表访问策略",
    "cases": [case for case in _SOURCE["cases"] if case.get("category") == "safety"],
}
