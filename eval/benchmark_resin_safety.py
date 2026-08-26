"""Material safety and policy benchmark."""

from eval.benchmark_v1 import SUITE as _SOURCE

SUITE = {
    "name": "resin_safety_policy",
    "version": "2.0.0",
    "profile": "resin",
    "description": "材料写操作、危险函数、越权和只读策略",
    "cases": [case for case in _SOURCE["cases"] if case.get("category") == "safety"],
}
