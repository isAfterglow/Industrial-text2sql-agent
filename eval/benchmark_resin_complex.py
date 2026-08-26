"""Material complex planning, repair, memory and boundary benchmark."""

from eval.benchmark_v1 import SUITE as _SOURCE

_COMPLEX_CATEGORIES = {
    "multi_turn", "long_term_memory", "repair_regression", "cross_temporal",
    "derived_metric", "capability_boundary", "clarification",
}

SUITE = {
    "name": "resin_complex_agent",
    "version": "2.0.0",
    "profile": "resin",
    "description": "材料复杂规划、修复、记忆、多轮和能力边界",
    "cases": [case for case in _SOURCE["cases"] if case.get("category") in _COMPLEX_CATEGORIES],
}
