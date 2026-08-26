"""Material basic/deterministic benchmark (no safety or complex planning cases)."""

from eval.benchmark_v1 import SUITE as _SOURCE

_BASIC_CATEGORIES = {"single_table", "cross_table", "temporal", "robustness", "full_table"}

SUITE = {
    "name": "resin_basic_deterministic",
    "version": "2.0.0",
    "profile": "resin",
    "description": "材料基础确定性、字段投影、Top-K、时序点和鲁棒表达",
    "cases": [case for case in _SOURCE["cases"] if case.get("category") in _BASIC_CATEGORIES],
}
