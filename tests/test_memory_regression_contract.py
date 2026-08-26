from eval.run_evaluation import load_suite
from pathlib import Path


def test_memory_regression_has_state_leakage_cases():
    suite = load_suite(Path("eval/benchmark_memory_regression.py"))
    assert len(suite["cases"]) >= 10
    assert {case["category"] for case in suite["cases"]} >= {
        "memory_conflict", "memory_scope", "memory_truncated_anchor",
        "memory_profile_boundary", "memory_parent_anchor",
    }
