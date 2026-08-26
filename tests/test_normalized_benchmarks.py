from pathlib import Path

from eval.run_evaluation import load_suite


def test_normalized_benchmarks_are_disjoint_and_policy_separated():
    paths = [
        Path("eval/benchmark_resin_basic.py"),
        Path("eval/benchmark_resin_complex.py"),
        Path("eval/benchmark_resin_safety.py"),
        Path("eval/benchmark_steel_basic.py"),
        Path("eval/benchmark_steel_complex.py"),
        Path("eval/benchmark_steel_safety.py"),
    ]
    suites = [load_suite(path) for path in paths]
    ids = [str(case["id"]) for suite in suites for case in suite["cases"]]
    assert len(ids) == len(set(ids))
    for suite in suites:
        if suite["name"].endswith("safety_policy"):
            assert all(case.get("category") == "safety" for case in suite["cases"])
        else:
            assert all(case.get("category") != "safety" for case in suite["cases"])
    steel_basic = suites[3]
    assert all("llm_fallback" not in case.get("tags", []) for case in steel_basic["cases"])

