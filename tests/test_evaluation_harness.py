from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from eval.harness import SUITES, metrics_from_results


class EvaluationHarnessTests(unittest.TestCase):
    def test_frozen_suite_contract(self) -> None:
        self.assertEqual(set(SUITES), {"resin_80", "steel_core_60", "steel_complex_10"})
        for config in SUITES.values():
            self.assertTrue(Path(config["suite"]).exists())

    def test_metrics_include_routing_repair_safety_and_few_shot(self) -> None:
        result = {
            "summary": {
                "cases": 2, "acceptable_pass": 2, "strict_pass": 1, "timeouts": 0,
                "worker_errors": 0, "model_role_counts": {"primary_3b": 2},
                "repaired_turns": 1, "repaired_success_turns": 1,
                "mean_turn_ms": 10.0, "p95_turn_ms": 20.0,
                "by_category": {"safety": {"cases": 1, "acceptable_pass": 1}},
            },
            "results": [{"turns": [
                {"result": {"few_shot_retrieval_diagnostics": {"few_shot_used": True}}},
                {"result": {"few_shot_retrieval_diagnostics": {"few_shot_used": False}}},
            ]}],
        }
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.json"
            path.write_text(json.dumps(result), encoding="utf-8")
            metrics = metrics_from_results(path, "resin_80")
        self.assertEqual(metrics["model_calls"], 2)
        self.assertEqual(metrics["repair_success_rate"], 1.0)
        self.assertEqual(metrics["safety_pass_rate"], 1.0)
        self.assertEqual(metrics["few_shot_hits"], 1)


if __name__ == "__main__":
    unittest.main()
