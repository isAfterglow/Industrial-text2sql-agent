#!/usr/bin/env python3
"""Versioned multi-suite evaluation Harness and baseline comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "eval" / "run_evaluation.py"
DEFAULT_OUTPUT = ROOT / "eval" / "harness_runs"
SUITES = {
    "resin_80": {
        "suite": ROOT / "eval" / "benchmark_v1.py",
        "description": "Resin 80-case engineering benchmark",
        "case_timeout": 90,
    },
    "steel_core_60": {
        "suite": ROOT / "eval" / "steel_benchmark_v1.py",
        "description": "Steel deterministic and safety benchmark",
        "case_timeout": 90,
    },
    "steel_complex_10": {
        "suite": ROOT / "eval" / "steel_agent_challenge_v1.py",
        "description": "Steel LLM planning and repair challenge",
        "case_timeout": 120,
    },
}

# These labels describe evaluation intent, rather than implementation route.
# They keep deterministic Profile coverage distinct from planning/memory flows.
_SEGMENTS = {
    "resin_80": {
        "basic": {"single_table", "cross_table", "temporal"},
        "complex": {"cross_temporal", "derived_metric", "full_table", "long_term_memory", "multi_turn", "repair_regression", "robustness"},
    },
    "steel_core_60": {"basic": set()},
    "steel_complex_10": {"complex": set()},
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _segment_metrics(summary: dict[str, Any], suite_key: str) -> dict[str, dict[str, Any]]:
    categories = dict(summary.get("by_category") or {})
    configured = _SEGMENTS.get(suite_key, {})
    if suite_key == "steel_core_60":
        configured = {"basic": {name for name in categories if name != "safety"}}
    elif suite_key == "steel_complex_10":
        configured = {"complex": set(categories)}
    segments: dict[str, dict[str, Any]] = {}
    for label, names in configured.items():
        values = [dict(categories[name]) for name in names if name in categories]
        cases = sum(int(value.get("cases", 0)) for value in values)
        acceptable = sum(int(value.get("acceptable_pass", 0)) for value in values)
        strict = sum(int(value.get("strict_pass", 0)) for value in values)
        segments[label] = {
            "cases": cases,
            "acceptable_pass": acceptable,
            "strict_pass": strict,
            "acceptable_rate": _rate(acceptable, cases),
            "strict_rate": _rate(strict, cases),
        }
    return segments


def metrics_from_results(path: Path, suite_key: str) -> dict[str, Any]:
    """Normalize the existing evaluator artifact into stable release metrics."""

    payload = _read_json(path)
    summary = dict(payload.get("summary") or {})
    results = list(payload.get("results") or [])
    turns = [turn for case in results for turn in case.get("turns", [])]
    few_shot_hits = sum(
        bool(turn.get("result", {}).get("few_shot_retrieval_diagnostics", {}).get("few_shot_used"))
        for turn in turns
    )
    safety = dict(summary.get("by_category", {}).get("safety", {}))
    model_roles = dict(summary.get("model_role_counts", {}))
    model_calls = sum(int(value) for value in model_roles.values())
    repaired = int(summary.get("repaired_turns", 0))
    repaired_success = int(summary.get("repaired_success_turns", 0))
    cases = int(summary.get("cases", len(results)))
    return {
        "suite": suite_key,
        "source_results": str(path),
        "cases": cases,
        "acceptable_pass": int(summary.get("acceptable_pass", 0)),
        "strict_pass": int(summary.get("strict_pass", 0)),
        "acceptable_rate": _rate(int(summary.get("acceptable_pass", 0)), cases),
        "strict_rate": _rate(int(summary.get("strict_pass", 0)), cases),
        "timeouts": int(summary.get("timeouts", 0)),
        "worker_errors": int(summary.get("worker_errors", 0)),
        "model_role_calls": model_roles,
        "model_calls": model_calls,
        "repair_attempts": repaired,
        "repair_successes": repaired_success,
        "repair_success_rate": _rate(repaired_success, repaired),
        "mean_turn_ms": float(summary.get("mean_turn_ms", 0.0)),
        "p95_turn_ms": float(summary.get("p95_turn_ms", 0.0)),
        "safety_cases": int(safety.get("cases", 0)),
        "safety_pass": int(safety.get("acceptable_pass", 0)),
        "safety_pass_rate": _rate(int(safety.get("acceptable_pass", 0)), int(safety.get("cases", 0))),
        "few_shot_hits": few_shot_hits,
        "few_shot_hit_rate": _rate(few_shot_hits, len(turns)),
        "segments": _segment_metrics(summary, suite_key),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(args: argparse.Namespace) -> int:
    paths = dict(item.split("=", 1) for item in args.result)
    unknown = sorted(set(paths) - set(SUITES))
    if unknown:
        raise ValueError("Unknown suite keys: " + ", ".join(unknown))
    missing = sorted(set(SUITES) - set(paths))
    if missing:
        raise ValueError("Missing suite artifacts: " + ", ".join(missing))
    payload = {
        "evaluation_summary_version": 1,
        "version": 1,
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suites": {key: metrics_from_results(Path(paths[key]), key) for key in SUITES},
    }
    write_json(Path(args.output), payload)
    print(f"Harness metrics: {args.output}")
    return 0


def _changes(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in SUITES:
        old = baseline["suites"][suite]
        new = candidate["suites"][suite]
        for field in ("acceptable_pass", "strict_pass", "safety_pass", "model_calls", "p95_turn_ms", "few_shot_hits"):
            rows.append({"suite": suite, "metric": field, "baseline": old.get(field), "candidate": new.get(field),
                         "delta": float(new.get(field, 0)) - float(old.get(field, 0))})
    return rows


def compare(args: argparse.Namespace) -> int:
    baseline = _read_json(Path(args.baseline))
    candidate = _read_json(Path(args.candidate))
    rows = _changes(baseline, candidate)
    lines = ["# Evaluation Comparison", "", f"- Baseline: `{baseline.get('label', '')}`", f"- Candidate: `{candidate.get('label', '')}`", "", "| Suite | Metric | Baseline | Candidate | Delta |", "|---|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['suite']} | {row['metric']} | {row['baseline']} | {row['candidate']} | {row['delta']:+.3f} |")
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Comparison: {args.output}")
    else:
        print(text)
    regressions = [
        row for row in rows
        if row["metric"] in {"acceptable_pass", "strict_pass", "safety_pass"} and row["delta"] < 0
    ]
    return 2 if regressions and args.fail_on_regression else 0


def regression_gate(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Evaluate stable PR gates while allowing LLM-path strict variance.

    Safety and acceptable semantics must never regress. Exact projection is a
    hard gate only for deterministic basic segments; complex strict variance is
    surfaced in the report but deliberately does not block a pull request.
    """

    failures: list[dict[str, Any]] = []
    notices: list[dict[str, Any]] = []
    for suite in SUITES:
        old = dict(baseline["suites"][suite])
        new = dict(candidate["suites"][suite])
        for metric in ("acceptable_pass", "safety_pass"):
            if int(new.get(metric, 0)) < int(old.get(metric, 0)):
                failures.append({"suite": suite, "metric": metric, "baseline": old.get(metric), "candidate": new.get(metric)})
        for metric in ("timeouts", "worker_errors"):
            if int(new.get(metric, 0)) > 0:
                failures.append({"suite": suite, "metric": metric, "baseline": 0, "candidate": new.get(metric)})
        for segment, old_values in dict(old.get("segments") or {}).items():
            new_values = dict(new.get("segments", {}).get(segment) or {})
            if not new_values:
                failures.append({"suite": suite, "metric": f"segment.{segment}.missing", "baseline": old_values, "candidate": None})
                continue
            if segment == "basic" and int(new_values.get("strict_pass", 0)) < int(old_values.get("strict_pass", 0)):
                failures.append({"suite": suite, "metric": "basic.strict_pass", "baseline": old_values.get("strict_pass"), "candidate": new_values.get("strict_pass")})
            if segment == "complex" and int(new_values.get("strict_pass", 0)) < int(old_values.get("strict_pass", 0)):
                notices.append({"suite": suite, "metric": "complex.strict_pass", "baseline": old_values.get("strict_pass"), "candidate": new_values.get("strict_pass")})
    return {"version": 1, "baseline": baseline.get("label", ""), "candidate": candidate.get("label", ""), "passed": not failures, "failures": failures, "notices": notices}


def gate(args: argparse.Namespace) -> int:
    outcome = regression_gate(_read_json(Path(args.baseline)), _read_json(Path(args.candidate)))
    text = json.dumps(outcome, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        write_json(Path(args.output), outcome)
        print(f"Regression gate: {args.output}")
    print(text)
    return 0 if outcome["passed"] else 2


def run(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve() / args.label
    artifacts: list[str] = []
    for key, config in SUITES.items():
        run_id = key
        command = [
            sys.executable, str(RUNNER), "--suite", str(config["suite"]),
            "--output-root", str(root), "--run-id", run_id,
            "--case-timeout", str(config["case_timeout"]),
            "--memory-mode", args.memory_mode,
        ]
        print("Running", key, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        artifacts.append(f"{key}={root / run_id / 'results.json'}")
    return collect(argparse.Namespace(label=args.label, result=artifacts, output=str(root / "metrics.json")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and compare versioned Agent benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run resin, steel core, and steel complex suites")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    run_parser.add_argument("--memory-mode", choices=("isolated", "production"), default="production")
    run_parser.set_defaults(handler=run)
    collect_parser = commands.add_parser("collect", help="Create a metrics artifact from existing evaluator results")
    collect_parser.add_argument("--label", required=True)
    collect_parser.add_argument("--result", action="append", required=True, metavar="SUITE=RESULTS_JSON")
    collect_parser.add_argument("--output", required=True)
    collect_parser.set_defaults(handler=collect)
    compare_parser = commands.add_parser("compare", help="Compare two harness metric artifacts")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--output", default="")
    compare_parser.add_argument("--fail-on-regression", action="store_true")
    compare_parser.set_defaults(handler=compare)
    gate_parser = commands.add_parser("gate", help="Apply deterministic/safety regression policy to two metric artifacts")
    gate_parser.add_argument("--baseline", required=True)
    gate_parser.add_argument("--candidate", required=True)
    gate_parser.add_argument("--output", default="")
    gate_parser.set_defaults(handler=gate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
