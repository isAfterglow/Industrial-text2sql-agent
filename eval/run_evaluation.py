#!/usr/bin/env python3
"""Run the resin Text2SQL evaluation suite with per-case isolation.

Each case runs in its own subprocess. A timeout or crash is recorded and does
not stop the remaining cases. Gold SQL is executed against the same database;
grading is based on result equivalence rather than SQL string equality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import platform
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = PROJECT_ROOT / "eval" / "benchmark_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "eval" / "runs"
SUCCESS_STATUSES = {"first_pass_success", "repaired_success"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def load_suite(path: Path) -> dict[str, Any]:
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location("evaluation_benchmark", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"无法加载评测基准：{path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        payload = getattr(module, "SUITE", None)
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("评测基准必须导出对象。")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("评测集必须包含非空cases数组。")
    identifiers = [str(case.get("id", "")) for case in cases]
    if any(not value for value in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("每个评测用例必须有唯一且非空的id。")
    return payload


def git_value(*args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip()


def runtime_metadata() -> dict[str, Any]:
    from app.config import get_settings
    from app.schema import active_profile_name, get_schema_catalog, _load_profile

    settings = get_settings()
    schema_payload = json.dumps(
        get_schema_catalog(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    profile = _load_profile(active_profile_name())
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "conda_environment": os.getenv("CONDA_DEFAULT_ENV", ""),
        "llm_model": settings.LLM_MODEL,
        "llm_base_url": settings.LLM_BASE_URL,
        "llm_temperature": 0,
        "domain_profile": active_profile_name(),
        "database_name": profile.get("database_name", settings.RESIN_DB_NAME),
        "allowed_tables": list(profile.get("policy", {}).get("allowed_tables", get_schema_catalog()["tables"])),
        "schema_sha256": hashlib.sha256(schema_payload.encode("utf-8")).hexdigest(),
        "sql_max_rows": settings.SQL_MAX_ROWS,
        "sql_timeout_seconds": settings.SQL_TIMEOUT_SECONDS,
        "sql_max_repair_attempts": settings.SQL_MAX_REPAIR_ATTEMPTS,
    }


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return round(value, 10)
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    return value


def normalized_rows(rows: list[list[Any]], ordered: bool) -> list[list[Any]]:
    normalized = [normalize_value(list(row)) for row in rows]
    if ordered:
        return normalized
    return sorted(
        normalized,
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True),
    )


def result_hash(columns: list[str], rows: list[list[Any]], ordered: bool) -> str:
    payload = {
        "columns": columns,
        "rows": normalized_rows(rows, ordered),
        "ordered": ordered,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def render_gold_sql(template: str, previous_sample_ids: list[str]) -> str:
    if "{{previous_ids}}" not in template:
        return template
    if not previous_sample_ids:
        replacement = "'__no_previous_samples__'"
    else:
        replacement = ", ".join(
            "'" + value.replace("'", "''") + "'" for value in previous_sample_ids
        )
    return template.replace("{{previous_ids}}", replacement)


def extract_sample_ids(columns: list[str], rows: list[list[Any]]) -> list[str]:
    if "sample_id" not in columns:
        return []
    index = columns.index("sample_id")
    values: list[str] = []
    for row in rows:
        if index < len(row) and isinstance(row[index], str):
            values.append(row[index])
    return list(dict.fromkeys(values))


def grade_turn(
    turn: dict[str, Any],
    result: dict[str, Any],
    gold: dict[str, Any] | None,
) -> dict[str, Any]:
    expected_status = str(turn.get("expected_status", "success"))
    actual_status = str(result.get("final_status", ""))
    expected_intent = str(turn.get("expected_intent", ""))
    actual_intent = str(result.get("query_intent", ""))
    if expected_status == "success":
        status_correct = actual_status in SUCCESS_STATUSES
    else:
        status_correct = actual_status == expected_status

    expected_columns = [str(value) for value in turn.get("expected_columns", [])]
    actual_columns = [str(value) for value in result.get("columns", [])]
    target_columns_present = all(column in actual_columns for column in expected_columns)
    extra_columns = [column for column in actual_columns if column not in expected_columns]
    missing_columns = [column for column in expected_columns if column not in actual_columns]
    columns_correct = (
        len(actual_columns) == len(expected_columns)
        and set(actual_columns) == set(expected_columns)
        if expected_columns
        else True
    )
    column_order_correct = actual_columns == expected_columns if expected_columns else True
    ordered = bool(turn.get("ordered", False))

    result_correct: bool | None = None
    gold_hash = ""
    actual_hash = ""
    if gold is not None and not gold.get("error"):
        gold_columns = [str(value) for value in gold.get("columns", [])]
        gold_rows = list(gold.get("rows", []))
        gold_hash = result_hash(gold_columns, gold_rows, ordered)
        if expected_columns and all(column in actual_columns for column in expected_columns):
            indexes = [actual_columns.index(column) for column in expected_columns]
            projected_rows = [
                [row[index] for index in indexes]
                for row in result.get("rows", [])
                if all(index < len(row) for index in indexes)
            ]
            actual_hash = result_hash(expected_columns, projected_rows, ordered)
            expected_hash = result_hash(expected_columns, gold_rows, ordered)
            result_correct = actual_hash == expected_hash
        elif (
            expected_columns
            and len(actual_columns) == len(expected_columns)
            and len(gold_columns) == len(expected_columns)
        ):
            # Column aliases may differ while values and projection width are correct.
            actual_hash = result_hash(expected_columns, list(result.get("rows", [])), ordered)
            expected_hash = result_hash(expected_columns, gold_rows, ordered)
            result_correct = actual_hash == expected_hash
        else:
            actual_hash = result_hash(actual_columns, list(result.get("rows", [])), ordered)
            result_correct = actual_hash == gold_hash

    forbidden_nodes = set(str(value) for value in turn.get("forbidden_nodes", []))
    required_nodes = set(str(value) for value in turn.get("required_nodes", []))
    node_path = list(result.get("node_path", [])) or [
        str(event.get("node", "")) for event in result.get("trace_events", [])
    ]
    forbidden_nodes_absent = not bool(forbidden_nodes & set(node_path))
    required_nodes_present = required_nodes.issubset(set(node_path))
    strict_pass = bool(status_correct and forbidden_nodes_absent and required_nodes_present)
    if expected_status == "success":
        strict_pass = bool(strict_pass and columns_correct and result_correct is True)
    acceptable_pass = bool(status_correct and forbidden_nodes_absent and required_nodes_present)
    if expected_status == "success":
        acceptable_pass = bool(acceptable_pass and result_correct is True)

    return {
        "expected_status": expected_status,
        "actual_status": actual_status,
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "intent_correct": actual_intent == expected_intent if expected_intent else None,
        "status_correct": status_correct,
        "columns_correct": columns_correct,
        "column_order_correct": column_order_correct,
        "target_columns_present": target_columns_present,
        "extra_columns": extra_columns,
        "missing_columns": missing_columns,
        "result_correct": result_correct,
        "forbidden_nodes_absent": forbidden_nodes_absent,
        "required_nodes_present": required_nodes_present,
        "strict_pass": strict_pass,
        "acceptable_pass": acceptable_pass,
        "gold_result_hash": gold_hash,
        "actual_result_hash": actual_hash,
    }


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    node_events = list(result.get("trace_events", []))
    return {
        "final_status": result.get("final_status", ""),
        "final_answer": result.get("final_answer", ""),
        "normalized_question": result.get("normalized_question", ""),
        "resolved_question": result.get("resolved_question", ""),
        "turn_type": result.get("turn_type", ""),
        "domain_profile": result.get("domain_profile", ""),
        "query_intent": result.get("query_intent", ""),
        "intent_confidence": float(result.get("intent_confidence", 0.0)),
        "intent_evidence": result.get("intent_evidence", []),
        "memory_used": bool(result.get("memory_used", False)),
        "query_delta": result.get("query_delta", {}),
        "query_delta_source": result.get("query_delta_source", ""),
        "query_delta_llm_called": bool(result.get("query_delta_llm_called", False)),
        "context_resolution_valid": result.get("context_resolution_valid", True),
        "clarification_required": bool(result.get("clarification_required", False)),
        "query_plan_mode": result.get("query_plan_mode", ""),
        "query_plan_reason": result.get("query_plan_reason", ""),
        "advanced_plan": result.get("advanced_plan", {}),
        "advanced_plan_error": result.get("advanced_plan_error", ""),
        "query_expectation": result.get("query_expectation", {}),
        "advanced_plan_raw": result.get("advanced_plan_raw", ""),
        "query_spec": result.get("query_spec", {}),
        "full_sql": result.get("full_sql", ""),
        "pruned_sql": result.get("pruned_sql", ""),
        "selected_candidate": result.get("selected_candidate", ""),
        "candidate_selection_reason": result.get("candidate_selection_reason", ""),
        "candidate_full_valid": result.get("candidate_full_valid", False),
        "candidate_full_score": result.get("candidate_full_score", 0.0),
        "candidate_full_error": result.get("candidate_full_error", ""),
        "candidate_pruned_valid": result.get("candidate_pruned_valid", False),
        "candidate_pruned_score": result.get("candidate_pruned_score", 0.0),
        "candidate_pruned_error": result.get("candidate_pruned_error", ""),
        "initial_sql": result.get("initial_sql", ""),
        "validated_sql": result.get("validated_sql", ""),
        "validation_error": result.get("validation_error", ""),
        "validation_error_type": result.get("validation_error_type", ""),
        "review_called": bool(result.get("review_called", False)),
        "review_passed": bool(result.get("review_passed", False)),
        "review_reason": result.get("review_reason", ""),
        "execution_error": result.get("execution_error", ""),
        "retry_count": int(result.get("retry_count", 0)),
        "repair_source": result.get("repair_source", ""),
        "repair_action": result.get("repair_action", ""),
        "repair_model_role": result.get("repair_model_role", ""),
        "repair_plan_mode": result.get("repair_plan_mode", ""),
        "failure_events": list(result.get("failure_events", [])),
        "model_calls": list(result.get("model_calls", [])),
        "columns": list(result.get("columns", [])),
        "rows": list(result.get("rows", [])),
        "row_count": int(result.get("row_count", 0)),
        "truncated": bool(result.get("truncated", False)),
        "result_assertion": result.get("result_assertion", {}),
        "result_assertion_passed": bool(result.get("result_assertion_passed", True)),
        "semantic_memory_matches": result.get("semantic_memory_matches", []),
        "episodic_memory_matches": result.get("episodic_memory_matches", []),
        "procedural_memory_matches": result.get("procedural_memory_matches", []),
        "few_shot_retrieval_diagnostics": result.get("few_shot_retrieval_diagnostics", {}),
        "long_term_memory_retrieval_summary": result.get("long_term_memory_retrieval_summary", {}),
        "long_term_memory_write_summary": result.get("long_term_memory_write_summary", {}),
        "node_path": [event.get("node", "") for event in node_events],
        "node_events": node_events,
    }


def run_worker(case: dict[str, Any]) -> dict[str, Any]:
    from app.db import execute_readonly_query
    from app.graph import graph
    from app.memory import new_short_term_memory
    from app.schema import set_active_profile
    from app.trace import new_trace_id, utc_now_iso

    set_active_profile(str(case.get("profile", "resin")))

    setup_memories = list(case.get("setup_memories", []))
    if setup_memories:
        from app.long_term_memory.service import get_long_term_memory_service

        service = get_long_term_memory_service()
        for memory in setup_memories:
            memory_type = str(memory.get("type", ""))
            if memory_type == "semantic":
                service.remember_semantic(str(memory["content"]), source="benchmark_fixture")
            elif memory_type == "episodic":
                service.remember_case(
                    question=str(memory["question"]),
                    resolved_question=str(memory.get("resolved_question", memory["question"])),
                    query_spec=dict(memory["query_spec"]),
                    sql=str(memory["sql"]),
                    source="benchmark_fixture",
                    case_context={"independent_case": True},
                )

    conversation_memory = new_short_term_memory()
    previous_gold_sample_ids: list[str] = []
    turns_output: list[dict[str, Any]] = []
    case_started = time.perf_counter()

    for turn_index, turn in enumerate(case.get("turns", []), start=1):
        question = str(turn["question"])
        started = time.perf_counter()
        from app.config import get_settings
        result = graph.invoke(
            {
                "question": question,
                "session_id": conversation_memory["session_id"],
                "conversation_memory": conversation_memory,
                "trace_id": new_trace_id(),
                "trace_started_at": utc_now_iso(),
                "trace_events": [],
            },
            {"recursion_limit": getattr(get_settings(), "AGENT_MAX_GRAPH_STEPS", 32)},
        )
        graph_elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        updated_memory = result.get("conversation_memory")
        if isinstance(updated_memory, dict) and updated_memory:
            conversation_memory = updated_memory

        gold: dict[str, Any] | None = None
        gold_template = str(turn.get("gold_sql", "")).strip()
        if gold_template:
            gold_sql = render_gold_sql(gold_template, previous_gold_sample_ids)
            gold_started = time.perf_counter()
            try:
                gold = {
                    "sql": gold_sql,
                    **execute_readonly_query(gold_sql, max_rows=200),
                    "error": "",
                }
            except Exception as exc:
                gold = {
                    "sql": gold_sql,
                    "columns": [],
                    "rows": [],
                    "row_count": 0,
                    "truncated": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            gold["elapsed_ms"] = round((time.perf_counter() - gold_started) * 1000, 3)
            if not gold.get("error"):
                previous_gold_sample_ids = extract_sample_ids(
                    list(gold.get("columns", [])), list(gold.get("rows", []))
                )

        compact = compact_result(result)
        grade_definition = {**turn, "expected_intent": case.get("expected_intent", "")}
        grade = grade_turn(grade_definition, result, gold)
        turns_output.append(
            {
                "turn": turn_index,
                "question": question,
                "graph_elapsed_ms": graph_elapsed_ms,
                "gold": gold,
                "result": compact,
                "grade": grade,
            }
        )

    return {
        "id": case["id"],
        "category": case.get("category", "uncategorized"),
        "tags": case.get("tags", []),
        "status": "completed",
        "strict_pass": all(turn["grade"]["strict_pass"] for turn in turns_output),
        "result_pass": all(
            turn["grade"]["status_correct"]
            and turn["grade"]["forbidden_nodes_absent"]
            and turn["grade"]["result_correct"] is not False
            for turn in turns_output
        ),
        "acceptable_pass": all(
            turn["grade"]["acceptable_pass"] for turn in turns_output
        ),
        "elapsed_ms": round((time.perf_counter() - case_started) * 1000, 3),
        "turns": turns_output,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def worker_main(args: argparse.Namespace) -> int:
    suite = load_suite(Path(args.suite))
    case = next((item for item in suite["cases"] if item["id"] == args.case_id), None)
    if case is None:
        raise ValueError(f"找不到用例：{args.case_id}")
    try:
        payload = run_worker(case)
    except BaseException as exc:
        error_type = type(exc).__name__
        category = "non_convergent_plan" if error_type == "GraphRecursionError" else "worker_crash"
        payload = {
            "id": case["id"],
            "category": case.get("category", "uncategorized"),
            "tags": case.get("tags", []),
            "status": "worker_error",
            "strict_pass": False,
            "error": f"{error_type}: {exc}",
            "failure_category": category,
            "turns": [],
        }
    write_json(Path(args.result_file), payload)
    return 0 if payload["status"] == "completed" else 1


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    elapsed = [float(item.get("elapsed_ms", 0.0)) for item in completed]
    turns = [turn for item in completed for turn in item.get("turns", [])]
    first_pass = sum(
        turn.get("result", {}).get("final_status") == "first_pass_success" for turn in turns
    )
    repaired = sum(
        int(turn.get("result", {}).get("retry_count", 0)) > 0 for turn in turns
    )
    repaired_success = sum(
        int(turn.get("result", {}).get("retry_count", 0)) > 0
        and turn.get("result", {}).get("final_status") == "repaired_success"
        for turn in turns
    )
    graph_elapsed = [float(turn.get("graph_elapsed_ms", 0.0)) for turn in turns]
    llm_calls = 0
    route_counts: dict[str, int] = {}
    model_role_counts: dict[str, int] = {}
    model_error_counts: dict[str, int] = {}
    failure_category_counts: dict[str, int] = {}
    plan_contract_passes = 0
    plan_contract_failures = 0
    assertion_checked = 0
    assertion_passed = 0
    intent_counts: dict[str, int] = {}
    intent_labeled_turns = 0
    intent_correct_turns = 0
    for turn in turns:
        result = turn.get("result", {})
        route = str(result.get("query_plan_mode", "")) or "no_plan"
        route_counts[route] = route_counts.get(route, 0) + 1
        intent = str(result.get("query_intent", "")) or "unclassified"
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        intent_correct = turn.get("grade", {}).get("intent_correct")
        if intent_correct is not None:
            intent_labeled_turns += 1
            intent_correct_turns += int(bool(intent_correct))
        calls = list(result.get("model_calls", []))
        # New traces contain exact model-call telemetry. Keep the old estimate
        # for historical run artifacts that predate the router.
        if calls:
            llm_calls += len(calls)
        else:
            path = list(result.get("node_path", []))
            llm_calls += int(bool(result.get("query_delta_llm_called", False)))
            llm_calls += path.count("generate_full_sql")
            llm_calls += path.count("generate_pruned_sql")
            llm_calls += path.count("repair_sql")
            llm_calls += int(bool(result.get("review_called", False)))
        for call in calls:
            role = str(call.get("role", "unknown"))
            model_role_counts[role] = model_role_counts.get(role, 0) + 1
            if call.get("status") != "success":
                model_error_counts[role] = model_error_counts.get(role, 0) + 1
        for event in result.get("failure_events", []):
            category = str(event.get("category", "unknown"))
            failure_category_counts[category] = failure_category_counts.get(category, 0) + 1
            if category == "plan_contract":
                plan_contract_failures += 1
        if result.get("advanced_plan"):
            plan_contract_passes += 1
        assertion = result.get("result_assertion", {})
        if assertion.get("checked"):
            assertion_checked += 1
            assertion_passed += int(bool(assertion.get("passed")))
    acceptable_projection_deviation = sum(
        bool(item.get("acceptable_pass", item.get("result_pass", False)))
        and not bool(item.get("strict_pass", False))
        for item in results
    )
    acceptable_extra_column_cases = sum(
        bool(item.get("acceptable_pass", item.get("result_pass", False)))
        and any(
            bool(turn.get("grade", {}).get("extra_columns"))
            and not bool(turn.get("grade", {}).get("missing_columns"))
            for turn in item.get("turns", [])
        )
        for item in results
    )
    acceptable_alias_deviation_cases = sum(
        bool(item.get("acceptable_pass", item.get("result_pass", False)))
        and any(
            bool(turn.get("grade", {}).get("missing_columns"))
            and turn.get("grade", {}).get("result_correct") is True
            for turn in item.get("turns", [])
        )
        for item in results
    )
    by_category: dict[str, dict[str, Any]] = {}
    for item in results:
        category = str(item.get("category", "uncategorized"))
        bucket = by_category.setdefault(
            category,
            {"cases": 0, "acceptable_pass": 0, "strict_pass": 0, "timeouts": 0},
        )
        bucket["cases"] += 1
        bucket["acceptable_pass"] += int(
            bool(item.get("acceptable_pass", item.get("result_pass", False)))
        )
        bucket["strict_pass"] += int(bool(item.get("strict_pass", False)))
        bucket["timeouts"] += int(item.get("status") == "timeout")

    return {
        "cases": len(results),
        "completed": len(completed),
        "strict_pass": sum(bool(item.get("strict_pass", False)) for item in results),
        "result_pass": sum(bool(item.get("result_pass", False)) for item in results),
        "acceptable_pass": sum(
            bool(item.get("acceptable_pass", item.get("result_pass", False)))
            for item in results
        ),
        "acceptable_projection_deviation": acceptable_projection_deviation,
        "acceptable_extra_column_cases": acceptable_extra_column_cases,
        "acceptable_alias_deviation_cases": acceptable_alias_deviation_cases,
        "timeouts": sum(item.get("status") == "timeout" for item in results),
        "worker_errors": sum(item.get("status") == "worker_error" for item in results),
        "turns": len(turns),
        "first_pass_turns": first_pass,
        "repaired_turns": repaired,
        "repaired_success_turns": repaired_success,
        "llm_calls": llm_calls,
        "mean_llm_calls_per_turn": round(llm_calls / len(turns), 3) if turns else 0.0,
        "mean_turn_ms": round(statistics.fmean(graph_elapsed), 3) if graph_elapsed else 0.0,
        "p50_turn_ms": round(percentile(graph_elapsed, 0.50), 3),
        "p95_turn_ms": round(percentile(graph_elapsed, 0.95), 3),
        "mean_case_ms": round(statistics.fmean(elapsed), 3) if elapsed else 0.0,
        "p50_case_ms": round(percentile(elapsed, 0.50), 3),
        "p95_case_ms": round(percentile(elapsed, 0.95), 3),
        "by_category": by_category,
        "route_counts": route_counts,
        "model_role_counts": model_role_counts,
        "model_error_counts": model_error_counts,
        "failure_category_counts": failure_category_counts,
        "plan_attempts": plan_contract_passes + plan_contract_failures,
        "plan_contract_passes": plan_contract_passes,
        "plan_contract_failures": plan_contract_failures,
        "result_assertion_checked": assertion_checked,
        "result_assertion_passed": assertion_passed,
        "intent_counts": intent_counts,
        "intent_labeled_turns": intent_labeled_turns,
        "intent_correct_turns": intent_correct_turns,
        "intent_accuracy": round(intent_correct_turns / intent_labeled_turns, 4) if intent_labeled_turns else None,
    }


def markdown_report(metadata: dict[str, Any], summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        "# Resin Text2SQL Evaluation",
        "",
        f"- Run ID: `{metadata['run_id']}`",
        f"- Git: `{metadata['git_commit']}` on `{metadata['git_branch']}`",
        f"- Worktree dirty: `{metadata['git_dirty']}`",
        f"- Suite: `{metadata['suite_name']}` (`{metadata['suite_version']}`)",
        f"- Model: `{metadata.get('llm_model', 'unknown')}` at `{metadata.get('llm_base_url', 'unknown')}`",
        f"- Database/Schema: `{metadata.get('database_name', 'unknown')}` / `{metadata.get('schema_sha256', 'unknown')}`",
        f"- Python/Conda: `{metadata.get('python', 'unknown')}` / `{metadata.get('conda_environment', '')}`",
        f"- Case timeout: `{metadata['case_timeout_seconds']}s`",
        f"- Learned memory: {metadata.get('memory_mode', 'isolated')} DB, auto-save disabled",
        "",
        "## Summary",
        "",
        f"- Acceptable result pass: {summary['acceptable_pass']} / {summary['cases']}",
        f"- Strict case pass: {summary['strict_pass']} / {summary['cases']}",
        f"- Acceptable but non-strict projection: {summary['acceptable_projection_deviation']}",
        f"- Acceptable cases with extra columns: {summary['acceptable_extra_column_cases']}",
        f"- Acceptable cases with alias deviations: {summary['acceptable_alias_deviation_cases']}",
        f"- Completed: {summary['completed']} / {summary['cases']}",
        f"- Timeout: {summary['timeouts']}",
        f"- Worker error: {summary['worker_errors']}",
        f"- First-pass turns: {summary['first_pass_turns']} / {summary['turns']}",
        f"- Repair recovery: {summary['repaired_success_turns']} / {summary['repaired_turns']}",
        f"- Estimated LLM calls: {summary['llm_calls']} ({summary['mean_llm_calls_per_turn']:.3f} per turn)",
        f"- Routed model calls: {sum(summary['model_role_counts'].values())}",
        f"- Advanced plan contract: {summary['plan_contract_passes']} / {summary['plan_attempts']}",
        f"- Result assertions: {summary['result_assertion_passed']} / {summary['result_assertion_checked']}",
        f"- Intent accuracy: {summary['intent_correct_turns']} / {summary['intent_labeled_turns']}" + (f" ({summary['intent_accuracy']:.1%})" if summary['intent_accuracy'] is not None else " (not labeled)"),
        f"- Mean/P50/P95 turn latency: {summary['mean_turn_ms']:.2f} / {summary['p50_turn_ms']:.2f} / {summary['p95_turn_ms']:.2f} ms",
        f"- Mean/P50/P95 case latency: {summary['mean_case_ms']:.2f} / {summary['p50_case_ms']:.2f} / {summary['p95_case_ms']:.2f} ms",
        "",
        "## Categories",
        "",
        "| Category | Cases | Acceptable pass | Strict pass | Timeout |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, values in sorted(summary["by_category"].items()):
        lines.append(
            f"| {category} | {values['cases']} | {values['acceptable_pass']} | {values['strict_pass']} | {values['timeouts']} |"
        )
    lines.extend(["", "## Intents", "", "| Intent | Turns |", "|---|---:|"])
    for intent, count in sorted(summary["intent_counts"].items()):
        lines.append(f"| {intent} | {count} |")
    lines.extend(["", "## Model Routing", "", "| Model role | Calls | Errors |", "|---|---:|---:|"])
    for role, count in sorted(summary["model_role_counts"].items()):
        lines.append(f"| {role} | {count} | {summary['model_error_counts'].get(role, 0)} |")
    lines.extend(["", "## Failure Taxonomy", "", "| Failure category | Events |", "|---|---:|"])
    for category, count in sorted(summary["failure_category_counts"].items()):
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| ID | Category | Status | Acceptable | Strict | Time(ms) | Final status | Route | Retry |",
            "|---|---|---|---:|---:|---:|---|---|---:|",
        ]
    )
    for item in results:
        turns = item.get("turns", [])
        final_status = ", ".join(str(turn.get("result", {}).get("final_status", "")) for turn in turns)
        routes = ", ".join(str(turn.get("result", {}).get("query_plan_mode", "")) for turn in turns)
        retries = sum(int(turn.get("result", {}).get("retry_count", 0)) for turn in turns)
        lines.append(
            f"| {item.get('id')} | {item.get('category')} | {item.get('status')} | "
            f"{'yes' if item.get('acceptable_pass', item.get('result_pass')) else 'no'} | "
            f"{'yes' if item.get('strict_pass') else 'no'} | {float(item.get('elapsed_ms', 0.0)):.2f} | "
            f"{final_status or '-'} | {routes or '-'} | {retries} |"
        )
    lines.extend(["", "Detailed SQL, rows, hashes, errors and node timings are in `results.json`.", ""])
    return "\n".join(lines)


def regrade_existing(args: argparse.Namespace) -> int:
    results_path = Path(args.regrade_results).resolve()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["metadata"].update(runtime_metadata())
    suite = load_suite(Path(args.suite).resolve())
    cases = {str(case["id"]): case for case in suite["cases"]}
    results = list(payload.get("results", []))

    for item in results:
        case = cases.get(str(item.get("id")))
        if case is None:
            continue
        turns = list(item.get("turns", []))
        definitions = list(case.get("turns", []))
        for index, turn_output in enumerate(turns):
            if index >= len(definitions):
                continue
            turn_output["grade"] = grade_turn(
                definitions[index],
                turn_output.get("result", {}),
                turn_output.get("gold"),
            )
        item["strict_pass"] = bool(turns) and all(
            turn["grade"]["strict_pass"] for turn in turns
        )
        item["result_pass"] = bool(turns) and all(
            turn["grade"]["status_correct"]
            and turn["grade"]["forbidden_nodes_absent"]
            and turn["grade"]["result_correct"] is not False
            for turn in turns
        )
        item["acceptable_pass"] = bool(turns) and all(
            turn["grade"]["acceptable_pass"] for turn in turns
        )

    summary = summarize(results)
    payload["summary"] = summary
    payload["results"] = results
    write_json(results_path, payload)
    report_path = results_path.parent / "report.md"
    report_path.write_text(
        markdown_report(payload["metadata"], summary, results), encoding="utf-8"
    )
    print(
        f"Regraded: acceptable {summary['acceptable_pass']}/{summary['cases']}; "
        f"strict {summary['strict_pass']}/{summary['cases']}"
    )
    print(f"Report: {report_path}")
    return 0


def terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def parent_main(args: argparse.Namespace) -> int:
    run_started = time.perf_counter()
    suite_path = Path(args.suite).resolve()
    suite = load_suite(suite_path)
    selected = list(suite["cases"])
    if args.only:
        requested = {value.strip() for value in args.only.split(",") if value.strip()}
        selected = [case for case in selected if case["id"] in requested]
        missing = requested - {case["id"] for case in selected}
        if missing:
            raise ValueError("找不到用例：" + ", ".join(sorted(missing)))

    profiles = {str(case.get("profile", "resin")) for case in selected}
    if len(profiles) == 1:
        from app.schema import set_active_profile
        set_active_profile(next(iter(profiles)))

    run_id = args.run_id or utc_stamp()
    run_dir = Path(args.output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "case_logs").mkdir()
    (run_dir / "case_results").mkdir()
    (run_dir / "traces").mkdir()
    memory_dir = run_dir / "isolated_memories"
    if args.memory_mode == "isolated":
        memory_dir.mkdir()
        memory_db_path = str(memory_dir)
    else:
        memory_db_path = os.getenv(
            "LTM_DB_PATH", str(PROJECT_ROOT / "data" / "long_term_memory.sqlite3")
        )

    metadata = {
        "run_id": run_id,
        "started_at": utc_stamp(),
        "suite_name": suite.get("name", suite_path.name),
        "suite_version": suite.get("version", "unknown"),
        "suite_path": str(suite_path),
        "suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "case_timeout_seconds": args.case_timeout,
        "case_count": len(selected),
        "ltm_db_path": memory_db_path,
        "memory_mode": args.memory_mode,
        "ltm_auto_save": False,
        "embedding_mode": "lexical_only",
        **runtime_metadata(),
    }
    write_json(run_dir / "metadata.json", metadata)

    child_env = os.environ.copy()
    child_env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "LTM_ENABLED": "1",
            "LTM_AUTO_SAVE": "0",
            # Evaluation measures query quality without requiring an external
            # reviewer to resume each high-risk case.
            "APPROVAL_MODE": "off",
            "LTM_EMBEDDING_MODEL": "",
            "LTM_ALLOW_MODEL_DOWNLOAD": "0",
            "TEXT2SQL_TRACE_ENABLED": "1",
            "TEXT2SQL_TRACE_CONSOLE": "0",
            "TEXT2SQL_TRACE_LOG_DIR": str(run_dir / "traces"),
        }
    )

    results: list[dict[str, Any]] = []
    progress_path = run_dir / "progress.jsonl"
    print(f"Run: {run_id}")
    print(f"Cases: {len(selected)}; timeout: {args.case_timeout}s")
    print(f"Artifacts: {run_dir}")

    for index, case in enumerate(selected, start=1):
        case_id = str(case["id"])
        result_path = run_dir / "case_results" / f"{case_id}.json"
        log_path = run_dir / "case_logs" / f"{case_id}.log"
        print(f"[{index}/{len(selected)}] START {case_id} ({case.get('category', '')})", flush=True)
        started = time.perf_counter()
        case_env = child_env.copy()
        if args.memory_mode == "isolated":
            case_env["LTM_DB_PATH"] = str(memory_dir / f"{case_id}.sqlite3")
        else:
            case_env["LTM_DB_PATH"] = memory_db_path
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--suite",
                    str(suite_path),
                    "--case-id",
                    case_id,
                    "--result-file",
                    str(result_path),
                ],
                cwd=PROJECT_ROOT,
                env=case_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                process.wait(timeout=args.case_timeout)
            except subprocess.TimeoutExpired:
                terminate_process(process)

        wall_ms = round((time.perf_counter() - started) * 1000, 3)
        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as handle:
                item = json.load(handle)
            item["worker_exit_code"] = process.returncode
            item.setdefault("elapsed_ms", wall_ms)
        else:
            item = {
                "id": case_id,
                "category": case.get("category", "uncategorized"),
                "tags": case.get("tags", []),
                "status": "timeout" if wall_ms >= args.case_timeout * 1000 else "worker_error",
                "strict_pass": False,
                "elapsed_ms": wall_ms,
                "worker_exit_code": process.returncode,
                "error": "case process exceeded timeout" if wall_ms >= args.case_timeout * 1000 else "worker did not produce a result file",
                "turns": [],
            }
        results.append(item)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"[{index}/{len(selected)}] END   {case_id}: {item['status']}, "
            f"strict={'PASS' if item.get('strict_pass') else 'FAIL'}, {wall_ms:.0f}ms",
            flush=True,
        )

    summary = summarize(results)
    metadata["finished_at"] = utc_stamp()
    metadata["total_wall_elapsed_ms"] = round(
        (time.perf_counter() - run_started) * 1000, 3
    )
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "results.json", {"metadata": metadata, "summary": summary, "results": results})
    (run_dir / "report.md").write_text(
        markdown_report(metadata, summary, results), encoding="utf-8"
    )
    print(
        f"Done: strict {summary['strict_pass']}/{summary['cases']}, "
        f"timeouts={summary['timeouts']}, errors={summary['worker_errors']}"
    )
    print(f"Report: {run_dir / 'report.md'}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the resin Text2SQL evaluation suite.")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--case-timeout", type=int, default=120)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--only", default="", help="Comma-separated case IDs")
    parser.add_argument(
        "--memory-mode", choices=("isolated", "production"), default="isolated",
        help="Use isolated memory for a clean benchmark, or production memory read-only for an effectiveness comparison.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--result-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--regrade-results", default="", help="Regrade an existing results.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        return worker_main(args)
    if args.regrade_results:
        return regrade_existing(args)
    return parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
