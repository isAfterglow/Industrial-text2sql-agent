import json
import os
import threading
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from app.schema import set_active_profile


_LOG_LOCK = threading.Lock()


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    return raw.strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def trace_enabled() -> bool:
    return _env_bool(
        "TEXT2SQL_TRACE_ENABLED",
        True,
    )


def console_trace_enabled() -> bool:
    return _env_bool(
        "TEXT2SQL_TRACE_CONSOLE",
        True,
    )


def verbose_trace_enabled() -> bool:
    return _env_bool(
        "TEXT2SQL_TRACE_VERBOSE",
        False,
    )


def trace_log_dir() -> Path:
    return Path(
        os.getenv(
            "TEXT2SQL_TRACE_LOG_DIR",
            "logs",
        )
    )


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(timespec="milliseconds")


def new_trace_id() -> str:
    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


def safe_json_value(
    value: Any,
    *,
    max_string_length: int = 12000,
    max_list_items: int = 50,
) -> Any:
    """Convert common DB/LLM values into JSON-safe values."""

    if value is None or isinstance(
        value,
        (bool, int, float),
    ):
        return value

    if isinstance(value, str):
        if len(value) <= max_string_length:
            return value
        return (
            value[:max_string_length]
            + "...<truncated>"
        )

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): safe_json_value(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        result = [
            safe_json_value(
                item,
                max_string_length=max_string_length,
                max_list_items=max_list_items,
            )
            for item in items[:max_list_items]
        ]
        if len(items) > max_list_items:
            result.append(
                f"...<{len(items) - max_list_items} more items>"
            )
        return result

    return str(value)


def _row_preview(
    rows: list[Any],
    limit: int = 5,
) -> list[Any]:
    return safe_json_value(
        rows[:limit],
        max_list_items=limit,
    )


def summarize_node_input(
    node_name: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    common = {
        "question": state.get("question", ""),
        "normalized_question": state.get(
            "normalized_question",
            "",
        ),
        "retry_count": state.get(
            "retry_count",
            0,
        ),
        "resolved_question": state.get("resolved_question", ""),
        "turn_type": state.get("turn_type", ""),
        "memory_used": state.get("memory_used", False),
    }

    if node_name == "load_schema":
        return {
            "question": common["question"],
            "session_id": state.get("session_id", ""),
        }

    if node_name == "extract_query_delta":
        return {
            **common,
            "conversation_memory": state.get("conversation_memory", {}),
        }

    if node_name == "resolve_conversation_context":
        return {
            **common,
            "conversation_memory": state.get("conversation_memory", {}),
            "query_delta": state.get("query_delta", {}),
        }

    if node_name in {
        "build_query_plan",
        "generate_simple_sql",
        "generate_full_sql",
        "build_robust_schema",
        "generate_pruned_sql",
        "select_sql_candidate",
    }:
        return {
            **common,
            "full_sql": state.get("full_sql", ""),
            "pruned_sql": state.get("pruned_sql", ""),
            "robust_schema_tables": state.get(
                "robust_schema_tables", []
            ),
        }

    if node_name == "validate_sql":
        return {
            **common,
            "raw_sql": state.get(
                "raw_sql",
                "",
            ),
        }

    if node_name == "review_sql":
        return {
            **common,
            "validated_sql": state.get(
                "validated_sql",
                "",
            ),
        }

    if node_name == "repair_sql":
        return {
            **common,
            "raw_sql": state.get(
                "raw_sql",
                "",
            ),
            "validated_sql": state.get(
                "validated_sql",
                "",
            ),
            "validation_error": state.get(
                "validation_error",
                "",
            ),
            "review_reason": state.get(
                "review_reason",
                "",
            ),
            "execution_error": state.get(
                "execution_error",
                "",
            ),
        }

    if node_name == "execute_sql":
        return {
            **common,
            "validated_sql": state.get(
                "validated_sql",
                "",
            ),
        }

    if node_name == "update_session_memory":
        return {
            **common,
            "query_spec": state.get("query_spec", {}),
            "validated_sql": state.get("validated_sql", ""),
            "row_count": state.get("row_count", 0),
        }

    if node_name in {
        "format_result",
        "format_error",
    }:
        return {
            **common,
            "validated_sql": state.get(
                "validated_sql",
                "",
            ),
            "validation_error": state.get(
                "validation_error",
                "",
            ),
            "execution_error": state.get(
                "execution_error",
                "",
            ),
            "row_count": state.get(
                "row_count",
                0,
            ),
        }

    return common


def summarize_node_output(
    node_name: str,
    output: dict[str, Any],
) -> dict[str, Any]:
    if node_name == "load_schema":
        return {
            "normalized_question": output.get(
                "normalized_question",
                "",
            ),
            "schema_context": output.get(
                "schema_context",
                "",
            ),
        }

    if node_name == "extract_query_delta":
        return {
            "query_delta_source": output.get("query_delta_source", ""),
            "query_delta_llm_called": output.get("query_delta_llm_called", False),
            "query_delta": output.get("query_delta", {}),
        }

    if node_name == "resolve_conversation_context":
        return {
            "turn_type": output.get("turn_type", ""),
            "memory_used": output.get("memory_used", False),
            "resolved_question": output.get("resolved_question", ""),
            "context_resolution": output.get("context_resolution", {}),
            "context_resolution_valid": output.get("context_resolution_valid", True),
            "current_turn_coverage": output.get("current_turn_coverage", {}),
            "inherited_fields": output.get("inherited_fields", []),
            "overridden_fields": output.get("overridden_fields", []),
            "resolved_query_spec": output.get("resolved_query_spec", {}),
        }

    if node_name == "build_query_plan":
        return {
            "query_plan_mode": output.get("query_plan_mode", ""),
            "query_plan_reason": output.get("query_plan_reason", ""),
            "query_spec": output.get("query_spec", {}),
            "deterministic_sql": output.get("deterministic_sql", ""),
        }

    if node_name == "generate_simple_sql":
        return {
            "selected_candidate": output.get("selected_candidate", ""),
            "selected_sql": output.get("raw_sql", ""),
            "candidate_selection_reason": output.get(
                "candidate_selection_reason", ""
            ),
        }

    if node_name == "generate_full_sql":
        return {
            "field_hint": output.get("field_hint", ""),
            "full_generator_raw_output": output.get(
                "full_generator_raw_output", ""
            ),
            "full_sql": output.get("full_sql", ""),
        }

    if node_name == "build_robust_schema":
        return {
            "forward_schema_tables": output.get(
                "forward_schema_tables", []
            ),
            "forward_schema_columns": output.get(
                "forward_schema_columns", []
            ),
            "backward_schema_tables": output.get(
                "backward_schema_tables", []
            ),
            "backward_schema_columns": output.get(
                "backward_schema_columns", []
            ),
            "accepted_backward_tables": output.get(
                "accepted_backward_tables", []
            ),
            "rejected_backward_tables": output.get(
                "rejected_backward_tables", []
            ),
            "robust_schema_tables": output.get(
                "robust_schema_tables", []
            ),
            "robust_schema_columns": output.get(
                "robust_schema_columns", []
            ),
            "robust_schema_context": output.get(
                "robust_schema_context", ""
            ),
        }

    if node_name == "generate_pruned_sql":
        return {
            "pruned_generator_raw_output": output.get(
                "pruned_generator_raw_output", ""
            ),
            "pruned_sql": output.get("pruned_sql", ""),
        }

    if node_name == "select_sql_candidate":
        return {
            "candidate_full_valid": output.get(
                "candidate_full_valid", False
            ),
            "candidate_full_score": output.get(
                "candidate_full_score", 0.0
            ),
            "candidate_full_error": output.get(
                "candidate_full_error", ""
            ),
            "candidate_pruned_valid": output.get(
                "candidate_pruned_valid", False
            ),
            "candidate_pruned_score": output.get(
                "candidate_pruned_score", 0.0
            ),
            "candidate_pruned_error": output.get(
                "candidate_pruned_error", ""
            ),
            "selected_candidate": output.get(
                "selected_candidate", ""
            ),
            "candidate_selection_reason": output.get(
                "candidate_selection_reason", ""
            ),
            "selected_sql": output.get("raw_sql", ""),
        }

    if node_name == "validate_sql":
        return {
            "valid": not bool(
                output.get(
                    "validation_error"
                )
            ),
            "validated_sql": output.get(
                "validated_sql",
                "",
            ),
            "validation_error": output.get(
                "validation_error",
                "",
            ),
            "validation_error_type": output.get(
                "validation_error_type",
                "",
            ),
            "validation_repairable": output.get(
                "validation_repairable",
                True,
            ),
        }

    if node_name == "review_sql":
        return {
            "review_called": output.get(
                "review_called",
                False,
            ),
            "review_passed": output.get(
                "review_passed",
                False,
            ),
            "review_reason": output.get(
                "review_reason",
                "",
            ),
            "review_note": output.get(
                "review_note",
                "",
            ),
            "review_input_summary": output.get(
                "review_input_summary",
                "",
            ),
        }

    if node_name == "repair_sql":
        return {
            "repair_source": output.get(
                "repair_source",
                "",
            ),
            "repair_reason": output.get(
                "last_repair_reason",
                "",
            ),
            "repair_action": output.get(
                "repair_action",
                "",
            ),
            "repair_bad_sql": output.get(
                "repair_bad_sql",
                "",
            ),
            "repair_raw_output": output.get(
                "repair_raw_output",
                "",
            ),
            "repaired_sql": output.get(
                "raw_sql",
                "",
            ),
            "retry_count": output.get(
                "retry_count",
                0,
            ),
        }

    if node_name == "execute_sql":
        return {
            "execution_error": output.get(
                "execution_error",
                "",
            ),
            "columns": output.get(
                "columns",
                [],
            ),
            "row_count": output.get(
                "row_count",
                0,
            ),
            "truncated": output.get(
                "truncated",
                False,
            ),
            "rows_preview": _row_preview(
                output.get("rows", [])
            ),
        }

    if node_name == "update_session_memory":
        return {
            "memory_update_summary": output.get("memory_update_summary", {}),
        }

    if node_name in {
        "format_result",
        "format_error",
    }:
        return {
            "final_status": output.get(
                "final_status",
                "",
            ),
            "final_answer_preview": safe_json_value(
                output.get(
                    "final_answer",
                    "",
                ),
                max_string_length=2000,
            ),
        }

    return safe_json_value(output)


def infer_event_status(
    node_name: str,
    output: dict[str, Any],
) -> str:
    if node_name == "validate_sql" and output.get(
        "validation_error"
    ):
        return "rejected"

    if node_name == "review_sql" and not output.get(
        "review_passed",
        False,
    ):
        return "rejected"

    if node_name == "execute_sql" and output.get(
        "execution_error"
    ):
        return "failed"

    if node_name == "format_error":
        return "failed"

    return "ok"


def append_jsonl(
    path: Path,
    record: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    payload = json.dumps(
        safe_json_value(record),
        ensure_ascii=False,
    )

    with _LOG_LOCK:
        with path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(payload + "\n")


def append_node_event(
    event: dict[str, Any],
) -> None:
    if not trace_enabled():
        return

    append_jsonl(
        trace_log_dir()
        / "node_events.jsonl",
        event,
    )


def _console_value(
    value: Any,
    max_length: int = 900,
) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            safe_json_value(value),
            ensure_ascii=False,
            indent=2,
        )

    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def print_node_event(
    event: dict[str, Any],
) -> None:
    if not (
        trace_enabled()
        and console_trace_enabled()
    ):
        return

    status = event.get("status", "ok")
    symbol = {
        "ok": "✓",
        "rejected": "!",
        "failed": "✗",
    }.get(status, "•")

    print(
        f"\n[TRACE] {symbol} "
        f"{event['node']} "
        f"{event['elapsed_ms']:.2f} ms"
    )

    output = event.get("output", {})

    # Normal mode prints the fields that are most useful for debugging.
    preferred_keys = {
        "load_schema": [
            "normalized_question",
        ],
        "extract_query_delta": [
            "query_delta_source",
            "query_delta_llm_called",
            "query_delta",
        ],
        "resolve_conversation_context": [
            "turn_type",
            "memory_used",
            "resolved_question",
            "context_resolution_valid",
            "current_turn_coverage",
            "inherited_fields",
            "overridden_fields",
        ],
        "build_query_plan": [
            "query_plan_mode",
            "query_plan_reason",
            "query_spec",
            "deterministic_sql",
        ],
        "generate_simple_sql": [
            "selected_candidate",
            "candidate_selection_reason",
            "selected_sql",
        ],
        "generate_full_sql": [
            "field_hint",
            "full_sql",
        ],
        "build_robust_schema": [
            "forward_schema_tables",
            "forward_schema_columns",
            "backward_schema_tables",
            "backward_schema_columns",
            "accepted_backward_tables",
            "rejected_backward_tables",
            "robust_schema_tables",
            "robust_schema_columns",
        ],
        "generate_pruned_sql": [
            "pruned_sql",
        ],
        "select_sql_candidate": [
            "candidate_full_valid",
            "candidate_full_score",
            "candidate_full_error",
            "candidate_pruned_valid",
            "candidate_pruned_score",
            "candidate_pruned_error",
            "selected_candidate",
            "candidate_selection_reason",
            "selected_sql",
        ],
        "validate_sql": [
            "valid",
            "validated_sql",
            "validation_error_type",
            "validation_error",
        ],
        "review_sql": [
            "review_called",
            "review_passed",
            "review_reason",
            "review_note",
        ],
        "repair_sql": [
            "repair_source",
            "repair_action",
            "repaired_sql",
        ],
        "execute_sql": [
            "execution_error",
            "columns",
            "row_count",
            "truncated",
            "rows_preview",
        ],
        "update_session_memory": [
            "memory_update_summary",
        ],
        "format_result": [
            "final_status",
        ],
        "format_error": [
            "final_status",
        ],
    }.get(event["node"], list(output))

    if verbose_trace_enabled():
        preferred_keys = list(output)

    for key in preferred_keys:
        value = output.get(key)
        if (
            value is None
            or value == ""
            or value is False
            or value == []
        ):
            continue
        print(
            f"  {key}: "
            f"{_console_value(value)}"
        )


def traced_node(
    node_name: str,
    node_function: Callable[
        [dict[str, Any]],
        dict[str, Any],
    ],
) -> Callable[
    [dict[str, Any]],
    dict[str, Any],
]:
    """Wrap a LangGraph node with timing and structured trace output."""

    def wrapped(
        state: dict[str, Any],
    ) -> dict[str, Any]:
        # LangGraph can run nodes in fresh ContextVar contexts. Profile is
        # request state, so restore it before every node rather than relying on
        # the router node's ContextVar mutation to survive graph boundaries.
        profile = state.get("domain_profile")
        if profile:
            set_active_profile(str(profile))
        if not trace_enabled():
            return node_function(state)

        trace_id = state.get(
            "trace_id"
        ) or new_trace_id()
        trace_started_at = state.get(
            "trace_started_at"
        ) or utc_now_iso()
        started_at = utc_now_iso()
        started = time.perf_counter()
        node_input = summarize_node_input(
            node_name,
            state,
        )

        try:
            output = node_function(state)
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000
            event = {
                "trace_id": trace_id,
                "node": node_name,
                "status": infer_event_status(
                    node_name,
                    output,
                ),
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "elapsed_ms": round(
                    elapsed_ms,
                    3,
                ),
                "input": safe_json_value(
                    node_input
                ),
                "output": safe_json_value(
                    summarize_node_output(
                        node_name,
                        output,
                    )
                ),
            }
            append_node_event(event)
            print_node_event(event)

            return {
                **output,
                "trace_id": trace_id,
                "trace_started_at": (
                    trace_started_at
                ),
                "trace_events": [event],
            }
        except Exception as exc:
            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000
            event = {
                "trace_id": trace_id,
                "node": node_name,
                "status": "failed",
                "started_at": started_at,
                "finished_at": utc_now_iso(),
                "elapsed_ms": round(
                    elapsed_ms,
                    3,
                ),
                "input": safe_json_value(
                    node_input
                ),
                "output": {},
                "exception": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            append_node_event(event)
            print_node_event(event)
            raise

    wrapped.__name__ = (
        f"traced_{node_name}"
    )
    return wrapped


def infer_final_status(
    result: dict[str, Any],
) -> str:
    if result.get("final_status"):
        return str(result["final_status"])

    if result.get("execution_error"):
        return "execution_failed"
    if result.get("validation_error"):
        if result.get(
            "validation_error_type"
        ) == "policy":
            return "policy_rejected"
        return "validation_failed"
    if not result.get(
        "review_passed",
        False,
    ):
        return "review_failed"
    if result.get("retry_count", 0) > 0:
        return "repaired_success"
    return "first_pass_success"


def build_trace_record(
    result: dict[str, Any],
    total_elapsed_ms: float,
) -> dict[str, Any]:
    return {
        "trace_id": (
            result.get("trace_id")
            or new_trace_id()
        ),
        "started_at": result.get(
            "trace_started_at",
            "",
        ),
        "finished_at": utc_now_iso(),
        "question": result.get(
            "question",
            "",
        ),
        "normalized_question": result.get(
            "normalized_question",
            "",
        ),
        "resolved_question": result.get("resolved_question", ""),
        "session_id": result.get("session_id", ""),
        "turn_type": result.get("turn_type", ""),
        "memory_used": result.get("memory_used", False),
        "query_delta": result.get("query_delta", {}),
        "query_delta_source": result.get("query_delta_source", ""),
        "query_delta_llm_called": result.get("query_delta_llm_called", False),
        "query_plan_mode": result.get("query_plan_mode", ""),
        "advanced_plan": result.get("advanced_plan", {}),
        "model_calls": result.get("model_calls", []),
        "failure_events": result.get("failure_events", []),
        "result_assertion": result.get("result_assertion", {}),
        "context_resolution_valid": result.get("context_resolution_valid", True),
        "current_turn_coverage": result.get("current_turn_coverage", {}),
        "inherited_fields": result.get("inherited_fields", []),
        "overridden_fields": result.get("overridden_fields", []),
        "memory_update_summary": result.get("memory_update_summary", {}),
        "status": infer_final_status(result),
        "retry_count": result.get(
            "retry_count",
            0,
        ),
        "total_elapsed_ms": round(
            total_elapsed_ms,
            3,
        ),
        "initial_sql": result.get(
            "initial_sql",
            "",
        ),
        "full_sql": result.get("full_sql", ""),
        "pruned_sql": result.get("pruned_sql", ""),
        "selected_candidate": result.get(
            "selected_candidate", ""
        ),
        "candidate_selection_reason": result.get(
            "candidate_selection_reason", ""
        ),
        "candidate_full_valid": result.get(
            "candidate_full_valid", False
        ),
        "candidate_full_score": result.get(
            "candidate_full_score", 0.0
        ),
        "candidate_full_error": result.get(
            "candidate_full_error", ""
        ),
        "candidate_pruned_valid": result.get(
            "candidate_pruned_valid", False
        ),
        "candidate_pruned_score": result.get(
            "candidate_pruned_score", 0.0
        ),
        "candidate_pruned_error": result.get(
            "candidate_pruned_error", ""
        ),
        "forward_schema_tables": result.get(
            "forward_schema_tables", []
        ),
        "backward_schema_tables": result.get(
            "backward_schema_tables", []
        ),
        "robust_schema_tables": result.get(
            "robust_schema_tables", []
        ),
        "final_sql": (
            result.get("validated_sql")
            or result.get("raw_sql", "")
        ),
        "validation_error": result.get(
            "validation_error",
            "",
        ),
        "validation_error_type": result.get(
            "validation_error_type",
            "",
        ),
        "review_called": result.get(
            "review_called",
            False,
        ),
        "review_passed": result.get(
            "review_passed",
            False,
        ),
        "review_reason": result.get(
            "review_reason",
            "",
        ),
        "execution_error": result.get(
            "execution_error",
            "",
        ),
        "columns": result.get(
            "columns",
            [],
        ),
        "row_count": result.get(
            "row_count",
            0,
        ),
        "rows_preview": _row_preview(
            result.get("rows", [])
        ),
        "events": result.get(
            "trace_events",
            [],
        ),
    }


def save_trace_record(
    result: dict[str, Any],
    total_elapsed_ms: float,
) -> dict[str, Any]:
    record = build_trace_record(
        result,
        total_elapsed_ms,
    )

    if trace_enabled():
        append_jsonl(
            trace_log_dir()
            / "traces.jsonl",
            record,
        )
        if record["status"] not in {
            "first_pass_success",
            "repaired_success",
        }:
            append_jsonl(
                trace_log_dir()
                / "errors.jsonl",
                record,
            )

    return record


def print_trace_summary(
    record: dict[str, Any],
) -> None:
    if not (
        trace_enabled()
        and console_trace_enabled()
    ):
        return

    print("\n" + "-" * 80)
    print("TRACE SUMMARY")
    print(
        f"trace_id: {record['trace_id']}"
    )
    print(
        f"status: {record['status']}"
    )
    print(
        "total_elapsed_ms: "
        f"{record['total_elapsed_ms']:.2f}"
    )
    print(
        "node_path: "
        + " -> ".join(
            event["node"]
            for event in record.get(
                "events",
                []
            )
        )
    )
    print(
        f"retry_count: {record['retry_count']}"
    )
    print(
        f"row_count: {record['row_count']}"
    )
    print("-" * 80)
