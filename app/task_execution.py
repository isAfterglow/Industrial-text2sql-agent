"""Process-safe execution entrypoint used by local and RQ workers."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.api_result import public_result
from app.config import get_settings
from app.graph import graph
from app.llm import model_call_scope
from app.long_term_memory import get_long_term_memory_service
from app.request_context import RequestIdentity, identity_scope
from app.schema import set_active_profile
from app.task_store import AgentTaskStore
from app.trace import save_trace_record, trace_event_sink, utc_now_iso
from app.metrics import observe_model_calls, observe_task
try:
    from langgraph.errors import GraphRecursionError
except ImportError:  # pragma: no cover - older LangGraph compatibility
    GraphRecursionError = RuntimeError


def run_agent_task(task_id: str, store_path: str | None = None) -> None:
    """Run one durable task. Safe to invoke in a separate RQ worker process."""
    store = AgentTaskStore(store_path or get_settings().AGENT_TASK_DB_PATH)
    task = store.get_task(task_id)
    if not task:
        return
    if task["cancel_requested"]:
        store.update_status(task_id, "cancelled")
        return
    store.update_status(task_id, "running")
    started = time.perf_counter()
    identity = RequestIdentity(task["user_id"], task["tenant_id"], str(task["input"].get("actor_role", "analyst")))
    try:
        with identity_scope(identity), model_call_scope(task_id):
            set_active_profile(task["profile"])
            payload: dict[str, Any] = dict(task["input"])
            graph_input: dict[str, Any] = {
                "question": task["question"], "requested_profile": task["profile"],
                "session_id": f"{identity.tenant_id}:{identity.user_id}:{task['session_id']}",
                "trace_id": task["trace_id"], "trace_started_at": utc_now_iso(), "trace_events": [],
                "current_span_id": "",
                "force_approval": bool(payload.get("force_approval", False)),
            }
            if payload.get("approval_mode"):
                graph_input["approval_mode"] = payload["approval_mode"]
            if task["approval_id"]:
                graph_input["approval_request"] = {"approval_id": task["approval_id"]}
                approved = get_long_term_memory_service().get_approval_request(task["approval_id"])
                if approved:
                    graph_input["approved_execution_plan"] = {**dict(approved.get("payload") or {}), "decision": dict(approved.get("decision") or {})}

            def event_sink(event: dict[str, Any]) -> None:
                store.append_event(task_id, {"type": "node", "event": event})

            with trace_event_sink(event_sink):
                result = graph.invoke(graph_input, {"recursion_limit": getattr(get_settings(), "AGENT_MAX_GRAPH_STEPS", 32)})
        elapsed_ms = (time.perf_counter() - started) * 1000
        if store.is_cancel_requested(task_id):
            store.update_status(task_id, "cancelled", result={"elapsed_ms": round(elapsed_ms, 3)})
            observe_task("cancelled", task["profile"], elapsed_ms)
            return
        public = public_result(result, elapsed_ms)
        observe_model_calls(list(result.get("model_calls", [])))
        public["trace"] = save_trace_record(result, elapsed_ms)
        terminal_status = "approval_required" if result.get("approval_required") else "completed"
        store.update_status(task_id, terminal_status, result=public)
        observe_task(terminal_status, task["profile"], elapsed_ms)
    except GraphRecursionError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        store.update_status(task_id, "failed", result={
            "elapsed_ms": round(elapsed_ms, 3),
            "final_status": "agent_budget_exhausted",
            "failure_events": [{"stage": "orchestration", "category": "non_convergent_plan", "error_type": "graph_recursion_limit", "repairable": False, "message": str(exc)[:500]}],
        }, error_message=f"agent_budget_exhausted: {exc}")
        observe_task("failed", task["profile"], elapsed_ms)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        store.update_status(task_id, "failed", result={"elapsed_ms": round(elapsed_ms, 3)}, error_message=f"{type(exc).__name__}: {exc}")
        observe_task("failed", task["profile"], elapsed_ms)
