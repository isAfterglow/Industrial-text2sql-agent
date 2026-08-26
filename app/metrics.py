"""Low-friction Prometheus metrics for API, Agent tasks and Trace events."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


TASKS_TOTAL = Counter(
    "text2sql_tasks_total",
    "Agent tasks by terminal status.",
    ("status", "profile"),
)
TASK_LATENCY_SECONDS = Histogram(
    "text2sql_task_latency_seconds",
    "End-to-end Agent task latency.",
    ("profile",),
)
TASK_QUEUE_DEPTH = Gauge(
    "text2sql_task_queue_depth",
    "Tasks currently queued or running in the durable task store.",
)
TRACE_EVENTS_TOTAL = Counter(
    "text2sql_trace_events_total",
    "Trace node events by node and status.",
    ("node", "status"),
)
TRACE_NODE_LATENCY_SECONDS = Histogram(
    "text2sql_trace_node_latency_seconds",
    "Agent node latency.",
    ("node",),
)
MODEL_CALLS_TOTAL = Counter(
    "text2sql_model_calls_total",
    "Model calls by role and outcome.",
    ("role", "outcome"),
)
MODEL_TOKENS_TOTAL = Counter(
    "text2sql_model_tokens_total",
    "Estimated or provider-reported model tokens by role and direction.",
    ("role", "direction", "estimated"),
)


def observe_trace_event(event: dict) -> None:
    """Record a bounded set of labels; metrics must never break execution."""
    try:
        node = str(event.get("node", "unknown"))[:80] or "unknown"
        status = str(event.get("status", "unknown"))[:40] or "unknown"
        TRACE_EVENTS_TOTAL.labels(node=node, status=status).inc()
        TRACE_NODE_LATENCY_SECONDS.labels(node=node).observe(max(0.0, float(event.get("elapsed_ms", 0.0))) / 1000.0)
    except Exception:
        return


def observe_task(status: str, profile: str, elapsed_ms: float) -> None:
    try:
        TASKS_TOTAL.labels(status=str(status), profile=str(profile)).inc()
        TASK_LATENCY_SECONDS.labels(profile=str(profile)).observe(max(0.0, float(elapsed_ms)) / 1000.0)
    except Exception:
        return


def observe_model_calls(calls: list[dict]) -> None:
    for call in calls:
        try:
            role = str(call.get("role", "unknown"))[:80] or "unknown"
            outcome = "error" if call.get("status") == "error" or call.get("error") else "success"
            MODEL_CALLS_TOTAL.labels(role=role, outcome=outcome).inc()
            estimated = str(bool(call.get("tokens_estimated", True))).lower()
            MODEL_TOKENS_TOTAL.labels(role=role, direction="prompt", estimated=estimated).inc(float(call.get("prompt_tokens", 0) or 0))
            MODEL_TOKENS_TOTAL.labels(role=role, direction="completion", estimated=estimated).inc(float(call.get("completion_tokens", 0) or 0))
        except Exception:
            continue


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
