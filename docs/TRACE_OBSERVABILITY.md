# AgentTrace v1

The Text2SQL Agent writes request-scoped, append-only JSONL trace events. The
same event is also published to the task SSE stream, so an in-flight query and
its durable history have one `trace_id`.

Each `AgentTrace v1` event contains:

- identity and topology: `trace_id`, `span_id`, `parent_span_id`, `node`;
- timing and outcome: timestamps, `elapsed_ms`, `status`, `event_type`,
  `error_code` when applicable;
- execution context: project, Profile, route, intent, retry count, model
  roles, safety decision and approval ID;
- bounded `input_summary` and `output_summary` for diagnosis.

Raw state is never serialized wholesale. Text is length-bounded, SQL rows are
previewed, and clients should use the summaries rather than treating traces as
an audit export of user data.

Node events are stored in `logs/node_events.jsonl`; completed requests are
stored in `logs/traces.jsonl`. The authenticated endpoint
`GET /api/tasks/{task_id}/trace` returns the request timeline and a replay-safe
summary (node path, failures, rejections, total node time). The task SSE stream
continues to emit the exact node events while a task is running.

Trace persistence and publishing are best-effort: an observability failure
must never interrupt SQL validation, approval, or execution.
