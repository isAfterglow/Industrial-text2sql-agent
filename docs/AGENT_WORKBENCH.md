# Agent Workbench

The workbench turns the existing Text2SQL graph into a browser-facing service
without duplicating the graph, model router, Guard, memory, or approval logic.

## Architecture

```text
React workbench --HTTP/SSE--> FastAPI task service --> background graph invocation
                                      |                       |
                                      v                       v
                               SQLite task/event store   existing Agent nodes
                                      |                       |
                                      +------ SSE events -----+
```

Each task receives a durable ID, session ID, trace ID, and selected Profile.
Node events are written to the task store as they occur. SSE consumers can
disconnect and reconnect with `?after=<sequence>` without losing the timeline.

## Start

Use two terminals from the project root:

```bash
conda run --no-capture-output -n scitime-agent uvicorn app.api:app --reload --port 8000
cd web && npm run dev
```

Open `http://127.0.0.1:5173`. FastAPI documentation is at
`http://127.0.0.1:8000/docs`.

## API Surface

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/tasks` | Start an asynchronous Agent task. |
| `GET` | `/api/tasks/{id}` | Retrieve result, SQL, rows, diagnostics and Trace. |
| `GET` | `/api/tasks/{id}/events` | Receive node and task events through SSE. |
| `GET` | `/api/approvals` | Read the approval queue. |
| `POST` | `/api/approvals/{id}/decision` | Approve, reject, or submit an edited `AdvancedPlan`. |
| `POST` | `/api/approvals/{id}/resume` | Re-run an approved request through Guard and execution. |
| `GET` | `/api/memories` | Inspect governed long-term memory by Profile. |

The selected `profile` is passed as `requested_profile`; it deliberately
overrides automatic vocabulary routing for a workbench user who has selected a
known data domain.

## Approval Flow

The workbench never edits SQL. It submits a constrained structured
`AdvancedPlan`, which is parsed and compiled server-side before the existing
Guard sees it. The approval record remains immutable. See
`docs/APPROVAL_WORKFLOW.md` for promotion governance and risk policy.
