# Multi-user execution platform

The workbench uses JWT identity on every API request. Tasks, approval records,
session keys, audit events and newly written long-term memories are scoped by
`tenant_id` and `user_id`. A user cannot read or cancel another user's task;
reviewers can decide approval requests only inside their tenant; admins can
inspect all tenant tasks.

Roles: `analyst` submits and follows tasks, `reviewer` approves or rejects
risky plans, and `admin` has tenant-wide operational visibility. The API never
accepts an approval actor from the browser; it records the JWT subject.

For development, the bootstrap accounts are `analyst_a`, `analyst_b`,
`reviewer`, and `admin` in tenant `demo`. Their password is configured by
`AUTH_DEMO_PASSWORD` (default: `agent-demo-password`). Set a strong
`JWT_SECRET`, disable bootstrap users, and use an external identity provider
before deployment.

## Queue and limits

`TASK_QUEUE_MODE=auto` uses Redis/RQ when `REDIS_URL` is reachable and falls
back to the local executor for a single-process development run. Set it to
`redis` in a multi-instance deployment to fail closed when Redis is absent.

```bash
redis-server --save '' --appendonly no
REDIS_URL=redis://127.0.0.1:6379/0 python -m app.worker
REDIS_URL=redis://127.0.0.1:6379/0 uvicorn app.api:app --port 8000
```

The durable SQLite task store keeps status and SSE events across API restarts.
Workers check cancellation before execution. Per-user active-task and
per-minute limits protect the queue; Redis leases cap 3B, 7B and DeepSeek model
concurrency across workers. If Redis is unavailable, model caps use a local
semaphore and are therefore only process-wide.

Authenticated long-term memory is private by default. Shared few-shot examples
must be promoted deliberately by an administrator rather than being retrieved
across tenants automatically.
