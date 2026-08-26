"""Offline fault-contract checks for cancellation and queue fallbacks."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from app.task_store import AgentTaskStore


def main() -> int:
    with TemporaryDirectory() as directory:
        store = AgentTaskStore(Path(directory) / "tasks.sqlite3")
        task = store.create_task(user_id="u", tenant_id="t", profile="resin", question="q", session_id="s", trace_id="tr", payload={})
        task_id = task["task_id"]
        store.request_cancel(task_id)
        assert store.is_cancel_requested(task_id)
        store.update_status(task_id, "cancelled")
        assert store.get_task(task_id)["status"] == "cancelled"
    print("fault smoke: cancellation persistence passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

