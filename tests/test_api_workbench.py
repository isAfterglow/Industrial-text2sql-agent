from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app
from app.auth import issue_token
from app.request_context import RequestIdentity
from app.trace import publish_trace_event


class ApiWorkbenchTests(unittest.TestCase):
    def test_task_api_persists_result_and_node_events(self) -> None:
        with TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(
                AGENT_TASK_DB_PATH=str(Path(temp_dir) / "tasks.sqlite3"),
                AGENT_MAX_CONCURRENT_TASKS=1,
                USER_MAX_ACTIVE_TASKS=3,
                USER_TASKS_PER_MINUTE=20,
                TASK_QUEUE_MODE="local",
            )

            def fake_invoke(state, _config):
                publish_trace_event({
                    "trace_id": state["trace_id"], "node": "load_schema", "status": "ok",
                    "elapsed_ms": 1.0, "input": {}, "output": {},
                })
                return {
                    **state,
                    "final_status": "first_pass_success",
                    "final_answer": "ok",
                    "domain_profile": state["requested_profile"],
                    "validated_sql": "SELECT 1",
                    "columns": ["value"], "rows": [[1]], "row_count": 1,
                    "trace_events": [], "model_calls": [], "failure_events": [],
                }

            with patch("app.api.get_settings", return_value=settings), patch(
                "app.task_queue.get_settings", return_value=settings
            ), patch("app.task_execution.get_settings", return_value=settings), patch(
                "app.task_execution.graph.invoke", side_effect=fake_invoke
            ):
                with TestClient(app) as client:
                    headers = {"Authorization": "Bearer " + issue_token(RequestIdentity("unit-analyst", "unit-tenant", "analyst"))}
                    self.assertEqual(client.get("/health").json(), {"status": "ok"})
                    response = client.post("/api/tasks", json={
                        "question": "test query", "profile": "resin",
                    }, headers=headers)
                    self.assertEqual(response.status_code, 202)
                    task_id = response.json()["task_id"]
                    task = {}
                    for _ in range(30):
                        task = client.get(f"/api/tasks/{task_id}", headers=headers).json()
                        if task["status"] == "completed":
                            break
                        time.sleep(0.05)
                    self.assertEqual(task["status"], "completed")
                    self.assertEqual(task["result"]["sql"], "SELECT 1")
                    self.assertTrue(client.get(f"/api/tasks/{task_id}/events", headers=headers).text)
                    trace = client.get(f"/api/tasks/{task_id}/trace", headers=headers).json()
                    self.assertEqual(trace["trace_id"], task["trace_id"])


if __name__ == "__main__":
    unittest.main()
