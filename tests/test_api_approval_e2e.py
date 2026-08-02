from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.api import app
from app.long_term_memory.config import get_long_term_memory_settings
from app.long_term_memory.service import LongTermMemoryService
from app.nodes import approval_gate
from app.schema import set_active_profile


class ApiApprovalEndToEndTests(unittest.TestCase):
    def test_force_approval_decision_and_resume(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = LongTermMemoryService(replace(
                get_long_term_memory_settings(), enabled=False, db_path=root / "memory.sqlite3"
            ))
            settings = SimpleNamespace(AGENT_TASK_DB_PATH=str(root / "tasks.sqlite3"), AGENT_MAX_CONCURRENT_TASKS=1)
            approval_settings = SimpleNamespace(APPROVAL_MODE="risk")

            def fake_invoke(state, _config):
                set_active_profile("resin")
                gate = approval_gate({
                    **state,
                    "domain_profile": "resin", "resolved_question": state["question"],
                    "query_intent": "lookup", "query_plan_mode": "deterministic",
                    "query_spec": {"query_type": "sample_lookup"},
                    "validated_sql": "SELECT sample_id FROM material_static LIMIT 1",
                    "delivery_policy": {}, "intent_related_tables": ["material_static"],
                    "retry_count": 0, "failure_events": [], "model_calls": [],
                    "approval_mode": state.get("approval_mode", "risk"),
                })
                if gate.get("approval_required"):
                    return {**state, **gate, "final_status": "approval_required", "final_answer": "Approval required", "trace_events": []}
                return {**state, **gate, "final_status": "first_pass_success", "final_answer": "Executed", "validated_sql": "SELECT 1", "columns": ["value"], "rows": [[1]], "row_count": 1, "trace_events": []}

            with patch("app.api.get_settings", return_value=settings), patch(
                "app.api.get_long_term_memory_service", return_value=service
            ), patch("app.api.graph.invoke", side_effect=fake_invoke), patch(
                "app.nodes.get_long_term_memory_service", return_value=service
            ), patch("app.nodes.get_settings", return_value=approval_settings):
                with TestClient(app) as client:
                    response = client.post("/api/tasks", json={
                        "question": "approval test", "profile": "resin", "force_approval": True,
                    })
                    task_id = response.json()["task_id"]
                    task = {}
                    for _ in range(30):
                        task = client.get(f"/api/tasks/{task_id}").json()
                        if task["status"] == "approval_required":
                            break
                        time.sleep(0.05)
                    self.assertEqual(task["status"], "approval_required")
                    approval_id = task["result"]["approval_request"]["approval_id"]
                    decision = client.post(f"/api/approvals/{approval_id}/decision?profile=resin", json={
                        "action": "approved", "actor": "reviewer", "comment": "verified",
                    })
                    self.assertEqual(decision.json()["status"], "approved")
                    resumed = client.post(f"/api/approvals/{approval_id}/resume?profile=resin")
                    self.assertEqual(resumed.status_code, 202)


if __name__ == "__main__":
    unittest.main()
