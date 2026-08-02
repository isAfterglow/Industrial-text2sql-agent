from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.approval import assess_approval_risk
from app.long_term_memory.config import get_long_term_memory_settings
from app.long_term_memory.service import LongTermMemoryService
from app.nodes import approval_gate
from app.schema import set_active_profile


class ApprovalWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        set_active_profile("resin")
        self.service = LongTermMemoryService(
            replace(
                get_long_term_memory_settings(),
                enabled=False,
                db_path=Path(self.temp_dir.name) / "approval.sqlite3",
            )
        )
        self.settings = SimpleNamespace(APPROVAL_MODE="risk")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _state(self) -> dict:
        return {
            "question": "查询复杂分析结果",
            "resolved_question": "查询复杂分析结果",
            "domain_profile": "resin",
            "query_intent": "aggregate",
            "query_plan_mode": "rsl",
            "query_spec": {"query_type": "complex_or_uncertain"},
            "advanced_plan": {},
            "validated_sql": "SELECT sample_id FROM material_static LIMIT 3",
            "delivery_policy": {"full_result_requested": False},
            "intent_related_tables": ["material_static"],
            "trace_id": "trace-test",
            "retry_count": 0,
            "failure_events": [],
            "model_calls": [],
            "approval_mode": "risk",
            "approval_request": {},
            "approval_decision": {},
        }

    def test_repaired_and_full_table_queries_are_risky(self) -> None:
        self.assertIn("repaired_sql", assess_approval_risk({"retry_count": 1})["reasons"])
        self.assertIn(
            "full_table_or_export",
            assess_approval_risk({"query_spec": {"query_type": "full_table"}})["reasons"],
        )

    def test_approved_request_can_resume_and_decision_is_immutable(self) -> None:
        with patch("app.nodes.get_long_term_memory_service", return_value=self.service), patch(
            "app.nodes.get_settings", return_value=self.settings
        ):
            pending = approval_gate(self._state())
            self.assertTrue(pending["approval_required"])
            approval_id = pending["approval_request"]["approval_id"]
            first = self.service.decide_approval_request(
                approval_id, {"action": "approved", "actor": "reviewer_a"}
            )
            second = self.service.decide_approval_request(
                approval_id, {"action": "rejected", "actor": "reviewer_b"}
            )
            self.assertEqual(first["status"], "approved")
            self.assertEqual(second["status"], "approved")

            resumed_state = self._state()
            self.settings.APPROVAL_MODE = "off"
            resumed_state["approval_mode"] = "off"
            resumed_state["approval_request"] = {"approval_id": approval_id}
            resumed = approval_gate(resumed_state)
            self.assertFalse(resumed["approval_required"])
            self.assertTrue(resumed["approval_approved"])

    def test_candidate_requires_three_validations_and_named_promotion(self) -> None:
        candidate = self.service.remember_candidate_case(
            question="复杂案例",
            resolved_question="复杂案例",
            query_spec={"query_type": "temporal_aggregate"},
            sql="SELECT sample_id FROM thermal_response",
        ).record
        for index in range(3):
            self.service.record_candidate_validation(
                candidate.memory_id,
                question=f"独立变体{index}",
                plan={"family": "test", "variant": index},
                evidence=f"trace-{index}",
                validator=f"reviewer_{index}",
            )
        promoted = self.service.promote_candidate(
            candidate.memory_id,
            evidence="change-review-123",
            approver="reviewer_lead",
            approval_reason="three independent validations passed",
        )
        self.assertEqual(promoted.record.memory_type, "episodic")
        self.assertEqual(
            promoted.record.metadata["promotion_review"]["approver"], "reviewer_lead"
        )

    def test_edited_plan_is_recompiled_and_approved(self) -> None:
        state = self._state()
        state["advanced_plan"] = {
            "family": "group_topk",
            "group_columns": ["load_type_name"],
            "metric": "usage_kwh",
            "output_columns": ["load_type_name", "reading_id", "usage_kwh"],
            "limit": 1,
        }
        state["domain_profile"] = "steel_industry"
        set_active_profile("steel_industry")
        with patch("app.nodes.get_long_term_memory_service", return_value=self.service), patch(
            "app.nodes.get_settings", return_value=self.settings
        ):
            pending = approval_gate(state)
            approved = approval_gate({
                **state,
                "approval_request": pending["approval_request"],
                "approval_decision": {
                    "action": "edited_plan", "actor": "reviewer_a",
                    "advanced_plan": state["advanced_plan"],
                },
            })
        self.assertTrue(approved["approval_approved"])
        self.assertTrue(approved["raw_sql"].startswith("WITH ranked AS"))
        revalidated = approval_gate({
            **state,
            **approved,
            "validated_sql": approved["raw_sql"],
        })
        self.assertEqual(revalidated["approval_summary"]["action"], "edited_plan_revalidated")


if __name__ == "__main__":
    unittest.main()
