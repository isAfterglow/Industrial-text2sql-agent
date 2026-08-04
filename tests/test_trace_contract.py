from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.trace import AGENT_TRACE_SCHEMA_VERSION, load_trace_timeline, save_trace_record, traced_node


class TraceContractTests(unittest.TestCase):
    def test_traced_node_emits_versioned_parent_linked_event(self) -> None:
        with TemporaryDirectory() as temp_dir:
            old_dir = os.environ.get("TEXT2SQL_TRACE_LOG_DIR")
            old_console = os.environ.get("TEXT2SQL_TRACE_CONSOLE")
            os.environ["TEXT2SQL_TRACE_LOG_DIR"] = temp_dir
            os.environ["TEXT2SQL_TRACE_CONSOLE"] = "0"
            try:
                result = traced_node("unit_node", lambda _: {"query_plan_mode": "deterministic"})({
                    "trace_id": "trace-unit", "trace_started_at": "2026-08-04T00:00:00Z",
                    "current_span_id": "root-span", "domain_profile": "resin", "question": "test",
                })
                event = result["trace_events"][0]
                self.assertEqual(event["schema_version"], AGENT_TRACE_SCHEMA_VERSION)
                self.assertEqual(event["parent_span_id"], "root-span")
                self.assertEqual(event["event_type"], "node.ok")
                self.assertEqual(event["route"], "deterministic")
                self.assertTrue(event["span_id"])
                self.assertEqual(load_trace_timeline("trace-unit", Path(temp_dir)), [event])
                record = save_trace_record(result, 5.0)
                self.assertEqual(record["trace_summary"]["node_path"], ["unit_node"])
            finally:
                if old_dir is None:
                    os.environ.pop("TEXT2SQL_TRACE_LOG_DIR", None)
                else:
                    os.environ["TEXT2SQL_TRACE_LOG_DIR"] = old_dir
                if old_console is None:
                    os.environ.pop("TEXT2SQL_TRACE_CONSOLE", None)
                else:
                    os.environ["TEXT2SQL_TRACE_CONSOLE"] = old_console


if __name__ == "__main__":
    unittest.main()
