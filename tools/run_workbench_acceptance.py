#!/usr/bin/env python3
"""Run representative API workbench flows against a locally running server."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "approval_required", "failed"}


def request(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = Request(base_url + path, method=method, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} -> {exc.code}: {detail}") from exc


def wait_task(base_url: str, task_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        task = request(base_url, "GET", f"/api/tasks/{task_id}")
        if task["status"] in TERMINAL:
            return task
        time.sleep(0.5)
    raise TimeoutError(f"Task {task_id} did not finish within {timeout_seconds}s")


def resume_if_needed(base_url: str, task: dict[str, Any], *, edit_plan: bool, timeout_seconds: int) -> tuple[dict[str, Any], str]:
    if task["status"] != "approval_required":
        return task, "not_required"
    result = task.get("result", {})
    approval = result.get("approval_request", {})
    approval_id = str(approval.get("approval_id", ""))
    if not approval_id:
        raise RuntimeError("Approval-required task has no approval ID")
    body: dict[str, Any] = {"action": "approved", "actor": "acceptance_runner", "comment": "End-to-end workbench acceptance."}
    if edit_plan:
        plan = dict(approval.get("payload", {}).get("advanced_plan", {}))
        if plan:
            body["action"] = "edited_plan"
            body["advanced_plan"] = plan
    request(base_url, "POST", f"/api/approvals/{approval_id}/decision?profile={task['profile']}", body)
    resumed = request(base_url, "POST", f"/api/approvals/{approval_id}/resume?profile={task['profile']}")
    return wait_task(base_url, str(resumed["task_id"]), timeout_seconds), str(body["action"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run three real workbench acceptance flows")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=150)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    scenarios = [
        ("material_complex", {"question": "原始孔隙率小于0.35的样本中，平均背面温度最高的5个是哪些？", "profile": "resin"}, False),
        ("steel_complex", {"question": "找出每种负荷类型中耗电量最高的一笔读数，返回负荷类型、读数编号和耗电量。", "profile": "steel_industry"}, True),
        ("forced_approval", {"question": "查询样本305的原始材料密度。", "profile": "resin", "force_approval": True}, False),
    ]
    results: list[dict[str, Any]] = []
    for name, payload, edit_plan in scenarios:
        created = request(args.base_url, "POST", "/api/tasks", payload)
        initial_task = wait_task(args.base_url, str(created["task_id"]), args.timeout)
        initial_result = initial_task.get("result", {})
        task, approval_action = resume_if_needed(
            args.base_url, initial_task, edit_plan=edit_plan, timeout_seconds=args.timeout
        )
        result = task.get("result", {})
        outcome = {
            "scenario": name, "task_id": task["task_id"], "status": task["status"],
            "final_status": result.get("final_status"), "row_count": result.get("row_count"),
            "elapsed_ms": result.get("elapsed_ms"), "query_plan_mode": result.get("query_plan_mode"),
            "model_roles": [call.get("role") for call in result.get("model_calls", [])],
            "initial_status": initial_task["status"],
            "initial_model_roles": [call.get("role") for call in initial_result.get("model_calls", [])],
            "approval_action": approval_action,
            "trace_events": len(result.get("trace", {}).get("events", [])),
        }
        if task["status"] != "completed":
            raise RuntimeError(f"{name} failed: {outcome}")
        results.append(outcome)
        print(json.dumps(outcome, ensure_ascii=False))
    output = Path(args.output) if args.output else ROOT / "eval" / "runs" / (
        "workbench-acceptance-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"scenarios": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Acceptance report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
