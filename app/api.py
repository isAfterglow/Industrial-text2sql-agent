"""FastAPI boundary for the Text2SQL Agent workbench."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.advanced_plan import compile_advanced_analysis_plan, parse_advanced_plan
from app.approval import normalize_approval_decision
from app.config import get_settings
from app.graph import graph
from app.long_term_memory import get_long_term_memory_service
from app.schema import set_active_profile
from app.task_store import AgentTaskStore
from app.trace import new_trace_id, safe_json_value, save_trace_record, trace_event_sink, utc_now_iso


ProfileName = Literal["resin", "steel_industry"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    profile: ProfileName = "resin"
    session_id: str | None = Field(default=None, max_length=128)
    approval_mode: Literal["off", "risk", "always"] | None = None
    force_approval: bool = False


class ApprovalDecisionRequest(BaseModel):
    action: Literal["approved", "rejected", "edited_plan"]
    actor: str = Field(min_length=1, max_length=120)
    comment: str = Field(default="", max_length=1000)
    advanced_plan: dict[str, Any] | None = None


def _public_result(result: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    """Expose useful output without returning large schema or prompt internals."""

    return safe_json_value({
        "final_status": result.get("final_status", ""),
        "final_answer": result.get("final_answer", ""),
        "profile": result.get("domain_profile", ""),
        "query_intent": result.get("query_intent", ""),
        "query_plan_mode": result.get("query_plan_mode", ""),
        "query_spec": result.get("query_spec", {}),
        "advanced_plan": result.get("advanced_plan", {}),
        "sql": result.get("validated_sql") or result.get("raw_sql", ""),
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "row_count": result.get("row_count", 0),
        "truncated": result.get("truncated", False),
        "retry_count": result.get("retry_count", 0),
        "repair_source": result.get("repair_source", ""),
        "model_calls": result.get("model_calls", []),
        "failure_events": result.get("failure_events", []),
        "few_shot": result.get("few_shot_retrieval_diagnostics", {}),
        "approval_required": result.get("approval_required", False),
        "approval_request": result.get("approval_request", {}),
        "approval_summary": result.get("approval_summary", {}),
        "validation_error": result.get("validation_error", ""),
        "execution_error": result.get("execution_error", ""),
        "elapsed_ms": round(elapsed_ms, 3),
    })


class AgentTaskRunner:
    def __init__(self) -> None:
        settings = get_settings()
        self.store = AgentTaskStore(settings.AGENT_TASK_DB_PATH)
        self.executor = ThreadPoolExecutor(max_workers=max(1, settings.AGENT_MAX_CONCURRENT_TASKS))

    def submit(self, request: QueryRequest, *, approval_id: str = "") -> dict[str, Any]:
        trace_id = new_trace_id()
        session_id = request.session_id or "session-" + uuid.uuid4().hex[:12]
        payload = request.model_dump()
        task = self.store.create_task(
            profile=request.profile, question=request.question, session_id=session_id,
            trace_id=trace_id, payload=payload, approval_id=approval_id,
        )
        self.executor.submit(self._run, task["task_id"])
        return task

    def _run(self, task_id: str) -> None:
        task = self.store.get_task(task_id)
        if not task:
            return
        payload = dict(task["input"])
        self.store.update_status(task_id, "running")
        started = time.perf_counter()
        try:
            set_active_profile(task["profile"])
            graph_input: dict[str, Any] = {
                "question": task["question"],
                "requested_profile": task["profile"],
                "session_id": task["session_id"],
                "trace_id": task["trace_id"],
                "trace_started_at": utc_now_iso(),
                "trace_events": [],
                "force_approval": bool(payload.get("force_approval", False)),
            }
            if payload.get("approval_mode"):
                graph_input["approval_mode"] = payload["approval_mode"]
            if task["approval_id"]:
                graph_input["approval_request"] = {"approval_id": task["approval_id"]}
                approved = get_long_term_memory_service().get_approval_request(task["approval_id"])
                if approved:
                    graph_input["approved_execution_plan"] = {
                        **dict(approved.get("payload") or {}),
                        "decision": dict(approved.get("decision") or {}),
                    }

            def event_sink(event: dict[str, Any]) -> None:
                self.store.append_event(task_id, {"type": "node", "event": event})

            with trace_event_sink(event_sink):
                result = graph.invoke(graph_input, {"recursion_limit": 32})
            elapsed_ms = (time.perf_counter() - started) * 1000
            trace = save_trace_record(result, elapsed_ms)
            public = _public_result(result, elapsed_ms)
            public["trace"] = trace
            status = "approval_required" if result.get("approval_required") else "completed"
            self.store.update_status(task_id, status, result=public)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            self.store.update_status(
                task_id, "failed", result={"elapsed_ms": round(elapsed_ms, 3)},
                error_message=f"{type(exc).__name__}: {exc}",
            )


runner: AgentTaskRunner | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runner
    runner = AgentTaskRunner()
    yield
    if runner:
        runner.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Text2SQL Agent API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _runner() -> AgentTaskRunner:
    if runner is None:
        raise RuntimeError("API lifespan has not initialized the task runner")
    return runner


def _service(profile: str):
    set_active_profile(profile)
    return get_long_term_memory_service()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/tasks", status_code=202)
def create_task(request: QueryRequest) -> dict[str, Any]:
    return _runner().submit(request)


@app.get("/api/tasks")
def list_tasks(limit: int = Query(default=30, ge=1, le=200)) -> list[dict[str, Any]]:
    return _runner().store.list_tasks(limit=limit)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict[str, Any]:
    task = _runner().store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
    if not _runner().store.get_task(task_id):
        raise HTTPException(status_code=404, detail="Task not found")

    async def stream():
        sequence = after
        while True:
            events = _runner().store.events_after(task_id, sequence)
            for event in events:
                sequence = event["sequence"]
                yield f"id: {sequence}\nevent: {event['payload'].get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            task = _runner().store.get_task(task_id)
            if task and task["status"] in {"completed", "approval_required", "failed"}:
                yield f"event: terminal\ndata: {json.dumps({'status': task['status']}, ensure_ascii=False)}\n\n"
                return
            yield ": keepalive\n\n"
            await asyncio.sleep(0.7)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/approvals")
def list_approvals(profile: ProfileName = "resin", status: str | None = None,
                   limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    return _service(profile).list_approval_requests(status=status, limit=limit)


@app.get("/api/approvals/{approval_id}")
def get_approval(approval_id: str, profile: ProfileName = "resin") -> dict[str, Any]:
    record = _service(profile).get_approval_request(approval_id)
    if not record:
        raise HTTPException(status_code=404, detail="Approval request not found")
    return record


@app.post("/api/approvals/{approval_id}/decision")
def decide_approval(approval_id: str, request: ApprovalDecisionRequest,
                    profile: ProfileName = "resin") -> dict[str, Any]:
    service = _service(profile)
    stored = service.get_approval_request(approval_id)
    if not stored:
        raise HTTPException(status_code=404, detail="Approval request not found")
    decision = request.model_dump()
    if request.action == "edited_plan":
        if not request.advanced_plan:
            raise HTTPException(status_code=422, detail="edited_plan requires advanced_plan")
        try:
            set_active_profile(str(stored["profile"]))
            plan = parse_advanced_plan(json.dumps(request.advanced_plan, ensure_ascii=False))
            compile_advanced_analysis_plan(plan)
            decision["advanced_plan"] = plan
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid advanced plan: {exc}") from exc
    return service.decide_approval_request(approval_id, normalize_approval_decision(decision)) or stored


@app.post("/api/approvals/{approval_id}/resume", status_code=202)
def resume_approval(approval_id: str, profile: ProfileName = "resin") -> dict[str, Any]:
    service = _service(profile)
    request = service.get_approval_request(approval_id)
    if not request:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if request["status"] not in {"approved", "edited_plan"}:
        raise HTTPException(status_code=409, detail="Only approved requests can resume")
    payload = dict(request.get("payload") or {})
    return _runner().submit(QueryRequest(
        question=str(payload.get("question") or ""), profile=str(request["profile"]), approval_mode="risk"
    ), approval_id=approval_id)


@app.get("/api/memories")
def list_memories(profile: ProfileName = "resin", memory_type: str | None = None,
                  limit: int = Query(default=100, ge=1, le=200)) -> list[dict[str, Any]]:
    return [record.to_public_dict() for record in _service(profile).list_memories(memory_type, limit)]
