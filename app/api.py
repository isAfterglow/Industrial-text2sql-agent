"""FastAPI boundary for the Text2SQL Agent workbench."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.advanced_plan import compile_advanced_analysis_plan, parse_advanced_plan
from app.approval import normalize_approval_decision
from app.config import get_settings
from app.auth import LoginRequest, RegisterRequest, current_user, get_user_store, issue_token, require_roles
from app.long_term_memory import get_long_term_memory_service
from app.request_context import RequestIdentity, identity_scope
from app.schema import set_active_profile
from app.task_queue import TaskDispatcher
from app.trace import safe_json_value


ProfileName = Literal["resin", "steel_industry"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    profile: ProfileName = "resin"
    session_id: str | None = Field(default=None, max_length=128)
    approval_mode: Literal["off", "risk", "always"] | None = None
    force_approval: bool = False


class ApprovalDecisionRequest(BaseModel):
    action: Literal["approved", "rejected", "edited_plan"]
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


runner: TaskDispatcher | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runner
    runner = TaskDispatcher()
    yield
    if runner:
        runner.shutdown()


app = FastAPI(title="Text2SQL Agent API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _runner() -> TaskDispatcher:
    if runner is None:
        raise RuntimeError("API lifespan has not initialized the task runner")
    return runner


def _service(profile: str, identity: RequestIdentity):
    with identity_scope(identity):
        set_active_profile(profile)
        return get_long_term_memory_service()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(request: LoginRequest) -> dict[str, Any]:
    identity = get_user_store().authenticate(request.username, request.password)
    if not identity:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"token": issue_token(identity), "user": identity.__dict__}

@app.get("/api/auth/me")
def me(identity: RequestIdentity = Depends(current_user)) -> dict[str, str]:
    return identity.__dict__

@app.post("/api/tasks", status_code=202)
def create_task(request: QueryRequest, identity: RequestIdentity = Depends(require_roles("analyst", "admin"))) -> dict[str, Any]:
    try:
        return _runner().submit(user_id=identity.user_id, tenant_id=identity.tenant_id, role=identity.role, request=request)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.get("/api/tasks")
def list_tasks(limit: int = Query(default=30, ge=1, le=200), identity: RequestIdentity = Depends(current_user)) -> list[dict[str, Any]]:
    return _runner().store.list_tasks(tenant_id=identity.tenant_id, user_id=None if identity.role == "admin" else identity.user_id, limit=limit)


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, identity: RequestIdentity = Depends(current_user)) -> dict[str, Any]:
    task = _runner().store.get_task(task_id)
    if not task or task["tenant_id"] != identity.tenant_id or (identity.role != "admin" and task["user_id"] != identity.user_id):
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str, identity: RequestIdentity = Depends(current_user)) -> dict[str, Any]:
    task = get_task(task_id, identity)
    _runner().store.request_cancel(task_id)
    return _runner().store.get_task(task_id) or task


@app.get("/api/tasks/{task_id}/events")
async def task_events(task_id: str, after: int = Query(default=0, ge=0), identity: RequestIdentity = Depends(current_user)) -> StreamingResponse:
    owner = _runner().store.get_task(task_id)
    if not owner or owner["tenant_id"] != identity.tenant_id or (identity.role != "admin" and owner["user_id"] != identity.user_id):
        raise HTTPException(status_code=404, detail="Task not found")

    async def stream():
        sequence = after
        while True:
            events = _runner().store.events_after(task_id, sequence)
            for event in events:
                sequence = event["sequence"]
                yield f"id: {sequence}\nevent: {event['payload'].get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            task = _runner().store.get_task(task_id)
            if task and task["status"] in {"completed", "approval_required", "failed", "cancelled"}:
                yield f"event: terminal\ndata: {json.dumps({'status': task['status']}, ensure_ascii=False)}\n\n"
                return
            yield ": keepalive\n\n"
            await asyncio.sleep(0.7)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/approvals")
def list_approvals(profile: ProfileName = "resin", status: str | None = None,
                   limit: int = Query(default=50, ge=1, le=200), identity: RequestIdentity = Depends(current_user)) -> list[dict[str, Any]]:
    with identity_scope(identity):
        service = _service(profile, identity)
        if identity.role in {"reviewer", "admin"}:
            return service.list_tenant_approval_requests(identity.tenant_id, status=status, limit=limit)
        return service.list_approval_requests(status=status, limit=limit, user_id=identity.user_id, tenant_id=identity.tenant_id)


@app.get("/api/approvals/{approval_id}")
def get_approval(approval_id: str, profile: ProfileName = "resin", identity: RequestIdentity = Depends(current_user)) -> dict[str, Any]:
    with identity_scope(identity): record = _service(profile, identity).get_approval_request(approval_id)
    if not record or record["tenant_id"] != identity.tenant_id or (identity.role == "analyst" and record["user_id"] != identity.user_id):
        raise HTTPException(status_code=404, detail="Approval request not found")
    return record


@app.post("/api/approvals/{approval_id}/decision")
def decide_approval(approval_id: str, request: ApprovalDecisionRequest,
                    profile: ProfileName = "resin", identity: RequestIdentity = Depends(require_roles("reviewer", "admin"))) -> dict[str, Any]:
    with identity_scope(identity):
        service = _service(profile, identity); stored = service.get_approval_request(approval_id)
    if not stored or stored["tenant_id"] != identity.tenant_id:
        raise HTTPException(status_code=404, detail="Approval request not found")
    decision = request.model_dump(); decision["actor"] = identity.user_id
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
def resume_approval(approval_id: str, profile: ProfileName = "resin", identity: RequestIdentity = Depends(require_roles("reviewer", "admin"))) -> dict[str, Any]:
    with identity_scope(identity):
        service = _service(profile, identity); request = service.get_approval_request(approval_id)
    if not request or request["tenant_id"] != identity.tenant_id:
        raise HTTPException(status_code=404, detail="Approval request not found")
    if request["status"] not in {"approved", "edited_plan"}:
        raise HTTPException(status_code=409, detail="Only approved requests can resume")
    payload = dict(request.get("payload") or {})
    return _runner().submit(user_id=str(request["user_id"]), tenant_id=identity.tenant_id, role="analyst", request=QueryRequest(question=str(payload.get("question") or ""), profile=str(request["profile"]), approval_mode="risk"), approval_id=approval_id)


@app.get("/api/memories")
def list_memories(profile: ProfileName = "resin", memory_type: str | None = None,
                  limit: int = Query(default=100, ge=1, le=200), identity: RequestIdentity = Depends(current_user)) -> list[dict[str, Any]]:
    with identity_scope(identity):
        return [record.to_public_dict() for record in _service(profile, identity).list_memories(memory_type, limit)]
