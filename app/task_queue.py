"""Durable RQ dispatch with an explicit process-local development fallback."""
from __future__ import annotations
import time, uuid
from concurrent.futures import ThreadPoolExecutor
from app.config import get_settings
from app.task_execution import run_agent_task
from app.task_store import AgentTaskStore

class TaskDispatcher:
    def __init__(self) -> None:
        self.settings = get_settings(); self.store = AgentTaskStore(self.settings.AGENT_TASK_DB_PATH)
        self.executor = ThreadPoolExecutor(max_workers=max(1, self.settings.AGENT_MAX_CONCURRENT_TASKS))
        self.queue = self._queue()
    def _queue(self):
        if self.settings.TASK_QUEUE_MODE == "local": return None
        try:
            from redis import Redis
            from rq import Queue
            connection = Redis.from_url(self.settings.REDIS_URL, socket_connect_timeout=1, protocol=2)
            connection.ping(); return Queue(self.settings.TASK_QUEUE_NAME, connection=connection)
        except Exception:
            if self.settings.TASK_QUEUE_MODE == "redis": raise RuntimeError("TASK_QUEUE_MODE=redis but Redis/RQ is unavailable")
            return None
    @property
    def mode(self) -> str: return "rq" if self.queue is not None else "local"
    def submit(self, *, user_id: str, tenant_id: str, role: str, request, approval_id: str = "") -> dict:
        if self.store.active_count(user_id, tenant_id) >= self.settings.USER_MAX_ACTIVE_TASKS: raise ValueError("Active task quota exceeded")
        recent = [task for task in self.store.list_tasks(user_id=user_id, tenant_id=tenant_id, limit=200) if task["created_at"] >= time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time()-60))]
        if len(recent) >= self.settings.USER_TASKS_PER_MINUTE: raise ValueError("Task rate limit exceeded")
        from app.trace import new_trace_id
        session_id = request.session_id or "session-" + uuid.uuid4().hex[:12]
        payload = request.model_dump(); payload["actor_role"] = role
        task = self.store.create_task(user_id=user_id, tenant_id=tenant_id, profile=request.profile, question=request.question, session_id=session_id, trace_id=new_trace_id(), payload=payload, approval_id=approval_id)
        if self.queue:
            job = self.queue.enqueue("app.task_execution.run_agent_task", task["task_id"], str(self.store.db_path), job_timeout=self.settings.TASK_JOB_TIMEOUT_SECONDS)
            self.store.set_queue_job(task["task_id"], job.id)
        else: self.executor.submit(run_agent_task, task["task_id"], str(self.store.db_path))
        return self.store.get_task(task["task_id"]) or task
    def shutdown(self) -> None: self.executor.shutdown(wait=False, cancel_futures=True)
