from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

from dotenv import dotenv_values
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.model_limiter import model_slot


ModelRole = Literal["primary_3b", "deepseek_api", "fallback_7b"]
# LangGraph may execute adjacent nodes in different execution contexts. Each
# evaluation worker is a process, so a lock-protected process-local log keeps
# call telemetry intact across nodes without leaking across benchmark cases.
_MODEL_LOG_LOCK = Lock()
_model_call_logs: dict[str, list[dict[str, object]]] = {}
_MODEL_LOG_SCOPE: ContextVar[str] = ContextVar("model_log_scope", default="global")


class ModelRouteError(RuntimeError):
    """Raised when a configured model route cannot be used."""


@dataclass(frozen=True)
class ModelTarget:
    role: ModelRole
    model: str
    api_key: str
    base_url: str
    timeout_seconds: int


def reset_model_call_log() -> None:
    with _MODEL_LOG_LOCK:
        _model_call_logs[_MODEL_LOG_SCOPE.get()] = []


def model_call_log() -> list[dict[str, object]]:
    with _MODEL_LOG_LOCK:
        return list(_model_call_logs.get(_MODEL_LOG_SCOPE.get(), []))


def _append_call(record: dict[str, object]) -> None:
    with _MODEL_LOG_LOCK:
        _model_call_logs.setdefault(_MODEL_LOG_SCOPE.get(), []).append(record)


def _usage_fields(response: object, messages: list[BaseMessage], text: str) -> dict[str, int | bool]:
    """Normalize provider usage; local Ollama often omits it, so estimate."""
    metadata = getattr(response, "response_metadata", None) or {}
    usage = getattr(response, "usage_metadata", None) or (metadata.get("token_usage", {}) if isinstance(metadata, dict) else {})
    prompt = usage.get("input_tokens", usage.get("prompt_tokens")) if isinstance(usage, dict) else None
    completion = usage.get("output_tokens", usage.get("completion_tokens")) if isinstance(usage, dict) else None
    estimated = prompt is None or completion is None
    prompt_tokens = int(prompt or sum(len(str(m.content)) for m in messages) / 4)
    completion_tokens = int(completion or len(text) / 4)
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens, "tokens_estimated": estimated}


@contextmanager
def model_call_scope(scope: str):
    token = _MODEL_LOG_SCOPE.set(scope)
    try:
        yield
    finally:
        _MODEL_LOG_SCOPE.reset(token)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """创建 OpenAI-compatible Chat Model。"""

    settings = get_settings()

    return ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
        temperature=0,
        max_retries=1,
        timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
    )


@lru_cache(maxsize=4)
def _get_client(target: ModelTarget) -> ChatOpenAI:
    return ChatOpenAI(
        model=target.model,
        api_key=target.api_key,
        base_url=target.base_url,
        temperature=0,
        max_retries=1,
        timeout=target.timeout_seconds,
    )


def _deepseek_target() -> ModelTarget:
    settings = get_settings()
    values: dict[str, str] = {}
    if settings.DEEPSEEK_ENV_FILE:
        path = Path(settings.DEEPSEEK_ENV_FILE).expanduser()
        if path.is_file():
            values = {
                key: value.strip()
                for key, value in dotenv_values(path).items()
                if isinstance(value, str) and value.strip()
            }
    api_key = settings.DEEPSEEK_API_KEY or values.get("OPENAI_API_KEY", "")
    model = settings.DEEPSEEK_MODEL or values.get("LLM_MODEL", "")
    base_url = settings.DEEPSEEK_BASE_URL or values.get("OPENAI_BASE_URL", "")
    if not api_key or not model or not base_url:
        raise ModelRouteError("DeepSeek API route is not fully configured")
    return ModelTarget(
        role="deepseek_api",
        model=model,
        api_key=api_key,
        base_url=base_url,
        timeout_seconds=settings.DEEPSEEK_REQUEST_TIMEOUT_SECONDS,
    )


def _target_for(role: ModelRole) -> ModelTarget:
    settings = get_settings()
    if role == "primary_3b":
        return ModelTarget(
            role=role,
            model=settings.LLM_MODEL,
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
            timeout_seconds=settings.LLM_REQUEST_TIMEOUT_SECONDS,
        )
    if role == "fallback_7b":
        return ModelTarget(
            role=role,
            model=settings.LLM_7B_MODEL,
            api_key=settings.LLM_7B_API_KEY or settings.LLM_API_KEY,
            base_url=settings.LLM_7B_BASE_URL or settings.LLM_BASE_URL,
            timeout_seconds=settings.LLM_7B_REQUEST_TIMEOUT_SECONDS,
        )
    return _deepseek_target()


def repair_model_role(repair_attempt: int) -> ModelRole:
    """First repair uses 3B, then DeepSeek, then the local 7B fallback."""

    if repair_attempt <= 1:
        return "primary_3b"
    if repair_attempt == 2:
        return "deepseek_api"
    return "fallback_7b"


def invoke_model(
    messages: list[BaseMessage],
    *,
    purpose: str,
    repair_attempt: int = 0,
) -> str:
    """Invoke the configured route and retain redacted route telemetry."""

    requested_role: ModelRole = (
        "deepseek_api"
        if purpose == "advanced_plan_completion"
        else repair_model_role(repair_attempt)
        if purpose == "repair"
        else "primary_3b"
    )
    attempted_roles = [requested_role]
    try:
        target = _target_for(requested_role)
    except ModelRouteError as exc:
        if requested_role != "deepseek_api":
            _append_call({
                "purpose": purpose,
                "requested_role": requested_role,
                "role": requested_role,
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "elapsed_ms": 0.0,
            })
            raise
        attempted_roles.append("fallback_7b")
        target = _target_for("fallback_7b")

    started = perf_counter()
    try:
        with model_slot(target.role) as queue_wait_ms:
            response = _get_client(target).invoke(messages)
        content = response.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                item if isinstance(item, str) else str(item.get("text", ""))
                for item in content
            )
        else:
            text = str(content)
        _append_call({
            "purpose": purpose,
            "requested_role": requested_role,
            "role": target.role,
            "model": target.model,
            "status": "success",
            "fallback_used": len(attempted_roles) > 1,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "queue_wait_ms": queue_wait_ms,
            **_usage_fields(response, messages, text),
        })
        return text
    except Exception as exc:
        _append_call({
            "purpose": purpose,
            "requested_role": requested_role,
            "role": target.role,
            "model": target.model,
            "status": "error",
            "error_type": type(exc).__name__,
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        })
        if requested_role == "deepseek_api":
            fallback = _target_for("fallback_7b")
            retry_started = perf_counter()
            try:
                with model_slot(fallback.role) as queue_wait_ms:
                    response = _get_client(fallback).invoke(messages)
                content = response.content
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "\n".join(
                        item if isinstance(item, str) else str(item.get("text", ""))
                        for item in content
                    )
                else:
                    text = str(content)
                _append_call({
                    "purpose": purpose,
                    "requested_role": requested_role,
                    "role": fallback.role,
                    "model": fallback.model,
                    "status": "success",
                    "fallback_used": True,
                    "elapsed_ms": round((perf_counter() - retry_started) * 1000, 3),
                    "queue_wait_ms": queue_wait_ms,
                    **_usage_fields(response, messages, text),
                })
                return text
            except Exception as fallback_exc:
                _append_call({
                    "purpose": purpose,
                    "requested_role": requested_role,
                    "role": fallback.role,
                    "model": fallback.model,
                    "status": "error",
                    "error_type": type(fallback_exc).__name__,
                    "fallback_used": True,
                    "elapsed_ms": round((perf_counter() - retry_started) * 1000, 3),
                })
        raise
