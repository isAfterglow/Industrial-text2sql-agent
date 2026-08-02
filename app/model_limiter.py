"""Cross-worker model concurrency limits with Redis leases and local fallback."""

from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from threading import BoundedSemaphore
from typing import Iterator

from app.config import get_settings


_LOCAL = {
    "primary_3b": BoundedSemaphore(1),
    "fallback_7b": BoundedSemaphore(1),
    "deepseek_api": BoundedSemaphore(3),
}


def _limit(role: str) -> int:
    settings = get_settings()
    return max(1, {
        "primary_3b": settings.MODEL_PRIMARY_3B_CONCURRENCY,
        "fallback_7b": settings.MODEL_FALLBACK_7B_CONCURRENCY,
        "deepseek_api": settings.MODEL_DEEPSEEK_CONCURRENCY,
    }.get(role, 1))


def _redis_client():
    url = get_settings().REDIS_URL
    if not url:
        return None
    try:
        import redis
        client = redis.Redis.from_url(url, decode_responses=True, protocol=2)
        client.ping()
        return client
    except Exception:
        return None


@contextmanager
def model_slot(role: str) -> Iterator[float]:
    """Yield queue wait milliseconds; lease ensures local models are not oversubscribed."""

    started = time.perf_counter()
    client = _redis_client()
    limit = _limit(role)
    token = uuid.uuid4().hex
    key = ""
    if client is not None:
        deadline = time.monotonic() + max(5, get_settings().LLM_7B_REQUEST_TIMEOUT_SECONDS + 10)
        while time.monotonic() < deadline:
            for index in range(limit):
                candidate = f"text2sql:model-slot:{role}:{index}"
                if client.set(candidate, token, nx=True, ex=max(30, get_settings().LLM_7B_REQUEST_TIMEOUT_SECONDS + 15)):
                    key = candidate
                    break
            if key:
                break
            time.sleep(0.05)
        if not key:
            raise TimeoutError(f"Timed out waiting for {role} model slot")
        try:
            yield round((time.perf_counter() - started) * 1000, 3)
        finally:
            try:
                if client.get(key) == token:
                    client.delete(key)
            except Exception:
                pass
        return

    semaphore = _LOCAL[role]
    semaphore.acquire()
    try:
        yield round((time.perf_counter() - started) * 1000, 3)
    finally:
        semaphore.release()
