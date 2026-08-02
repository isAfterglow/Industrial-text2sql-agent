"""Request-scoped identity shared by API, workers, approvals and memory."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestIdentity:
    user_id: str = "system"
    tenant_id: str = "default"
    role: str = "admin"


_IDENTITY: ContextVar[RequestIdentity] = ContextVar("agent_request_identity", default=RequestIdentity())


def current_identity() -> RequestIdentity:
    return _IDENTITY.get()


@contextmanager
def identity_scope(identity: RequestIdentity):
    token = _IDENTITY.set(identity)
    try:
        yield
    finally:
        _IDENTITY.reset(token)
