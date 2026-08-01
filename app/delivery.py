"""Separate query semantics from bounded result delivery."""

from __future__ import annotations

from typing import Any


def build_delivery_policy(question: str, query_spec: dict[str, Any], max_rows: int) -> dict[str, Any]:
    explicit_projection = bool(query_spec.get("strict_projection"))
    is_full_table = str(query_spec.get("query_type", "")) == "full_table"
    return {
        "projection_mode": "strict" if explicit_projection else "helpful",
        "semantic_limit": query_spec.get("limit"),
        "transport_limit": max_rows,
        "pagination": "cursor" if is_full_table else "none",
        "full_result_requested": is_full_table,
    }
