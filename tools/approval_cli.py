"""Operational CLI for execution approval and few-shot memory governance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.advanced_plan import compile_advanced_analysis_plan, parse_advanced_plan
from app.approval import normalize_approval_decision
from app.graph import graph
from app.long_term_memory import get_long_term_memory_service
from app.schema import set_active_profile
from app.trace import new_trace_id, utc_now_iso


def _json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _plan_from_file(path: str) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def _service(profile: str):
    set_active_profile(profile)
    return get_long_term_memory_service()


def _request_or_exit(service, approval_id: str) -> dict[str, Any]:
    request = service.get_approval_request(approval_id)
    if not request:
        raise SystemExit(f"Approval request not found: {approval_id}")
    return request


def _decide(args: argparse.Namespace) -> None:
    service = _service(args.profile)
    request = _request_or_exit(service, args.approval_id)
    decision: dict[str, Any] = {
        "action": args.action,
        "actor": args.actor,
        "comment": args.comment or "",
    }
    if args.action == "edited_plan":
        if not args.plan_file:
            raise SystemExit("edited_plan requires --plan-file")
        set_active_profile(str(request["profile"]))
        plan = parse_advanced_plan(json.dumps(_plan_from_file(args.plan_file), ensure_ascii=False))
        compile_advanced_analysis_plan(plan)
        decision["advanced_plan"] = plan
    decided = service.decide_approval_request(
        args.approval_id, normalize_approval_decision(decision)
    )
    _json(decided)


def _resume(args: argparse.Namespace) -> None:
    service = _service(args.profile)
    request = _request_or_exit(service, args.approval_id)
    if str(request.get("status")) not in {"approved", "edited_plan"}:
        raise SystemExit("Only approved or edited_plan requests can be resumed.")
    payload = dict(request.get("payload") or {})
    result = graph.invoke(
        {
            "question": payload.get("question", ""),
            "approval_mode": "risk",
            "approval_request": {"approval_id": args.approval_id},
            "trace_id": new_trace_id(),
            "trace_started_at": utc_now_iso(),
            "trace_events": [],
        },
        {"recursion_limit": 32},
    )
    _json({
        "final_status": result.get("final_status"),
        "final_answer": result.get("final_answer"),
        "approval_summary": result.get("approval_summary"),
        "validated_sql": result.get("validated_sql"),
        "row_count": result.get("row_count"),
    })


def _validate_candidate(args: argparse.Namespace) -> None:
    service = _service(args.profile)
    plan = _plan_from_file(args.plan_file)
    service.record_candidate_validation(
        args.memory_id,
        question=args.question,
        plan=plan,
        evidence=args.evidence,
        validator=args.validator,
    )
    record = service.repository.get(args.memory_id)
    _json(record.to_public_dict() if record else {"memory_id": args.memory_id})


def _promote(args: argparse.Namespace) -> None:
    service = _service(args.profile)
    result = service.promote_candidate(
        args.memory_id,
        evidence=args.evidence,
        approver=args.actor,
        approval_reason=args.reason,
    )
    _json(result.record.to_public_dict())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approval and memory-governance CLI")
    parser.add_argument("--profile", default="resin", choices=["resin", "steel_industry"])
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="List approval requests")
    listing.add_argument("--status", default=None)
    listing.add_argument("--limit", type=int, default=50)
    listing.set_defaults(handler=lambda args: _json(_service(args.profile).list_approval_requests(args.status, args.limit)))

    show = commands.add_parser("show", help="Show an approval request")
    show.add_argument("approval_id")
    show.set_defaults(handler=lambda args: _json(_request_or_exit(_service(args.profile), args.approval_id)))

    decide = commands.add_parser("decide", help="Approve, reject, or edit an immutable plan")
    decide.add_argument("approval_id")
    decide.add_argument("action", choices=["approved", "rejected", "edited_plan"])
    decide.add_argument("--actor", required=True)
    decide.add_argument("--comment", default="")
    decide.add_argument("--plan-file")
    decide.set_defaults(handler=_decide)

    resume = commands.add_parser("resume", help="Re-run an approved request through Guard and execution")
    resume.add_argument("approval_id")
    resume.set_defaults(handler=_resume)

    candidates = commands.add_parser("candidates", help="List non-retrievable candidate memories")
    candidates.add_argument("--limit", type=int, default=50)
    candidates.set_defaults(handler=lambda args: _json([item.to_public_dict() for item in _service(args.profile).list_memories("candidate_episodic", args.limit)]))

    validate = commands.add_parser("validate-candidate", help="Attach one independent validation")
    validate.add_argument("memory_id")
    validate.add_argument("--question", required=True)
    validate.add_argument("--plan-file", required=True)
    validate.add_argument("--evidence", required=True)
    validate.add_argument("--validator", required=True)
    validate.set_defaults(handler=_validate_candidate)

    promote = commands.add_parser("promote", help="Approve a verified candidate for few-shot retrieval")
    promote.add_argument("memory_id")
    promote.add_argument("--actor", required=True)
    promote.add_argument("--reason", required=True)
    promote.add_argument("--evidence", required=True)
    promote.set_defaults(handler=_promote)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
