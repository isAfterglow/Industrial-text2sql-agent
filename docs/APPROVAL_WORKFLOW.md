# Approval Workflow

## Execution Approval

The graph pauses after SQL Guard and before review/execution. `APPROVAL_MODE`
controls the gate:

- `off`: no automatic pause.
- `risk`: pause advanced plans, free-form/RSL plans, repaired SQL, full-table
  exports, and Profile-declared sensitive columns.
- `always`: pause every validated query.

The stored request is an immutable snapshot containing the trace ID, Profile,
QuerySpec, AdvancedPlan, compiled SQL, delivery policy, risk reasons, retry
count, model calls, failure events, and a plan fingerprint. A resumed request
must regenerate the same fingerprint; otherwise the graph opens a new request.

```bash
conda run -n scitime-agent python tools/approval_cli.py --profile steel_industry list --status pending
conda run -n scitime-agent python tools/approval_cli.py --profile steel_industry show approval-xxxxxxxxxxxxxxxx
conda run -n scitime-agent python tools/approval_cli.py --profile steel_industry decide approval-xxxxxxxxxxxxxxxx approved --actor reviewer_a --comment "business check passed"
conda run -n scitime-agent python tools/approval_cli.py --profile steel_industry resume approval-xxxxxxxxxxxxxxxx
```

`edited_plan` accepts a JSON AdvancedPlan, validates it, compiles it, and then
forces the graph through Guard again. Direct free-form SQL editing is not an
approval operation.

## Memory Governance

Successful complex queries enter `candidate_episodic`; this type is never used
by few-shot retrieval. Three independently recorded validations are required
before an identified reviewer can promote it to `episodic`.

```bash
conda run -n scitime-agent python tools/approval_cli.py --profile steel_industry candidates
conda run -n scitime-agent python tools/approval_cli.py --profile steel_industry validate-candidate mem-xxxxxxxxxxxxxxxx --question "..." --plan-file plan.json --evidence "run URL or trace ID" --validator reviewer_a
conda run -n scitime-agent python tools/approval_cli.py --profile steel_industry promote mem-xxxxxxxxxxxxxxxx --actor reviewer_b --reason "three independent validations" --evidence "review ticket ABC-123"
```

The promotion record stores the approver, reason, evidence, and validation
history. Only the promoted record can become a retrievable few-shot example.
