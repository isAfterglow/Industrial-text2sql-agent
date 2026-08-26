# Evaluation Layout V2

The old benchmark entry points remain available for historical baselines. New
runs use six disjoint suites so the score cannot hide routing differences:

| Suite | Domain | Scope | Cases |
|---|---|---|---:|
| `resin_basic` | resin | deterministic/basic, no policy cases | 45 |
| `resin_complex` | resin | planning, repair, memory, multi-turn, boundary | 27 |
| `resin_safety` | resin | policy refusal and dangerous operations | 10 |
| `steel_basic` | steel | deterministic fact/dimension queries | 29 |
| `steel_complex` | steel | advanced-plan/window/statistical challenge | 37 |
| `steel_safety` | steel | policy refusal, privilege and system-schema access | 6 |

The normalized release set currently contains 154 cases. Safety cases are
kept separate from functional scores and include destructive writes, system
schemas, file access, dangerous functions, privilege escalation and result
write-back attempts.

Run with:

```bash
python -m eval.harness run --label <label> --suite-set normalized \
  --memory-mode production
```

Each suite records strict/acceptable pass, safety, model routing, repair,
few-shot usage, latency and trace artifacts. Safety is reported as its own
suite rather than mixed into a material or steel functional score.
