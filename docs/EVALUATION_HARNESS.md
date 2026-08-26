# Evaluation Harness and Regression Gate

`eval/harness.py` converts the material and steel evaluator artifacts into a
versioned `EvaluationSummary v1`. A summary records the Git/schema/model
context, strict and acceptable result pass, safety pass, timeouts, worker
errors, repair recovery, LLM usage and latency. It also separates basic Profile
queries from complex planning/memory queries.

The frozen legacy baseline is `eval/baselines/v5.2-production-fewshot.json`.
The CI/self-hosted default uses the same six-suite normalized layout as the
current harness: `eval/baselines/v5.2-normalized-production.json`. Do not
overwrite either after a normal run; add a new reviewed baseline only when
intentionally releasing a new version.

## Commands

Collect existing evaluator output into one normalized summary:

```bash
python -m eval.harness collect --label candidate \
  --result resin_80=eval/runs/<resin-run>/results.json \
  --result steel_core_60=eval/runs/<steel-core-run>/results.json \
  --result steel_complex_10=eval/runs/<steel-complex-run>/results.json \
  --output eval/harness_runs/candidate/metrics.json
```

Run all three suites on a machine with the configured databases and model
router, then gate them against the baseline:

```bash
python -m eval.harness run --label candidate --memory-mode production
python -m eval.harness gate \
  --baseline eval/baselines/v5.2-production-fewshot.json \
  --candidate eval/harness_runs/candidate/metrics.json \
  --output eval/harness_runs/candidate/regression_gate.json
```

The gate fails for a lower acceptable pass, lower safety pass, timeout, worker
error, or lower strict pass on a basic deterministic segment. A lower strict
pass on the complex LLM/planning segment is reported as a notice, because it
needs trace review rather than an automatic claim of a functional regression.

`ci.yml` validates these contracts on every PR. `full-regression.yml` runs the
real database/model evaluation only on the `self-hosted, agent-eval` runner and
uploads reports, case logs, and node traces as artifacts.
