# Evaluation Harness

`eval/run_evaluation.py` remains the isolated per-case evaluator. The Harness
standardizes three frozen suites, turns their result artifacts into a single
release metric file, and compares that file to a versioned baseline.

| Key | Dataset | Primary purpose |
| --- | ---: | --- |
| `resin_80` | 80 cases | Material-domain accuracy, conversations and safety |
| `steel_core_60` | 60 cases | Profile portability and deterministic coverage |
| `steel_complex_10` | 10 cases | LLM planning, repair and structured-plan execution |

The tracked baseline is `eval/baselines/v5.2-production-fewshot.json`. It is a
reference, not a pass threshold: latency and model-call changes need context,
while lower strict, acceptable, or safety pass counts are regressions.

## Full Release Run

```bash
conda run --no-capture-output -n scitime-agent \
  python eval/harness.py run --label release-YYYYMMDD --memory-mode production
```

This creates three standard evaluator runs under `eval/harness_runs/<label>/`
and one normalized `metrics.json`.

## Compare With Baseline

```bash
conda run --no-capture-output -n scitime-agent \
  python eval/harness.py compare \
  --baseline eval/baselines/v5.2-production-fewshot.json \
  --candidate eval/harness_runs/release-YYYYMMDD/metrics.json \
  --output eval/harness_runs/release-YYYYMMDD/comparison.md \
  --fail-on-regression
```

The normalized artifact records strict and acceptable pass counts, safety pass
rate, model roles/calls, repair attempts/recovery, mean/P95 latency, and
few-shot hit rate. Evaluator child processes explicitly set `APPROVAL_MODE=off`;
approval quality belongs to the API end-to-end tests because benchmarks cannot
wait for a person.

## CI Scope

GitHub Actions runs fast, database-free checks: Python compilation, approval,
API/SSE, approval-resume and Harness contract tests, plus the frontend build.
Full benchmark execution needs the project database and local/Ollama models, so
it remains a deliberate release or nightly command rather than an unreliable
hosted CI job.
