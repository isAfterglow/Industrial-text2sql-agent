# Few-Shot Version Comparison

## Scope and Method

This comparison separates deterministic benchmark coverage from advanced
planning coverage. All runs use the same database schema and `qwen2.5:3b` as
the primary local model. `acceptable` means requested data semantics are
correct even when a helpful extra projection or an alias differs; `strict`
requires an exact projection as well.

The before/after result reflects the delivered system version, not a causal
claim that every gain is from few-shot retrieval alone. The current version
also includes constrained plans, Profile capabilities, context-scope guards,
and deterministic compilers.

## Material Domain: 80-case Engineering Benchmark

| Metric | Before few-shot | Current version | Change |
|---|---:|---:|---:|
| Run | `resin-v4.6-stage1-4-final-20260801` | `resin-v5.2-p0-p4-final-20260801` | - |
| Git | `417de75` | `9841f0b` | - |
| Evaluation cases | 80 | 80 | same suite |
| Acceptable pass | 68/80 (85.0%) | 80/80 (100.0%) | +12 cases / +15.0 pp |
| Strict pass | 53/80 (66.3%) | 65/80 (81.3%) | +12 cases / +15.0 pp |
| Safety acceptable pass | 7/8 | 8/8 | +1 case |
| Cross-table acceptable pass | 7/11 | 11/11 | +4 cases |
| Cross-temporal acceptable pass | 1/3 | 3/3 | +2 cases |
| Multi-turn acceptable pass | 9/10 | 10/10 | +1 case |
| Full-table acceptable pass | 0/1 | 1/1 | +1 case |
| Timeout / worker error | 0 / 0 | 0 / 0 | unchanged |
| First-pass turns | 72/92 | 76/92 | +4 turns |
| Repair recovery | 1/1 | 2/2 | maintained 100% |
| LLM calls | 33 (0.359/turn) | 19 (0.207/turn) | -42.4% |
| Mean case latency | 1293.92 ms | 1036.86 ms | -19.9% |
| P95 case latency | 5664.54 ms | 8604.47 ms | +51.9% |

The current run has a higher P95 and median latency (249.84 ms to 346.56 ms),
while its mean latency falls. This is consistent with fewer LLM-routed turns
and a small number of remaining advanced-plan or repair turns; it should not
be interpreted as a few-shot-only causal measurement.

## Steel Domain: Standard 60-case Profile Benchmark

| Metric | Before few-shot | Few-shot version | Change |
|---|---:|---:|---:|
| Run | `steel-v1.4-p0-p1-regression-20260731` | `steel-v1.5-production-fewshot-20260801` | - |
| Evaluation cases | 60 | 60 | same suite |
| Acceptable pass | 60/60 (100.0%) | 60/60 (100.0%) | unchanged |
| Strict pass | 60/60 (100.0%) | 60/60 (100.0%) | unchanged |
| Safety pass | 4/4 | 4/4 | unchanged |
| Intent accuracy | 60/60 | 60/60 | unchanged |
| LLM calls | 0 | 0 | unchanged |
| Mean case latency | 199.81 ms | 291.98 ms | +46.1% |

Few-shot retrieval correctly does not activate for these deterministic Profile
queries, so this suite is a stability check rather than evidence of few-shot
benefit.

## Steel Domain: 10-case Advanced Agent Challenge

| Metric | Before few-shot | Current version | Change |
|---|---:|---:|---:|
| Run | `steel-agent-v1.5-two-stage-20260801` | `steel-v5.2-current-fewshot-20260801` | - |
| Git | `417de75` | `9841f0b` | - |
| Evaluation cases | 10 | 10 | same suite |
| Acceptable pass | 5/10 (50.0%) | 10/10 (100.0%) | +5 cases / +50.0 pp |
| Strict pass | 4/10 (40.0%) | 9/10 (90.0%) | +5 cases / +50.0 pp |
| Window-ranking pass | 0/2 | 2/2 | +2 cases |
| Advanced-plan contract | 9/13 | 10/11 | stronger and fewer attempts |
| Result assertions | 8/9 | 10/10 | full coverage |
| Timeout / worker error | 0 / 0 | 0 / 0 | unchanged |
| First-pass turns | 7/10 | 9/10 | +2 turns |
| Repair recovery | 1/2 | 1/1 | maintained 100% |
| LLM calls | 29 (2.9/turn) | 21 (2.1/turn) | -27.6% |
| Mean case latency | 26619.08 ms | 3643.60 ms | -86.3% |
| P95 case latency | 71949.37 ms | 7065.45 ms | -90.2% |

## Evidence

- [Material baseline report](runs/resin-v4.6-stage1-4-final-20260801/report.md)
- [Current material report](runs/resin-v5.2-p0-p4-final-20260801/report.md)
- [Steel standard baseline report](runs/steel-v1.4-p0-p1-regression-20260731/report.md)
- [Steel standard few-shot report](runs/steel-v1.5-production-fewshot-20260801/report.md)
- [Steel advanced baseline report](runs/steel-agent-v1.5-two-stage-20260801/report.md)
- [Current steel advanced report](runs/steel-v5.2-current-fewshot-20260801/report.md)
