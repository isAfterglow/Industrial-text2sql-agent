# Release Evaluation Report

Run: `release-p0-p2-20260826`

## Dataset

The current normalized release benchmark contains 154 cases:

The repository also includes a third-domain portability smoke test for the
declarative `profiles/ecommerce.yaml` contract. It intentionally has no
business database or benchmark score; it validates Profile loading, aliases,
relationships, sensitive columns, read-only table allowlisting and SQL Guard
through the same core code.

| Suite | Cases |
|---|---:|
| Resin basic | 45 |
| Resin complex | 27 |
| Resin safety | 10 |
| Steel basic | 29 |
| Steel complex | 37 |
| Steel safety | 6 |

## Results

| Suite | Acceptable | Strict | Safety | Worker errors | Timeout |
|---|---:|---:|---:|---:|---:|
| Resin basic | 45/45 | 35/45 | - | 0 | 0 |
| Resin complex | 27/27 | 22/27 | - | 0 | 0 |
| Resin safety | 10/10 | 9/10 | 10/10 | 0 | 0 |
| Steel basic | 29/29 | 29/29 | - | 0 | 0 |
| Steel complex | 33/37 | 32/37 | - | 3 | 0 |
| Steel safety | 6/6 | 6/6 | 6/6 | 0 | 0 |

Overall acceptable pass is `150/154 (97.4%)`; strict pass is `133/154
(86.4%)`; safety policy pass is `16/16 (100%)`.

## Model usage

The run recorded 36 model calls and 36,682 total tokens:

| Role | Calls | Prompt | Completion | Total |
|---|---:|---:|---:|---:|
| Qwen 3B | 32 | 26,962 | 1,786 | 28,748 |
| DeepSeek API | 2 | 826 | 1,251 | 2,077 |
| Qwen 7B | 2 | 1,980 | 120 | 2,100 |

No monetary cost is claimed in this run because pricing is not configured.
Set `TEXT2SQL_COST_PROMPT_PER_1K_USD` and
`TEXT2SQL_COST_COMPLETION_PER_1K_USD` to calculate API cost in future runs.

## Known limitation

Three steel advanced-analysis cases are classified as non-convergent Agent
loops. They are isolated and recorded with Trace rather than being reported as
successful SQL. Basic, safety and all resin suites have zero worker errors.
