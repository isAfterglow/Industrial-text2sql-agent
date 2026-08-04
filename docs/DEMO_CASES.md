# Demo Cases

这两条案例来自冻结评测集，不是只替换参数的演示模板。建议在面试中先展示任务 Trace，再解释“模型在哪里发挥作用、程序如何约束它”。

## 1. Resin: Static Filter + Temporal Aggregate

**Question**：原始孔隙率小于 0.35 的样本中，平均背面温度最高的 5 个是哪些？

**Why it matters**：问题同时包含静态材料属性、时序响应、聚合和排序，不能从单表字段关键词直接回答。

```text
Profile(resin)
  -> resolve porosity_v / back_temperature / sample grain
  -> QuerySpec: static filter + temporal AVG + Top-K
  -> deterministic compiler builds a one-to-many join and GROUP BY
  -> SQL Guard verifies table, join and read-only constraints
  -> result assertion checks projection, order and result rows
```

对应评测用例：`cross_temporal_extra_001`，分类为 `cross_temporal`。最终 v5.2 运行中该类 `3/3` 可接受、`3/3` 严格通过。

演示命令：

```bash
conda run -n scitime-agent python eval/run_evaluation.py \
  --suite eval/benchmark_v1.py --only cross_temporal_extra_001 \
  --run-id demo-resin-cross-temporal --memory-mode production
```

观察 `eval/runs/demo-resin-cross-temporal/traces/node_events.jsonl` 中的 `build_query_plan`、`generate_simple_sql`、`validate_sql`、`execute_sql`、`validate_result_assertions`。这条路径强调：复杂语义不必都交给 LLM，自然语言被归约为可验证的结构化计划。

## 2. Steel: Window Analysis With Constrained Repair

**Question**：计算每个月碳强度相较上个月的变化率，按增幅从高到低列出前 3 个月。

**Why it matters**：需要识别碳强度派生公式、按月聚合、窗口函数 `LAG`、变化率、空值处理和 Top-K，超出了简单 QuerySpec 编译器的固定能力。

```text
Profile(steel_industry)
  -> intent / schema linking: usage_kwh, co2_tco2, calendar_dim.month
  -> 3B produces constrained AdvancedPlan
  -> program validates family, fields, joins and allowed window operation
  -> compiler emits SQL; Guard and result assertion verify it
  -> if contract/SQL fails, one budgeted repair is attempted
  -> 7B/API is only a fallback, never an unvalidated executor
```

对应评测用例：`steel_agent_002`，分类为 `window_change`。最终 v5.2 运行中该用例一次修复后成功；10 道高级分析题为 `10/10` 可接受、`9/10` 严格通过。

演示命令：

```bash
conda run -n scitime-agent python eval/run_evaluation.py \
  --suite eval/steel_agent_challenge_v1.py --only steel_agent_002 \
  --run-id demo-steel-window-change --memory-mode production
```

面试时应展示 `results.json` 中的 `advanced_plan`、`model_calls`、`failure_events` 和 `trace_id`。重点不是“LLM 会写复杂 SQL”，而是“LLM 只能提出白名单结构计划，失败可定位、可修复、可回归”。

## Related Operational Paths

- **Safety**：写操作、跨库系统表和危险函数在生成 SQL 前被拒绝，禁止进入 `execute_sql` 或修复节点。
- **Approval**：高风险计划经过 Guard 后进入审批快照；审批者修改的是 `AdvancedPlan`，服务端重新编译并再次 Guard。
- **Trace**：`GET /api/tasks/{task_id}/trace` 或任务 SSE 可查看单请求的节点拓扑、耗时、路由、审批和失败原因。
