# 失败案例复盘

这组案例来自实际评测产物，不是为某道题添加的特例规则。面试时应重点说明：系统如何发现失败、如何分类、如何恢复，以及为什么没有简单地放宽校验。

## 案例一：材料多轮查询的 Anchor 协议断裂

### 场景

用户先查询某个样本的静态属性，随后使用“它”追问另一张表中的属性：

```text
查询样本305的原始密度和原始孔隙率。
再查它的热解热和表面发射率。
```

### 失败表现

历史版本在第二轮解析结果引用时只保留了 `anchor_id`，丢失了结果 Anchor 的完整类型信息。编译器需要校验 `status`、`profile`、`schema_hash` 和 `sample_ids`，因此无法确认这个历史结果是否仍属于当前领域、当前 Schema 和有效结果集，最终 fail closed。

专项记录见 `eval/harness_runs/resin_complex_projectionfix_20260826/case_results/conversation_single_001.json`，其中曾出现：

```text
AttributeError: 'dict' object has no attribute 'add'
```

这不是数据库错误，而是上下文解析层的数据协议错误；同一问题会批量影响多轮题，因此不能通过给单题加规则解决。

### 定位与修复

在 `resolve_conversation_context()` 中保留完整 Typed Anchor，同时把解析方式作为元数据单独记录：

- Anchor 本体保留 `status/profile/schema_hash/sample_ids`
- 解析元数据记录 `resolution_method/confidence`
- 编译前检查过期、跨领域、Schema 变化和结果截断
- 只有通过校验后才把样本范围继承到新的 QuerySpec

此外修复了字段匹配协议中的字典/集合类型误用，避免多轮查询统一触发运行时异常。

### 验证结果

修复后的专项 10 题：

- 0 worker error
- 0 timeout
- 跨表追问、结果集筛选、父 Anchor、重排和修复回归均可执行
- `conversation_reorder_001` 从 worker error 恢复为 strict pass

### 面试回答重点

> 我们把结果集当成带 Schema 版本和生命周期的 Typed Anchor，而不是把上一轮 SQL 字符串拼到下一轮提示词里。Anchor 无效时系统宁可要求用户重新查询，也不会盲目继承范围。这个问题暴露的是跨节点状态协议不完整，而不是模型能力不足。

## 案例二：钢铁高级统计计划进入循环

### 场景

钢铁复杂集中的窗口函数、相关聚合和累计占比问题需要多步计划生成、契约校验和重试。实际失败记录包括：

- `steel_agent_001`
- `steel_agent_003`
- `steel_agent_004`

### 失败表现

三题最终记录为：

```text
GraphRecursionError: Recursion limit of 32 reached without hitting a stop condition
```

这说明 Agent 在“计划生成/校验/重试”节点之间没有在限定次数内收敛。它不是普通 SQL 执行超时，也不能简单通过无限提高 recursion limit 解决；无限重试会放大 CPU、模型调用和数据库压力。

### 定位与处置

当前系统已经具备以下保护：

- LangGraph recursion limit
- 每题独立 worker 隔离
- case timeout
- 失败事件分类和 Trace
- 评测进程跳过失败题，继续执行剩余题

因此该问题被记录为 `worker_error`，不会拖垮整套评测。后续合理的工程修复是为 Agent Loop 增加显式预算：计划尝试次数、修复尝试次数、总模型耗时和相同错误指纹计数；同一错误重复出现时直接转人工审批或返回结构化失败，而不是继续循环。

### 面试回答重点

> 我没有把 recursion limit 当成成功率开关。它首先是系统稳定性边界。遇到相同计划契约错误重复出现时，应根据错误指纹提前停止，并把上下文、失败阶段和候选计划落 Trace，交给人工或更强模型处理。这样失败是可观测、可恢复的，而不是 Agent 无限跑飞。

## 案例三：SQL 合法，但语义结果不满足契约

### 场景

钢铁复杂问题：

```text
找出功率因数低于各负荷类型平均值一个标准差的异常读数，返回读数编号、负荷类型和功率因数。
```

### 失败表现

模型生成的 SQL 语法合法，也包含分组平均值、标准差、Join 和异常条件，但结果级断言发现缺少问题要求的实体字段 `sample_id`。记录中的候选 SQL 为：

```sql
WITH stats AS (...)
SELECT er.reading_id, ltd.load_type_name, er.power_factor
FROM energy_readings AS er
JOIN load_type_dim AS ltd ON er.load_type_id = ltd.load_type_id
JOIN stats ON stats.group_key = ltd.load_type_name
WHERE er.power_factor < stats.average_metric - stats.stddev_metric
```

评测记录的失败分类包括：

- `plan_contract`
- `guard_generation`
- `validation_result_invariant`
- 缺少要求字段 `sample_id`

### 路由与修复过程

该题触发了修复链路：

1. 3B 计划契约校验失败
2. 进入修复计划
3. 结果级断言发现字段契约仍不满足
4. 调用 7B fallback
5. 重试 3 次仍未满足结果契约，最终标记 `failed`

这说明“SQL 能执行”不等于“回答正确”。Guard 负责安全和结构合法性，结果断言负责列、行、排序和结果语义；两者职责不能合并。

### 面试回答重点

> 这个案例中 SQL 不是语法错误，而是语义契约错误。我们没有因为 SQL 可执行就放行，而是用结果级断言检查用户要求的实体字段、维度字段和指标字段。修复失败后切换 7B 仍失败，系统会保留失败原因并停止重试，避免把错误答案伪装成成功。

## 三个案例体现的工程能力

| 问题类型 | 失败层 | 处理原则 |
|---|---|---|
| Anchor 协议断裂 | 多轮状态/编译前 | 完整类型化状态、Schema 校验、失效即澄清 |
| Agent 循环不收敛 | 编排/运行时 | 循环预算、错误指纹、worker 隔离、可观测失败 |
| SQL 合法但语义错误 | 结果校验/模型路由 | 结果断言、分级修复、失败不伪成功 |

## 证据索引

- 材料多轮历史失败：`eval/harness_runs/resin_complex_projectionfix_20260826/case_results/`
- 材料修复后专项：`eval/runs/resin_complex_postfix_20260826/report.md`
- 钢铁复杂失败：`eval/harness_runs/resin-generalization-20260826/steel_complex/case_results/`
- 钢铁复杂汇总：`eval/harness_runs/resin-generalization-20260826/steel_complex/report.md`
