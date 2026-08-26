# Memory Reliability V2

本轮记忆优化聚焦四个工程边界：

## 1. Anchor 原生编译

`QuerySpec.scope` 是多轮实体范围的规范表示。确定性编译器统一校验：

- `status=active`
- 当前 Profile
- 当前 Schema hash
- `entity_type=sample`、`entity_key=sample_id`

校验失败时编译器 fail closed，返回空 SQL，由上层进入澄清或重新规划。旧
`sample_ids` 只作为旧状态兼容输入。

## 2. Anchor 存储策略

小结果集使用 `storage_mode=inline_ids`，在 Anchor 中保存有上限的有序 ID。
超过上限或数据库返回截断结果时使用 `storage_mode=deferred_result_set`，保存
`lookup_key=anchor_id` 和 `entity_count`，并设置 `resolution_required=true`。
截断 Anchor 不允许直接作为后续完整集合执行范围，避免静默丢数据。

## 3. 记忆质量指标

Few-shot 检索在 State 和 Trace 中记录：

- `retrieval_latency_ms`
- `candidate_count`
- `compatible_count`
- `selected_count`
- `rejected_count`
- `useful_candidate`

原有诊断还保留每类拒绝原因、Schema/结构/质量分解和最终注入 ID。

## 4. 回归覆盖

`eval/benchmark_memory_regression.py` 扩展到 10 个多轮场景，覆盖单/多实体
引用、条件叠加、独立查询防泄漏、截断结果、展示列隔离、跨 Profile、父子
Anchor 和冲突澄清。

验证命令：

```bash
PYTHONPATH=. pytest -q
```

当前结果：19 passed（4 个 JWT 测试警告，不影响功能）。
