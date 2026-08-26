# Few-shot 审核记录 V1

本次只审核可泛化的查询结构，不按评测题逐题记忆。入选案例必须满足：

- 至少 3 条独立成功验证；
- 不绑定上一轮结果集或具体样本范围；
- SQL/Plan 已通过 Guard、执行和结果契约；
- 能把参数替换为同一结构的其他字段、分组或阈值；
- Profile 与 AdvancedPlan family 明确匹配。

审核通过 8 条：钢铁 6 条、材料 2 条，见 `eval/audited_few_shots_v1.json`。钢铁覆盖分组 Top-K、双维度窗口、组内阈值、累计贡献、相关性、条件对比；材料覆盖时序初末点和一对一跨表 Top-K。

这些记录只作为结构示例，不能复制历史 sample_id、LIMIT、业务参数或结果。候选记忆仍采用 `pending_review -> active` 生命周期。

