# Production Runbook

## Worker and SSE

本地开发可使用 `TASK_QUEUE_MODE=local`，API 进程内通过线程池执行任务；生产环境
使用 `TASK_QUEUE_MODE=redis`，启动 RQ worker：

```bash
python -m app.worker
```

提交任务后返回 `task_id`，通过以下接口轮询或接收 SSE：

```text
GET /api/tasks/{task_id}
GET /api/tasks/{task_id}/events?after=0
POST /api/tasks/{task_id}/cancel
```

任务节点事件持久化在 `agent_task_events`，事件序列在 SQLite 事务中分配，支持
API 和 Worker 并发写入。

## Observability

```text
GET /metrics
GET /api/tasks/{task_id}/trace
```

`/metrics` 暴露任务状态、端到端延迟、节点延迟、模型调用和队列深度；Trace 接口
保留完整节点时间线和失败节点。生产部署时应只允许内网 Prometheus 抓取 `/metrics`。

## Regression CI

Pull Request CI 运行完整离线 pytest 套件和评测契约门禁；需要真实数据库和 Ollama
的 120 题回归由 self-hosted workflow 执行，并通过 `eval.harness gate` 检查通过率、
安全题、超时和 worker 错误。

## Agent loop budgets

除了 LangGraph recursion limit，运行时还使用显式预算配置：

```text
AGENT_MAX_GRAPH_STEPS=32
AGENT_MAX_SAME_ERROR_REPEATS=2
AGENT_MAX_TOTAL_MODEL_TIME_SECONDS=90
```

达到预算时记录 `non_convergent_plan`/`agent_budget_exhausted`，停止重复
修复并保留 Trace，不把循环耗尽伪装成 SQL 成功。

## Model usage and cost telemetry

统一模型路由记录模型角色、用途、队列等待、耗时、prompt/completion/total
tokens。API provider 返回 usage 时使用服务端数值；本地 Ollama 未返回 usage
时记录估算值并标记 `tokens_estimated=true`。Prometheus 暴露按角色和估算标记
聚合的 token counter。

## Database read-only check

```bash
PYTHONPATH=. python tools/check_db_readonly.py
```

该脚本检查两个 Profile 的当前数据库用户和 `SHOW GRANTS`，发现 `INSERT/UPDATE/DELETE`
等权限时返回非零退出码。当前配置用户仅拥有两个业务库的 `SELECT, SHOW VIEW` 权限。
