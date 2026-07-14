import time

from langgraph.errors import GraphRecursionError

from app.graph import graph
from app.trace import (
    new_trace_id,
    print_trace_summary,
    save_trace_record,
    trace_enabled,
    utc_now_iso,
)


WELCOME_TEXT = """
树脂基防热材料 Text2SQL V0.6.4 Guard约束一致性版

流程：
1. 规范化问题，统一解析数值并构建静态QuerySpec或TemporalQuerySpec；
2. 可确定的静态查询、时序明细和标准时序聚合走确定性SQL快路径；
3. 无法完整规划的复杂查询进入RSL-SQL-inspired双候选链路；
4. 完整Schema生成SQL1，并进行正向/反向Schema Linking；
5. 置信度过滤反向噪声，形成表级和列级稳健裁剪Schema；
6. 裁剪Schema生成SQL2，并通过Guard与结构评分选择候选；
7. 执行统一安全、Schema、投影、数值、Top-K和聚合校验；
8. 必要时轻量语义审查并最多自动修复一次；
9. 执行数据库查询，输出节点输入、输出、耗时和JSONL日志。

默认日志：
- logs/node_events.jsonl：每次节点执行记录；
- logs/traces.jsonl：每次完整查询记录；
- logs/errors.jsonl：失败或安全拒绝记录。

环境变量：
- TEXT2SQL_TRACE_ENABLED=0：关闭Trace；
- TEXT2SQL_TRACE_CONSOLE=0：不在终端打印Trace；
- TEXT2SQL_TRACE_VERBOSE=1：终端显示更完整的节点输出；
- TEXT2SQL_TRACE_LOG_DIR=路径：修改日志目录。

输入 exit、quit 或 q 退出。
""".strip()


def main() -> None:
    print(WELCOME_TEXT)

    while True:
        print("\n" + "=" * 80)

        try:
            question = input(
                "请输入问题："
            ).strip()
        except KeyboardInterrupt:
            print("\n已退出。")
            break

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("已退出。")
            break

        trace_id = new_trace_id()
        started_at = utc_now_iso()
        started = time.perf_counter()

        try:
            result = graph.invoke(
                {
                    "question": question,
                    "trace_id": trace_id,
                    "trace_started_at": (
                        started_at
                    ),
                    "trace_events": [],
                },
                {
                    "recursion_limit": 24,
                },
            )

            total_elapsed_ms = (
                time.perf_counter() - started
            ) * 1000
            record = save_trace_record(
                result,
                total_elapsed_ms,
            )

            print(
                "\n"
                + result["final_answer"]
            )
            print_trace_summary(record)

        except GraphRecursionError:
            total_elapsed_ms = (
                time.perf_counter() - started
            ) * 1000
            print(
                "\nGraph超过最大执行步数，"
                "请检查重试路由。"
            )
            if trace_enabled():
                print(
                    "trace_id: "
                    f"{trace_id}; "
                    "partial node events are in "
                    "logs/node_events.jsonl; "
                    f"elapsed={total_elapsed_ms:.2f} ms"
                )

        except Exception as exc:
            total_elapsed_ms = (
                time.perf_counter() - started
            ) * 1000
            print(
                "\n程序运行失败："
                f"{type(exc).__name__}: {exc}"
            )
            if trace_enabled():
                print(
                    "trace_id: "
                    f"{trace_id}; "
                    "partial node events are in "
                    "logs/node_events.jsonl; "
                    f"elapsed={total_elapsed_ms:.2f} ms"
                )


if __name__ == "__main__":
    main()