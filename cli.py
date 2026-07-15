import time

from langgraph.errors import GraphRecursionError

from app.graph import graph
from app.memory import (
    format_memory_summary,
    new_short_term_memory,
    reset_short_term_memory,
)
from app.trace import (
    new_trace_id,
    print_trace_summary,
    save_trace_record,
    trace_enabled,
    utc_now_iso,
)


WELCOME_TEXT = """
树脂基防热材料 Text2SQL V0.7.1 QueryDelta短期记忆版

流程：
1. 保存最后一次成功QuerySpec、最近两轮成功对话和上一结果样本范围；
2. 将当前轮解析成QueryDelta：独立查询、同一样本、上一结果集合或修改上一查询；
3. 常见承接表达使用Schema字段识别和确定性算法，特别模糊时才调用一次轻量LLM；
4. LLM只提取字段替换/增加/删除与指代关系，不直接修改旧SQL；
5. 确定性State Reducer合并QueryDelta，并重新推导表、查询类型和SQL路径；
6. 当前轮字段、数量和样本范围必须通过Coverage检查，防止错误状态与错误SQL互相验证；
7. SQL继续经过原有只读策略、Schema、字段、数值、Top-K和聚合Guard；
8. 仅在语义覆盖与数据库执行均成功后更新记忆，失败轮次保留上一成功状态。

会话命令：
- /memory：查看当前结构化短期记忆；
- /reset：清空当前会话记忆，但保留session_id；
- /new：创建全新会话；
- exit、quit、q：退出。

推荐多轮示例：
- 查询样本305的热解热。
- 它的碳化密度是多少？
- 再返回表面发射率。

- 找出表面发射率最低的5个样本。
- 这些样本中碳化密度最高的是哪个？

默认日志：
- logs/node_events.jsonl：每次节点执行记录；
- logs/traces.jsonl：每次完整查询记录；
- logs/errors.jsonl：失败或安全拒绝记录。
""".strip()


def main() -> None:
    print(WELCOME_TEXT)
    conversation_memory = new_short_term_memory()
    print(
        "\n当前session_id: "
        + conversation_memory["session_id"]
    )

    while True:
        print("\n" + "=" * 80)

        try:
            question = input("请输入问题：").strip()
        except KeyboardInterrupt:
            print("\n已退出。")
            break

        if not question:
            continue

        lowered = question.lower()
        if lowered in {"exit", "quit", "q"}:
            print("已退出。")
            break

        if lowered == "/memory":
            print("\n" + format_memory_summary(conversation_memory))
            continue

        if lowered == "/reset":
            conversation_memory = reset_short_term_memory(
                conversation_memory,
                keep_session_id=True,
            )
            print("当前会话短期记忆已清空。")
            continue

        if lowered == "/new":
            conversation_memory = new_short_term_memory()
            print(
                "已创建新会话，session_id: "
                + conversation_memory["session_id"]
            )
            continue

        trace_id = new_trace_id()
        started_at = utc_now_iso()
        started = time.perf_counter()

        try:
            result = graph.invoke(
                {
                    "question": question,
                    "session_id": conversation_memory["session_id"],
                    "conversation_memory": conversation_memory,
                    "trace_id": trace_id,
                    "trace_started_at": started_at,
                    "trace_events": [],
                },
                {
                    "recursion_limit": 28,
                },
            )

            # 只有成功路径会经过update_session_memory；失败结果保留旧记忆。
            updated_memory = result.get("conversation_memory")
            if isinstance(updated_memory, dict) and updated_memory:
                conversation_memory = updated_memory

            total_elapsed_ms = (
                time.perf_counter() - started
            ) * 1000
            record = save_trace_record(
                result,
                total_elapsed_ms,
            )

            print("\n" + result["final_answer"])
            print_trace_summary(record)

        except GraphRecursionError:
            total_elapsed_ms = (
                time.perf_counter() - started
            ) * 1000
            print(
                "\nGraph超过最大执行步数，请检查重试路由。"
            )
            if trace_enabled():
                print(
                    "trace_id: "
                    f"{trace_id}; partial node events are in "
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
                    f"{trace_id}; partial node events are in "
                    "logs/node_events.jsonl; "
                    f"elapsed={total_elapsed_ms:.2f} ms"
                )


if __name__ == "__main__":
    main()