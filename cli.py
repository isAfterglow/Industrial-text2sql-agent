import time

from langgraph.errors import GraphRecursionError

from app.graph import graph
from app.memory import (
    cancel_pending_clarification,
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
树脂基防热材料 Text2SQL V0.7.4 指代安全短期记忆版

流程：
1. 最近两轮原始用户输入用于理解“它、它们、这些样本、其中”等指代，成功和失败轮次都保留；
2. 最后一次成功QuerySpec单独保存，失败查询不会覆盖可执行状态；
3. 结果范围保存当前集合和父集合两层，支持从Top-1继续回到原候选集合；
4. QueryDelta分别描述样本范围、返回字段、过滤、排序、时序聚合和数量变化；
5. 引用上一结果集合时只继承sample_id范围，不继承旧过滤、排序、聚合和LIMIT；
6. State Reducer按固定继承矩阵合并状态，并重新推导表、查询类型和SQL路径；
7. Coverage同时检查当前字段是否缺失和旧状态是否残留；
8. 结果集合保存原始顺序；只换展示字段时保持原顺序，明确最高/最低时才重新排序；
9. “它”回指最近单样本，“它们/这些样本/其中”回指最近样本集合；单样本和样本集合锚点互不覆盖；
10. 澄清最多允许两次无效回答，可输入/取消澄清退出；输入新的完整查询会自动放弃旧澄清；
11. 写入、删除和修改请求在记忆解析前直接按只读策略拒绝。

会话命令：
- /memory：查看成功查询状态、当前/父结果集合和最近两轮原始输入；
- /reset：清空当前会话短期记忆，但保留session_id；
- /new：创建全新会话；
- /取消澄清：只取消当前待澄清问题，不清空成功查询记忆；
- 出现澄清选项时，可直接输入A/B/C/D/E、补充明确字段/样本编号，或输入“取消”；
- exit、quit、q：退出。

重点回归示例：
- 查询原始密度最低的5个样本。
- 改成最高的10个。
- 数量改为3个。

- 找出峰值表面温度最高的10个样本。
- 这些样本中原始密度最高的是哪个？
- 再找出其中碳化密度最低的3个。

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

        if lowered in {"/取消澄清", "/cancel", "/cancel-clarification"}:
            conversation_memory = cancel_pending_clarification(conversation_memory)
            print("当前待澄清问题已取消，最后一次成功查询记忆仍保留。")
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

            # 成功路径提交QuerySpec；失败路径也会保留最近两轮原始输入。
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