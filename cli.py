import time

from langgraph.errors import GraphRecursionError

from app.graph import graph
from app.long_term_memory import get_long_term_memory_service
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
树脂基防热材料 Text2SQL V0.8.2 常见时序增强与结构感知Few-shot版

主流程：
1. Schema加载、样本编号规范化与只读安全预检查；
2. 语义长期记忆在QueryDelta前按需检索，可把领域别名改写为标准字段术语；
3. QueryDelta描述样本范围、返回字段、过滤、排序、时序聚合和数量变化；
4. 短期记忆保存最近输入、最后成功QuerySpec、单/多样本锚点和当前/父结果集合；
5. 完整新查询优先使用当前QuerySpec，显式代词才触发历史锚点；
6. 真正存在多种解释时进入澄清，支持取消、重试上限和新问题打断；
7. 复杂查询先以BGE-M3粗召回情节案例，再按QuerySignature硬过滤、结构重排和MMR选择最多2条Few-shot；
8. 没有结构兼容案例时拒绝注入Few-shot，避免语言相似但SQL结构不相似造成负迁移；
9. INITIAL、FINAL、峰值、质量损失率、背温抬升和字段间比较走确定性扩展编译路径；
10. 单句出现“前N个中再取前M个”等多阶段Top-K时停止猜测，并提示拆成多轮查询；
11. 简单查询走确定性SQL快路径，其他复杂查询走完整Schema与稳健裁剪Schema双候选；
12. SQL统一经过只读、Schema、字段归属、Top-K、聚合和样本范围Guard；
13. Guard或执行失败时检索程序性记忆，为一次自动修复提供经验提示；
14. 只有语义覆盖、Guard和数据库执行均成功后才更新短期记忆与高价值长期案例。

短期记忆命令：
- /memory：查看当前会话短期记忆；
- /reset：清空短期记忆，保留session_id；
- /new：创建新会话，长期记忆仍保留；
- /取消澄清：取消待澄清问题。

长期记忆命令：
- /remember 术语 -> 字段或标准术语
  示例：/remember 生料热导率 -> kv_list
- /remember-case：保存最后一次成功查询为情节记忆；
- /memories：查看全部长期记忆；
- /memories semantic|episodic|procedural：按类型查看；
- /approvals [pending]：查看本Profile的人工审批队列；
- /approval <id>：查看审批请求详情；
- /forget <memory_id或唯一前缀>：停用一条长期记忆；
- /ltm-status：查看SQLite、Schema版本与Embedding状态。

退出：exit、quit、q
""".strip()


def _handle_long_term_memory_command(
    question: str,
    conversation_memory: dict,
) -> bool:
    """处理长期记忆命令；已处理返回True。"""

    service = get_long_term_memory_service()
    lowered = question.lower()

    if lowered == "/ltm-status":
        print("\n" + service.status_summary())
        return True

    if lowered == "/memories":
        print("\n" + service.format_list(service.list_memories(limit=100)))
        return True

    if lowered.startswith("/memories "):
        memory_type = question.split(maxsplit=1)[1].strip().lower()
        try:
            records = service.list_memories(memory_type=memory_type, limit=100)
            print("\n" + service.format_list(records))
        except ValueError as exc:
            print(f"\n{exc}")
        return True

    if lowered.startswith("/remember "):
        payload = question.split(maxsplit=1)[1].strip()
        try:
            result = service.remember_semantic(payload)
            action = "新增" if result.created else "更新"
            print(
                f"\n已{action}语义记忆：{result.record.memory_id}\n"
                f"{result.record.content}"
            )
        except ValueError as exc:
            print(f"\n{exc}")
        return True

    if lowered == "/approvals" or lowered.startswith("/approvals "):
        status = question.split(maxsplit=1)[1].strip() if " " in question else None
        records = service.list_approval_requests(status=status or None)
        if not records:
            print("\n当前没有匹配的审批请求。")
        else:
            for item in records:
                print(f"\n[{item['approval_id']}] {item['status']} | {item['created_at']}")
                print(str(item["payload"].get("question", "")))
        return True

    if lowered.startswith("/approval "):
        approval_id = question.split(maxsplit=1)[1].strip()
        record = service.repository.get_approval_request(approval_id)
        print("\n" + (str(record) if record else "没有找到该审批请求。"))
        return True

    if lowered == "/remember-case":
        try:
            result = service.remember_case_from_short_memory(conversation_memory)
            action = "新增" if result.created else "更新"
            print(
                f"\n已{action}情节记忆：{result.record.memory_id}\n"
                f"{result.record.title}"
            )
        except ValueError as exc:
            print(f"\n{exc}")
        return True

    if lowered.startswith("/forget "):
        prefix = question.split(maxsplit=1)[1].strip()
        success, message = service.forget(prefix)
        if success:
            print(f"\n已停用长期记忆：{message}")
        else:
            print(f"\n{message}")
        return True

    return False


def main() -> None:
    print(WELCOME_TEXT)

    long_term_memory = get_long_term_memory_service()
    print("\n长期记忆状态：")
    print(long_term_memory.status_summary())

    conversation_memory = new_short_term_memory()
    print("\n当前session_id: " + conversation_memory["session_id"])

    while True:
        print("\n" + "=" * 80)

        try:
            question = input("请输入问题：").strip()
        except (KeyboardInterrupt, EOFError):
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
            print("当前会话短期记忆已清空，长期记忆保留。")
            continue

        if lowered == "/new":
            conversation_memory = new_short_term_memory()
            print("已创建新会话，session_id: " + conversation_memory["session_id"])
            continue

        if lowered in {"/取消澄清", "/cancel", "/cancel-clarification"}:
            conversation_memory = cancel_pending_clarification(conversation_memory)
            print("当前待澄清问题已取消，成功查询记忆与长期记忆均保留。")
            continue

        if _handle_long_term_memory_command(question, conversation_memory):
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
                {"recursion_limit": 32},
            )

            updated_memory = result.get("conversation_memory")
            if isinstance(updated_memory, dict) and updated_memory:
                conversation_memory = updated_memory

            total_elapsed_ms = (time.perf_counter() - started) * 1000
            record = save_trace_record(result, total_elapsed_ms)

            print("\n" + result["final_answer"])

            retrieval = result.get("long_term_memory_retrieval_summary", {})
            write_summary = result.get("long_term_memory_write_summary", {})
            if retrieval:
                print("\n长期记忆检索：" + str(retrieval))
            if write_summary.get("saved"):
                print("长期记忆写入：" + str(write_summary["saved"]))

            print_trace_summary(record)

        except GraphRecursionError:
            total_elapsed_ms = (time.perf_counter() - started) * 1000
            print("\nGraph超过最大执行步数，请检查重试路由。")
            if trace_enabled():
                print(
                    "trace_id: "
                    f"{trace_id}; partial node events are in "
                    "logs/node_events.jsonl; "
                    f"elapsed={total_elapsed_ms:.2f} ms"
                )

        except Exception as exc:
            total_elapsed_ms = (time.perf_counter() - started) * 1000
            print(f"\n程序运行失败：{type(exc).__name__}: {exc}")
            if trace_enabled():
                print(
                    "trace_id: "
                    f"{trace_id}; partial node events are in "
                    "logs/node_events.jsonl; "
                    f"elapsed={total_elapsed_ms:.2f} ms"
                )


if __name__ == "__main__":
    main()
