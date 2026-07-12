from langgraph.errors import GraphRecursionError

from app.graph import graph


WELCOME_TEXT = """
树脂基防热材料 Text2SQL V0.2

输入自然语言问题，系统将：
1. 生成只读SQL；
2. 执行安全与质量检查；
3. 校验或执行失败时自动修复一次；
4. 返回实际执行SQL和数据库结果。

输入 exit、quit 或 q 退出。
""".strip()


def main() -> None:
    print(WELCOME_TEXT)

    while True:
        print("\n" + "=" * 80)

        question = input(
            "请输入问题："
        ).strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("已退出。")
            break

        try:
            result = graph.invoke(
                {
                    "question": question,
                },
                {
                    # 当前流程最多只修复一次，
                    # 20步已经远高于正常执行所需步数。
                    # 这是额外的循环保护。
                    "recursion_limit": 20,
                },
            )

            print(
                "\n"
                + result["final_answer"]
            )

        except GraphRecursionError:
            print(
                "\nGraph运行超过最大步数。"
                "请检查retry_count和条件路由，"
                "避免出现无限修复循环。"
            )

        except Exception as exc:
            print(
                "\n程序运行失败："
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()