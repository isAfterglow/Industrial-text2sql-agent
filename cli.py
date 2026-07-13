from langgraph.errors import GraphRecursionError

from app.graph import graph


WELCOME_TEXT = """
树脂基防热材料 Text2SQL V0.4 精简通用版

流程：
1. 根据问题和Schema生成SQL；
2. 执行确定性安全与Schema检查；
3. 执行轻量语义审查；
4. 必要时自动修复一次；
5. 执行数据库查询并返回结果。

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

        try:
            result = graph.invoke(
                {
                    "question": question,
                },
                {
                    "recursion_limit": 24,
                },
            )

            print(
                "\n"
                + result["final_answer"]
            )

        except GraphRecursionError:
            print(
                "\nGraph超过最大执行步数，"
                "请检查重试路由。"
            )

        except Exception as exc:
            print(
                "\n程序运行失败："
                f"{type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()