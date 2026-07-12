from sqlalchemy import inspect

from app.config import get_settings
from app.db import get_engine, ping_database
from app.llm import get_llm


def check_database() -> None:
    settings = get_settings()

    print("=" * 70)
    print("1. 检查数据库连接")
    print("=" * 70)

    database_info = ping_database()

    print(f"数据库：{database_info['database_name']}")
    print(f"MySQL版本：{database_info['mysql_version']}")

    inspector = inspect(get_engine())

    print("\n检查数据表：")

    for table_name in settings.allowed_tables:
        exists = inspector.has_table(table_name)

        print(f"\n- {table_name}: {'存在' if exists else '不存在'}")

        if not exists:
            continue

        columns = inspector.get_columns(table_name)

        for column in columns:
            print(
                f"    {column['name']}: "
                f"{column['type']}"
            )


def check_llm() -> None:
    print("\n" + "=" * 70)
    print("2. 检查 Ollama LLM")
    print("=" * 70)

    response = get_llm().invoke(
        "只回复字符串 OK，不要输出其他内容。"
    )

    print(f"模型返回：{response.content}")


if __name__ == "__main__":
    check_database()
    check_llm()

    print("\n环境检查完成。")