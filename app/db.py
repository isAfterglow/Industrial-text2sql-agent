from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

from app.config import get_settings
from app.schema import _load_profile, active_profile_name


@lru_cache(maxsize=4)
def get_engine(profile_name: str | None = None) -> Engine:
    """创建并缓存 SQLAlchemy Engine。"""

    settings = get_settings()
    profile = _load_profile(profile_name or active_profile_name())

    database_url = URL.create(
        drivername="mysql+pymysql",
        username=settings.RESIN_DB_USER,
        password=settings.RESIN_DB_PASSWORD,
        host=settings.RESIN_DB_HOST,
        port=settings.RESIN_DB_PORT,
        database=profile.get("database_name", settings.RESIN_DB_NAME),
        query={"charset": "utf8mb4"},
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={
            "connect_timeout": 5,
            "read_timeout": settings.SQL_TIMEOUT_SECONDS,
            "write_timeout": settings.SQL_TIMEOUT_SECONDS,
        },
    )


def serialize_value(value: Any) -> Any:
    """将数据库值转换成适合状态保存和终端输出的类型。"""

    if value is None:
        return None

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return value


def ping_database() -> dict[str, Any]:
    """测试数据库连接。"""

    # The service can process resin and steel tasks in the same process.  Pass
    # the active Profile explicitly so the LRU cache cannot reuse an engine
    # created earlier under the implicit ``None`` key.
    engine = get_engine(active_profile_name())

    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    DATABASE() AS database_name,
                    VERSION() AS mysql_version
                """
            )
        ).mappings().one()

    return dict(row)


def execute_readonly_query(
    sql: str,
    max_rows: int,
) -> dict[str, Any]:
    """执行已经通过安全检查的只读 SQL。"""

    engine = get_engine(active_profile_name())

    with engine.connect() as connection:
        result = connection.execute(text(sql))

        columns = list(result.keys())

        # 多读取一行，用来判断是否发生结果截断。
        raw_rows = result.fetchmany(max_rows + 1)
        truncated = len(raw_rows) > max_rows
        raw_rows = raw_rows[:max_rows]

        rows = [
            [serialize_value(value) for value in row]
            for row in raw_rows
        ]

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
