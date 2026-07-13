from typing import Any, TypedDict


class Text2SQLState(TypedDict, total=False):
    """精简版Text2SQL LangGraph共享状态。"""

    # 用户输入
    question: str
    normalized_question: str

    # Schema
    schema_context: str

    # SQL
    initial_sql: str
    raw_sql: str
    validated_sql: str

    # 确定性校验
    validation_error: str
    validation_repairable: bool
    validation_error_type: str

    # 语义审查
    review_passed: bool
    review_reason: str
    review_note: str

    # 数据库执行
    execution_error: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool

    # 修复控制
    retry_count: int
    last_repair_reason: str

    # 最终输出
    final_answer: str