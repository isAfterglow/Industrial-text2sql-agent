import operator
from typing import Annotated, Any, TypedDict


class Text2SQLState(TypedDict, total=False):
    """Text2SQL LangGraph共享状态，包含V0.5可观测字段。"""

    # 用户输入
    question: str
    normalized_question: str

    # Schema
    schema_context: str
    generation_schema_context: str
    generation_relevant_tables: list[str]
    field_hint: str

    # SQL生成
    generator_raw_output: str
    initial_sql: str
    raw_sql: str
    validated_sql: str

    # 确定性校验
    validation_error: str
    validation_repairable: bool
    validation_error_type: str

    # 语义审查
    review_called: bool
    review_passed: bool
    review_reason: str
    review_note: str
    review_input_summary: str

    # 数据库执行
    execution_error: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool

    # 修复控制与修复观测
    retry_count: int
    last_repair_reason: str
    repair_source: str
    repair_action: str
    repair_bad_sql: str
    repair_raw_output: str

    # Trace
    trace_id: str
    trace_started_at: str
    trace_events: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    # 最终输出
    final_status: str
    final_answer: str