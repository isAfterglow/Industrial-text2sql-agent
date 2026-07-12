from langgraph.graph import END, START, StateGraph

from app.nodes import (
    execute_sql,
    format_error,
    format_result,
    generate_sql,
    load_schema,
    repair_sql,
    route_after_execution,
    route_after_validation,
    validate_sql,
)
from app.state import Text2SQLState


def build_graph():
    """构建Text2SQL V0.2 LangGraph。

    V0.2新增：
    1. repair_sql节点；
    2. validate_sql失败后的修复回路；
    3. execute_sql失败后的修复回路；
    4. retry_count控制的终止条件。
    """

    builder = StateGraph(Text2SQLState)

    # ==================================================
    # 注册节点
    # ==================================================

    builder.add_node(
        "load_schema",
        load_schema,
    )

    builder.add_node(
        "generate_sql",
        generate_sql,
    )

    builder.add_node(
        "validate_sql",
        validate_sql,
    )

    builder.add_node(
        "repair_sql",
        repair_sql,
    )

    builder.add_node(
        "execute_sql",
        execute_sql,
    )

    builder.add_node(
        "format_result",
        format_result,
    )

    builder.add_node(
        "format_error",
        format_error,
    )

    # ==================================================
    # 主流程
    # ==================================================

    builder.add_edge(
        START,
        "load_schema",
    )

    builder.add_edge(
        "load_schema",
        "generate_sql",
    )

    builder.add_edge(
        "generate_sql",
        "validate_sql",
    )

    # ==================================================
    # SQL校验后的条件路由
    # ==================================================

    builder.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            # 校验成功，执行SQL
            "execute": "execute_sql",

            # 校验失败但还有修复次数
            "repair": "repair_sql",

            # 已达到最大修复次数
            "error": "format_error",
        },
    )

    # 修复后的SQL必须重新经过完整SQL Guard，
    # 不能直接进入数据库执行。
    builder.add_edge(
        "repair_sql",
        "validate_sql",
    )

    # ==================================================
    # SQL执行后的条件路由
    # ==================================================

    builder.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            # 数据库执行成功
            "success": "format_result",

            # 执行失败但还有修复次数
            "repair": "repair_sql",

            # 执行失败且已经达到修复上限
            "error": "format_error",
        },
    )

    # ==================================================
    # 结束节点
    # ==================================================

    builder.add_edge(
        "format_result",
        END,
    )

    builder.add_edge(
        "format_error",
        END,
    )

    return builder.compile()


graph = build_graph()