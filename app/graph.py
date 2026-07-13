from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.nodes import (
    execute_sql,
    format_error,
    format_result,
    generate_sql,
    load_schema,
    repair_sql,
    review_sql,
    route_after_execution,
    route_after_review,
    route_after_validation,
    validate_sql,
)
from app.state import Text2SQLState


def build_graph():
    """构建精简通用Text2SQL工作流。"""

    builder = StateGraph(Text2SQLState)

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
        "review_sql",
        review_sql,
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

    builder.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "review": "review_sql",
            "repair": "repair_sql",
            "error": "format_error",
        },
    )

    builder.add_conditional_edges(
        "review_sql",
        route_after_review,
        {
            "execute": "execute_sql",
            "repair": "repair_sql",
            "error": "format_error",
        },
    )

    # 修复后的SQL必须重新经过确定性Guard和语义审查。
    builder.add_edge(
        "repair_sql",
        "validate_sql",
    )

    builder.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "success": "format_result",
            "repair": "repair_sql",
            "error": "format_error",
        },
    )

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