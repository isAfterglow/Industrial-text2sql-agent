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
from app.trace import traced_node


def build_graph():
    """构建带统一节点Trace的Text2SQL工作流。"""

    builder = StateGraph(Text2SQLState)

    builder.add_node(
        "load_schema",
        traced_node(
            "load_schema",
            load_schema,
        ),
    )
    builder.add_node(
        "generate_sql",
        traced_node(
            "generate_sql",
            generate_sql,
        ),
    )
    builder.add_node(
        "validate_sql",
        traced_node(
            "validate_sql",
            validate_sql,
        ),
    )
    builder.add_node(
        "review_sql",
        traced_node(
            "review_sql",
            review_sql,
        ),
    )
    builder.add_node(
        "repair_sql",
        traced_node(
            "repair_sql",
            repair_sql,
        ),
    )
    builder.add_node(
        "execute_sql",
        traced_node(
            "execute_sql",
            execute_sql,
        ),
    )
    builder.add_node(
        "format_result",
        traced_node(
            "format_result",
            format_result,
        ),
    )
    builder.add_node(
        "format_error",
        traced_node(
            "format_error",
            format_error,
        ),
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