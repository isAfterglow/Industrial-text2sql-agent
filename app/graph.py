from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.nodes import (
    build_query_plan,
    build_robust_schema,
    execute_sql,
    extract_query_delta,
    policy_precheck,
    retrieve_semantic_memory,
    retrieve_few_shot_memory,
    route_after_policy_precheck,
    request_clarification,
    route_after_context_resolution,
    format_error,
    format_result,
    generate_full_sql,
    generate_simple_sql,
    generate_pruned_sql,
    load_schema,
    repair_sql,
    resolve_conversation_context,
    route_after_query_plan,
    review_sql,
    route_after_execution,
    route_after_review,
    route_after_validation,
    select_sql_candidate,
    update_session_memory,
    update_long_term_memory,
    validate_sql,
)
from app.state import Text2SQLState
from app.trace import traced_node


def build_graph():
    """构建V0.8.1长期记忆、短期记忆与双Schema候选工作流。"""

    builder = StateGraph(Text2SQLState)

    for node_name, node_function in (
        ("load_schema", load_schema),
        ("policy_precheck", policy_precheck),
        ("retrieve_semantic_memory", retrieve_semantic_memory),
        ("extract_query_delta", extract_query_delta),
        ("resolve_conversation_context", resolve_conversation_context),
        ("retrieve_few_shot_memory", retrieve_few_shot_memory),
        ("request_clarification", request_clarification),
        ("build_query_plan", build_query_plan),
        ("generate_simple_sql", generate_simple_sql),
        ("generate_full_sql", generate_full_sql),
        ("build_robust_schema", build_robust_schema),
        ("generate_pruned_sql", generate_pruned_sql),
        ("select_sql_candidate", select_sql_candidate),
        ("validate_sql", validate_sql),
        ("review_sql", review_sql),
        ("repair_sql", repair_sql),
        ("execute_sql", execute_sql),
        ("update_session_memory", update_session_memory),
        ("update_long_term_memory", update_long_term_memory),
        ("format_result", format_result),
        ("format_error", format_error),
    ):
        builder.add_node(
            node_name,
            traced_node(
                node_name,
                node_function,
            ),
        )

    builder.add_edge(START, "load_schema")
    builder.add_edge("load_schema", "policy_precheck")
    builder.add_conditional_edges(
        "policy_precheck",
        route_after_policy_precheck,
        {"continue": "retrieve_semantic_memory", "error": "format_error"},
    )
    builder.add_edge("retrieve_semantic_memory", "extract_query_delta")
    builder.add_edge("extract_query_delta", "resolve_conversation_context")
    builder.add_conditional_edges(
        "resolve_conversation_context",
        route_after_context_resolution,
        {"continue": "retrieve_few_shot_memory", "clarify": "request_clarification"},
    )
    builder.add_edge("retrieve_few_shot_memory", "build_query_plan")
    builder.add_conditional_edges(
        "build_query_plan",
        route_after_query_plan,
        {
            "simple": "generate_simple_sql",
            "rsl": "generate_full_sql",
        },
    )
    builder.add_edge("generate_simple_sql", "validate_sql")
    builder.add_edge(
        "generate_full_sql",
        "build_robust_schema",
    )
    builder.add_edge(
        "build_robust_schema",
        "generate_pruned_sql",
    )
    builder.add_edge(
        "generate_pruned_sql",
        "select_sql_candidate",
    )
    builder.add_edge(
        "select_sql_candidate",
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

    builder.add_edge("repair_sql", "validate_sql")

    builder.add_conditional_edges(
        "execute_sql",
        route_after_execution,
        {
            "success": "update_session_memory",
            "repair": "repair_sql",
            "error": "format_error",
        },
    )

    builder.add_edge("update_session_memory", "update_long_term_memory")
    builder.add_edge("update_long_term_memory", "format_result")
    builder.add_edge("format_result", END)
    builder.add_edge("request_clarification", END)
    builder.add_edge("format_error", END)

    return builder.compile()


graph = build_graph()