from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from app.nodes import (
    build_query_plan,
    approval_gate,
    build_robust_schema,
    execute_sql,
    extract_query_delta,
    hydrate_session_memory,
    policy_precheck,
    retrieve_semantic_memory,
    retrieve_few_shot_memory,
    route_after_policy_precheck,
    request_clarification,
    route_after_context_resolution,
    format_error,
    format_result,
    format_unsupported_query,
    generate_full_sql,
    generate_structured_query_spec,
    generate_simple_sql,
    generate_pruned_sql,
    identify_query_intent,
    load_schema,
    repair_sql,
    resolve_conversation_context,
    route_after_query_plan,
    route_after_structured_query_spec,
    regenerate_advanced_plan,
    route_after_regenerated_plan,
    review_sql,
    route_after_execution,
    route_after_result_assertions,
    route_after_review,
    route_after_validation,
    route_after_approval,
    select_sql_candidate,
    update_session_memory,
    persist_session_memory,
    update_long_term_memory,
    validate_sql,
    validate_result_assertions,
    format_approval_required,
)
from app.state import Text2SQLState
from app.trace import traced_node


def build_graph():
    """构建V0.8.2常见时序增强、结构感知Few-shot与双Schema候选工作流。"""

    builder = StateGraph(Text2SQLState)

    for node_name, node_function in (
        ("load_schema", load_schema),
        ("hydrate_session_memory", hydrate_session_memory),
        ("identify_query_intent", identify_query_intent),
        ("policy_precheck", policy_precheck),
        ("retrieve_semantic_memory", retrieve_semantic_memory),
        ("extract_query_delta", extract_query_delta),
        ("resolve_conversation_context", resolve_conversation_context),
        ("retrieve_few_shot_memory", retrieve_few_shot_memory),
        ("request_clarification", request_clarification),
        ("build_query_plan", build_query_plan),
        ("generate_simple_sql", generate_simple_sql),
        ("generate_structured_query_spec", generate_structured_query_spec),
        ("regenerate_advanced_plan", regenerate_advanced_plan),
        ("generate_full_sql", generate_full_sql),
        ("build_robust_schema", build_robust_schema),
        ("generate_pruned_sql", generate_pruned_sql),
        ("select_sql_candidate", select_sql_candidate),
        ("validate_sql", validate_sql),
        ("approval_gate", approval_gate),
        ("review_sql", review_sql),
        ("repair_sql", repair_sql),
        ("execute_sql", execute_sql),
        ("validate_result_assertions", validate_result_assertions),
        ("update_session_memory", update_session_memory),
        ("persist_session_memory", persist_session_memory),
        ("update_long_term_memory", update_long_term_memory),
        ("format_result", format_result),
        ("format_approval_required", format_approval_required),
        ("format_unsupported_query", format_unsupported_query),
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
    builder.add_edge("load_schema", "hydrate_session_memory")
    builder.add_edge("hydrate_session_memory", "identify_query_intent")
    builder.add_edge("identify_query_intent", "policy_precheck")
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
            "rsl": "generate_structured_query_spec",
            "unsupported": "format_unsupported_query",
        },
    )
    builder.add_conditional_edges(
        "generate_structured_query_spec",
        route_after_structured_query_spec,
        {
            "structured": "generate_simple_sql",
            "regenerate": "regenerate_advanced_plan",
            "sql": "generate_full_sql",
        },
    )
    builder.add_conditional_edges(
        "regenerate_advanced_plan",
        route_after_regenerated_plan,
        {"structured": "generate_simple_sql", "error": "format_error"},
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
            "review": "approval_gate",
            "repair": "repair_sql",
            "error": "format_error",
        },
    )
    builder.add_conditional_edges(
        "approval_gate",
        route_after_approval,
        {
            "review": "review_sql",
            "revalidate": "validate_sql",
            "pending": "format_approval_required",
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
            "success": "validate_result_assertions",
            "repair": "repair_sql",
            "error": "format_error",
        },
    )

    builder.add_conditional_edges(
        "validate_result_assertions",
        route_after_result_assertions,
        {
            "success": "update_session_memory",
            "repair": "repair_sql",
            "error": "format_error",
        },
    )

    builder.add_edge("update_session_memory", "persist_session_memory")
    builder.add_edge("persist_session_memory", "update_long_term_memory")
    builder.add_edge("update_long_term_memory", "format_result")
    builder.add_edge("format_result", END)
    builder.add_edge("format_approval_required", END)
    builder.add_edge("request_clarification", END)
    builder.add_edge("format_unsupported_query", END)
    builder.add_edge("format_error", END)

    return builder.compile()


graph = build_graph()
