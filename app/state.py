import operator
from typing import Annotated, Any, TypedDict


class Text2SQLState(TypedDict, total=False):
    """Text2SQL LangGraph共享状态，包含V0.8.1长期记忆、V0.7短期记忆与V0.6双候选字段。"""

    # 用户输入与会话
    question: str
    normalized_question: str
    resolved_question: str
    session_id: str
    domain_profile: str
    query_intent: str
    intent_confidence: float
    intent_evidence: list[str]
    intent_related_tables: list[str]

    # V0.7.3 澄清感知短期记忆与上下文解析
    conversation_memory: dict[str, Any]
    query_delta: dict[str, Any]
    query_delta_source: str
    query_delta_llm_called: bool
    query_delta_llm_raw_output: str
    resolved_query_spec: dict[str, Any]
    turn_type: str
    memory_used: bool
    context_resolution: dict[str, Any]
    context_resolution_valid: bool
    clarification_required: bool
    clarification_cancelled: bool
    clarification_question: str
    pending_clarification: dict[str, Any]
    policy_precheck_failed: bool
    current_turn_coverage: dict[str, Any]
    inherited_fields: list[str]
    overridden_fields: list[str]
    memory_update_summary: dict[str, Any]
    session_store_summary: dict[str, Any]


    # V0.8.1 持久化长期记忆与Few-shot检索
    memory_augmented_question: str
    long_term_memory_enabled: bool
    semantic_memory_matches: list[dict[str, Any]]
    semantic_memory_applied_ids: list[str]
    semantic_memory_hint: str
    episodic_memory_matches: list[dict[str, Any]]
    few_shot_context: str
    few_shot_retrieval_diagnostics: dict[str, Any]
    advanced_plan_memory_matches: list[dict[str, Any]]
    advanced_plan_memory_diagnostics: dict[str, Any]
    query_signature: dict[str, Any]
    procedural_memory_matches: list[dict[str, Any]]
    procedural_memory_context: str
    long_term_memory_retrieval_summary: dict[str, Any]
    long_term_memory_write_summary: dict[str, Any]

    # 基础Schema与字段提示
    schema_context: str
    field_hint: str

    # 基础查询QuerySpec与确定性快路径
    query_spec: dict[str, Any]
    query_plan_mode: str
    query_plan_reason: str
    deterministic_sql: str
    advanced_plan: dict[str, Any]
    advanced_plan_raw: str
    advanced_plan_error: str
    query_expectation: dict[str, Any]
    query_spec_json_raw: str
    query_spec_json_error: str
    unsupported_query: bool
    unsupported_query_reason: str
    unsupported_query_suggestions: list[str]

    # RSL-SQL-inspired：完整Schema候选
    full_schema_context: str
    full_generator_raw_output: str
    full_sql: str

    # 正向、反向与稳健Schema Linking
    forward_schema_tables: list[str]
    forward_schema_columns: list[str]
    backward_schema_tables: list[str]
    backward_schema_columns: list[str]
    accepted_backward_tables: list[str]
    rejected_backward_tables: list[str]
    robust_schema_context: str
    robust_schema_tables: list[str]
    robust_schema_columns: list[str]

    # RSL-SQL-inspired：稳健裁剪Schema候选
    pruned_generator_raw_output: str
    pruned_sql: str

    # 双候选Guard评估与选择
    candidate_full_valid: bool
    candidate_full_normalized_sql: str
    candidate_full_error: str
    candidate_full_error_type: str
    candidate_full_score: float
    candidate_pruned_valid: bool
    candidate_pruned_normalized_sql: str
    candidate_pruned_error: str
    candidate_pruned_error_type: str
    candidate_pruned_score: float
    selected_candidate: str
    candidate_selection_reason: str

    # 兼容后续既有节点
    generation_schema_context: str
    generation_relevant_tables: list[str]
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
    result_assertion: dict[str, Any]
    result_assertion_passed: bool

    # Human-in-the-loop approval; decisions edit plans rather than raw SQL.
    approval_required: bool
    approval_mode: str
    force_approval: bool
    approval_request: dict[str, Any]
    approval_decision: dict[str, Any]
    approval_approved: bool
    approval_summary: dict[str, Any]

    # 修复控制与修复观测
    retry_count: int
    last_repair_reason: str
    repair_source: str
    repair_action: str
    repair_bad_sql: str
    repair_raw_output: str
    repair_model_role: str
    repair_plan_mode: str

    # Stable, append-only failure telemetry for routing and evaluation.
    failure_events: Annotated[list[dict[str, Any]], operator.add]
    model_calls: list[dict[str, Any]]

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
