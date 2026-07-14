import re
from typing import Any, Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)
from tabulate import tabulate

from app.config import get_settings
from app.db import execute_readonly_query
from app.llm import get_llm
from app.schema import (
    build_compact_sql_context,
    build_query_spec,
    compile_query_spec_sql,
    build_question_field_hint,
    build_robust_schema_linking,
    build_schema_context,
    extract_requested_sample_ids,
    extract_sql_schema_elements,
    infer_relevant_tables,
    normalize_question_sample_ids,
)
from app.sql_guard import (
    assess_deterministic_semantic_coverage,
    build_sql_review_summary,
    clean_llm_sql,
    normalize_sample_id_literals,
    validate_and_normalize_sql,
    validate_question_policy,
)
from app.state import Text2SQLState


def message_content_to_text(content: Any) -> str:
    """兼容常见OpenAI-compatible接口的content格式。"""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)

        if parts:
            return "\n".join(parts)

    return str(content)


def invoke_text(
    system_prompt: str,
    user_prompt: str,
) -> str:
    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    return message_content_to_text(
        response.content
    ).strip()


def parse_review_line(
    text: str,
) -> tuple[bool | None, str]:
    """只接受一条PASS:原因或FAIL:原因。"""

    cleaned = text.strip()
    code_block = re.search(
        r"```(?:text)?\s*(.*?)```",
        cleaned,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if code_block:
        cleaned = code_block.group(1).strip()

    status_matches = list(
        re.finditer(
            r"(?:^|\n)\s*(PASS|FAIL)\s*(?::|：)\s*([^\n]+)",
            cleaned,
            flags=re.IGNORECASE,
        )
    )

    statuses = {
        match.group(1).upper()
        for match in status_matches
    }

    if len(status_matches) != 1 or len(statuses) != 1:
        return None, "审查器没有返回唯一可信的PASS/FAIL结论。"

    match = status_matches[0]
    reason = match.group(2).strip()

    if not reason:
        reason = "未提供具体原因。"

    return (
        match.group(1).upper() == "PASS",
        reason,
    )


def review_complex_sql(
    question: str,
    schema_context: str,
    sql: str,
) -> tuple[bool | None, str]:
    """仅审查确定性规则无法充分覆盖的复杂SQL。"""

    summary = build_sql_review_summary(sql)

    system_prompt = """
你只负责审查一条复杂SQL是否满足用户问题。

优先核对：
1. SELECT实际返回字段；
2. WHERE实际过滤字段和值；
3. ORDER BY实际排序字段和方向；
4. 聚合函数与GROUP BY；
5. 是否因多余的一对多JOIN产生重复行。

不要因为SQL写法不够简洁而判错。
不要把ORDER BY字段误认为SELECT返回字段。
不得要求增加用户问题中没有出现的数值、过滤条件或业务约束。
如果SQL已经完整包含用户明确给出的条件，不得自行补充“>0”等条件。

只能输出一行：
PASS: 简短原因
或
FAIL: 简短且具体的原因
""".strip()

    user_prompt = f"""
数据库Schema：
{schema_context}

用户问题：
{question}

候选SQL：
{sql}

{summary}
""".strip()

    result = parse_review_line(
        invoke_text(
            system_prompt,
            user_prompt,
        )
    )

    if result[0] is not None:
        return result

    retry = invoke_text(
        "只能输出一行PASS:原因或FAIL:原因。",
        f"用户问题：{question}\nSQL：{sql}",
    )
    return parse_review_line(retry)


def load_schema(
    state: Text2SQLState,
) -> dict[str, Any]:
    return {
        "normalized_question": (
            normalize_question_sample_ids(
                state["question"]
            )
        ),
        "schema_context": build_schema_context(),
        "query_spec": {},
        "query_plan_mode": "",
        "query_plan_reason": "",
        "deterministic_sql": "",
        "full_schema_context": "",
        "full_generator_raw_output": "",
        "full_sql": "",
        "forward_schema_tables": [],
        "forward_schema_columns": [],
        "backward_schema_tables": [],
        "backward_schema_columns": [],
        "accepted_backward_tables": [],
        "rejected_backward_tables": [],
        "robust_schema_context": "",
        "robust_schema_tables": [],
        "robust_schema_columns": [],
        "pruned_generator_raw_output": "",
        "pruned_sql": "",
        "candidate_full_valid": False,
        "candidate_full_normalized_sql": "",
        "candidate_full_error": "",
        "candidate_full_error_type": "",
        "candidate_full_score": 0.0,
        "candidate_pruned_valid": False,
        "candidate_pruned_normalized_sql": "",
        "candidate_pruned_error": "",
        "candidate_pruned_error_type": "",
        "candidate_pruned_score": 0.0,
        "selected_candidate": "",
        "candidate_selection_reason": "",
        "generation_schema_context": "",
        "generation_relevant_tables": [],
        "field_hint": "",
        "generator_raw_output": "",
        "initial_sql": "",
        "raw_sql": "",
        "validated_sql": "",
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "retry_count": 0,
        "last_repair_reason": "",
        "repair_source": "",
        "repair_action": "",
        "repair_bad_sql": "",
        "repair_raw_output": "",
        "final_status": "",
        "final_answer": "",
    }


def build_query_plan(
    state: Text2SQLState,
) -> dict[str, Any]:
    """构建基础查询QuerySpec，决定快路径或RSL路径。"""

    spec = build_query_spec(
        state["normalized_question"]
    )
    deterministic_sql = (
        compile_query_spec_sql(spec)
        if spec.get("eligible")
        else ""
    )
    return {
        "query_spec": spec,
        "query_plan_mode": spec.get("mode", "rsl"),
        "query_plan_reason": spec.get("reason", ""),
        "deterministic_sql": deterministic_sql,
    }


def route_after_query_plan(
    state: Text2SQLState,
) -> Literal["simple", "rsl"]:
    if (
        state.get("query_plan_mode") == "deterministic"
        and state.get("deterministic_sql")
    ):
        return "simple"
    return "rsl"


def generate_simple_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """将可信QuerySpec编译结果送入统一Guard。"""

    sql = state.get("deterministic_sql", "")
    return {
        "selected_candidate": "deterministic",
        "candidate_selection_reason": state.get(
            "query_plan_reason", "基础查询确定性快路径。"
        ),
        "generator_raw_output": "",
        "initial_sql": sql,
        "raw_sql": sql,
        "validated_sql": "",
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
    }


def _generation_system_prompt(
    candidate_mode: str,
) -> str:
    mode_instruction = (
        "当前提供的是完整Schema。先保证字段和表选择正确，"
        "仍然只使用回答问题所需的最少数据表。"
        if candidate_mode == "full"
        else
        "当前提供的是正向与反向Schema Linking合并后的稳健裁剪Schema。"
        "严格限制在该Schema中生成更聚焦的候选SQL。"
    )

    return f"""
你是材料数据库Text2SQL生成器。

{mode_instruction}

生成一条MySQL只读SELECT查询。

要求：
1. 只输出SQL；
2. 使用Schema中的真实表名和真实字段；
3. ms、mtp、tr只能作为真实表名之后的别名；
4. 严格使用提供的业务字段对应关系；
5. 只返回用户要求的字段，只使用必要数据表；
6. 不增加用户未要求的LIKE、IS NOT NULL或其他过滤；
7. 普通样本级Top-K直接返回sample_id和所需字段，使用ORDER BY目标字段加LIMIT，不使用MAX、GROUP BY或IN子查询；
8. 只有一个样本多行的时序峰值才使用MAX并按sample_id分组；
9. 峰值查询不得用固定point_index代替完整序列峰值；
10. 时序明细不聚合；
11. 用户明确指定样本编号时，必须使用对应sample_id等值或IN过滤；
12. 用户没有指定样本编号时，禁止添加任何固定sample_id过滤；
13. 固定样本查询禁止使用LIKE；
14. 用户明确请求某个白名单表全部数据时，只查询该表并显式列出全部字段；
15. 禁止SELECT *、写操作和跨库查询；
16. 科学计数法是一个完整数值，例如2e-12不得拆成2和12；
17. 用户没有要求数量时，不得自行添加LIMIT 1或其他限制性LIMIT，系统会统一添加资源上限。
""".strip()


def _generate_candidate_sql(
    question: str,
    schema_context: str,
    field_hint: str,
    candidate_mode: str,
) -> tuple[str, str]:
    user_prompt = f"""
数据库Schema：
{schema_context}

用户问题：
{question}

{field_hint}

只输出一条完整SQL。
""".strip()

    raw_output = invoke_text(
        _generation_system_prompt(
            candidate_mode
        ),
        user_prompt,
    )
    cleaned_sql = normalize_sample_id_literals(
        clean_llm_sql(raw_output)
    )
    return raw_output, cleaned_sql


def generate_full_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """使用完整Schema生成第一条候选SQL。"""

    question = state["normalized_question"]
    if validate_question_policy(question) is not None:
        return {
            "full_schema_context": state.get(
                "schema_context", ""
            ),
            "full_generator_raw_output": "",
            "full_sql": "",
        }

    field_hint = build_question_field_hint(
        question
    )
    full_context = state["schema_context"]
    raw_output, sql = _generate_candidate_sql(
        question=question,
        schema_context=full_context,
        field_hint=field_hint,
        candidate_mode="full",
    )

    return {
        "field_hint": field_hint,
        "full_schema_context": full_context,
        "full_generator_raw_output": raw_output,
        "full_sql": sql,
    }


def build_robust_schema(
    state: Text2SQLState,
) -> dict[str, Any]:
    """将问题正向链接与SQL1反向链接合并为稳健裁剪Schema。"""

    linking = build_robust_schema_linking(
        question=state["normalized_question"],
        preliminary_sql=state.get(
            "full_sql", ""
        ),
    )

    return {
        "forward_schema_tables": linking[
            "forward_tables"
        ],
        "forward_schema_columns": linking[
            "forward_columns"
        ],
        "backward_schema_tables": linking[
            "backward_tables"
        ],
        "backward_schema_columns": linking[
            "backward_columns"
        ],
        "accepted_backward_tables": linking[
            "accepted_backward_tables"
        ],
        "rejected_backward_tables": linking[
            "rejected_backward_tables"
        ],
        "robust_schema_context": linking[
            "context"
        ],
        "robust_schema_tables": linking[
            "robust_tables"
        ],
        "robust_schema_columns": linking[
            "robust_columns"
        ],
        # 保留V0.5可观测字段兼容性
        "generation_schema_context": linking[
            "context"
        ],
        "generation_relevant_tables": linking[
            "robust_tables"
        ],
    }


def generate_pruned_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """使用稳健裁剪Schema生成第二条候选SQL。"""

    question = state["normalized_question"]
    if validate_question_policy(question) is not None:
        return {
            "pruned_generator_raw_output": "",
            "pruned_sql": "",
        }

    raw_output, sql = _generate_candidate_sql(
        question=question,
        schema_context=state[
            "robust_schema_context"
        ],
        field_hint=state.get(
            "field_hint", ""
        ),
        candidate_mode="pruned",
    )

    return {
        "pruned_generator_raw_output": raw_output,
        "pruned_sql": sql,
    }


def _sql_complexity_metrics(
    sql: str,
) -> dict[str, int]:
    try:
        tree = sqlglot.parse_one(
            sql,
            read="mysql",
        )
    except ParseError:
        return {
            "joins": 99,
            "subqueries": 99,
            "groups": 99,
            "aggregates": 99,
            "tables": 99,
        }

    return {
        "joins": len(
            list(tree.find_all(exp.Join))
        ),
        "subqueries": len(
            list(tree.find_all(exp.Subquery))
        ),
        "groups": len(
            list(tree.find_all(exp.Group))
        ),
        "aggregates": len(
            list(tree.find_all(exp.AggFunc))
        ),
        "tables": len(
            list(tree.find_all(exp.Table))
        ),
    }


def _canonical_sql(sql: str) -> str:
    try:
        return sqlglot.parse_one(
            sql,
            read="mysql",
        ).sql(dialect="mysql")
    except ParseError:
        return sql.strip()


def _evaluate_candidate(
    label: str,
    sql: str,
    question: str,
    query_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    result = validate_and_normalize_sql(
        sql=sql,
        allowed_tables=set(
            settings.allowed_tables
        ),
        max_rows=settings.SQL_MAX_ROWS,
        question=question,
        query_spec=query_spec,
    )

    if not result.valid:
        severity = {
            "policy": 1000.0,
            "generation": 220.0,
            "syntax": 200.0,
            "schema": 150.0,
            "semantic": 120.0,
            "resource": 100.0,
        }.get(result.error_type, 130.0)
        error_lines = max(
            1,
            result.error.count("\n") + 1,
        )
        return {
            "label": label,
            "valid": False,
            "normalized_sql": "",
            "error": result.error,
            "error_type": result.error_type,
            "repairable": result.repairable,
            "score": -severity - error_lines,
            "covered": False,
            "coverage_reason": "",
            "tables": [],
            "columns": [],
            "metrics": _sql_complexity_metrics(sql),
        }

    normalized_sql = result.sql
    covered, coverage_reason = (
        assess_deterministic_semantic_coverage(
            question=question,
            sql=normalized_sql,
        )
    )
    used_tables, used_columns = (
        extract_sql_schema_elements(
            normalized_sql
        )
    )
    expected_tables = infer_relevant_tables(
        question
    )
    metrics = _sql_complexity_metrics(
        normalized_sql
    )

    score = 100.0
    if covered:
        score += 15.0

    if expected_tables:
        if used_tables == expected_tables:
            score += 10.0
        elif expected_tables.issubset(
            used_tables
        ):
            score += 4.0
        else:
            score -= 12.0 * len(
                expected_tables - used_tables
            )
        score -= 5.0 * len(
            used_tables - expected_tables
        )

    score -= 1.5 * metrics["joins"]
    score -= 3.0 * metrics["subqueries"]
    score -= 0.5 * metrics["groups"]
    score -= 0.25 * metrics["aggregates"]

    # 分数完全相同时，稳健裁剪候选作为轻微、可解释的tie-break。
    if label == "pruned":
        score += 0.1

    return {
        "label": label,
        "valid": True,
        "normalized_sql": normalized_sql,
        "error": "",
        "error_type": "",
        "repairable": True,
        "score": round(score, 3),
        "covered": covered,
        "coverage_reason": coverage_reason,
        "tables": sorted(used_tables),
        "columns": sorted(used_columns),
        "metrics": metrics,
    }


def select_sql_candidate(
    state: Text2SQLState,
) -> dict[str, Any]:
    """先用Guard淘汰，再用确定性覆盖与复杂度选择候选。"""

    question = state["normalized_question"]
    full_eval = _evaluate_candidate(
        label="full",
        sql=state.get("full_sql", ""),
        question=question,
        query_spec=state.get("query_spec"),
    )
    pruned_eval = _evaluate_candidate(
        label="pruned",
        sql=state.get("pruned_sql", ""),
        question=question,
        query_spec=state.get("query_spec"),
    )

    if full_eval["valid"] and not pruned_eval["valid"]:
        selected = full_eval
        reason = "仅完整Schema候选通过确定性Guard。"
    elif pruned_eval["valid"] and not full_eval["valid"]:
        selected = pruned_eval
        reason = "仅稳健裁剪Schema候选通过确定性Guard。"
    elif full_eval["valid"] and pruned_eval["valid"]:
        if _canonical_sql(
            full_eval["normalized_sql"]
        ) == _canonical_sql(
            pruned_eval["normalized_sql"]
        ):
            selected = pruned_eval
            reason = (
                "两个候选规范化后等价，选择稳健裁剪Schema候选。"
            )
        elif (
            full_eval["score"]
            > pruned_eval["score"]
        ):
            selected = full_eval
            reason = (
                "两个候选均通过Guard，完整Schema候选的"
                "确定性覆盖与结构评分更高。"
            )
        else:
            selected = pruned_eval
            reason = (
                "两个候选均通过Guard，稳健裁剪Schema候选的"
                "确定性覆盖与结构评分更高或并列。"
            )
    else:
        # 两条都失败时，只选择错误风险较低的一条进入既有一次修复。
        if full_eval["score"] > pruned_eval["score"]:
            selected = full_eval
        else:
            selected = pruned_eval
        reason = (
            "两个候选均未通过Guard，选择错误严重度较低的候选"
            "进入现有一次修复链路。"
        )

    selected_label = selected["label"]
    selected_original_sql = (
        state.get("full_sql", "")
        if selected_label == "full"
        else state.get("pruned_sql", "")
    )
    selected_sql = (
        selected["normalized_sql"]
        if selected["valid"]
        else selected_original_sql
    )

    return {
        "candidate_full_valid": full_eval["valid"],
        "candidate_full_normalized_sql": full_eval[
            "normalized_sql"
        ],
        "candidate_full_error": full_eval["error"],
        "candidate_full_error_type": full_eval[
            "error_type"
        ],
        "candidate_full_score": full_eval["score"],
        "candidate_pruned_valid": pruned_eval["valid"],
        "candidate_pruned_normalized_sql": pruned_eval[
            "normalized_sql"
        ],
        "candidate_pruned_error": pruned_eval["error"],
        "candidate_pruned_error_type": pruned_eval[
            "error_type"
        ],
        "candidate_pruned_score": pruned_eval["score"],
        "selected_candidate": selected_label,
        "candidate_selection_reason": reason,
        "generator_raw_output": (
            state.get("full_generator_raw_output", "")
            if selected_label == "full"
            else state.get("pruned_generator_raw_output", "")
        ),
        "initial_sql": selected_sql,
        "raw_sql": selected_sql,
        "validated_sql": "",
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
    }



def validate_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    settings = get_settings()

    result = validate_and_normalize_sql(
        sql=state.get("raw_sql", ""),
        allowed_tables=set(
            settings.allowed_tables
        ),
        max_rows=settings.SQL_MAX_ROWS,
        question=state["normalized_question"],
        query_spec=state.get("query_spec"),
    )

    if not result.valid:
        return {
            "validation_error": result.error,
            "validation_repairable": result.repairable,
            "validation_error_type": result.error_type,
            "validated_sql": "",
            "execution_error": "",
        }

    return {
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "validated_sql": result.sql,
        "execution_error": "",
    }


def review_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """基础SQL由确定性Guard放行，复杂SQL才调用一次LLM审查。"""

    covered, coverage_reason = (
        assess_deterministic_semantic_coverage(
            question=state[
                "normalized_question"
            ],
            sql=state["validated_sql"],
            query_spec=state.get("query_spec"),
        )
    )

    if covered:
        return {
            "review_called": False,
            "review_passed": True,
            "review_reason": (
                "确定性语义检查通过。"
            ),
            "review_note": coverage_reason,
            "review_input_summary": "",
        }

    review_input_summary = (
        build_sql_review_summary(
            state["validated_sql"]
        )
    )
    passed, reason = review_complex_sql(
        question=state[
            "normalized_question"
        ],
        schema_context=state[
            "schema_context"
        ],
        sql=state["validated_sql"],
    )

    if passed is None:
        return {
            "review_called": True,
            "review_passed": False,
            "review_reason": (
                "复杂SQL语义审查未能返回可信结论。"
            ),
            "review_note": "review_unavailable",
            "review_input_summary": review_input_summary,
        }

    return {
        "review_called": True,
        "review_passed": passed,
        "review_reason": reason,
        "review_note": "llm_review",
        "review_input_summary": review_input_summary,
    }


def build_explicit_repair_action(
    question: str,
    reason: str,
) -> str:
    """把高可信Guard错误转换为简短、明确的修复动作。"""

    requested_samples = (
        extract_requested_sample_ids(
            question
        )
    )

    if (
        "用户未指定的固定sample_id过滤" in reason
        or (
            "用户没有指定任何样本" in reason
            and "sample_id" in reason
        )
    ):
        return (
            "用户没有指定任何样本。"
            "删除整个固定sample_id过滤谓词；"
            "不要改成IN、LIKE、其他样本编号或其他固定样本条件。"
        )

    if (
        "固定样本过滤不能使用LIKE" in reason
        and requested_samples
    ):
        return (
            "删除sample_id LIKE谓词，"
            "改为用户指定sample_id的等值或IN条件："
            + ", ".join(
                sorted(requested_samples)
            )
            + "。"
        )

    if (
        "用户明确指定了样本" in reason
        and "没有使用对应的sample_id" in reason
        and requested_samples
    ):
        return (
            "加入用户指定sample_id的等值或IN过滤："
            + ", ".join(
                sorted(requested_samples)
            )
            + "。不得使用LIKE。"
        )

    if (
        "IN子查询中使用LIMIT" in reason
        or "Top-K查询缺少顶层ORDER BY" in reason
    ):
        return (
            "这是普通Top-K结构错误。"
            "丢弃IN子查询、派生表包装和无意义GROUP BY，"
            "从正确字段所属的最少表重新生成："
            "顶层SELECT返回sample_id及用户要求字段，"
            "顶层ORDER BY目标字段并使用正确LIMIT。"
        )

    if (
        "字段归属错误" in reason
        or "未知字段" in reason
    ):
        return (
            "根据确定性字段提示重新选择真实字段所属表。"
            "不要沿用原SQL中的错误表或嵌套子查询；"
            "使用包含目标字段的最少必要表重新生成。"
        )

    if "多个来源中存在" in reason:
        return (
            "为所有可能歧义的字段添加正确表别名；"
            "如果这是普通Top-K，优先改写成单层ORDER BY加LIMIT，"
            "不要保留不必要的派生表JOIN。"
        )

    if (
        "不能限制point_index" in reason
        or "固定point_index代替峰值" in reason
    ):
        return (
            "删除WHERE、JOIN ON和HAVING中的固定point_index条件，"
            "在完整thermal_response序列上按sample_id分组并使用MAX。"
        )

    if "全部数据" in reason:
        return (
            "用户明确请求白名单表全部数据。"
            "只查询该目标表，显式列出该表全部字段，"
            "删除WHERE、GROUP BY、聚合和其他表JOIN。"
        )

    if (
        "普通样本级字段Top-K" in reason
        or "样本级Top-K必须返回sample_id" in reason
    ):
        return (
            "这是每个样本一行字段的普通Top-K。"
            "返回sample_id及用户要求字段，"
            "直接按目标字段ORDER BY并LIMIT；"
            "删除MAX、GROUP BY和子查询。"
        )

    return (
        "只修复错误原因中明确指出的问题，"
        "使用正确字段所属的最少必要表，"
        "不要增加用户未要求的过滤、字段或数据表。"
        "科学计数法必须作为一个完整数值保留，"
        "不得把指数部分解释成LIMIT或其他条件。"
    )


def repair_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """只根据可信的Guard、数据库或一次语义审查错误重写SQL。"""

    if state.get("validation_error"):
        source = "确定性Guard"
        reason = state["validation_error"]
        bad_sql = state.get("raw_sql", "")
    elif state.get("execution_error"):
        source = "数据库执行"
        reason = state["execution_error"]
        bad_sql = (
            state.get("validated_sql")
            or state.get("raw_sql", "")
        )
    else:
        source = "复杂SQL语义审查"
        reason = state.get(
            "review_reason",
            "语义不一致",
        )
        bad_sql = (
            state.get("validated_sql")
            or state.get("raw_sql", "")
        )

    next_retry_count = (
        state.get("retry_count", 0) + 1
    )
    question = state["normalized_question"]
    compact_context = (
        state.get("robust_schema_context")
        or build_compact_sql_context(
            question
        )
    )
    repair_action = (
        build_explicit_repair_action(
            question=question,
            reason=reason,
        )
    )

    system_prompt = """
你是MySQL SQL修复器。

只输出一条完整SELECT SQL。
根据明确错误修复原SQL，不要解释，不要复制错误结构。
必须使用真实表名、真实字段和最少必要表。
不得用AS别名伪装错误字段。
普通样本级Top-K返回sample_id，直接使用目标字段ORDER BY和LIMIT，不使用MAX、GROUP BY或IN子查询。
标准时序查询必须区分“样本内聚合”和“聚合结果排名”：峰值使用MAX，平均值使用AVG，均按sample_id分组；“峰值最低”仍是MAX后ASC排序。
最终值表示每个样本最大point_index对应的记录，不能用MIN或MAX(目标值)替代。
指定样本的point_index明细查询直接使用sample_id和point_index条件，不使用聚合。
用户明确指定样本编号时，必须使用对应sample_id等值或IN过滤，禁止LIKE。
用户没有指定样本编号时，禁止保留或新增任何固定sample_id过滤。
""".strip()

    user_prompt = f"""
{compact_context}

用户问题：
{question}

错误来源：{source}
错误原因：
{reason}

必须执行的修复动作：
{repair_action}

错误SQL：
{bad_sql}

请从零输出修复后的完整SQL。
""".strip()

    repair_raw_output = invoke_text(
        system_prompt,
        user_prompt,
    )
    repaired_sql = normalize_sample_id_literals(
        clean_llm_sql(
            repair_raw_output
        )
    )

    return {
        "raw_sql": repaired_sql,
        "validated_sql": "",
        "selected_candidate": "repair",
        "candidate_selection_reason": (
            state.get("candidate_selection_reason", "")
            + " 修复器基于已选候选重写SQL。"
        ).strip(),
        "retry_count": next_retry_count,
        "last_repair_reason": (
            f"{source}: {reason}"
        ),
        "repair_source": source,
        "repair_action": repair_action,
        "repair_bad_sql": bad_sql,
        "repair_raw_output": repair_raw_output,
        "validation_error": "",
        "validation_repairable": True,
        "validation_error_type": "",
        "review_called": False,
        "review_passed": False,
        "review_reason": "",
        "review_note": "",
        "review_input_summary": "",
        "execution_error": "",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
    }


def execute_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    settings = get_settings()

    try:
        result = execute_readonly_query(
            sql=state["validated_sql"],
            max_rows=settings.SQL_MAX_ROWS,
        )
    except Exception as exc:
        return {
            "execution_error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
        }

    return {
        "execution_error": "",
        **result,
    }


def shorten_cell(
    value: Any,
    max_length: int = 100,
) -> Any:
    if value is None:
        return None

    text = str(value)
    if len(text) <= max_length:
        return value

    return text[:max_length] + "..."


def format_result(
    state: Text2SQLState,
) -> dict[str, Any]:
    rows = state.get("rows", [])
    columns = state.get("columns", [])

    if rows:
        result_text = tabulate(
            [
                [
                    shorten_cell(value)
                    for value in row
                ]
                for row in rows
            ],
            headers=columns,
            tablefmt="github",
            stralign="left",
            numalign="right",
        )
    else:
        result_text = (
            "查询执行成功，但没有返回符合条件的数据。"
        )

    retry_count = state.get("retry_count", 0)
    execution_notice = (
        f"本次SQL经过{retry_count}次自动修复后执行成功。"
        if retry_count > 0
        else "本次SQL首次生成后执行成功。"
    )

    review_section = ""
    if state.get("review_reason"):
        review_section = (
            "\n\n语义校验："
            + state["review_reason"]
        )

    if (
        state.get("review_note")
        and state.get("review_note")
        not in {
            "llm_review",
            "review_unavailable",
        }
    ):
        review_section += (
            "\n校验说明："
            + state["review_note"]
        )

    truncate_notice = ""
    if state.get("truncated", False):
        truncate_notice = (
            "\n\n注意：结果超过最大返回行数，"
            f"当前只展示前{get_settings().SQL_MAX_ROWS}行。"
        )

    return {
        "final_status": (
            "repaired_success"
            if retry_count > 0
            else "first_pass_success"
        ),
        "final_answer": f"""
查询执行成功。

{execution_notice}

实际执行 SQL：

```sql
{state["validated_sql"]}
```

查询结果：

{result_text}{truncate_notice}{review_section}
""".strip()
    }


def format_error(
    state: Text2SQLState,
) -> dict[str, Any]:
    error = (
        state.get("validation_error")
        or state.get("execution_error")
        or state.get("review_reason")
        or "未知错误"
    )

    if (
        state.get("validation_error_type")
        == "policy"
    ):
        description = (
            "该请求违反只读、白名单或跨库安全策略，"
            "系统不会尝试改写。"
        )
    elif state.get("retry_count", 0) > 0:
        description = (
            "系统已经自动修复一次，"
            "但仍未通过确定性校验、复杂语义审查或数据库执行。"
        )
    else:
        description = (
            "本次查询未能生成可执行结果。"
        )

    final_status = (
        "policy_rejected"
        if state.get("validation_error_type") == "policy"
        else "failed"
    )

    return {
        "final_status": final_status,
        "final_answer": f"""
本次查询没有成功执行。

{description}

错误信息：

{error}

首次SQL：

```sql
{state.get("initial_sql") or "未生成SQL"}
```

最后SQL：

```sql
{state.get("raw_sql") or "未生成SQL"}
```
""".strip()
    }


def route_after_validation(
    state: Text2SQLState,
) -> Literal["review", "repair", "error"]:
    if not state.get("validation_error"):
        return "review"

    if not state.get(
        "validation_repairable",
        True,
    ):
        return "error"

    if (
        state.get("retry_count", 0)
        < get_settings().SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"


def route_after_review(
    state: Text2SQLState,
) -> Literal["execute", "repair", "error"]:
    if state.get("review_passed", False):
        return "execute"

    if (
        state.get("review_note")
        == "review_unavailable"
    ):
        return "error"

    if (
        state.get("retry_count", 0)
        < get_settings().SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"


def route_after_execution(
    state: Text2SQLState,
) -> Literal["success", "repair", "error"]:
    if not state.get("execution_error"):
        return "success"

    if (
        state.get("retry_count", 0)
        < get_settings().SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"