import re
from typing import Any, Literal

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
    build_generation_schema_context,
    build_question_field_hint,
    build_schema_context,
    extract_requested_sample_ids,
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


def generate_sql(
    state: Text2SQLState,
) -> dict[str, Any]:
    """根据问题、Schema和确定性字段提示直接生成SQL。"""

    system_prompt = """
你是材料数据库Text2SQL生成器。

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
15. 禁止SELECT *、写操作和跨库查询。
""".strip()

    question = state["normalized_question"]

    # 危险请求不调用LLM，下一节点由Guard返回policy错误。
    if validate_question_policy(
        question
    ) is not None:
        return {
            "initial_sql": "",
            "raw_sql": "",
            "validated_sql": "",
            "validation_error": "",
            "review_called": False,
            "review_passed": False,
            "review_reason": "",
            "review_note": "",
            "review_input_summary": "",
            "execution_error": "",
        }

    field_hint = build_question_field_hint(
        question
    )

    generation_context = (
        build_generation_schema_context(
            question
        )
    )

    user_prompt = f"""
数据库Schema：
{generation_context}

用户问题：
{question}

{field_hint}

只输出一条完整SQL。
""".strip()

    generator_raw_output = invoke_text(
        system_prompt,
        user_prompt,
    )
    sql = normalize_sample_id_literals(
        clean_llm_sql(
            generator_raw_output
        )
    )

    return {
        "generation_schema_context": generation_context,
        "generation_relevant_tables": sorted(
            infer_relevant_tables(question)
        ),
        "field_hint": field_hint,
        "generator_raw_output": generator_raw_output,
        "initial_sql": sql,
        "raw_sql": sql,
        "validated_sql": "",
        "validation_error": "",
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
        build_compact_sql_context(
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
只有时序峰值才使用MAX并按sample_id分组，且不得固定point_index。
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