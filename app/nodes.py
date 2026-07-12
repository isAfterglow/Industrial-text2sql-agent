import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from tabulate import tabulate

from app.config import get_settings
from app.db import execute_readonly_query
from app.llm import get_llm
from app.schema import build_schema_context
from app.sql_guard import (
    clean_llm_sql,
    validate_and_normalize_sql,
)
from app.state import Text2SQLState


def message_content_to_text(content: Any) -> str:
    """将不同模型可能返回的content格式统一转换成字符串。

    OpenAI-compatible接口通常返回字符串，但部分模型或版本
    可能返回内容块列表，因此这里增加一个兼容处理。
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue

            if isinstance(item, dict):
                text_value = item.get("text")

                if isinstance(text_value, str):
                    text_parts.append(text_value)

        if text_parts:
            return "\n".join(text_parts)

    return str(content)


def normalize_sample_id_mentions(question: str) -> str:
    """将用户输入的简写样本编号转换为数据库中的标准格式。

    示例：
    样本100       -> 样本 sample_000100
    样本 7        -> 样本 sample_000007
    sample 25     -> sample sample_000025
    sample_id 120 -> sample_id sample_000120

    已经是sample_000100格式的内容不会被重复转换。
    """

    normalized_question = question

    chinese_pattern = re.compile(
        r"样本\s*(?!sample_)(\d+)\b",
        flags=re.IGNORECASE,
    )

    normalized_question = chinese_pattern.sub(
        lambda match: (
            f"样本 sample_{int(match.group(1)):06d}"
        ),
        normalized_question,
    )

    english_pattern = re.compile(
        r"\b(sample|sample_id)\s*"
        r"(?!sample_)(\d+)\b",
        flags=re.IGNORECASE,
    )

    normalized_question = english_pattern.sub(
        lambda match: (
            f"{match.group(1)} "
            f"sample_{int(match.group(2)):06d}"
        ),
        normalized_question,
    )

    return normalized_question


def load_schema(state: Text2SQLState) -> dict[str, Any]:
    """加载数据库Schema，并初始化本轮Graph运行状态。"""

    return {
        "schema_context": build_schema_context(),
        "initial_sql": "",
        "raw_sql": "",
        "validated_sql": "",
        "retry_count": 0,
        "last_repair_reason": "",
        "validation_error": "",
        "execution_error": "",
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "final_answer": "",
    }


def generate_sql(state: Text2SQLState) -> dict[str, Any]:
    """根据用户问题和Schema首次生成SQL。

    V0.2仍然不使用with_structured_output，
    避免本地3B模型在函数调用和结构化输出方面出现类型不稳定。
    """

    original_question = state["question"]
    normalized_question = normalize_sample_id_mentions(
        original_question
    )
    schema_context = state["schema_context"]

    system_prompt = """
你是一个专门负责树脂基防热材料数据库的Text2SQL模型。

你的唯一任务是：
根据用户问题和数据库Schema，生成一条可以在MySQL执行的只读SELECT查询。

必须遵守以下规则：

1. 只输出SQL，不输出Markdown代码围栏。
2. 不输出解释、分析过程或多个候选SQL。
3. 只能使用Schema中存在的表和字段。
4. 禁止INSERT、UPDATE、DELETE、DROP、ALTER、CREATE等写操作。
5. 只连接回答问题必需的数据表。
6. 只返回用户明确要求的字段，不要擅自增加字段。
7. 简单Top-K查询直接使用ORDER BY和LIMIT，不要使用IN子查询。
8. “最高”或“最大”通常使用DESC。
9. “最低”或“最小”通常使用ASC。
10. 查询峰值表温时使用MAX(surface_temperature)。
11. 查询峰值背温时使用MAX(back_temperature)。
12. thermal_response是一对多表，聚合响应数据时按sample_id分组。
13. material_static和material_thermal_property都是一条样本一行，
    不要为了去重而使用GROUP BY。
14. point_index是序列点编号，不是物理时间。
15. 用户要求某段point_index范围时，必须使用WHERE范围条件，
    LIMIT不能代替point_index范围。
""".strip()

    user_prompt = f"""
以下是数据库Schema、字段语义和SQL示例：

{schema_context}

用户原始问题：
{original_question}

规范化后的问题：
{normalized_question}

请只输出一条MySQL SELECT查询。
""".strip()

    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    raw_content = message_content_to_text(
        response.content
    )

    generated_sql = clean_llm_sql(raw_content)

    return {
        "initial_sql": generated_sql,
        "raw_sql": generated_sql,
        "validated_sql": "",
        "validation_error": "",
        "execution_error": "",
    }


def validate_sql(state: Text2SQLState) -> dict[str, Any]:
    """调用SQL Guard进行安全检查和质量检查。

    SQL Guard同时检查：
    - 是否只读
    - 是否访问白名单表
    - 是否跨库
    - 返回字段是否正确
    - 是否存在冗余字段或JOIN
    - Top-K排序字段和方向是否正确
    - GROUP BY和聚合是否合理
    """

    settings = get_settings()

    result = validate_and_normalize_sql(
        sql=state.get("raw_sql", ""),
        question=state["question"],
        allowed_tables=set(
            settings.allowed_tables
        ),
        max_rows=settings.SQL_MAX_ROWS,
    )

    if not result.valid:
        return {
            "validation_error": result.error,
            "execution_error": "",
            "validated_sql": "",
        }

    return {
        "validation_error": "",
        "execution_error": "",
        "validated_sql": result.sql,
    }


def repair_sql(state: Text2SQLState) -> dict[str, Any]:
    """根据SQL Guard或数据库返回的错误自动修复SQL。

    修复节点不会绕过SQL Guard。
    修复后的SQL必须重新进入validate_sql节点完整校验。
    """

    current_sql = state.get("raw_sql", "")
    schema_context = state["schema_context"]
    question = state["question"]

    validation_error = state.get(
        "validation_error",
        "",
    )
    execution_error = state.get(
        "execution_error",
        "",
    )

    repair_reason = (
        validation_error
        or execution_error
        or "未提供具体错误信息"
    )

    current_retry_count = state.get(
        "retry_count",
        0,
    )
    next_retry_count = current_retry_count + 1

    system_prompt = """
你是一个SQL修复器，负责修复树脂基防热材料数据库的MySQL SELECT查询。

你会收到：
1. 用户原始问题；
2. 数据库Schema；
3. 当前错误SQL；
4. SQL Guard或数据库给出的具体错误。

你的任务不是解释错误，而是重新生成一条更简单、更准确的SQL。

必须遵守：

1. 只输出一条MySQL SELECT查询。
2. 不要输出Markdown代码围栏。
3. 不要输出解释、分析过程或多个候选SQL。
4. 必须解决错误信息中指出的全部问题。
5. 只能使用Schema中存在的表和字段。
6. 只连接回答问题所必需的数据表。
7. 只返回用户明确要求的字段。
8. 简单Top-K使用ORDER BY和LIMIT，不使用IN子查询。
9. 排序必须针对用户要求比较的字段。
10. “最高/最大”使用DESC，“最低/最小”使用ASC。
11. 峰值表温使用MAX(surface_temperature)。
12. 峰值背温使用MAX(back_temperature)。
13. 访问thermal_response并做样本级聚合时，按sample_id分组。
14. 不要使用无意义的GROUP BY。
15. 不要尝试绕过SQL Guard。
16. 不得生成任何写入、删除或修改数据库的操作。
""".strip()

    user_prompt = f"""
数据库Schema和业务语义：

{schema_context}

用户问题：

{question}

当前错误SQL：

{current_sql}

校验或执行错误：

{repair_reason}

这是第{next_retry_count}次修复。

请重新生成一条完整、简洁、准确的MySQL SELECT查询。
只输出SQL。
""".strip()

    response = get_llm().invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )

    raw_content = message_content_to_text(
        response.content
    )

    repaired_sql = clean_llm_sql(raw_content)

    return {
        # 用修复后的SQL覆盖当前待校验SQL
        "raw_sql": repaired_sql,

        # 修复后必须重新校验，因此清空旧的validated_sql
        "validated_sql": "",

        # 记录已经修复过多少次，防止无限循环
        "retry_count": next_retry_count,

        # 保存触发本次修复的原因，便于最终调试
        "last_repair_reason": repair_reason,

        # 清空旧错误，等待下一次validate_sql或execute_sql产生新结果
        "validation_error": "",
        "execution_error": "",
    }


def execute_sql(state: Text2SQLState) -> dict[str, Any]:
    """执行已经通过SQL Guard的只读SQL。"""

    settings = get_settings()
    validated_sql = state["validated_sql"]

    try:
        query_result = execute_readonly_query(
            sql=validated_sql,
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
        **query_result,
    }


def shorten_cell(
    value: Any,
    max_length: int = 100,
) -> Any:
    """缩短终端中的超长字段，避免表格格式被破坏。"""

    if value is None:
        return None

    text = str(value)

    if len(text) <= max_length:
        return value

    return text[:max_length] + "..."


def format_result(
    state: Text2SQLState,
) -> dict[str, Any]:
    """将成功执行的SQL和查询结果格式化为终端文本。"""

    columns = state.get("columns", [])
    rows = state.get("rows", [])
    sql = state.get("validated_sql", "")
    truncated = state.get("truncated", False)
    retry_count = state.get("retry_count", 0)

    if not rows:
        result_text = (
            "查询执行成功，但没有返回符合条件的数据。"
        )
    else:
        display_rows = [
            [
                shorten_cell(value)
                for value in row
            ]
            for row in rows
        ]

        result_text = tabulate(
            display_rows,
            headers=columns,
            tablefmt="github",
            stralign="left",
            numalign="right",
        )

    if retry_count > 0:
        repair_notice = (
            f"本次SQL经过 {retry_count} 次自动修复后执行成功。\n\n"
        )
    else:
        repair_notice = (
            "本次SQL首次生成即通过校验。\n\n"
        )

    truncate_notice = ""

    if truncated:
        truncate_notice = (
            "\n\n注意：查询结果超过最大返回行数，"
            f"当前只展示前 "
            f"{get_settings().SQL_MAX_ROWS} 行。"
        )

    final_answer = f"""
查询执行成功。

{repair_notice}实际执行 SQL：

```sql
{sql}
查询结果：

{result_text}{truncate_notice}
""".strip()

    return {
        "final_answer": final_answer,
    }
    
def format_error(
    state: Text2SQLState,
    ) -> dict[str, Any]:
    """格式化最终仍未通过校验或执行失败的结果。"""

    validation_error = state.get(
        "validation_error",
        "",
    )
    execution_error = state.get(
        "execution_error",
        "",
    )

    final_error = (
        validation_error
        or execution_error
        or "未知错误"
    )

    initial_sql = state.get(
        "initial_sql",
        "",
    )
    current_sql = state.get(
        "raw_sql",
        "",
    )
    retry_count = state.get(
        "retry_count",
        0,
    )
    last_repair_reason = state.get(
        "last_repair_reason",
        "",
    )

    if retry_count > 0:
        retry_description = (
            f"系统已经自动修复 {retry_count} 次，"
            "但修复后的SQL仍未通过检查或执行。"
        )
    else:
        retry_description = (
            "该错误不满足自动修复条件，"
            "或者尚未执行自动修复。"
        )

    if (
        initial_sql
        and current_sql
        and initial_sql != current_sql
    ):
        sql_history = f"""

    首次生成 SQL：

    {initial_sql}

    最后一次修复 SQL：

    {current_sql}

    """.strip()
    else:
        sql_history = f"""
        模型生成的 SQL：

        {current_sql or initial_sql or "未生成SQL"}

        """.strip()

    repair_reason_text = ""

    if last_repair_reason:
        repair_reason_text = f"""

        触发最近一次修复的错误：

        {last_repair_reason}
        """

    final_answer = f"""

    本次查询没有成功执行。

    {retry_description}

    最终错误信息：
    
    {final_error}
    {sql_history}
    {repair_reason_text}
    """.strip()

    return {
        "final_answer": final_answer,
    }
    
def route_after_validation(
    state: Text2SQLState,
    ) -> Literal["execute", "repair", "error"]:
    """根据SQL Guard结果选择执行、修复或结束。
    只有校验失败且修复次数未达到上限时，
    才能进入repair_sql。
    """

    validation_error = state.get(
        "validation_error",
        "",
    )

    if not validation_error:
        return "execute"

    settings = get_settings()
    retry_count = state.get(
        "retry_count",
        0,
    )

    if (
        retry_count
        < settings.SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"

def route_after_execution(
    state: Text2SQLState,
    ) -> Literal["success", "repair", "error"]:
    """根据数据库执行结果选择成功、修复或结束。"""

    execution_error = state.get(
        "execution_error",
        "",
    )

    if not execution_error:
        return "success"

    settings = get_settings()
    retry_count = state.get(
        "retry_count",
        0,
    )

    if (
        retry_count
        < settings.SQL_MAX_REPAIR_ATTEMPTS
    ):
        return "repair"

    return "error"

