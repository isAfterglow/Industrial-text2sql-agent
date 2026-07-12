import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.config import get_settings


FORBIDDEN_PATTERN = re.compile(
    r"""
    \b(
        INSERT
        |UPDATE
        |DELETE
        |DROP
        |ALTER
        |CREATE
        |TRUNCATE
        |REPLACE
        |MERGE
        |CALL
        |GRANT
        |REVOKE
        |SET
        |USE
        |LOAD_FILE
        |SLEEP
        |BENCHMARK
    )\b
    |INTO\s+OUTFILE
    |INTO\s+DUMPFILE
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


BANNED_AST_KEYS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "truncate",
    "merge",
    "command",
    "transaction",
    "grant",
    "revoke",
    "set",
    "use",
}

FIELD_PHRASES = [
    # 峰值响应字段放前面，避免被“表温”“背温”提前匹配。
    (
        ("峰值表温", "表温峰值", "最大表面温度"),
        "surface_temperature",
        "max",
    ),
    (
        ("峰值背温", "背温峰值", "最大背面温度"),
        "back_temperature",
        "max",
    ),

    # material_static
    (
        ("原始材料密度", "原始密度"),
        "rhov_i",
        None,
    ),
    (
        ("碳化材料密度", "碳化密度"),
        "rhoc_i",
        None,
    ),
    (
        ("原始材料孔隙率", "原始孔隙率"),
        "porosity_v",
        None,
    ),
    (
        ("碳化材料孔隙率", "碳化孔隙率"),
        "porosity_c",
        None,
    ),
    (
        ("原始材料渗透率", "原始渗透率"),
        "permeability_v",
        None,
    ),
    (
        ("碳化材料渗透率", "碳化渗透率"),
        "permeability_c",
        None,
    ),

    # material_thermal_property
    (
        ("原始热导率",),
        "kv_list",
        None,
    ),
    (
        ("碳化热导率",),
        "kc_list",
        None,
    ),
    (
        ("原始比热容",),
        "cpv_list",
        None,
    ),
    (
        ("碳化比热容",),
        "cpc_list",
        None,
    ),
    (
        ("热解热",),
        "pyrolysis_heat",
        None,
    ),
    (
        ("表面发射率", "发射率"),
        "surface_emissivity",
        None,
    ),

    # thermal_response
    (
        ("表面温度", "表温"),
        "surface_temperature",
        None,
    ),
    (
        ("背面温度", "背温"),
        "back_temperature",
        None,
    ),
    (
        ("质量",),
        "mass",
        None,
    ),
    (
        ("point_index", "序列点"),
        "point_index",
        None,
    ),
    (
        ("样本编号", "sample_id"),
        "sample_id",
        None,
    ),
]


DIRECTION_WORDS = {
    "最高": "desc",
    "最大": "desc",
    "最低": "asc",
    "最小": "asc",
}


EXPLICIT_PROJECTION_WORDS = (
    "返回",
    "显示",
    "同时返回",
    "并显示",
    "列出",
)


AGGREGATION_REQUEST_WORDS = (
    "平均",
    "均值",
    "总数",
    "数量",
    "计数",
    "求和",
    "总和",
    "最大值",
    "最小值",
    "统计",
)


@dataclass
class SQLValidationResult:
    valid: bool
    sql: str = ""
    error: str = ""

@dataclass
class RankingSpec:
    """用户要求按照哪个字段、哪个方向排序。"""

    column: str
    direction: str
    aggregate: str | None = None


@dataclass
class QuestionRequirements:
    """从用户问题中提取的确定性要求。"""

    requested_columns: set[str] = field(default_factory=set)

    # 例如：
    # 背温峰值 -> back_temperature 必须使用 MAX
    required_aggregates: dict[str, str] = field(
        default_factory=dict
    )

    ranking: RankingSpec | None = None
    top_k: int | None = None
    explicit_projection: bool = False


@dataclass
class ProjectionInfo:
    """一个 SELECT 投影或别名对应的源字段和聚合函数。"""

    columns: set[str]
    aggregates: set[str]


def clean_llm_sql(content: str) -> str:
    """清理模型可能生成的 Markdown 代码围栏。"""

    content = content.strip()

    code_block = re.search(
        r"```(?:sql)?\s*(.*?)```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if code_block:
        content = code_block.group(1).strip()

    content = re.sub(
        r"^\s*(SQL|SQL查询|查询语句)\s*[:：]\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    return content.strip()

def extract_question_requirements(
    question: str,
) -> QuestionRequirements:
    """从用户问题中提取字段、排序和Top-K要求。

    这不是通用中文语义解析器，只针对当前材料数据库
    常见查询做确定性检查。
    """

    requirements = QuestionRequirements(
        explicit_projection=any(
            word in question
            for word in EXPLICIT_PROJECTION_WORDS
        )
    )

    # 提取用户明确提到的字段。
    for phrases, column, aggregate in FIELD_PHRASES:
        if any(
            phrase.lower() in question.lower()
            for phrase in phrases
        ):
            requirements.requested_columns.add(column)

            if aggregate:
                requirements.required_aggregates[
                    column
                ] = aggregate

    # “最高的几个样本”通常默认应该返回 sample_id。
    if "样本" in question:
        requirements.requested_columns.add("sample_id")

    # 识别排序目标。
    for phrases, column, aggregate in FIELD_PHRASES:
        for phrase in phrases:
            escaped_phrase = re.escape(phrase)

            # 例如：原始比热容最高
            match = re.search(
                rf"{escaped_phrase}.{{0,10}}?"
                r"(最高|最大|最低|最小)",
                question,
                flags=re.IGNORECASE,
            )

            # 兼容：最高的原始比热容
            if not match:
                match = re.search(
                    r"(最高|最大|最低|最小)"
                    rf".{{0,6}}?{escaped_phrase}",
                    question,
                    flags=re.IGNORECASE,
                )

            if match:
                requirements.ranking = RankingSpec(
                    column=column,
                    direction=DIRECTION_WORDS[
                        match.group(1)
                    ],
                    aggregate=aggregate,
                )
                break

        if requirements.ranking:
            break

    # 识别 Top-K。
    # 支持：
    # 最高的7个样本
    # 最高的3条样本
    # 前10个样本
    top_k_match = re.search(
        r"(?:前\s*)?(\d+)\s*"
        r"(?:个|条|组|项)?\s*样本",
        question,
    )

    if not top_k_match:
        top_k_match = re.search(
            r"(?:最高|最大|最低|最小)"
            r"的?\s*(\d+)",
            question,
        )

    if top_k_match:
        requirements.top_k = int(
            top_k_match.group(1)
        )

    return requirements

def get_projection_info(
    tree: exp.Expression,
) -> tuple[
    set[str],
    dict[str, ProjectionInfo],
]:
    """获取顶层 SELECT 返回的源字段和别名信息。"""

    selected_columns: set[str] = set()
    alias_map: dict[str, ProjectionInfo] = {}

    for projection in tree.expressions:
        columns = {
            column.name
            for column in projection.find_all(exp.Column)
        }

        aggregates = {
            aggregate.key.lower()
            for aggregate in projection.find_all(
                exp.AggFunc
            )
        }

        selected_columns.update(columns)

        output_name = projection.alias_or_name

        if output_name:
            alias_map[output_name] = ProjectionInfo(
                columns=columns,
                aggregates=aggregates,
            )

    return selected_columns, alias_map

def get_first_order_info(
    tree: exp.Expression,
    alias_map: dict[str, ProjectionInfo],
) -> tuple[
    ProjectionInfo | None,
    str | None,
]:
    """获取顶层 ORDER BY 的第一个排序目标。"""

    order = tree.args.get("order")

    if order is None or not order.expressions:
        return None, None

    ordered_expression = order.expressions[0]
    expression = ordered_expression.this

    # ORDER BY peak_back_temperature
    # 需要还原别名背后的真实字段。
    if (
        isinstance(expression, exp.Column)
        and not expression.table
        and expression.name in alias_map
    ):
        projection_info = alias_map[
            expression.name
        ]
    else:
        projection_info = ProjectionInfo(
            columns={
                column.name
                for column in expression.find_all(
                    exp.Column
                )
            },
            aggregates={
                aggregate.key.lower()
                for aggregate in expression.find_all(
                    exp.AggFunc
                )
            },
        )

    direction = (
        "desc"
        if ordered_expression.args.get("desc")
        else "asc"
    )

    return projection_info, direction

def build_table_columns() -> dict[str, set[str]]:
    settings = get_settings()

    return {
        settings.RESIN_TABLE_STATIC: {
            "sample_id",
            "rhov_i",
            "rhoc_i",
            "porosity_v",
            "porosity_c",
            "permeability_v",
            "permeability_c",
        },
        settings.RESIN_TABLE_MATERIAL_THERMAL_PROPERTY: {
            "sample_id",
            "kv_list",
            "kc_list",
            "cpv_list",
            "cpc_list",
            "pyrolysis_heat",
            "surface_emissivity",
        },
        settings.RESIN_TABLE_THERMAL_RESPONSE: {
            "sample_id",
            "point_index",
            "surface_temperature",
            "back_temperature",
            "mass",
        },
    }
    
def get_referenced_tables_outside_join(
    tree: exp.Expression,
    table_columns: dict[str, set[str]],
) -> tuple[set[str], set[str]]:
    """区分SQL中出现的表和真正参与业务表达式的表。

    只在 JOIN ON 中出现，但没有参与返回、筛选、
    排序、分组或聚合的表，视为潜在冗余表。
    """

    alias_to_table = {
        table.alias_or_name: table.name
        for table in tree.find_all(exp.Table)
    }

    all_tables = set(alias_to_table.values())

    clause_roots: list[exp.Expression] = list(
        tree.expressions
    )

    for clause_name in (
        "where",
        "group",
        "order",
        "having",
        "qualify",
    ):
        clause = tree.args.get(clause_name)

        if clause is not None:
            clause_roots.append(clause)

    referenced_tables: set[str] = set()

    for clause_root in clause_roots:
        for column in clause_root.find_all(
            exp.Column
        ):
            if column.table:
                table_name = alias_to_table.get(
                    column.table
                )

                if table_name:
                    referenced_tables.add(table_name)

                continue

            # 没写表别名时，根据字段归属推断。
            owners = [
                table_name
                for table_name in all_tables
                if column.name
                in table_columns.get(
                    table_name,
                    set(),
                )
            ]

            if len(owners) == 1:
                referenced_tables.add(owners[0])

    return all_tables, referenced_tables

def projection_contains_star(
    tree: exp.Expression,
) -> bool:
    """检测 SELECT *，但允许 COUNT(*)。"""

    for projection in tree.expressions:
        stars = list(
            projection.find_all(exp.Star)
        )

        if not stars:
            continue

        # COUNT(*) 可以保留。
        if list(
            projection.find_all(exp.Count)
        ):
            continue

        return True

    return False

def get_limit_value(
    tree: exp.Expression,
) -> int | None:
    limit = tree.args.get("limit")

    if (
        limit is None
        or limit.expression is None
    ):
        return None

    expression = limit.expression

    if (
        isinstance(expression, exp.Literal)
        and expression.is_int
    ):
        return int(expression.this)

    return None


def set_limit(
    tree: exp.Expression,
    limit_value: int,
) -> None:
    tree.set(
        "limit",
        exp.Limit(
            expression=exp.Literal.number(
                limit_value
            )
        ),
    )
    
def validate_sql_quality(
    tree: exp.Expression,
    question: str,
    max_rows: int,
) -> list[str]:
    """检查可执行但低质量或语义错误的 SQL。"""

    errors: list[str] = []

    settings = get_settings()
    table_columns = build_table_columns()

    requirements = extract_question_requirements(
        question
    )

    selected_columns, alias_map = (
        get_projection_info(tree)
    )

    # ------------------------------------------------
    # 1. 禁止 SELECT *
    # ------------------------------------------------
    if projection_contains_star(tree):
        errors.append(
            "禁止使用 SELECT *，"
            "请只返回用户需要的字段。"
        )

    # ------------------------------------------------
    # 2. 检查明确要求返回的字段
    # ------------------------------------------------
    if requirements.explicit_projection:
        missing_columns = (
            requirements.requested_columns
            - selected_columns
        )

        unexpected_columns = (
            selected_columns
            - requirements.requested_columns
        )

        if missing_columns:
            errors.append(
                "SQL 没有返回用户明确要求的字段："
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        if unexpected_columns:
            errors.append(
                "SQL 返回了用户没有要求的"
                "冗余字段："
                + ", ".join(
                    sorted(unexpected_columns)
                )
            )

    # ------------------------------------------------
    # 3. 检查峰值字段是否使用 MAX
    # ------------------------------------------------
    for (
        column_name,
        aggregate_name,
    ) in requirements.required_aggregates.items():
        matching_projections = [
            projection_info
            for projection_info
            in alias_map.values()
            if column_name
            in projection_info.columns
        ]

        if (
            matching_projections
            and not any(
                aggregate_name
                in projection_info.aggregates
                for projection_info
                in matching_projections
            )
        ):
            errors.append(
                f"字段 {column_name} "
                f"应使用 "
                f"{aggregate_name.upper()} "
                "聚合后返回。"
            )

    # ------------------------------------------------
    # 4. 检查 Top-K 的 ORDER BY
    # ------------------------------------------------
    if requirements.ranking:
        order_info, actual_direction = (
            get_first_order_info(
                tree=tree,
                alias_map=alias_map,
            )
        )

        if order_info is None:
            errors.append(
                "Top-K 查询缺少顶层 ORDER BY，"
                "不能保证返回的是最高或最低记录。"
            )
        else:
            expected_column = (
                requirements.ranking.column
            )

            target_matches = (
                expected_column
                in order_info.columns
            )

            if not target_matches:
                errors.append(
                    "ORDER BY 排序字段与用户要求"
                    "不一致："
                    f"应按 {expected_column} 排序。"
                )

            expected_direction = (
                requirements.ranking.direction
            )

            if (
                actual_direction
                != expected_direction
            ):
                expected_sql_direction = (
                    "DESC"
                    if expected_direction == "desc"
                    else "ASC"
                )

                errors.append(
                    "ORDER BY 方向错误，"
                    f"应使用 "
                    f"{expected_sql_direction}。"
                )

            # 只有排序字段匹配时，再检查该字段
            # 是否应该聚合，避免产生误导错误。
            if target_matches:
                expected_aggregate = (
                    requirements.ranking.aggregate
                )

                if (
                    expected_aggregate
                    and expected_aggregate
                    not in order_info.aggregates
                ):
                    errors.append(
                        "排序目标需要先聚合："
                        f"应使用 "
                        f"{expected_aggregate.upper()}"
                        f"({expected_column})。"
                    )

                if (
                    expected_aggregate is None
                    and order_info.aggregates
                ):
                    errors.append(
                        "该 Top-K 查询应直接按"
                        "样本字段排序，"
                        "不应对排序字段额外"
                        "使用聚合函数。"
                    )

    # ------------------------------------------------
    # 5. 简单 Top-K 禁止子查询
    # ------------------------------------------------
    has_subquery = any(
        tree.find_all(exp.Subquery)
    )

    asks_average_comparison = any(
        word in question
        for word in (
            "平均",
            "均值",
            "高于平均",
            "低于平均",
        )
    )

    if (
        requirements.ranking
        and has_subquery
        and not asks_average_comparison
    ):
        errors.append(
            "当前是简单 Top-K 查询，"
            "不应使用子查询或 IN 子查询。"
        )

    # ------------------------------------------------
    # 6. 检查冗余 JOIN
    # ------------------------------------------------
    (
        all_tables,
        referenced_tables,
    ) = get_referenced_tables_outside_join(
        tree=tree,
        table_columns=table_columns,
    )

    redundant_tables = (
        all_tables - referenced_tables
    )

    if redundant_tables:
        errors.append(
            "SQL 连接了未参与返回、筛选、"
            "排序或聚合的冗余表："
            + ", ".join(
                sorted(redundant_tables)
            )
        )

    # ------------------------------------------------
    # 7. 检查无意义 GROUP BY
    # ------------------------------------------------
    group = tree.args.get("group")

    has_aggregate = any(
        tree.find_all(exp.AggFunc)
    )

    if (
        group is not None
        and not has_aggregate
    ):
        errors.append(
            "SQL 使用了 GROUP BY，"
            "但没有使用任何聚合函数；"
            "请删除不必要的 GROUP BY。"
        )

    asks_explicit_aggregation = any(
        word in question
        for word in AGGREGATION_REQUEST_WORDS
    )

    response_table = (
        settings.RESIN_TABLE_THERMAL_RESPONSE
    )

    used_tables = {
        table.name
        for table in tree.find_all(exp.Table)
    }

    if (
        group is not None
        and response_table not in used_tables
        and not asks_explicit_aggregation
    ):
        errors.append(
            "material_static 和 "
            "material_thermal_property "
            "都是一条样本一行，"
            "当前查询不需要 GROUP BY。"
        )

    # ------------------------------------------------
    # 8. LIMIT采用确定性修正
    # ------------------------------------------------
    if requirements.top_k is not None:
        set_limit(
            tree,
            min(
                requirements.top_k,
                max_rows,
            ),
        )
    else:
        current_limit = get_limit_value(tree)

        if (
            current_limit is None
            or current_limit > max_rows
        ):
            set_limit(tree, max_rows)

    return list(dict.fromkeys(errors))


def validate_and_normalize_sql(
    sql: str,
    question: str,
    allowed_tables: set[str],
    max_rows: int,
) -> SQLValidationResult:
    """校验 SQL 是否为安全的单条只读查询。"""

    cleaned_sql = clean_llm_sql(sql)

    if not cleaned_sql:
        return SQLValidationResult(
            valid=False,
            error="模型没有生成 SQL。",
        )

    if FORBIDDEN_PATTERN.search(cleaned_sql):
        return SQLValidationResult(
            valid=False,
            error="SQL 中包含禁止使用的操作或函数。",
        )

    try:
        statements = sqlglot.parse(
            cleaned_sql,
            read="mysql",
        )
    except ParseError as exc:
        return SQLValidationResult(
            valid=False,
            error=f"SQL 语法解析失败：{exc}",
        )

    if len(statements) != 1:
        return SQLValidationResult(
            valid=False,
            error="只允许执行一条 SQL。",
        )

    tree = statements[0]

    if tree.key != "select":
        return SQLValidationResult(
            valid=False,
            error=(
                "Text2SQL V0.1 只允许一条普通 "
                "SELECT 查询，暂不允许 UNION、"
                "INTERSECT 或其他复合查询。"
            ),
        )

    for node in tree.walk():
        node_key = getattr(node, "key", "")

        if node_key in BANNED_AST_KEYS:
            return SQLValidationResult(
                valid=False,
                error=f"SQL 语法树中发现禁止操作：{node_key}",
            )

    table_nodes = list(tree.find_all(exp.Table))

    if not table_nodes:
        return SQLValidationResult(
            valid=False,
            error="SQL 没有访问任何允许的数据表。",
        )

    used_tables: set[str] = set()

    for table in table_nodes:
        # 第一版禁止跨库查询。
        if table.db or table.catalog:
            return SQLValidationResult(
                valid=False,
                error="第一版禁止使用数据库名前缀或跨库查询。",
            )

        used_tables.add(table.name)

    unknown_tables = used_tables - allowed_tables

    if unknown_tables:
        return SQLValidationResult(
            valid=False,
            error=(
                "SQL 使用了不在白名单中的表："
                + ", ".join(sorted(unknown_tables))
            ),
        )
        
    quality_errors = validate_sql_quality(
    tree=tree,
    question=question,
    max_rows=max_rows,
)

    if quality_errors:
        return SQLValidationResult(
            valid=False,
            error=(
                "SQL 质量检查未通过：\n- "
                + "\n- ".join(quality_errors)
            ),
        )

    # LIMIT 已经由 validate_sql_quality
    # 根据用户Top-K或最大行数统一设置。
    normalized_sql = tree.sql(dialect="mysql").rstrip(";")

    # 查询未设置 LIMIT 时，添加默认限制。
    if tree.args.get("limit") is None:
        normalized_sql = f"{normalized_sql} LIMIT {max_rows}"

    return SQLValidationResult(
        valid=True,
        sql=normalized_sql,
    )