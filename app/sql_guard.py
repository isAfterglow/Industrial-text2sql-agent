import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.schema import (
    get_column_owner_map,
    get_schema_catalog,
    infer_question_ranking_column,
    infer_requested_output_columns,
    match_question_semantic_columns,
)


FORBIDDEN_PATTERN = re.compile(
    r"""
    \b(
        INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|
        TRUNCATE|REPLACE|MERGE|CALL|GRANT|REVOKE|
        SET|USE|LOAD_FILE|SLEEP|BENCHMARK
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


@dataclass
class SQLValidationResult:
    valid: bool
    sql: str = ""
    error: str = ""
    repairable: bool = True
    error_type: str = "schema"


def clean_llm_sql(content: str) -> str:
    """移除Markdown代码围栏和常见SQL前缀。"""

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


def normalize_sample_id_literals(sql: str) -> str:
    """统一SQL中的sample_id字面量格式。"""

    pattern = re.compile(
        r"""
        (?P<prefix>
            (?:\b\w+\.)?sample_id\s*=\s*
        )
        (?P<quote>['"]?)
        (?:sample_)?
        (?P<number>\d+)
        (?P=quote)
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    def replace(match: re.Match[str]) -> str:
        number = int(match.group("number"))
        return (
            f"{match.group('prefix')}"
            f"'sample_{number:06d}'"
        )

    return pattern.sub(replace, sql)


def build_declared_table_alias_map() -> dict[str, str]:
    """返回Schema声明的推荐别名到真实物理表的映射。"""

    catalog = get_schema_catalog()

    return {
        str(info["alias"]): table_name
        for table_name, info
        in catalog["tables"].items()
        if info.get("alias")
    }


def normalize_declared_table_aliases(
    tree: exp.Expression,
) -> bool:
    """把误写成物理表名的ms/mtp/tr确定性改回真实表。

    例如：
    FROM ms
    会改为：
    FROM material_static AS ms

    只处理Schema明确声明的别名，不猜测其他拼写错误。
    """

    alias_map = build_declared_table_alias_map()
    changed = False

    for table in tree.find_all(exp.Table):
        if table.db or table.catalog:
            continue

        original_name = table.name
        real_table = alias_map.get(
            original_name
        )

        if real_table is None:
            continue

        existing_alias = table.args.get(
            "alias"
        )

        table.set(
            "this",
            exp.Identifier(
                this=real_table
            ),
        )

        if existing_alias is None:
            table.set(
                "alias",
                exp.TableAlias(
                    this=exp.Identifier(
                        this=original_name
                    )
                ),
            )

        changed = True

    return changed


def _top_level_selected_columns(
    tree: exp.Select,
) -> set[str]:
    """返回顶层SELECT表达式实际引用的物理字段名。"""

    columns: set[str] = set()

    for projection in tree.expressions:
        for column in projection.find_all(
            exp.Column
        ):
            if nearest_select(column) is tree:
                columns.add(column.name)

    return columns


def _extract_output_request_text(
    question: str,
) -> str:
    """提取“返回/显示”等词之后的输出要求文本。"""

    markers = (
        "只返回",
        "只显示",
        "同时返回",
        "并返回",
        "返回",
        "显示",
        "列出",
    )

    positions = [
        (
            question.find(marker),
            len(marker),
        )
        for marker in markers
        if question.find(marker) >= 0
    ]

    if not positions:
        return ""

    position, marker_length = min(
        positions,
        key=lambda item: item[0],
    )

    return question[
        position + marker_length:
    ]


def validate_question_field_semantics(
    tree: exp.Select,
    question: str,
) -> list[str]:
    """确定性核对问题中的业务字段和明确输出字段。"""

    matches = match_question_semantic_columns(
        question
    )
    requested_outputs = (
        infer_requested_output_columns(
            question
        )
    )

    errors: list[str] = []
    used_columns = {
        column.name
        for column in tree.find_all(
            exp.Column
        )
    }
    selected_columns = (
        _top_level_selected_columns(
            tree
        )
    )

    for expected_column, terms in matches.items():
        if expected_column not in used_columns:
            errors.append(
                "用户问题中的"
                f"“{'、'.join(terms)}”"
                f"对应真实字段{expected_column}，"
                "但SQL没有使用该字段。"
            )

    for expected_column in sorted(
        requested_outputs
    ):
        if expected_column not in selected_columns:
            errors.append(
                "用户明确要求返回字段"
                f"{expected_column}，"
                "但顶层SELECT没有返回该真实字段。"
            )

    if requested_outputs:
        ranking_info = (
            infer_question_ranking_column(
                question
            )
        )
        allowed_selected = set(
            requested_outputs
        ) | {"sample_id"}

        # 排名字段即使未明确要求展示，返回它也不属于无关字段。
        if ranking_info is not None:
            allowed_selected.add(
                ranking_info[0]
            )

        unrelated_selected = (
            selected_columns
            - allowed_selected
        )
        if unrelated_selected:
            errors.append(
                "顶层SELECT返回了用户未要求的无关字段："
                + ", ".join(
                    sorted(
                        unrelated_selected
                    )
                )
                + "。"
            )

    catalog = get_schema_catalog()
    response_table = next(
        table_name
        for table_name, info
        in catalog["tables"].items()
        if info["grain"]
        == "many_rows_per_sample"
    )
    response_columns = set(
        catalog["tables"][
            response_table
        ]["columns"]
    ) - {"sample_id"}

    used_tables = {
        table.name
        for table in tree.find_all(
            exp.Table
        )
    }
    expected_columns = set(matches)

    if (
        response_table in used_tables
        and not (
            expected_columns
            & response_columns
        )
    ):
        errors.append(
            f"用户问题没有要求任何时序响应字段，"
            f"但SQL连接了{response_table}；"
            "这会把每个样本展开为多条响应记录。"
        )

    return list(dict.fromkeys(errors))


def validate_in_subquery_projection(
    tree: exp.Expression,
) -> list[str]:
    """检查IN左侧字段与子查询首个输出字段是否一致。"""

    errors: list[str] = []

    for in_expression in tree.find_all(
        exp.In
    ):
        left = _unwrap_parentheses(
            in_expression.this
        )
        inner = _subquery_select(
            in_expression.args.get(
                "query"
            )
        )

        if (
            not isinstance(
                left,
                exp.Column,
            )
            or inner is None
            or not inner.expressions
        ):
            continue

        projection = inner.expressions[0]
        inner_column = next(
            projection.find_all(
                exp.Column
            ),
            None,
        )

        if (
            inner_column is not None
            and left.name
            != inner_column.name
        ):
            errors.append(
                "IN条件左右字段不一致："
                f"外层使用{left.name}，"
                f"子查询返回{inner_column.name}。"
            )

    return errors



def _simplify_redundant_predicate(
    expression: exp.Expression,
) -> exp.Expression | None:
    """删除基于Schema不变式的sample_id LIKE 'sample_%'。"""

    expression = _unwrap_parentheses(
        expression
    )

    if isinstance(expression, exp.And):
        left = _simplify_redundant_predicate(
            expression.this
        )
        right = _simplify_redundant_predicate(
            expression.expression
        )

        if left is None:
            return right
        if right is None:
            return left

        expression.set("this", left)
        expression.set("expression", right)
        return expression

    if isinstance(expression, exp.Like):
        left = _unwrap_parentheses(
            expression.this
        )
        right = _unwrap_parentheses(
            expression.expression
        )

        if (
            isinstance(left, exp.Column)
            and left.name == "sample_id"
            and isinstance(right, exp.Literal)
            and right.is_string
            and right.this == "sample_%"
        ):
            return None

    return expression


def normalize_redundant_predicates(
    tree: exp.Select,
) -> bool:
    """删除不会改变结果的固定sample_id格式过滤。"""

    where = tree.args.get("where")
    if where is None:
        return False

    simplified = _simplify_redundant_predicate(
        where.this
    )

    if simplified is None:
        tree.set("where", None)
        return True

    if simplified is not where.this:
        tree.set(
            "where",
            exp.Where(this=simplified),
        )
        return True

    return False


def extract_requested_limit(
    question: str,
) -> int | None:
    """提取用户明确要求的最大返回数量。"""

    patterns = (
        r"最多(?:返回)?\s*(\d+)\s*条",
        r"(?:最高|最大|最低|最小)(?:的)?\s*(\d+)\s*(?:个|条|项|组)?",
        r"(?:前|top\s*)\s*(\d+)\s*(?:个|条|项|组)?",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )
        if match:
            return int(match.group(1))

    return None


def _question_numbers_without_sample_id(
    question: str,
) -> set[str]:
    cleaned = re.sub(
        r"sample_\d{6}",
        "",
        question,
        flags=re.IGNORECASE,
    )

    return {
        number.lstrip("+")
        for number in re.findall(
            r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?",
            cleaned,
        )
    }


def validate_question_numeric_values(
    tree: exp.Select,
    question: str,
) -> list[str]:
    """检查SQL条件和LIMIT中的数值是否来自用户问题。"""

    expected_numbers = (
        _question_numbers_without_sample_id(
            question
        )
    )

    if not expected_numbers:
        return []

    actual_numbers: set[str] = set()

    clause_names = [
        "where",
        "having",
    ]
    if extract_requested_limit(question) is not None:
        clause_names.append("limit")

    for clause_name in clause_names:
        clause = tree.args.get(
            clause_name
        )
        if clause is None:
            continue

        for literal in clause.find_all(
            exp.Literal
        ):
            if literal.is_number:
                actual_numbers.add(
                    str(literal.this).lstrip("+")
                )

    missing = expected_numbers - actual_numbers
    if missing:
        return [
            "用户问题中的数值没有完整体现在SQL条件或LIMIT中："
            + ", ".join(sorted(missing))
            + "。"
        ]

    unexpected = actual_numbers - expected_numbers
    if unexpected:
        return [
            "SQL加入了用户问题中未出现的数值条件："
            + ", ".join(sorted(unexpected))
            + "。"
        ]

    return []


def validate_unrequested_predicates(
    tree: exp.Select,
    question: str,
) -> list[str]:
    """拦截小模型常擅自加入的非空过滤。"""

    where = tree.args.get("where")
    if where is None:
        return []

    where_sql = where.sql(
        dialect="mysql"
    ).upper()

    if (
        "IS NOT NULL" in where_sql
        and not re.search(
            r"非空|不为空|有值",
            question,
        )
    ):
        return [
            "SQL加入了用户未要求的IS NOT NULL过滤条件。"
        ]

    return []

def build_table_columns() -> dict[str, set[str]]:
    catalog = get_schema_catalog()

    return {
        table_name: set(info["columns"])
        for table_name, info
        in catalog["tables"].items()
    }


def build_column_owners(
    table_columns: dict[str, set[str]],
) -> dict[str, set[str]]:
    owners: dict[str, set[str]] = {}

    for table_name, columns in table_columns.items():
        for column in columns:
            owners.setdefault(
                column,
                set(),
            ).add(table_name)

    return owners


def nearest_select(
    node: exp.Expression,
) -> exp.Select | None:
    """返回节点所属的最近一层SELECT作用域。"""

    parent = node.parent

    while parent is not None:
        if isinstance(parent, exp.Select):
            return parent

        parent = parent.parent

    return None


def direct_scope_nodes(
    select: exp.Select,
    node_type,
) -> list[exp.Expression]:
    """返回仅属于当前SELECT作用域的节点。"""

    return [
        node
        for node in select.find_all(node_type)
        if nearest_select(node) is select
    ]


def get_subquery_outputs(
    subquery: exp.Subquery,
) -> set[str]:
    """获取派生表对外暴露的列名。"""

    inner = subquery.this

    if not isinstance(inner, exp.Select):
        return set()

    outputs: set[str] = set()

    for projection in inner.expressions:
        output_name = projection.alias_or_name

        if output_name:
            outputs.add(output_name)

    return outputs


def validate_select_scope(
    select: exp.Select,
    table_columns: dict[str, set[str]],
    column_owners: dict[str, set[str]],
) -> list[str]:
    """在单个SELECT作用域内校验字段归属和歧义。

    这样不会把内层子查询的别名误判为外层未知字段。
    """

    errors: list[str] = []

    physical_aliases: dict[str, str] = {}

    for table in direct_scope_nodes(
        select,
        exp.Table,
    ):
        physical_aliases[
            table.alias_or_name
        ] = table.name

    derived_aliases: dict[str, set[str]] = {}

    for subquery in direct_scope_nodes(
        select,
        exp.Subquery,
    ):
        if subquery.alias:
            derived_aliases[
                subquery.alias
            ] = get_subquery_outputs(
                subquery
            )

    select_aliases = {
        projection.alias
        for projection in select.expressions
        if projection.alias
    }

    physical_tables = set(
        physical_aliases.values()
    )

    for column in direct_scope_nodes(
        select,
        exp.Column,
    ):
        name = column.name

        # ORDER BY / HAVING可以使用本层SELECT别名。
        if (
            not column.table
            and name in select_aliases
        ):
            continue

        if column.table:
            qualifier = column.table

            if qualifier in physical_aliases:
                table_name = physical_aliases[
                    qualifier
                ]

                if name not in table_columns.get(
                    table_name,
                    set(),
                ):
                    actual = sorted(
                        column_owners.get(
                            name,
                            set(),
                        )
                    )

                    if actual:
                        errors.append(
                            f"字段归属错误：{name}不属于"
                            f"{table_name}，实际属于"
                            + ", ".join(actual)
                            + "。"
                        )
                    else:
                        errors.append(
                            f"未知字段：{name}。"
                        )

                continue

            if qualifier in derived_aliases:
                outputs = derived_aliases[
                    qualifier
                ]

                if outputs and name not in outputs:
                    errors.append(
                        f"派生表{qualifier}没有输出字段"
                        f"{name}。"
                    )

                continue

            errors.append(
                f"未知表或派生表别名：{qualifier}。"
            )
            continue

        physical_owners = (
            column_owners.get(name, set())
            & physical_tables
        )

        derived_owner_count = sum(
            1
            for outputs in derived_aliases.values()
            if not outputs or name in outputs
        )

        total_owner_count = (
            len(physical_owners)
            + derived_owner_count
        )

        if total_owner_count > 1:
            errors.append(
                f"字段{name}在当前查询的多个来源中存在，"
                "必须使用表别名限定。"
            )
        elif total_owner_count == 0:
            errors.append(
                f"未知字段：{name}。"
            )

    return errors


def validate_column_ownership(
    tree: exp.Expression,
) -> list[str]:
    """按SELECT作用域校验全部字段。"""

    table_columns = build_table_columns()
    column_owners = build_column_owners(
        table_columns
    )

    errors: list[str] = []

    for select in tree.find_all(exp.Select):
        errors.extend(
            validate_select_scope(
                select=select,
                table_columns=table_columns,
                column_owners=column_owners,
            )
        )

    return list(dict.fromkeys(errors))


def validate_join_structure(
    tree: exp.Expression,
) -> list[str]:
    """检查明显错误的JOIN结构。"""

    errors: list[str] = []

    for join in tree.find_all(exp.Join):
        kind = str(
            join.args.get("kind") or ""
        ).upper()

        if kind == "CROSS":
            errors.append(
                "禁止CROSS JOIN。"
            )
            continue

        if (
            join.args.get("on") is None
            and join.args.get("using") is None
        ):
            errors.append(
                "JOIN缺少ON或USING连接条件。"
            )

    return errors


def projection_contains_star(
    tree: exp.Expression,
) -> bool:
    """禁止SELECT *，但允许COUNT(*)。"""

    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            stars = list(
                projection.find_all(exp.Star)
            )

            if not stars:
                continue

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


def extract_simple_topk_intent(
    question: str,
) -> tuple[int, str, bool] | None:
    """提取通用Top-K数量、方向和是否需要聚合。

    这里只识别稳定的数量/方向语义，不识别具体业务字段。
    """

    count: int | None = None

    patterns = (
        r"(?:最高|最大|最低|最小)(?:的)?\s*(\d+)\s*(?:个|条|项|组)?",
        r"(?:前|top\s*)\s*(\d+)\s*(?:个|条|项|组)?",
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )
        if match:
            count = int(match.group(1))
            break

    if count is None or count <= 0:
        return None

    if re.search(r"最高|最大", question):
        direction = "desc"
    elif re.search(r"最低|最小", question):
        direction = "asc"
    elif re.search(r"降序|从高到低", question):
        direction = "desc"
    elif re.search(r"升序|从低到高", question):
        direction = "asc"
    else:
        return None

    requires_aggregation = bool(
        re.search(
            r"峰值|平均|均值|总和|合计",
            question,
        )
    )

    return count, direction, requires_aggregation


def _unwrap_parentheses(
    expression: exp.Expression,
) -> exp.Expression:
    while isinstance(expression, exp.Paren):
        expression = expression.this

    return expression


def _subquery_select(
    expression: exp.Expression | None,
) -> exp.Select | None:
    if expression is None:
        return None

    expression = _unwrap_parentheses(expression)

    if isinstance(expression, exp.Subquery):
        expression = expression.this

    if isinstance(expression, exp.Select):
        return expression

    return None


def _direct_table_aliases(
    select: exp.Select,
) -> dict[str, str]:
    return {
        table.alias_or_name: table.name
        for table in direct_scope_nodes(
            select,
            exp.Table,
        )
    }


def _adapt_order_to_outer_scope(
    order: exp.Order,
    inner: exp.Select,
    outer: exp.Select,
) -> exp.Order:
    """把内层ORDER BY中的表别名映射到外层同一物理表别名。"""

    order_copy = order.copy()
    inner_aliases = _direct_table_aliases(inner)
    outer_aliases = _direct_table_aliases(outer)

    outer_alias_by_table = {
        table_name: alias
        for alias, table_name in outer_aliases.items()
    }

    for column in order_copy.find_all(exp.Column):
        if not column.table:
            continue

        table_name = inner_aliases.get(
            column.table
        )
        outer_alias = outer_alias_by_table.get(
            table_name
        )

        if outer_alias:
            column.set(
                "table",
                exp.Identifier(this=outer_alias),
            )

    return order_copy


def _has_direct_aggregate(
    select: exp.Select,
) -> bool:
    return any(
        nearest_select(aggregate) is select
        for aggregate in select.find_all(
            exp.AggFunc
        )
    )


def _remove_nonaggregate_group(
    select: exp.Select,
) -> None:
    if (
        select.args.get("group") is not None
        and not _has_direct_aggregate(select)
        and select.args.get("having") is None
    ):
        select.set("group", None)


def _normalize_in_topk(
    tree: exp.Select,
    predicate: exp.Expression,
    count: int,
) -> bool:
    """化简WHERE ... IN (SELECT ... ORDER BY ... LIMIT N)。"""

    predicate = _unwrap_parentheses(predicate)

    if not isinstance(predicate, exp.In):
        return False

    inner = _subquery_select(
        predicate.args.get("query")
    )
    if inner is None:
        return False

    inner_order = inner.args.get("order")
    inner_limit = get_limit_value(inner)

    if (
        inner_order is None
        or inner_limit is None
        or inner_limit != count
        or inner.args.get("where") is not None
        or inner.args.get("having") is not None
        or inner.args.get("group") is not None
        or _has_direct_aggregate(inner)
    ):
        return False

    outer_column = _unwrap_parentheses(
        predicate.this
    )
    first_projection = (
        inner.expressions[0]
        if inner.expressions
        else None
    )
    inner_column = (
        next(
            first_projection.find_all(
                exp.Column
            ),
            None,
        )
        if first_projection is not None
        else None
    )

    if (
        not isinstance(
            outer_column,
            exp.Column,
        )
        or inner_column is None
        or outer_column.name
        != inner_column.name
    ):
        return False

    outer_tables = set(
        _direct_table_aliases(tree).values()
    )
    inner_tables = set(
        _direct_table_aliases(inner).values()
    )

    if not (outer_tables & inner_tables):
        return False

    tree.set("where", None)
    tree.set(
        "order",
        _adapt_order_to_outer_scope(
            inner_order,
            inner,
            tree,
        ),
    )
    set_limit(tree, count)
    _remove_nonaggregate_group(tree)

    return True


def _find_scalar_aggregate(
    select: exp.Select,
) -> tuple[str, exp.Column] | None:
    for aggregate in select.find_all(
        exp.AggFunc
    ):
        if nearest_select(aggregate) is not select:
            continue

        if isinstance(aggregate, exp.Min):
            function_name = "min"
        elif isinstance(aggregate, exp.Max):
            function_name = "max"
        else:
            continue

        column = next(
            aggregate.find_all(exp.Column),
            None,
        )
        if column is not None:
            return function_name, column

    return None


def _normalize_scalar_topk(
    tree: exp.Select,
    predicate: exp.Expression,
    count: int,
    direction: str,
) -> bool:
    """化简column = (SELECT MIN/MAX(column) ...)形式的Top-K。"""

    predicate = _unwrap_parentheses(predicate)

    if not isinstance(predicate, exp.EQ):
        return False

    left = _unwrap_parentheses(
        predicate.this
    )
    right = _unwrap_parentheses(
        predicate.expression
    )

    outer_column: exp.Column | None = None
    inner: exp.Select | None = None

    if isinstance(left, exp.Column):
        candidate = _subquery_select(right)
        if candidate is not None:
            outer_column = left
            inner = candidate

    if outer_column is None and isinstance(
        right,
        exp.Column,
    ):
        candidate = _subquery_select(left)
        if candidate is not None:
            outer_column = right
            inner = candidate

    if outer_column is None or inner is None:
        return False

    if (
        inner.args.get("where") is not None
        or inner.args.get("having") is not None
        or inner.args.get("group") is not None
    ):
        return False

    aggregate_info = _find_scalar_aggregate(
        inner
    )
    if aggregate_info is None:
        return False

    _, aggregate_column = aggregate_info

    if aggregate_column.name != outer_column.name:
        return False

    tree.set("where", None)
    tree.set(
        "order",
        exp.Order(
            expressions=[
                exp.Ordered(
                    this=outer_column.copy(),
                    desc=(direction == "desc"),
                )
            ]
        ),
    )
    set_limit(tree, count)
    _remove_nonaggregate_group(tree)

    return True


def normalize_common_topk_sql(
    tree: exp.Expression,
    question: str,
) -> bool:
    """化简小模型常生成的简单Top-K复杂SQL。

    仅在问题明确是非聚合Top-K、且WHERE整体就是可安全识别的
    IN-LIMIT子查询或MIN/MAX标量子查询时改写。
    """

    intent = extract_simple_topk_intent(
        question
    )

    if intent is None:
        return False

    count, direction, requires_aggregation = intent

    if requires_aggregation:
        return False

    if not isinstance(tree, exp.Select):
        return False

    where = tree.args.get("where")
    if where is None:
        return False

    predicate = where.this

    if _normalize_in_topk(
        tree,
        predicate,
        count,
    ):
        return True

    return _normalize_scalar_topk(
        tree,
        predicate,
        count,
        direction,
    )


def validate_mysql_limit_in_subquery(
    tree: exp.Expression,
) -> list[str]:
    """拦截当前MySQL版本不支持的IN子查询LIMIT组合。"""

    errors: list[str] = []

    for in_expression in tree.find_all(exp.In):
        inner = _subquery_select(
            in_expression.args.get("query")
        )

        if (
            inner is not None
            and get_limit_value(inner) is not None
        ):
            errors.append(
                "当前MySQL版本不支持IN子查询中使用LIMIT；"
                "请改为直接ORDER BY ... LIMIT，或使用派生表JOIN。"
            )

    return errors


def check_simple_topk_shape(
    question: str,
    sql: str,
) -> tuple[bool | None, str]:
    """确定性检查Top-K数量和排序方向。

    返回None表示当前问题不是可识别的Top-K，不参与判断。
    """

    intent = extract_simple_topk_intent(
        question
    )
    if intent is None:
        return None, "不是可确定检查的Top-K问题。"

    count, direction, requires_aggregation = intent

    try:
        tree = sqlglot.parse_one(
            sql,
            read="mysql",
        )
    except ParseError as exc:
        return False, f"Top-K检查无法解析SQL：{exc}"

    if not isinstance(tree, exp.Select):
        return False, "Top-K查询不是SELECT。"

    actual_limit = get_limit_value(tree)
    if actual_limit != count:
        return (
            False,
            f"Top-K数量不一致：用户要求{count}条，"
            f"SQL顶层LIMIT为{actual_limit}。",
        )

    order = tree.args.get("order")
    if order is None or not order.expressions:
        return False, "Top-K查询缺少顶层ORDER BY。"

    first_order = order.expressions[0]
    actual_direction = (
        "desc"
        if first_order.args.get("desc")
        else "asc"
    )

    if actual_direction != direction:
        expected = (
            "DESC"
            if direction == "desc"
            else "ASC"
        )
        return False, f"Top-K排序方向错误，应使用{expected}。"

    ranking_info = infer_question_ranking_column(
        question
    )
    if ranking_info is not None:
        expected_ranking_column = (
            ranking_info[0]
        )

        order_columns = {
            column.name
            for column in first_order.find_all(
                exp.Column
            )
        }

        if (
            len(order_columns) == 1
            and next(iter(order_columns))
            in {
                projection.alias
                for projection in tree.expressions
                if projection.alias
            }
        ):
            alias_name = next(
                iter(order_columns)
            )
            projection = next(
                (
                    item
                    for item in tree.expressions
                    if item.alias
                    == alias_name
                ),
                None,
            )
            if projection is not None:
                order_columns = {
                    column.name
                    for column
                    in projection.find_all(
                        exp.Column
                    )
                }

        if (
            expected_ranking_column
            not in order_columns
        ):
            return (
                False,
                "Top-K排序字段错误："
                f"用户要求按{expected_ranking_column}排序，"
                "但SQL顶层ORDER BY没有使用该真实字段。",
            )

    if requires_aggregation:
        where = tree.args.get("where")
        if (
            "峰值" in question
            and where is not None
            and any(
                column.name
                == "point_index"
                for column
                in where.find_all(
                    exp.Column
                )
            )
        ):
            return (
                False,
                "峰值必须基于完整响应序列使用MAX聚合，"
                "不能用固定point_index代替峰值。",
            )
        direct_aggregates = [
            aggregate
            for aggregate in tree.find_all(exp.AggFunc)
            if nearest_select(aggregate) is tree
        ]

        if "峰值" in question and not any(
            isinstance(aggregate, exp.Max)
            for aggregate in direct_aggregates
        ):
            return False, "峰值Top-K查询缺少MAX聚合。"

        if re.search(r"平均|均值", question) and not any(
            isinstance(aggregate, exp.Avg)
            for aggregate in direct_aggregates
        ):
            return False, "平均值Top-K查询缺少AVG聚合。"

        if tree.args.get("group") is None:
            return False, "样本级聚合Top-K查询缺少GROUP BY。"

    return True, "Top-K数量、方向、排序字段和基本聚合结构通过确定性检查。"



def assess_deterministic_semantic_coverage(
    question: str,
    sql: str,
) -> tuple[bool, str]:
    """判断当前SQL是否已被确定性规则充分覆盖。

    覆盖充分时不再调用LLM审查，避免正确SQL被小模型误杀。
    """

    try:
        tree = sqlglot.parse_one(
            sql,
            read="mysql",
        )
    except ParseError as exc:
        return False, f"无法解析SQL：{exc}"

    if not isinstance(tree, exp.Select):
        return False, "不是普通SELECT。"

    if len(list(tree.find_all(exp.Select))) != 1:
        return False, "包含子查询或多层SELECT。"

    if (
        tree.args.get("with") is not None
        or tree.args.get("having") is not None
        or any(
            isinstance(node, exp.Window)
            for node in tree.walk()
        )
    ):
        return False, "包含CTE、HAVING或窗口函数。"

    semantic_matches = (
        match_question_semantic_columns(
            question
        )
    )
    requested_outputs = (
        infer_requested_output_columns(
            question
        )
    )

    if (
        not semantic_matches
        and "全部静态材料参数"
        not in question
    ):
        return False, "问题缺少可确定映射的业务字段。"

    semantic_errors = (
        validate_question_field_semantics(
            tree,
            question,
        )
        + validate_question_numeric_values(
            tree,
            question,
        )
        + validate_unrequested_predicates(
            tree,
            question,
        )
    )
    if semantic_errors:
        return False, semantic_errors[0]

    topk_status, topk_reason = (
        check_simple_topk_shape(
            question,
            sql,
        )
    )
    if topk_status is False:
        return False, topk_reason

    direct_aggregates = [
        aggregate
        for aggregate in tree.find_all(
            exp.AggFunc
        )
        if nearest_select(aggregate) is tree
    ]

    if direct_aggregates:
        if "峰值" in question and not all(
            isinstance(item, exp.Max)
            for item in direct_aggregates
        ):
            return False, "峰值聚合不是MAX。"

        if re.search(r"平均|均值", question) and not all(
            isinstance(item, exp.Avg)
            for item in direct_aggregates
        ):
            return False, "平均值聚合不是AVG。"

        if tree.args.get("group") is None:
            return False, "聚合查询缺少GROUP BY。"

    if len(list(tree.find_all(exp.Table))) > 3:
        return False, "使用的数据表过多。"

    if requested_outputs:
        selected = _top_level_selected_columns(
            tree
        )
        ranking = infer_question_ranking_column(
            question
        )
        allowed = set(requested_outputs) | {
            "sample_id"
        }
        if ranking is not None:
            allowed.add(ranking[0])

        if selected - allowed:
            return False, "返回字段超出用户明确要求。"

    reason_parts = [
        "字段映射、返回字段、数值条件和表结构均通过确定性检查。"
    ]
    if topk_status is True:
        reason_parts.append(topk_reason)

    return True, " ".join(reason_parts)

def build_sql_review_summary(sql: str) -> str:
    """生成供LLM语义审查使用的确定性SQL摘要。

    摘要只描述SQL实际使用的表、返回表达式、过滤、排序、
    分组、聚合和LIMIT，不尝试理解用户自然语言。
    """

    try:
        tree = sqlglot.parse_one(
            sql,
            read="mysql",
        )
    except ParseError as exc:
        return f"SQL摘要生成失败：{exc}"

    if not isinstance(tree, exp.Select):
        return (
            "SQL类型："
            + tree.key
        )

    def render_expressions(
        expressions: list[exp.Expression],
    ) -> str:
        if not expressions:
            return "无"

        return "; ".join(
            expression.sql(
                dialect="mysql"
            )
            for expression in expressions
        )

    direct_tables = direct_scope_nodes(
        tree,
        exp.Table,
    )
    table_items = []

    for table in direct_tables:
        if table.alias and table.alias != table.name:
            table_items.append(
                f"{table.name} AS {table.alias}"
            )
        else:
            table_items.append(table.name)

    joins = [
        join.sql(dialect="mysql")
        for join in tree.args.get(
            "joins",
            [],
        )
    ]

    where = tree.args.get("where")
    group = tree.args.get("group")
    order = tree.args.get("order")
    having = tree.args.get("having")

    direct_aggregates = [
        aggregate.sql(
            dialect="mysql"
        )
        for aggregate in tree.find_all(
            exp.AggFunc
        )
        if nearest_select(aggregate) is tree
    ]

    lines = [
        "SQL结构摘要：",
        "- 顶层数据表："
        + (", ".join(table_items) or "无"),
        "- 返回表达式："
        + render_expressions(
            list(tree.expressions)
        ),
        "- JOIN："
        + ("; ".join(joins) or "无"),
        "- WHERE："
        + (
            where.this.sql(dialect="mysql")
            if where is not None
            else "无"
        ),
        "- GROUP BY："
        + (
            render_expressions(
                list(group.expressions)
            )
            if group is not None
            else "无"
        ),
        "- HAVING："
        + (
            having.this.sql(dialect="mysql")
            if having is not None
            else "无"
        ),
        "- ORDER BY："
        + (
            render_expressions(
                list(order.expressions)
            )
            if order is not None
            else "无"
        ),
        "- 顶层聚合："
        + (
            ", ".join(direct_aggregates)
            or "无"
        ),
        "- 顶层LIMIT："
        + (
            str(get_limit_value(tree))
            if get_limit_value(tree) is not None
            else "无"
        ),
    ]

    if any(
        table.name == "thermal_response"
        for table in tree.find_all(
            exp.Table
        )
    ):
        lines.append(
            "- 粒度提醒：SQL使用了thermal_response，"
            "该表一个样本通常对应多条时序记录。"
        )

    return "\n".join(lines)


def validate_and_normalize_sql(
    sql: str,
    allowed_tables: set[str],
    max_rows: int,
    question: str = "",
) -> SQLValidationResult:
    """只执行确定性的安全、Schema和资源检查。"""

    cleaned_sql = normalize_sample_id_literals(
        clean_llm_sql(sql)
    )

    if not cleaned_sql:
        return SQLValidationResult(
            valid=False,
            error="模型没有生成SQL。",
            repairable=True,
            error_type="generation",
        )

    if FORBIDDEN_PATTERN.search(cleaned_sql):
        return SQLValidationResult(
            valid=False,
            error="SQL中包含禁止使用的操作或函数。",
            repairable=False,
            error_type="policy",
        )

    try:
        statements = sqlglot.parse(
            cleaned_sql,
            read="mysql",
        )
    except ParseError as exc:
        return SQLValidationResult(
            valid=False,
            error=f"SQL语法解析失败：{exc}",
            repairable=True,
            error_type="syntax",
        )

    if len(statements) != 1:
        return SQLValidationResult(
            valid=False,
            error="只允许执行一条SQL。",
            repairable=False,
            error_type="policy",
        )

    tree = statements[0]

    if tree.key != "select":
        return SQLValidationResult(
            valid=False,
            error="只允许普通SELECT查询。",
            repairable=False,
            error_type="policy",
        )

    normalize_declared_table_aliases(
        tree
    )
    normalize_redundant_predicates(
        tree
    )

    if question:
        normalize_common_topk_sql(
            tree,
            question,
        )

    for node in tree.walk():
        key = getattr(node, "key", "")

        if key in BANNED_AST_KEYS:
            return SQLValidationResult(
                valid=False,
                error=(
                    "SQL语法树中发现禁止操作："
                    f"{key}"
                ),
                repairable=False,
                error_type="policy",
            )

    tables = list(
        tree.find_all(exp.Table)
    )

    if not tables:
        return SQLValidationResult(
            valid=False,
            error="SQL没有访问任何数据表。",
            repairable=True,
            error_type="schema",
        )

    used_tables: set[str] = set()

    for table in tables:
        if table.db or table.catalog:
            return SQLValidationResult(
                valid=False,
                error="禁止跨库查询。",
                repairable=False,
                error_type="policy",
            )

        used_tables.add(table.name)

    unknown_tables = (
        used_tables - allowed_tables
    )

    if unknown_tables:
        return SQLValidationResult(
            valid=False,
            error=(
                "SQL使用了Schema中不存在的表："
                + ", ".join(
                    sorted(unknown_tables)
                )
                + "。这通常是模型把别名当成表名，"
                "或生成了不存在的表名。"
            ),
            repairable=True,
            error_type="schema",
        )

    errors: list[str] = []

    if projection_contains_star(tree):
        errors.append(
            "禁止使用SELECT *，请明确返回字段。"
        )

    errors.extend(
        validate_column_ownership(tree)
    )
    errors.extend(
        validate_join_structure(tree)
    )
    errors.extend(
        validate_in_subquery_projection(
            tree
        )
    )
    errors.extend(
        validate_mysql_limit_in_subquery(tree)
    )

    if question:
        errors.extend(
            validate_question_field_semantics(
                tree,
                question,
            )
        )
        errors.extend(
            validate_unrequested_predicates(
                tree,
                question,
            )
        )
        errors.extend(
            validate_question_numeric_values(
                tree,
                question,
            )
        )

    errors = list(dict.fromkeys(errors))

    if errors:
        return SQLValidationResult(
            valid=False,
            error=(
                "SQL确定性检查未通过：\n- "
                + "\n- ".join(errors)
            ),
            repairable=True,
            error_type="schema",
        )

    requested_limit = (
        extract_requested_limit(question)
        if question
        else None
    )

    if requested_limit is not None:
        set_limit(
            tree,
            min(
                max(requested_limit, 1),
                max_rows,
            ),
        )
    else:
        # 用户未指定数量时，不保留模型擅自生成的LIMIT 1等值。
        set_limit(
            tree,
            max_rows,
        )

    return SQLValidationResult(
        valid=True,
        sql=tree.sql(
            dialect="mysql"
        ).rstrip(";"),
        repairable=True,
        error_type="none",
    )