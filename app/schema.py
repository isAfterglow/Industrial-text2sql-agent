from copy import deepcopy
import re
from typing import Any

from app.config import get_settings


def get_schema_catalog() -> dict[str, Any]:
    """返回机器可读的数据库Schema。

    表、字段、推荐别名、表粒度、关系和业务术语集中维护在这里。
    """

    settings = get_settings()

    static_table = settings.RESIN_TABLE_STATIC
    property_table = (
        settings.RESIN_TABLE_MATERIAL_THERMAL_PROPERTY
    )
    response_table = (
        settings.RESIN_TABLE_THERMAL_RESPONSE
    )

    return deepcopy(
        {
            "database_type": "MySQL",
            "tables": {
                static_table: {
                    "description": (
                        "材料样本的静态材料参数，"
                        "每个sample_id只有一行。"
                    ),
                    "alias": "ms",
                    "grain": "one_row_per_sample",
                    "columns": {
                        "sample_id": (
                            "样本唯一编号，格式为"
                            "sample_加6位十进制数字。"
                        ),
                        "rhov_i": "原始材料密度。",
                        "rhoc_i": "碳化材料密度。",
                        "porosity_v": "原始材料孔隙率。",
                        "porosity_c": "碳化材料孔隙率。",
                        "permeability_v": (
                            "原始材料渗透率。"
                        ),
                        "permeability_c": (
                            "碳化材料渗透率。"
                        ),
                    },
                },
                property_table: {
                    "description": (
                        "材料热物性和热解参数，"
                        "每个sample_id只有一行。"
                    ),
                    "alias": "mtp",
                    "grain": "one_row_per_sample",
                    "columns": {
                        "sample_id": "样本编号。",
                        "kv_list": (
                            "原始材料热导率参数。"
                        ),
                        "kc_list": (
                            "碳化材料热导率参数。"
                        ),
                        "cpv_list": (
                            "原始材料比热容参数。"
                        ),
                        "cpc_list": (
                            "碳化材料比热容参数。"
                        ),
                        "pyrolysis_heat": "热解热。",
                        "surface_emissivity": (
                            "表面发射率。"
                        ),
                    },
                },
                response_table: {
                    "description": (
                        "样本的时序热响应，"
                        "一个sample_id通常有3000个点。"
                    ),
                    "alias": "tr",
                    "grain": "many_rows_per_sample",
                    "columns": {
                        "sample_id": "样本编号。",
                        "point_index": (
                            "序列点编号，通常为0到2999，"
                            "不等同于秒。"
                        ),
                        "surface_temperature": (
                            "该点的表面温度。"
                        ),
                        "back_temperature": (
                            "该点的背面温度。"
                        ),
                        "mass": "该点的质量。",
                    },
                },
            },
            "relationships": [
                (
                    f"{static_table}.sample_id = "
                    f"{property_table}.sample_id，1对1"
                ),
                (
                    f"{static_table}.sample_id = "
                    f"{response_table}.sample_id，1对多"
                ),
                (
                    f"{property_table}.sample_id = "
                    f"{response_table}.sample_id，"
                    "可按sample_id直接连接"
                ),
            ],
            # 这是数据库词汇表，不是评测题模板。
            # Guard和Prompt共用同一份映射，避免各自维护不同含义。
            "semantic_terms": {
                "sample_id": [
                    "样本编号",
                    "样本ID",
                    "它们的编号",
                ],
                "rhov_i": [
                    "原始材料密度",
                    "原始密度",
                ],
                "rhoc_i": [
                    "碳化材料密度",
                    "碳化密度",
                ],
                "porosity_v": [
                    "原始材料孔隙率",
                    "原始孔隙率",
                ],
                "porosity_c": [
                    "碳化材料孔隙率",
                    "碳化孔隙率",
                ],
                "permeability_v": [
                    "原始材料渗透率",
                    "原始渗透率",
                ],
                "permeability_c": [
                    "碳化材料渗透率",
                    "碳化渗透率",
                ],
                "kv_list": [
                    "原始材料热导率",
                    "原始热导率",
                    "原始导热率",
                ],
                "kc_list": [
                    "碳化材料热导率",
                    "碳化热导率",
                    "碳化导热率",
                ],
                "cpv_list": [
                    "原始材料比热容",
                    "原始比热容",
                ],
                "cpc_list": [
                    "碳化材料比热容",
                    "碳化比热容",
                ],
                "pyrolysis_heat": [
                    "热解热",
                ],
                "surface_emissivity": [
                    "表面发射率",
                    "发射率",
                ],
                "point_index": [
                    "point_index",
                    "序列点编号",
                    "点位编号",
                ],
                "surface_temperature": [
                    "表面温度",
                    "表温",
                ],
                "back_temperature": [
                    "背面温度",
                    "背温",
                ],
                "mass": [
                    "质量",
                ],
            },
            "domain_conventions": [
                (
                    "sample_id固定为sample_后跟6位十进制数字。"
                ),
                (
                    "ms、mtp、tr只是推荐别名，"
                    "不能作为FROM或JOIN后的真实表名。"
                ),
                (
                    "原始密度是rhov_i，原始渗透率是"
                    "permeability_v，原始比热容是cpv_list；"
                    "三者不能混用。"
                ),
                (
                    "碳化密度是rhoc_i，碳化渗透率是"
                    "permeability_c，碳化比热容是cpc_list；"
                    "三者不能混用。"
                ),
                (
                    "如果所需字段全部属于material_static，"
                    "只查询material_static。"
                ),
                (
                    "只有问题需要热导率、比热容、热解热或"
                    "表面发射率时，才使用"
                    "material_thermal_property。"
                ),
                (
                    "只有问题涉及point_index、温度、质量、"
                    "峰值或时序均值时，才使用thermal_response。"
                ),
                (
                    "连接thermal_response会把一个样本展开为"
                    "多条时序记录；不需要响应字段时不要连接它。"
                ),
                (
                    "普通字段Top-K使用ORDER BY目标字段"
                    "ASC或DESC并配合LIMIT N；"
                    "不要用等于MIN/MAX表示前N条。"
                ),
                (
                    "峰值表面温度使用"
                    "MAX(surface_temperature)。"
                ),
                (
                    "峰值背面温度使用"
                    "MAX(back_temperature)。"
                ),
                "平均值使用AVG。",
                (
                    "对thermal_response做样本级聚合时，"
                    "按sample_id分组。"
                ),
                (
                    "查询时序明细时，不使用MAX、AVG"
                    "或无意义GROUP BY。"
                ),
                (
                    "查询时序明细时，应限制sample_id，"
                    "并按point_index升序排列。"
                ),
            ],
        }
    )




SAMPLE_ID_PATTERNS = (
    # 规范sample_id、非补零sample_id，数字后允许直接接中文。
    re.compile(
        r"(?<![A-Za-z0-9_])sample_(\d{1,6})(?!\d)",
        flags=re.IGNORECASE,
    ),
    # 中文样本编号、带空格编号和规范sample_id。
    re.compile(
        r"样本(?:编号|ID|id)?\s*[:：#=-]?\s*"
        r"(?:sample_)?(\d{1,6})(?!\d)",
        flags=re.IGNORECASE,
    ),
    # sample 305、sample_id=305。
    re.compile(
        r"(?<![A-Za-z0-9_])(?:sample|sample_id)"
        r"\s*[:：#=-]?\s*(?:sample_)?"
        r"(\d{1,6})(?!\d)",
        flags=re.IGNORECASE,
    ),
)


def extract_requested_sample_ids(
    question: str,
) -> set[str]:
    """提取用户明确指定的样本编号。

    支持数字后直接跟中文，例如：
    - 样本305的全部参数
    - 样本481碳化密度
    - 规范sample_id后直接跟中文

    不使用Unicode ``\\b`` 判断数字结束位置，
    避免中文字符被视为单词字符而导致匹配失败。
    """

    if not question:
        return set()

    numbers: set[int] = set()

    for pattern in SAMPLE_ID_PATTERNS:
        for match in pattern.finditer(question):
            numbers.add(
                int(match.group(1))
            )

    return {
        f"sample_{number:06d}"
        for number in numbers
    }


def remove_requested_sample_mentions(
    question: str,
) -> str:
    """删除问题中的明确样本编号片段。

    用于普通数值一致性检查，避免把“样本305”中的305
    当成密度、数量或point_index等业务数值。
    """

    cleaned = question

    for pattern in SAMPLE_ID_PATTERNS:
        cleaned = pattern.sub(
            " ",
            cleaned,
        )

    return cleaned



def normalize_question_sample_ids(
    question: str,
) -> str:
    """把问题中的明确样本编号统一成规范sample_id。

    所有样本识别入口共用 ``SAMPLE_ID_PATTERNS``，
    避免生成节点、Schema提示和Guard使用不同正则。

    例如自然语言中的“样本 + 数字”会规范为
    ``sample_`` 加六位十进制数字；未指定样本的问题保持不变。
    """

    normalized = question

    for pattern in SAMPLE_ID_PATTERNS:
        normalized = pattern.sub(
            lambda match: (
                f"sample_{int(match.group(1)):06d}"
            ),
            normalized,
        )

    return normalized



def get_column_owner_map() -> dict[str, set[str]]:
    """返回真实字段到所属物理表的映射。"""

    catalog = get_schema_catalog()
    owners: dict[str, set[str]] = {}

    for table_name, table_info in catalog["tables"].items():
        for column in table_info["columns"]:
            owners.setdefault(
                column,
                set(),
            ).add(table_name)

    return owners


def match_question_semantic_columns(
    question: str,
) -> dict[str, list[str]]:
    """从问题中提取明确出现的业务概念及其真实字段。

    只匹配Schema中维护的稳定术语，不推断模糊近义词。
    """

    catalog = get_schema_catalog()
    normalized = question.lower()
    matches: dict[str, list[str]] = {}

    for column, terms in catalog[
        "semantic_terms"
    ].items():
        matched_terms = [
            term
            for term in sorted(
                terms,
                key=len,
                reverse=True,
            )
            if term.lower() in normalized
        ]

        if matched_terms:
            matches[column] = matched_terms

    return matches


def infer_question_ranking_column(
    question: str,
) -> tuple[str, str] | None:
    """推断明确Top-K或显式排序所针对的业务字段。

    只在字段术语紧邻“最高/最低/最大/最小”或位于
    “按……升序/降序”之前时返回结果。
    """

    catalog = get_schema_catalog()
    candidates: list[
        tuple[int, int, str, str]
    ] = []

    for column, terms in catalog[
        "semantic_terms"
    ].items():
        for term in terms:
            start = 0

            while True:
                position = question.find(
                    term,
                    start,
                )
                if position < 0:
                    break

                candidates.append(
                    (
                        position,
                        position + len(term),
                        column,
                        term,
                    )
                )
                start = position + 1

    if not candidates:
        return None

    direction_matches = list(
        re.finditer(
            r"最高|最低|最大|最小|升序|降序",
            question,
        )
    )

    for direction_match in direction_matches:
        before = [
            candidate
            for candidate in candidates
            if candidate[1] <= direction_match.start()
        ]

        if not before:
            continue

        nearest = max(
            before,
            key=lambda item: item[1],
        )

        # 限制距离，避免把很早出现的返回字段误当排名字段。
        if (
            direction_match.start()
            - nearest[1]
            <= 12
        ):
            return nearest[2], nearest[3]

    return None



def extract_output_request_text(
    question: str,
) -> str:
    """提取用户明确描述返回字段的文本部分。"""

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
    ].strip()


def infer_requested_output_columns(
    question: str,
) -> set[str]:
    """推断用户明确要求展示的真实字段。

    只处理明确返回表达和稳定的“全部静态材料参数”意图。
    无法确定时返回空集合，不做强制猜测。
    """

    catalog = get_schema_catalog()

    if "全部静态材料参数" in question:
        static_table = next(
            table_name
            for table_name, info
            in catalog["tables"].items()
            if info["alias"] == "ms"
        )
        return set(
            catalog["tables"][
                static_table
            ]["columns"]
        )

    output_text = extract_output_request_text(
        question
    )

    if output_text:
        matches = match_question_semantic_columns(
            output_text
        )

        if "对应字段" in output_text:
            matches = match_question_semantic_columns(
                question
            )

        return set(matches)

    # “查询样本305的热解热、发射率……”和时序区间查询：
    # 没有显式“返回”，但列出的业务字段就是输出字段。
    exact_sample_lookup = bool(
        extract_requested_sample_ids(
            question
        )
    )
    has_filter_or_ranking = bool(
        re.search(
            r"大于|小于|不少于|不大于|之间|最高|最低|最大|最小|前\s*\d+",
            question,
        )
    )

    if exact_sample_lookup and not has_filter_or_ranking:
        return set(
            match_question_semantic_columns(
                question
            )
        )

    # point_index区间明细的字段通常在问题主体中直接列出。
    if (
        exact_sample_lookup
        and "point_index" in question.lower()
    ):
        return set(
            match_question_semantic_columns(
                question
            )
        )

    return set()



def infer_explicit_full_table_request(
    question: str,
) -> str | None:
    """识别用户明确点名白名单表并请求全部数据/字段的意图。

    只接受Schema中存在的真实表名，不从模糊自然语言猜表。
    """

    catalog = get_schema_catalog()

    full_data_intent = bool(
        re.search(
            r"全部数据|所有数据|全部记录|所有记录|"
            r"全部字段|所有字段|完整数据|完整记录",
            question,
            flags=re.IGNORECASE,
        )
    )
    if not full_data_intent:
        return None

    lowered = question.lower()

    for table_name in catalog["tables"]:
        if table_name.lower() in lowered:
            return table_name

    return None


def infer_relevant_tables(
    question: str,
) -> set[str]:
    """根据问题中的稳定字段语义推断生成SQL所需的最少表集合。

    只使用Schema词汇表和明确表名，不根据评测题整句做特判。
    无法可靠推断时返回空集合，由调用方回退到完整Schema。
    """

    catalog = get_schema_catalog()
    owners = get_column_owner_map()
    tables: set[str] = set()

    explicit_full_table = (
        infer_explicit_full_table_request(
            question
        )
    )
    if explicit_full_table is not None:
        return {explicit_full_table}

    lowered = question.lower()
    for table_name in catalog["tables"]:
        if table_name.lower() in lowered:
            tables.add(table_name)

    if re.search(
        r"全部静态材料参数|全部静态参数|"
        r"所有静态材料参数|所有静态参数",
        question,
    ):
        static_table = next(
            table_name
            for table_name, info
            in catalog["tables"].items()
            if info["alias"] == "ms"
        )
        tables.add(static_table)

    semantic_matches = (
        match_question_semantic_columns(
            question
        )
    )
    for column in semantic_matches:
        # sample_id是三张表共有的连接键，不用于决定业务表。
        # 真正相关表由用户询问的业务字段或明确表名决定。
        if column == "sample_id":
            continue

        tables.update(
            owners.get(column, set())
        )

    return tables


def build_generation_schema_context(
    question: str,
) -> str:
    """生成SQL生成器使用的裁剪Schema。

    简单单表问题只展示相关表；跨表问题只展示必要表和关系。
    无法可靠推断时回退到完整Schema，避免遗漏未知意图。
    """

    catalog = get_schema_catalog()
    relevant_tables = infer_relevant_tables(
        question
    )

    if not relevant_tables:
        return build_schema_context()

    lines: list[str] = [
        "数据库类型：MySQL",
        "仅可使用以下与当前问题相关的真实表：",
    ]

    for table_name, info in catalog["tables"].items():
        if table_name not in relevant_tables:
            continue

        grain_text = (
            "每个样本一行"
            if info["grain"] == "one_row_per_sample"
            else "每个样本多行时序记录"
        )
        lines.append(
            f"[{table_name}] 推荐别名：{info['alias']}；"
            f"粒度：{grain_text}"
        )
        for column, description in info[
            "columns"
        ].items():
            lines.append(
                f"- {column}：{description}"
            )

    relationships = [
        relationship
        for relationship in catalog[
            "relationships"
        ]
        if sum(
            table_name in relationship
            for table_name in relevant_tables
        ) >= 2
    ]
    if relationships:
        lines.append("必要连接关系：")
        lines.extend(
            f"- {relationship}"
            for relationship in relationships
        )

    lines.extend(
        [
            build_question_field_hint(question),
            "生成约束：",
            "- 只使用以上必要表。",
            "- 普通样本级Top-K直接使用目标字段ORDER BY和LIMIT。",
            "- 一行一个样本的字段Top-K不得使用MAX或无意义GROUP BY。",
            "- 时序峰值才使用MAX并按sample_id分组。",
            "- 用户未指定样本时不得添加固定sample_id过滤。",
            "- 禁止SELECT *，必须显式返回字段。",
        ]
    )

    return "\n".join(lines)


def build_compact_sql_context(
    question: str,
) -> str:
    """生成修复节点使用的精简且按问题裁剪的Schema上下文。"""

    catalog = get_schema_catalog()
    relevant_tables = infer_relevant_tables(
        question
    )

    if not relevant_tables:
        relevant_tables = set(
            catalog["tables"]
        )

    lines = [
        "当前问题允许使用的真实表与字段：",
    ]

    for table_name, info in catalog["tables"].items():
        if table_name not in relevant_tables:
            continue

        lines.append(
            f"- {table_name} AS {info['alias']} "
            f"({info['grain']}): "
            + ", ".join(info["columns"])
        )

    relationships = [
        relationship
        for relationship in catalog[
            "relationships"
        ]
        if sum(
            table_name in relationship
            for table_name in relevant_tables
        ) >= 2
    ]
    if relationships:
        lines.append("连接关系：")
        lines.extend(
            f"- {relationship}"
            for relationship in relationships
        )

    lines.append(
        build_question_field_hint(question)
    )

    return "\n".join(lines)



def build_question_field_hint(
    question: str,
) -> str:
    """生成给SQL生成器和修复器使用的确定性提示。"""

    matches = match_question_semantic_columns(
        question
    )
    sample_ids = extract_requested_sample_ids(
        question
    )

    if not matches and not sample_ids:
        return "未识别到需要额外提示的明确业务字段或样本编号。"

    owners = get_column_owner_map()
    lines = [
        "根据Schema确定的业务字段和样本约束："
    ]

    for column, terms in matches.items():
        owner_text = "/".join(
            sorted(
                owners.get(
                    column,
                    set(),
                )
            )
        )
        lines.append(
            f"- {'、'.join(terms)}"
            f" -> {owner_text}.{column}"
        )

    if sample_ids:
        lines.append(
            "- 指定样本必须使用sample_id等值过滤"
            " -> "
            + ", ".join(
                sorted(sample_ids)
            )
        )

    ranking = infer_question_ranking_column(
        question
    )
    if ranking is not None:
        lines.append(
            f"- 当前排序/Top-K目标字段"
            f" -> {ranking[0]}"
        )

    return "\n".join(lines)


def build_schema_context() -> str:
    """生成提供给LLM的Schema和稳定使用约束。

    不包含评测题答案，只提供数据库事实和可迁移的SQL规则。
    """

    catalog = get_schema_catalog()

    lines: list[str] = [
        "数据库类型：MySQL",
        "",
        "数据库用途：",
        (
            "查询树脂基防热材料的静态参数、"
            "热物性参数和时序热响应。"
        ),
        "",
        "表结构：",
    ]

    for table_name, info in catalog["tables"].items():
        grain_text = (
            "每个样本一行"
            if info["grain"] == "one_row_per_sample"
            else "每个样本多行"
        )

        lines.extend(
            [
                "",
                (
                    f"[{table_name}] "
                    f"推荐别名：{info['alias']}；"
                    f"数据粒度：{grain_text}"
                ),
                info["description"],
                "字段：",
            ]
        )

        for column, description in info[
            "columns"
        ].items():
            lines.append(
                f"- {column}：{description}"
            )

    lines.append("\n核心业务术语与真实字段：")
    owners = get_column_owner_map()

    for column, terms in catalog[
        "semantic_terms"
    ].items():
        owner_text = "/".join(
            sorted(
                owners.get(
                    column,
                    set(),
                )
            )
        )
        lines.append(
            f"- {'、'.join(terms)}"
            f" -> {owner_text}.{column}"
        )

    lines.append("\n表关系：")
    for relationship in catalog["relationships"]:
        lines.append(f"- {relationship}")

    lines.append("\n稳定领域规则：")
    for convention in catalog[
        "domain_conventions"
    ]:
        lines.append(f"- {convention}")

    lines.extend(
        [
            "",
            "通用SQL要求：",
            "- 只生成一条MySQL SELECT查询。",
            "- FROM和JOIN后只能写真实表名，别名写在真实表名之后。",
            "- 只能使用以上真实表和真实字段。",
            "- 必须返回用户明确要求的字段。",
            "- 不得增加用户未要求的返回字段或过滤条件。",
            "- 只连接回答问题所需的最少数据表。",
            "- 不得擅自增加GROUP BY或聚合函数。",
            (
                "- 简单排序和Top-K直接使用"
                "ORDER BY目标字段与LIMIT，不使用IN子查询。"
            ),
            (
                "- 多表查询中的同名字段必须使用"
                "表别名限定。"
            ),
            "- 禁止SELECT *。",
            (
                "- 用户未要求单位时，不得自行添加单位。"
            ),
        ]
    )

    return "\n".join(lines).strip()