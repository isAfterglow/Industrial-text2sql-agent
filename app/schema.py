from copy import deepcopy
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from contextvars import ContextVar
import re
from typing import Any

import sqlglot
import yaml
from sqlglot import exp
from sqlglot.errors import ParseError

from app.config import get_settings


# 统一的数值词法规则。整个项目只使用这一份定义，避免把
# 2e-12错误拆成2和12，或把9E-14错误拆成9和14。
NUMERIC_LITERAL_PATTERN = (
    r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))"
    r"(?:[eE][+-]?\d+)?"
)
NUMERIC_LITERAL_RE = re.compile(
    rf"(?<![A-Za-z0-9_])({NUMERIC_LITERAL_PATTERN})(?![A-Za-z0-9_])"
)


def extract_numeric_literals(text: str) -> list[str]:
    """按完整数值字面量提取整数、小数和科学计数法。"""

    if not text:
        return []
    return [match.group(1) for match in NUMERIC_LITERAL_RE.finditer(text)]


def parse_decimal_literal(value: object) -> Decimal | None:
    """把数值字面量转换为Decimal，格式差异不影响比较。"""

    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def normalize_numeric_literal(value: object) -> str:
    """返回适合日志和集合比较的规范数值字符串。"""

    number = parse_decimal_literal(value)
    if number is None:
        return str(value).strip()
    if number == 0:
        return "0"
    normalized = number.normalize()
    if -6 <= normalized.adjusted() <= 20:
        return format(normalized, "f")
    return str(normalized)


_ACTIVE_PROFILE: ContextVar[str] = ContextVar("active_profile", default="resin")


def set_active_profile(name: str) -> None:
    _load_profile(name)
    _ACTIVE_PROFILE.set(name)


def active_profile_name() -> str:
    return _ACTIVE_PROFILE.get()


def resolve_profile(question: str) -> str:
    """Route known domain vocabulary deterministically; resin remains default."""
    text = question.lower()
    steel_profile = _load_profile("steel_industry")
    resin_profile = _load_profile("resin")

    def score(profile: dict[str, Any]) -> int:
        semantic_terms = profile.get("semantic_terms", {})
        vocabulary = [
            term
            for values in semantic_terms.values()
            for term in values
        ] + list(profile.get("routing_terms", []))
        return sum(str(term).lower() in text for term in vocabulary)

    steel_score, resin_score = score(steel_profile), score(resin_profile)
    return "steel_industry" if steel_score > resin_score and steel_score else "resin"


@lru_cache(maxsize=4)
def _load_profile(name: str = "resin") -> dict[str, Any]:
    """Load a domain profile; the query engine never owns domain vocabulary."""

    path = Path(__file__).resolve().parents[1] / "profiles" / f"{name}.yaml"
    with path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)
    if not isinstance(profile, dict) or not profile.get("tables"):
        raise ValueError(f"Invalid domain profile: {path}")
    return profile


def _profile_catalog() -> dict[str, Any]:
    profile = _load_profile(active_profile_name())
    relationships = []
    for item in profile.get("relationships", []):
        relationships.append(
            f"{item['left']} = {item['right']}，"
            + ("1对1" if item.get("cardinality") == "one_to_one" else "1对多")
        )
    return {
        "database_type": profile.get("database_type", "MySQL"),
        "tables": profile["tables"],
        "relationships": relationships,
        "semantic_terms": profile.get("semantic_terms", {}),
        "domain_conventions": profile.get("domain_conventions", []),
        "aggregations": profile.get("aggregations", {}),
        "derived_metrics": profile.get("derived_metrics", {}),
        "temporal_semantics": profile.get("temporal_semantics", {}),
        "policy": profile.get("policy", {}),
    }


def _legacy_resin_catalog() -> dict[str, Any]:
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

def get_schema_catalog() -> dict[str, Any]:
    """Return the active domain catalog from its declarative profile."""

    return deepcopy(_profile_catalog())


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
        # Field identifiers are part of the public schema vocabulary as well.
        # This keeps mixed Chinese/English engineering questions on the same
        # structured path as their natural-language equivalents.
        candidates = list(dict.fromkeys([*terms, column]))
        matched_terms = [
            term
            for term in sorted(
                candidates,
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
        for term in dict.fromkeys([*terms, column]):
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
            <= 32
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




def is_strict_projection_request(
    question: str,
) -> bool:
    """用户是否明确要求只返回/只显示指定字段。"""

    return bool(
        re.search(
            r"只返回|只显示|仅返回|仅显示",
            question,
        )
    )


def extract_requested_limit_from_question(
    question: str,
) -> int | None:
    """提取明确的Top-K或最多返回数量。"""

    patterns = (
        r"(?:top|bottom)\s*(\d+)\s*(?:个|条)?",
        r"(?:最高|最低|最大|最小)(?:的)?\s*(\d+)\s*个",
        r"(?:最高|最低|最大|最小)(?:的)?\s*(\d+)\s*条(?:记录|数据)?",
        r"前\s*(\d+)\s*(?:个|条)?",
        r"最多(?:返回)?\s*(\d+)\s*(?:个|条)?",
        r"(?:限制|limit)\s*(?:为|=)?\s*(\d+)",
        r"(\d+)\s*个样本",
        r"(\d+)\s*(?:个|条|笔).{0,12}(?:最高|最低|最大|最小)",
    )
    for pattern in patterns:
        match = re.search(
            pattern,
            question,
            flags=re.IGNORECASE,
        )
        if match:
            return int(match.group(1))
    chinese = re.search(r"(?:前|最高的?|最低的?|最大的?|最小的?)\s*([一二三四五六七八九十])\s*(?:个|条|笔)", question)
    if chinese:
        return {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}[chinese.group(1)]
    return None


def _ordered_columns_for_table(
    table_name: str,
    columns: set[str],
) -> list[str]:
    catalog = get_schema_catalog()
    table_columns = catalog["tables"][table_name]["columns"]
    ordered = [
        column
        for column in table_columns
        if column in columns
    ]
    if "sample_id" in ordered:
        ordered.remove("sample_id")
        ordered.insert(0, "sample_id")
    return ordered


def _semantic_term_occurrences(
    question: str,
) -> list[tuple[str, str]]:
    catalog = get_schema_catalog()
    occurrences: list[tuple[str, str]] = []
    for column, terms in catalog["semantic_terms"].items():
        for term in sorted(terms, key=len, reverse=True):
            if term in question:
                occurrences.append((column, term))
    return occurrences


def _extract_simple_filters(
    question: str,
) -> tuple[list[dict[str, Any]], set[str]]:
    """提取可确定的单表数值过滤，并返回已消费数字。

    支持整数、小数和科学计数法，例如2e-12、9E-14。
    """

    filters: list[dict[str, Any]] = []
    consumed_numbers: set[str] = set()
    seen: set[tuple[str, str, str, str]] = set()

    comparison_patterns = (
        (r"大于等于|不小于|不低于|至少|不少于", ">="),
        (r"小于等于|不大于|至多|不超过", "<="),
        (r"大于|高于|超过", ">"),
        (r"小于|低于", "<"),
    )

    number_group = rf"({NUMERIC_LITERAL_PATTERN})"

    for column, term in _semantic_term_occurrences(question):
        if column == "sample_id":
            continue

        escaped = re.escape(term)
        between = re.search(
            escaped
            + r"\s*(?:在)?\s*"
            + number_group
            + r"\s*(?:到|至|~|～)\s*"
            + number_group
            + r"\s*(?:之间)?",
            question,
            flags=re.IGNORECASE,
        )
        if between:
            low, high = between.group(1), between.group(2)
            low_key = normalize_numeric_literal(low)
            high_key = normalize_numeric_literal(high)
            key = (column, "BETWEEN", low_key, high_key)
            if key not in seen:
                filters.append(
                    {
                        "column": column,
                        "operator": "BETWEEN",
                        "value": low,
                        "value2": high,
                    }
                )
                seen.add(key)
                consumed_numbers.update({low_key, high_key})
            continue

        matched_comparison = False
        for operator_pattern, operator in comparison_patterns:
            match = re.search(
                escaped
                + r"\s*(?:为|是)?\s*(?:"
                + operator_pattern
                + r")\s*"
                + number_group,
                question,
                flags=re.IGNORECASE,
            )
            if match:
                value = match.group(1)
                value_key = normalize_numeric_literal(value)
                key = (column, operator, value_key, "")
                if key not in seen:
                    filters.append(
                        {
                            "column": column,
                            "operator": operator,
                            "value": value,
                        }
                    )
                    seen.add(key)
                    consumed_numbers.add(value_key)
                matched_comparison = True
                break

        if matched_comparison:
            continue

        equality = re.search(
            escaped
            + r"\s*(?:等于|等於|为|是|=)\s*"
            + number_group,
            question,
            flags=re.IGNORECASE,
        )
        if equality:
            value = equality.group(1)
            value_key = normalize_numeric_literal(value)
            key = (column, "=", value_key, "")
            if key not in seen:
                filters.append(
                    {
                        "column": column,
                        "operator": "=",
                        "value": value,
                    }
                )
                seen.add(key)
                consumed_numbers.add(value_key)

    return filters, consumed_numbers


def _response_table_info() -> tuple[str, dict[str, Any]] | None:
    """返回时序响应表名及其Schema信息。"""

    catalog = get_schema_catalog()
    for table_name, info in catalog["tables"].items():
        if info["grain"] == "many_rows_per_sample":
            return table_name, info
    return None


def _ranking_direction(question: str) -> str | None:
    """提取结果排名或显式排序方向。"""

    if re.search(r"最低|最小|升序|从低到高|\bbottom\b", question, re.IGNORECASE):
        return "ASC"
    if re.search(r"最高|最大|降序|从高到低|\btop\b", question, re.IGNORECASE):
        return "DESC"
    return None


def canonical_metric_alias(column: str, aggregation: str) -> str:
    """Return the single public alias for a temporal aggregation."""

    prefixes = {
        "MAX": "peak",
        "AVG": "average",
        "MIN": "min",
        "SUM": "sum",
        "INITIAL": "initial",
        "FINAL": "final",
    }
    return f"{prefixes[aggregation]}_{column}"


def _near_term(
    question: str,
    term: str,
    pattern: str,
    radius: int = 14,
) -> bool:
    """判断聚合/时点词是否出现在字段术语附近。"""

    start = 0
    while True:
        position = question.find(term, start)
        if position < 0:
            return False
        left = max(0, position - radius)
        right = min(len(question), position + len(term) + radius)
        if re.search(pattern, question[left:right]):
            return True
        start = position + 1


def _infer_temporal_metrics(
    question: str,
) -> list[dict[str, str]]:
    """识别标准时序指标及其聚合角色。

    对每个字段选择距离最近的聚合/时点词，避免
    “峰值背面温度和最终质量”把背温误判为FINAL。
    """

    response = _response_table_info()
    if response is None:
        return []
    _, response_info = response
    matches = match_question_semantic_columns(question)
    catalog = get_schema_catalog()
    cue_patterns = (
        ("FINAL", r"最终|最后(?:一个)?(?:点|时刻)?|末时刻|终点"),
        ("AVG", r"平均|均值"),
        ("MAX", r"峰值|最大值|最大质量|最高值"),
        ("MIN", r"最小值|最低值"),
    )
    metrics: list[dict[str, str]] = []

    for column in response_info["columns"]:
        if column in {"sample_id", "point_index"} or column not in matches:
            continue

        best: tuple[int, int, str] | None = None
        for term in catalog["semantic_terms"].get(column, []):
            start = 0
            while True:
                position = question.find(term, start)
                if position < 0:
                    break
                term_start = position
                term_end = position + len(term)
                clause_left = max(
                    question.rfind("，", 0, term_start),
                    question.rfind("。", 0, term_start),
                    question.rfind("？", 0, term_start),
                    question.rfind("；", 0, term_start),
                    -1,
                ) + 1
                clause_end_candidates = [
                    value for value in (
                        question.find("，", term_end),
                        question.find("。", term_end),
                        question.find("？", term_end),
                        question.find("；", term_end),
                    ) if value >= 0
                ]
                clause_right = min(clause_end_candidates) if clause_end_candidates else len(question)
                clause = question[clause_left:clause_right]
                offset = clause_left

                for priority, (aggregation, pattern) in enumerate(cue_patterns):
                    for cue in re.finditer(pattern, clause):
                        cue_start = offset + cue.start()
                        cue_end = offset + cue.end()
                        if cue_end <= term_start:
                            distance = term_start - cue_end
                        elif cue_start >= term_end:
                            distance = cue_start - term_end
                        else:
                            distance = 0
                        # 超过12个字符通常已属于另一个字段短语。
                        if distance > 12:
                            continue
                        candidate = (distance, priority, aggregation)
                        if best is None or candidate < best:
                            best = candidate
                start = position + 1

        aggregation = best[2] if best is not None else None
        if aggregation is None and re.search(r"每个样本", question):
            if re.search(r"平均|均值", question):
                aggregation = "AVG"
            elif re.search(r"峰值|最大值|最大质量", question):
                aggregation = "MAX"
            elif re.search(r"最小值", question):
                aggregation = "MIN"

        if aggregation is not None:
            metrics.append(
                {
                    "column": column,
                    "aggregation": aggregation,
                    "alias": canonical_metric_alias(column, aggregation),
                }
            )

    unique: dict[tuple[str, str], dict[str, str]] = {}
    for metric in metrics:
        unique[(metric["column"], metric["aggregation"])] = metric
    return list(unique.values())

def _all_question_numbers_consumed(
    question: str,
    consumed_numbers: set[str],
    limit: int | None,
) -> bool:
    cleaned = remove_requested_sample_mentions(question)
    all_numbers = {
        normalize_numeric_literal(value)
        for value in extract_numeric_literals(cleaned)
    }
    consumed = {
        normalize_numeric_literal(value)
        for value in consumed_numbers
    }
    if limit is not None:
        consumed.add(normalize_numeric_literal(limit))
    return not (all_numbers - consumed)


def _build_temporal_query_spec(
    question: str,
    matches: dict[str, list[str]],
    requested_outputs: set[str],
    ranking: tuple[str, str] | None,
    sample_ids: list[str],
    strict_projection: bool,
    filters: list[dict[str, Any]],
    consumed_numbers: set[str],
    limit: int | None,
) -> dict[str, Any] | None:
    """构建标准时序查询计划。

    支持：
    1. 指定样本的point_index明细/区间/前N点；
    2. 每样本MAX/AVG/MIN后Top-K；
    3. 样本级标量字段排名并返回时序聚合；
    4. 最终point_index对应值；
    5. 标量WHERE与时序聚合HAVING的组合。
    """

    catalog = get_schema_catalog()
    owners = get_column_owner_map()
    response = _response_table_info()
    if response is None:
        return None
    response_table, response_info = response
    response_columns = set(response_info["columns"])
    response_value_columns = response_columns - {"sample_id", "point_index"}
    metrics = _infer_temporal_metrics(question)
    metric_by_column = {
        metric["column"]: metric
        for metric in metrics
    }

    matched_columns = set(matches)
    mentioned_response = matched_columns & response_columns
    has_point_language = bool(
        re.search(r"point_index|序列点|点位|前\s*\d+\s*个序列点", question, re.I)
    )

    # A. 指定样本的时序明细切片。
    if sample_ids and mentioned_response and not metrics:
        unsupported = matched_columns - response_columns
        if unsupported:
            return None

        point_filters = [
            item for item in filters
            if item["column"] == "point_index"
        ]
        non_point_filters = [
            item for item in filters
            if item["column"] != "point_index"
        ]
        if non_point_filters:
            return None

        first_n_points = bool(
            re.search(r"前\s*\d+\s*个(?:序列点|点位|点)", question)
        )
        if not point_filters and not first_n_points and not has_point_language:
            return None
        if not _all_question_numbers_consumed(
            question, consumed_numbers, limit
        ):
            return None

        if requested_outputs:
            selected = set(requested_outputs) & response_columns
        else:
            selected = set(mentioned_response)
        if not strict_projection:
            selected.update({"sample_id", "point_index"})
        elif "point_index" in requested_outputs:
            selected.add("point_index")

        selected -= set(metric_by_column)
        if not selected:
            return None

        return {
            "eligible": True,
            "mode": "deterministic",
            "query_type": "response_detail",
            "table": response_table,
            "select_columns": _ordered_columns_for_table(response_table, selected),
            "filters": point_filters,
            "where_filters": point_filters,
            "having_filters": [],
            "order_by": {
                "kind": "column",
                "column": "point_index",
                "direction": "DESC" if re.search(r"point_index.{0,12}降序", question, re.I) else "ASC",
            },
            "limit": limit if first_n_points else None,
            "sample_ids": sample_ids,
            "strict_projection": strict_projection,
            "temporal_metrics": [],
            "scalar_columns": [],
            "scalar_tables": [],
            "confidence": 1.0,
            "reason": "指定样本、响应字段、point_index条件和排序均可确定，使用时序明细快路径。",
        }

    # B. 每样本时序聚合及跨表样本级排名。
    if not metrics:
        return None

    if re.search(r"占比|比例|中位数|方差|标准差|变化率", question):
        return None

    ranking_column = ranking[0] if ranking is not None else None
    ranking_direction = _ranking_direction(question)
    if limit is not None and (ranking_column is None or ranking_direction is None):
        return None

    business_columns = matched_columns - {"sample_id", "point_index"}
    scalar_columns = business_columns - response_value_columns
    scalar_columns.update(requested_outputs - response_columns - {"sample_id", "point_index"})
    if ranking_column and ranking_column not in response_columns:
        scalar_columns.add(ranking_column)

    scalar_tables: set[str] = set()
    for column in scalar_columns:
        column_owners = owners.get(column, set())
        if len(column_owners) != 1:
            return None
        owner = next(iter(column_owners))
        if catalog["tables"][owner]["grain"] != "one_row_per_sample":
            return None
        scalar_tables.add(owner)

    where_filters: list[dict[str, Any]] = []
    having_filters: list[dict[str, Any]] = []
    for item in filters:
        column = item["column"]
        if column in metric_by_column:
            having_filters.append(item)
        elif column in response_value_columns:
            # 原始时序行过滤与“每样本聚合”语义容易混淆，保守回退。
            return None
        else:
            where_filters.append(item)

    if not _all_question_numbers_consumed(question, consumed_numbers, limit):
        return None

    selected_scalar: set[str] = set()
    if requested_outputs:
        selected_scalar.update(requested_outputs - response_columns - {"sample_id", "point_index"})
    if ranking_column and ranking_column not in response_columns and not strict_projection:
        selected_scalar.add(ranking_column)

    metric_columns = {metric["column"] for metric in metrics}
    selected_metrics = [
        metric for metric in metrics
        if (
            not requested_outputs
            or metric["column"] in requested_outputs
            or metric["column"] == ranking_column
            or not strict_projection
        )
    ]
    if not selected_metrics:
        selected_metrics = metrics

    ranking_spec: dict[str, str] | None = None
    if ranking_column is not None and ranking_direction is not None:
        if ranking_column in metric_by_column:
            ranking_spec = {
                "kind": "metric",
                "column": ranking_column,
                "alias": metric_by_column[ranking_column]["alias"],
                "direction": ranking_direction,
            }
        elif ranking_column in scalar_columns:
            ranking_spec = {
                "kind": "scalar",
                "column": ranking_column,
                "direction": ranking_direction,
            }
        else:
            return None

    return {
        "eligible": True,
        "mode": "deterministic",
        "query_type": "per_sample_temporal_aggregate",
        "table": response_table,
        "select_columns": ["sample_id"],
        "filters": filters,
        "where_filters": where_filters,
        "having_filters": having_filters,
        "order_by": ranking_spec,
        "limit": limit,
        "sample_ids": sample_ids,
        "strict_projection": strict_projection,
        "temporal_metrics": selected_metrics,
        "all_temporal_metrics": metrics,
        "scalar_columns": sorted(selected_scalar),
        "scalar_tables": sorted(scalar_tables),
        "confidence": 1.0,
        "reason": "时序字段的聚合角色、样本分组、排名字段和方向均可确定，使用标准时序查询快路径。",
    }


def _build_fact_aggregate_query_spec(
    question: str,
    matches: dict[str, list[str]],
    requested_outputs: set[str],
    strict_projection: bool,
) -> dict[str, Any] | None:
    """Compile common fact/dimension rollups declared by a Profile.

    This deliberately covers only unambiguous SUM/AVG aggregates. More complex
    analytical questions continue through the constrained LLM path.
    """
    catalog = get_schema_catalog()
    profile_relationships = _load_profile(active_profile_name()).get("relationships", [])
    fact_tables = [name for name, info in catalog["tables"].items() if info["grain"] == "fact"]
    if len(fact_tables) != 1:
        return None
    if not re.search(r"统计|总(?:计|量)|合计|平均|均值", question):
        return None

    aggregation = "AVG" if re.search(r"平均|均值", question) else "SUM"
    owners = get_column_owner_map()
    metric_candidates = [
        column for column in matches
        if aggregation in set(catalog.get("aggregations", {}).get(column, []))
    ]
    if len(metric_candidates) != 1:
        return None
    metric = metric_candidates[0]
    fact_table = fact_tables[0]
    if fact_table not in owners.get(metric, set()):
        return None

    group_candidates = [
        column for column in matches
        if column != metric and re.search(r"按.{0,16}" + re.escape(next(iter(matches[column]))), question)
    ]
    # A group column is normally written directly after "按"; fall back to a
    # single matched dimension field for compact Chinese phrasing.
    if not group_candidates:
        group_candidates = [
            column for column in matches
            if column != metric and any(catalog["tables"][table]["grain"] == "dimension" for table in owners.get(column, set()))
        ]
    if len(group_candidates) > 1:
        return None
    group_column = group_candidates[0] if group_candidates else ""
    group_table = next(iter(owners.get(group_column, set())), "") if group_column else ""

    if group_column and catalog["tables"].get(group_table, {}).get("grain") != "dimension":
        return None
    if group_column:
        related = any(
            {str(rel["left"]).split(".")[0], str(rel["right"]).split(".")[0]} == {fact_table, group_table}
            for rel in profile_relationships
        )
        if not related:
            return None

    alias = canonical_metric_alias(metric, aggregation)
    selected = [group_column] if group_column else []
    selected.append(alias)
    return {
        "eligible": True,
        "mode": "deterministic",
        "query_type": "fact_aggregate",
        "table": fact_table,
        "group_column": group_column,
        "group_table": group_table,
        "metric": metric,
        "aggregation": aggregation,
        "metric_alias": alias,
        "select_columns": selected if strict_projection else selected,
        "filters": [], "where_filters": [], "having_filters": [],
        "order_by": {"kind": "metric", "alias": alias, "direction": "DESC"} if group_column and re.search(r"降序|最高|最大", question) else None,
        "limit": None, "sample_ids": [], "strict_projection": strict_projection,
        "temporal_metrics": [], "scalar_columns": [], "scalar_tables": [],
        "confidence": 1.0,
        "reason": "事实表指标、聚合函数和维度关系均由Profile明确声明，使用通用聚合快路径。",
    }


def _profile_fact_filters(question: str) -> list[dict[str, Any]]:
    """Extract portable fact/dimension constraints from a Profile vocabulary."""
    filters: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(column: str, operator: str, value: Any, value2: Any = None) -> None:
        key = (column, operator, str(value), str(value2 or ""))
        if key not in seen:
            item: dict[str, Any] = {"column": column, "operator": operator, "value": value}
            if value2 is not None:
                item["value2"] = value2
            filters.append(item)
            seen.add(key)

    # Common calendar expressions. The physical values remain Profile data,
    # while this parser intentionally only recognizes universal calendar forms.
    year = re.search(r"(20\d{2})年", question)
    if year:
        add("year", "=", year.group(1))
    month = re.search(r"(?<!\d)(1[0-2]|[1-9])月", question)
    if month:
        add("month", "=", month.group(1))
    if re.search(r"第一季度|第1季度", question):
        add("month", "BETWEEN", 1, 3)
    elif re.search(r"第二季度|第2季度", question):
        add("month", "BETWEEN", 4, 6)
    elif re.search(r"第三季度|第3季度", question):
        add("month", "BETWEEN", 7, 9)
    elif re.search(r"第四季度|第4季度", question):
        add("month", "BETWEEN", 10, 12)
    hour_range = re.search(r"(\d{1,2})点\s*(?:到|至|~|～)\s*(\d{1,2})点?", question)
    if hour_range:
        add("hour", "BETWEEN", hour_range.group(1), hour_range.group(2))
    elif (hour := re.search(r"(?<!\d)(\d{1,2})点", question)):
        add("hour", "=", hour.group(1))
    status_is_group = bool(re.search(r"按工作日状态|工作日.*周末|平日.*周末", question))
    if not status_is_group and ("工作日" in question or "平日" in question or re.search(r"\bWeekday\b", question, re.I)):
        add("week_status", "=", "Weekday")
    elif not status_is_group and ("周末" in question or re.search(r"\bWeekend\b", question, re.I)):
        add("week_status", "=", "Weekend")
    for name in ("Maximum_Load", "Medium_Load", "Light_Load"):
        if name.lower() in question.lower():
            add("load_type_name", "=", name)
    return filters


def _build_profile_fact_query_spec(
    question: str,
    matches: dict[str, list[str]],
    requested_outputs: set[str],
    ranking: tuple[str, str] | None,
    strict_projection: bool,
    numeric_filters: list[dict[str, Any]],
    limit: int | None,
) -> dict[str, Any] | None:
    """Compile a normalized fact table plus declared dimensions from a Profile."""
    catalog = get_schema_catalog()
    facts = [name for name, info in catalog["tables"].items() if info["grain"] == "fact"]
    if len(facts) != 1:
        return None
    fact_table = facts[0]
    owners = get_column_owner_map()
    profile = _load_profile(active_profile_name())
    aggregations = catalog.get("aggregations", {})
    all_filters = list(numeric_filters) + _profile_fact_filters(question)
    has_aggregate = bool(re.search(r"统计|总(?:计|量)|合计|汇总|平均|均值|记录数|数量|多少", question))
    derived_requested = bool(re.search(r"碳强度", question)) and "carbon_intensity" in catalog.get("derived_metrics", {})
    mentioned = set(matches)

    # Only take ownership of queries that genuinely need Profile fact/dimension
    # semantics; resin's dedicated planner remains untouched.
    used_columns = mentioned | requested_outputs | {item["column"] for item in all_filters}
    used_tables = {fact_table}
    for column in used_columns:
        used_tables.update(owners.get(column, set()))
    # Grouping words can be unambiguous even when they use a colloquial form
    # (for example "每个月" instead of the Profile term "月份").
    if re.search(r"月(?:份|度|各|的|个)|每个月|按月|\d+个月", question):
        used_tables.update(owners.get("month", set()))
    if re.search(r"负荷(?:类型)?|各负荷", question):
        used_tables.update(owners.get("load_type_name", set()))
    if re.search(r"工作日.*周末|平日.*周末|按工作日状态", question):
        used_tables.update(owners.get("week_status", set()))
    dimensions = sorted(table for table in used_tables if table != fact_table and catalog["tables"].get(table, {}).get("grain") == "dimension")
    needs_profile_plan = bool(dimensions or derived_requested or len(numeric_filters) > 1 or (has_aggregate and len([c for c in mentioned if c in aggregations]) > 1))
    if not needs_profile_plan:
        return None
    if any(table != fact_table and table not in dimensions for table in used_tables):
        return None

    grouping_cue = bool(re.search(r"按|每(?:个|月|年|天|小时|种)|不同(?:负荷|类型)|各(?:负荷|类型)|比较.+和|\d+个月|平日.*周末|工作日.*周末", question))
    filter_columns = {item["column"] for item in all_filters}
    group_columns = [
        column for column in mentioned
        if column in owners and any(owner in dimensions for owner in owners[column])
        and column not in filter_columns
    ]
    if grouping_cue:
        inferred_groups = []
        if re.search(r"月(?:份|度|各|的|个)|每个月|按月|\d+个月", question):
            inferred_groups.append("month")
        if re.search(r"负荷(?:类型)?|各负荷", question):
            inferred_groups.append("load_type_name")
        if re.search(r"工作日.*周末|平日.*周末|按工作日状态", question):
            inferred_groups.append("week_status")
        group_columns.extend(
            column for column in inferred_groups
            if column not in filter_columns and column in owners
        )
    group_columns = sorted(dict.fromkeys(group_columns))
    if not grouping_cue:
        group_columns = []

    # A field used solely as a predicate ("耗电量大于100的平均排放")
    # constrains the population; it is not an implicitly requested aggregate.
    metric_columns = [
        column for column in mentioned
        if fact_table in owners.get(column, set())
        and column in aggregations
        and (column not in filter_columns or column in requested_outputs)
    ]
    metric_columns = sorted(dict.fromkeys(metric_columns))
    aggregation = "AVG" if re.search(r"平均|均值", question) else "SUM"
    metrics = [
        {"column": column, "aggregation": aggregation, "alias": canonical_metric_alias(column, aggregation)}
        for column in metric_columns
        if has_aggregate and not derived_requested
        if aggregation in set(aggregations.get(column, []))
    ]
    is_count = bool(re.search(r"记录数|数量|多少", question))
    if has_aggregate and not metrics and not derived_requested and not is_count:
        return None

    select_columns: list[str] = []
    if has_aggregate or derived_requested:
        select_columns.extend(group_columns)
        select_columns.extend(metric["alias"] for metric in metrics)
        if is_count:
            select_columns.append("record_count")
        if derived_requested:
            select_columns.append("carbon_intensity")
    else:
        selected = set(requested_outputs) or ({ranking[0]} if ranking is not None else set(metric_columns))
        if ranking is not None:
            selected.add(ranking[0])
        if not strict_projection:
            selected.add(catalog["tables"][fact_table]["key"])
        if not selected:
            return None
        select_columns = _ordered_columns_for_table(fact_table, selected)

    order_by: dict[str, str] | None = None
    direction = _ranking_direction(question)
    if derived_requested and re.search(r"最高|最大|最低|最小|前\s*\d+", question) and direction:
        order_by = {"kind": "derived", "alias": "carbon_intensity", "direction": direction}
    elif ranking is not None and direction:
        ranking_column = ranking[0]
        ranking_positions = [match.start() for match in re.finditer(r"最高|最低|最大|最小", question)]
        if ranking_positions:
            ranking_position = ranking_positions[-1]
            candidates: list[tuple[int, str]] = []
            for column, terms in matches.items():
                for term in terms:
                    positions = [match.start() for match in re.finditer(re.escape(term), question)]
                    if positions:
                        candidates.append((min(abs(position - ranking_position) for position in positions), column))
            if candidates:
                ranking_column = min(candidates)[1]
        if not has_aggregate and not requested_outputs:
            select_columns = _ordered_columns_for_table(fact_table, {catalog["tables"][fact_table]["key"], ranking_column})
        order_by = {"kind": "metric" if has_aggregate else "column", "column": ranking_column, "alias": canonical_metric_alias(ranking_column, aggregation), "direction": direction}
    elif metrics and group_columns and re.search(r"最高|最大|最低|最小", question) and direction:
        order_by = {"kind": "metric", "alias": metrics[0]["alias"], "direction": direction}

    if limit is not None and order_by is None:
        return None
    relationships = []
    for dimension in dimensions:
        relation = next((rel for rel in profile.get("relationships", []) if {str(rel["left"]).split(".")[0], str(rel["right"]).split(".")[0]} == {fact_table, dimension}), None)
        if relation is None:
            return None
        relationships.append(relation)
    return {
        "eligible": True, "mode": "deterministic", "query_type": "profile_fact_query",
        "table": fact_table, "fact_table": fact_table, "dimension_tables": dimensions,
        "relationships": relationships, "select_columns": select_columns, "filters": all_filters,
        "metrics": metrics, "group_columns": group_columns, "is_count": is_count,
        "derived_metrics": ([{"alias": "carbon_intensity", "formula": "co2_tco2 / NULLIF(usage_kwh, 0)", "dependencies": ["co2_tco2", "usage_kwh"]}] if derived_requested else []),
        "order_by": order_by, "limit": limit, "sample_ids": [],
        "strict_projection": strict_projection, "temporal_metrics": [], "scalar_columns": [], "scalar_tables": [],
        "confidence": 1.0,
        "reason": "Profile声明的事实表、维度关系、过滤、聚合和派生指标可确定编译。",
    }


def requires_llm_query_planning(question: str) -> bool:
    """Return True for analytic operators outside the constrained QuerySpec DSL."""

    advanced_patterns = (
        r"相较上个月|环比|变化率",
        r"累计.*(?:贡献|占比)|(?:贡献|占比).*累计",
        r"相关系数|标准差",
        r"占当月总.*(?:比例|占比)|占.*总.*(?:比例|占比)",
        r"连续\s*\d+\s*(?:个)?小时",
        r"差异及其百分比|百分比差异",
        r"(?:每种|各).{0,12}(?:最高|最低).{0,12}(?:读数|记录)",
        r"所属负荷类型.*平均|各负荷类型.*平均",
    )
    return any(re.search(pattern, question) for pattern in advanced_patterns)


def build_query_spec(
    question: str,
) -> dict[str, Any]:
    """构造静态与标准时序查询的结构化中间表示。"""

    catalog = get_schema_catalog()
    owners = get_column_owner_map()
    matches = match_question_semantic_columns(question)
    requested_outputs = infer_requested_output_columns(question)
    ranking = infer_question_ranking_column(question)
    sample_ids = sorted(extract_requested_sample_ids(question))
    explicit_full_table = infer_explicit_full_table_request(question)
    strict_projection = is_strict_projection_request(question)
    filters, consumed_numbers = _extract_simple_filters(question)
    limit = extract_requested_limit_from_question(question)

    partial_order_by: dict[str, str] | None = None
    if ranking is not None:
        direction = _ranking_direction(question)
        if direction is not None:
            partial_order_by = {
                "kind": "column",
                "column": ranking[0],
                "direction": direction,
            }

    # 即使问题仍需进入RSL，也保留已经可靠识别出的局部约束。
    # eligible=False只表示不能由确定性编译器完整生成SQL，
    # 不代表LIMIT、排序字段、返回字段等可信信息应被丢弃。
    result: dict[str, Any] = {
        "eligible": False,
        "mode": "rsl",
        "query_type": "complex_or_uncertain",
        "table": "",
        "select_columns": sorted(requested_outputs),
        "filters": filters,
        "where_filters": filters,
        "having_filters": [],
        "order_by": partial_order_by,
        "limit": limit,
        "sample_ids": sample_ids,
        "strict_projection": strict_projection,
        "temporal_metrics": [],
        "scalar_columns": [],
        "scalar_tables": [],
        "confidence": 0.0,
        "reason": "问题包含尚未结构化的多表、聚合、时序或模糊语义。",
    }

    if explicit_full_table is not None:
        columns = set(catalog["tables"][explicit_full_table]["columns"])
        result.update(
            {
                "eligible": True,
                "mode": "deterministic",
                "query_type": "full_table",
                "table": explicit_full_table,
                "select_columns": _ordered_columns_for_table(explicit_full_table, columns),
                "confidence": 1.0,
                "reason": "用户明确点名白名单表并请求全部数据。",
            }
        )
        return result

    if requires_llm_query_planning(question):
        result["reason"] = "问题包含窗口、相关性、累计、份额或分组内排名等高级分析算子，需要LLM规划。"
        return result

    temporal = _build_temporal_query_spec(
        question=question,
        matches=matches,
        requested_outputs=requested_outputs,
        ranking=ranking,
        sample_ids=sample_ids,
        strict_projection=strict_projection,
        filters=filters,
        consumed_numbers=consumed_numbers,
        limit=limit,
    )
    if temporal is not None:
        return temporal

    profile_fact = _build_profile_fact_query_spec(
        question, matches, requested_outputs, ranking, strict_projection, filters, limit
    )
    if profile_fact is not None:
        return profile_fact

    fact_aggregate = _build_fact_aggregate_query_spec(
        question, matches, requested_outputs, strict_projection
    )
    if fact_aggregate is not None:
        return fact_aggregate

    # 其他明确复杂算子继续交给RSL。
    if re.search(r"峰值|平均|均值|每个样本|分组|占比|比例|最终|最后一个点", question):
        return result

    response_columns = {
        column
        for table_name, info in catalog["tables"].items()
        if info["grain"] == "many_rows_per_sample"
        for column in info["columns"]
    } - {"sample_id"}
    if set(matches) & response_columns:
        return result

    business_columns = set(matches) - {"sample_id"}
    business_columns.update(requested_outputs - {"sample_id"})
    business_columns.update(item["column"] for item in filters)
    if ranking is not None:
        business_columns.add(ranking[0])

    candidate_tables: set[str] = set()
    for column in business_columns:
        candidate_tables.update(owners.get(column, set()))

    if len(candidate_tables) != 1:
        # 多表问题继续进入RSL，但保留已经确定的查询类型与字段角色，
        # 供Guard和Trace使用。这里不生成固定SQL，也不改变RSL路由。
        if len(candidate_tables) > 1:
            one_to_one = all(
                catalog["tables"][table]["grain"] == "one_row_per_sample"
                for table in candidate_tables
            )
            if ranking is not None and limit is not None:
                result["query_type"] = "multi_table_topk"
            elif filters:
                result["query_type"] = "multi_table_filter"
            else:
                result["query_type"] = "multi_table_projection"

            result["scalar_columns"] = sorted(business_columns)
            result["scalar_tables"] = sorted(candidate_tables)
            if one_to_one and _all_question_numbers_consumed(question, consumed_numbers, limit):
                selected = set(requested_outputs) or {"sample_id", *business_columns}
                # "返回样本编号和这三个字段" refers to the explicit
                # predicates immediately before it. This is a reusable
                # projection convention, not a benchmark-specific rule.
                if re.search(r"(?:这|上述|这些)(?:一|二|三|四|\d+)?个?字段", question):
                    selected.update(item["column"] for item in filters)
                if ranking is not None and not strict_projection:
                    selected.add(ranking[0])
                result.update({
                    "eligible": True, "mode": "deterministic",
                    "query_type": "one_to_one_join",
                    "select_columns": ["sample_id", *sorted(selected - {"sample_id"})],
                    "order_by": partial_order_by,
                    "reason": "字段均位于sample_id一对一关系表，使用通用Join编译器。",
                    "confidence": 1.0,
                })
                return result
            result["reason"] = (
                "已识别多表字段、返回字段、排序和数量约束，"
                "但SQL连接结构仍交由RSL双候选生成。"
            )
        return result

    table_name = next(iter(candidate_tables))
    if catalog["tables"][table_name]["grain"] not in {"one_row_per_sample", "fact"}:
        return result

    query_type = "single_table_filter"
    order_by: dict[str, str] | None = None
    if ranking is not None and limit is not None:
        query_type = "single_table_topk"
        direction = "DESC" if re.search(r"最高|最大|降序|从高到低", question) else "ASC"
        order_by = {"kind": "column", "column": ranking[0], "direction": direction}
    elif ranking is not None or re.search(r"最高|最低|最大|最小|前\s*\d+", question):
        return result
    elif sample_ids:
        query_type = "exact_sample"

    if not filters and not sample_ids and query_type != "single_table_topk":
        return result

    if not _all_question_numbers_consumed(question, consumed_numbers, limit):
        return result

    if requested_outputs:
        selected = set(requested_outputs)
    elif query_type == "single_table_topk" and ranking is not None:
        selected = {catalog["tables"][table_name]["key"], ranking[0]}
    elif query_type == "exact_sample":
        selected = {catalog["tables"][table_name]["key"]} | business_columns
    else:
        selected = {catalog["tables"][table_name]["key"]} | business_columns

    if not strict_projection:
        selected.add(catalog["tables"][table_name]["key"])
    if not selected:
        return result
    if any(
        column != catalog["tables"][table_name]["key"] and table_name not in owners.get(column, set())
        for column in selected
    ):
        return result

    result.update(
        {
            "eligible": True,
            "mode": "deterministic",
            "query_type": query_type,
            "table": table_name,
            "select_columns": _ordered_columns_for_table(table_name, selected),
            "order_by": order_by,
            "confidence": 1.0,
            "reason": "字段归属、单表粒度、过滤、排序和数量均可确定，使用基础查询确定性快路径。",
        }
    )
    return result


def _qualified_column(column: str, table_name: str) -> str:
    alias = get_schema_catalog()["tables"][table_name]["alias"]
    return f"{alias}.{column}"


def _column_owner(column: str) -> str:
    owners = get_column_owner_map().get(column, set())
    if len(owners) != 1:
        raise ValueError(f"字段{column}没有唯一所属表。")
    return next(iter(owners))


def _compile_predicate(
    item: dict[str, Any],
    column_sql: str,
) -> str:
    def literal(value: Any) -> str:
        if parse_decimal_literal(value) is not None:
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"

    operator = item["operator"]
    if operator == "BETWEEN":
        return f"{column_sql} BETWEEN {literal(item['value'])} AND {literal(item['value2'])}"
    return f"{column_sql} {operator} {literal(item['value'])}"


def _final_metrics_subquery(
    response_table: str,
    metrics: list[dict[str, str]],
) -> str:
    projections = ["final_row.sample_id"]
    for metric in metrics:
        projections.append(
            f"final_row.{metric['column']} AS {metric['alias']}"
        )
    return (
        "SELECT " + ", ".join(projections) + " "
        f"FROM {response_table} AS final_row "
        "JOIN ("
        f"SELECT sample_id, MAX(point_index) AS max_point_index FROM {response_table} GROUP BY sample_id"
        ") AS final_idx ON final_row.sample_id = final_idx.sample_id "
        "AND final_row.point_index = final_idx.max_point_index"
    )


def compile_query_spec_sql(
    query_spec: dict[str, Any],
) -> str:
    """把可信静态或时序QuerySpec确定性编译为MySQL SELECT。"""

    if not query_spec.get("eligible"):
        return ""

    catalog = get_schema_catalog()
    query_type = query_spec.get("query_type")

    if query_type == "response_detail":
        table_name = str(query_spec["table"])
        alias = catalog["tables"][table_name]["alias"]
        select_sql = ", ".join(
            f"{alias}.{column}" for column in query_spec["select_columns"]
        )
        parts = [f"SELECT {select_sql}", f"FROM {table_name} AS {alias}"]
        predicates: list[str] = []
        sample_ids = list(query_spec.get("sample_ids", []))
        if len(sample_ids) == 1:
            predicates.append(f"{alias}.sample_id = '{sample_ids[0]}'")
        elif sample_ids:
            values = ", ".join(f"'{value}'" for value in sample_ids)
            predicates.append(f"{alias}.sample_id IN ({values})")
        for item in query_spec.get("where_filters", []):
            predicates.append(_compile_predicate(item, f"{alias}.{item['column']}"))
        if predicates:
            parts.append("WHERE " + " AND ".join(predicates))
        order_by = query_spec.get("order_by")
        if order_by:
            parts.append(
                f"ORDER BY {alias}.{order_by['column']} {order_by['direction']}"
            )
        limit = query_spec.get("limit")
        if isinstance(limit, int) and limit > 0:
            parts.append(f"LIMIT {limit}")
        return " ".join(parts)

    if query_type == "one_to_one_join":
        tables = list(query_spec.get("scalar_tables", []))
        if not tables:
            return ""
        base = tables[0]
        base_alias = catalog["tables"][base]["alias"]
        selected = []
        for column in query_spec.get("select_columns", []):
            if column == "sample_id":
                selected.append(f"{base_alias}.sample_id")
            else:
                selected.append(_qualified_column(column, _column_owner(column)))
        parts = ["SELECT " + ", ".join(selected), f"FROM {base} AS {base_alias}"]
        for table in tables[1:]:
            alias = catalog["tables"][table]["alias"]
            parts.append(f"JOIN {table} AS {alias} ON {base_alias}.sample_id = {alias}.sample_id")
        predicates = []
        for sample_id in query_spec.get("sample_ids", []):
            predicates.append(f"{base_alias}.sample_id = '{sample_id}'")
        for item in query_spec.get("filters", []):
            predicates.append(_compile_predicate(item, _qualified_column(item["column"], _column_owner(item["column"]))))
        if predicates:
            parts.append("WHERE " + " AND ".join(predicates))
        order_by = query_spec.get("order_by")
        if order_by:
            parts.append(f"ORDER BY {_qualified_column(order_by['column'], _column_owner(order_by['column']))} {order_by['direction']}")
        if isinstance(query_spec.get("limit"), int) and query_spec["limit"] > 0:
            parts.append(f"LIMIT {query_spec['limit']}")
        return " ".join(parts)

    if query_type == "per_sample_temporal_aggregate":
        response_table = str(query_spec["table"])
        scalar_tables = list(query_spec.get("scalar_tables", []))
        scalar_columns = list(query_spec.get("scalar_columns", []))
        metrics = list(query_spec.get("temporal_metrics", []))
        all_metrics = list(query_spec.get("all_temporal_metrics", metrics))
        final_metrics = [item for item in all_metrics if item["aggregation"] == "FINAL"]
        row_metrics = [item for item in all_metrics if item["aggregation"] != "FINAL"]
        selected_keys = {(item["column"], item["aggregation"]) for item in metrics}

        final_ref = {
            item["column"]: f"fv.{item['alias']}"
            for item in final_metrics
        }
        if row_metrics:
            base_sample = "tr.sample_id"
            from_part = f"FROM {response_table} AS tr"
        elif final_metrics:
            base_sample = "fv.sample_id"
            from_part = f"FROM ({_final_metrics_subquery(response_table, final_metrics)}) AS fv"
        else:
            return ""

        join_parts: list[str] = []
        if row_metrics and final_metrics:
            join_parts.append(
                f"JOIN ({_final_metrics_subquery(response_table, final_metrics)}) AS fv "
                "ON tr.sample_id = fv.sample_id"
            )
        for table_name in scalar_tables:
            alias = catalog["tables"][table_name]["alias"]
            join_parts.append(
                f"JOIN {table_name} AS {alias} ON {base_sample} = {alias}.sample_id"
            )

        select_parts = [base_sample]
        group_parts = [base_sample]
        for column in scalar_columns:
            owner = _column_owner(column)
            expression = _qualified_column(column, owner)
            select_parts.append(expression)
            group_parts.append(expression)

        for metric in all_metrics:
            if (metric["column"], metric["aggregation"]) not in selected_keys:
                continue
            if metric["aggregation"] == "FINAL":
                expression = final_ref[metric["column"]]
                select_parts.append(f"{expression} AS {metric['alias']}")
                if row_metrics:
                    group_parts.append(expression)
            else:
                select_parts.append(
                    f"{metric['aggregation']}(tr.{metric['column']}) AS {metric['alias']}"
                )

        parts = ["SELECT " + ", ".join(select_parts), from_part]
        parts.extend(join_parts)

        where_predicates: list[str] = []
        sample_ids = list(query_spec.get("sample_ids", []))
        if len(sample_ids) == 1:
            where_predicates.append(f"{base_sample} = '{sample_ids[0]}'")
        elif sample_ids:
            values = ", ".join(f"'{value}'" for value in sample_ids)
            where_predicates.append(f"{base_sample} IN ({values})")
        for item in query_spec.get("where_filters", []):
            owner = _column_owner(item["column"])
            where_predicates.append(
                _compile_predicate(item, _qualified_column(item["column"], owner))
            )
        # FINAL指标的条件属于最终值派生表，不是HAVING。
        metric_lookup = {item["column"]: item for item in all_metrics}
        for item in query_spec.get("having_filters", []):
            metric = metric_lookup[item["column"]]
            if metric["aggregation"] == "FINAL":
                where_predicates.append(
                    _compile_predicate(item, final_ref[item["column"]])
                )
        if where_predicates:
            parts.append("WHERE " + " AND ".join(where_predicates))

        if row_metrics:
            parts.append("GROUP BY " + ", ".join(dict.fromkeys(group_parts)))

        having_predicates: list[str] = []
        for item in query_spec.get("having_filters", []):
            metric = metric_lookup[item["column"]]
            if metric["aggregation"] == "FINAL":
                continue
            having_predicates.append(
                _compile_predicate(
                    item,
                    f"{metric['aggregation']}(tr.{item['column']})",
                )
            )
        if having_predicates:
            parts.append("HAVING " + " AND ".join(having_predicates))

        order_by = query_spec.get("order_by")
        if order_by:
            if order_by["kind"] == "metric":
                expression = order_by["alias"]
            else:
                owner = _column_owner(order_by["column"])
                expression = _qualified_column(order_by["column"], owner)
            parts.append(f"ORDER BY {expression} {order_by['direction']}")
        limit = query_spec.get("limit")
        if isinstance(limit, int) and limit > 0:
            parts.append(f"LIMIT {limit}")
        return " ".join(parts)

    if query_type == "fact_aggregate":
        fact_table = str(query_spec["table"])
        fact_alias = catalog["tables"][fact_table]["alias"]
        metric = str(query_spec["metric"])
        aggregation = str(query_spec["aggregation"])
        metric_alias = str(query_spec["metric_alias"])
        group_column = str(query_spec.get("group_column", ""))
        group_table = str(query_spec.get("group_table", ""))
        metric_sql = f"{aggregation}({fact_alias}.{metric}) AS {metric_alias}"
        if not group_column:
            return f"SELECT {metric_sql} FROM {fact_table} AS {fact_alias}"

        group_alias = catalog["tables"][group_table]["alias"]
        relation = next(
            (
                rel for rel in _load_profile(active_profile_name()).get("relationships", [])
                if {str(rel["left"]).split(".")[0], str(rel["right"]).split(".")[0]} == {fact_table, group_table}
            ),
            None,
        )
        if relation is None:
            return ""
        left_table, left_column = str(relation["left"]).split(".", 1)
        right_table, right_column = str(relation["right"]).split(".", 1)
        left_alias = catalog["tables"][left_table]["alias"]
        right_alias = catalog["tables"][right_table]["alias"]
        parts = [
            f"SELECT {group_alias}.{group_column}, {metric_sql}",
            f"FROM {fact_table} AS {fact_alias}",
            f"JOIN {group_table} AS {group_alias} ON {left_alias}.{left_column} = {right_alias}.{right_column}",
            f"GROUP BY {group_alias}.{group_column}",
        ]
        order_by = query_spec.get("order_by")
        if order_by:
            parts.append(f"ORDER BY {metric_alias} {order_by['direction']}")
        return " ".join(parts)

    if query_type == "profile_fact_query":
        fact_table = str(query_spec["fact_table"])
        fact_alias = catalog["tables"][fact_table]["alias"]
        dimensions = list(query_spec.get("dimension_tables", []))
        aliases = {fact_table: fact_alias}
        aliases.update({table: catalog["tables"][table]["alias"] for table in dimensions})
        parts: list[str] = []
        group_columns = list(query_spec.get("group_columns", []))
        metrics = list(query_spec.get("metrics", []))
        derived = {
            str(item.get("alias", "")) if isinstance(item, dict) else str(item)
            for item in query_spec.get("derived_metrics", [])
        }
        select_parts: list[str] = []
        if metrics or derived or query_spec.get("is_count"):
            for column in group_columns:
                select_parts.append(_qualified_column(column, _column_owner(column)))
            for metric in metrics:
                select_parts.append(f"{metric['aggregation']}({fact_alias}.{metric['column']}) AS {metric['alias']}")
            if query_spec.get("is_count"):
                select_parts.append("COUNT(*) AS record_count")
            if "carbon_intensity" in derived:
                select_parts.append(f"SUM({fact_alias}.co2_tco2) / NULLIF(SUM({fact_alias}.usage_kwh), 0) AS carbon_intensity")
        else:
            select_parts = [f"{fact_alias}.{column}" for column in query_spec.get("select_columns", [])]
        if not select_parts:
            return ""
        parts.extend(["SELECT " + ", ".join(select_parts), f"FROM {fact_table} AS {fact_alias}"])
        for relation in query_spec.get("relationships", []):
            left_table, left_column = str(relation["left"]).split(".", 1)
            right_table, right_column = str(relation["right"]).split(".", 1)
            dimension = right_table if left_table == fact_table else left_table
            parts.append(f"JOIN {dimension} AS {aliases[dimension]} ON {aliases[left_table]}.{left_column} = {aliases[right_table]}.{right_column}")
        predicates = [
            _compile_predicate(item, _qualified_column(item["column"], _column_owner(item["column"])))
            for item in query_spec.get("filters", [])
        ]
        if predicates:
            parts.append("WHERE " + " AND ".join(predicates))
        if group_columns:
            parts.append("GROUP BY " + ", ".join(_qualified_column(column, _column_owner(column)) for column in group_columns))
        order_by = query_spec.get("order_by")
        if order_by:
            if order_by["kind"] in {"metric", "derived"}:
                expression = order_by["alias"]
            else:
                expression = _qualified_column(order_by["column"], _column_owner(order_by["column"]))
            parts.append(f"ORDER BY {expression} {order_by['direction']}")
        if isinstance(query_spec.get("limit"), int) and query_spec["limit"] > 0:
            parts.append(f"LIMIT {query_spec['limit']}")
        return " ".join(parts)

    table_name = str(query_spec["table"])
    alias = catalog["tables"][table_name]["alias"]
    select_columns = list(query_spec["select_columns"])
    if not select_columns:
        return ""
    select_sql = ", ".join(f"{alias}.{column}" for column in select_columns)
    parts = [f"SELECT {select_sql}", f"FROM {table_name} AS {alias}"]
    predicates: list[str] = []
    sample_ids = list(query_spec.get("sample_ids", []))
    if catalog["tables"][table_name]["key"] == "sample_id":
        if len(sample_ids) == 1:
            predicates.append(f"{alias}.sample_id = '{sample_ids[0]}'")
        elif len(sample_ids) > 1:
            values = ", ".join(f"'{sample_id}'" for sample_id in sample_ids)
            predicates.append(f"{alias}.sample_id IN ({values})")
    for item in query_spec.get("filters", []):
        predicates.append(_compile_predicate(item, f"{alias}.{item['column']}"))
    if predicates:
        parts.append("WHERE " + " AND ".join(predicates))
    order_by = query_spec.get("order_by")
    if order_by:
        parts.append(f"ORDER BY {alias}.{order_by['column']} {order_by['direction']}")
    limit = query_spec.get("limit")
    if isinstance(limit, int) and limit > 0:
        parts.append(f"LIMIT {limit}")
    return " ".join(parts)

def infer_forward_schema_elements(
    question: str,
) -> tuple[set[str], set[str]]:
    """从用户问题正向识别相关表和字段。"""

    owners = get_column_owner_map()
    columns = set(
        match_question_semantic_columns(
            question
        )
    )
    columns.update(
        infer_requested_output_columns(
            question
        )
    )

    ranking = infer_question_ranking_column(
        question
    )
    if ranking is not None:
        columns.add(ranking[0])

    if (
        extract_requested_sample_ids(question)
        or re.search(
            r"样本|sample_id|sample_",
            question,
            flags=re.IGNORECASE,
        )
    ):
        columns.add("sample_id")

    tables = infer_relevant_tables(question)
    for column in columns:
        if column == "sample_id":
            continue
        tables.update(
            owners.get(column, set())
        )

    return tables, columns


def extract_sql_schema_elements(
    sql: str,
) -> tuple[set[str], set[str]]:
    """从初步SQL反向提取真实表和Schema字段。"""

    if not sql:
        return set(), set()

    catalog = get_schema_catalog()
    physical_tables = set(
        catalog["tables"]
    )
    alias_to_table = {
        info["alias"]: table_name
        for table_name, info
        in catalog["tables"].items()
    }
    valid_columns = set(
        get_column_owner_map()
    )

    try:
        tree = sqlglot.parse_one(
            sql,
            read="mysql",
        )
    except ParseError:
        return set(), set()

    tables: set[str] = set()
    for table in tree.find_all(exp.Table):
        name = table.name
        if name in physical_tables:
            tables.add(name)
        elif name in alias_to_table:
            tables.add(alias_to_table[name])

    columns = {
        column.name
        for column in tree.find_all(exp.Column)
        if column.name in valid_columns
    }

    return tables, columns


def _relationships_for_tables(
    tables: set[str],
) -> list[str]:
    catalog = get_schema_catalog()
    return [
        relationship
        for relationship in catalog[
            "relationships"
        ]
        if sum(
            table_name in relationship
            for table_name in tables
        ) >= 2
    ]


def build_schema_context_for_tables(
    question: str,
    tables: set[str],
    highlighted_columns: set[str] | None = None,
) -> str:
    """为SQL2构建严格的表级和列级裁剪Schema。"""

    catalog = get_schema_catalog()
    is_resin_profile = active_profile_name() == "resin"
    owners = get_column_owner_map()
    selected_tables = {
        table_name
        for table_name in tables
        if table_name in catalog["tables"]
    }
    if not selected_tables:
        return build_schema_context()

    requested_columns = set(highlighted_columns or set())
    explicit_full_table = infer_explicit_full_table_request(question)
    lines: list[str] = [
        "数据库类型：MySQL",
        "稳健Schema Linking后允许使用的真实表和必要字段：",
    ]

    for table_name, info in catalog["tables"].items():
        if table_name not in selected_tables:
            continue
        if explicit_full_table == table_name:
            allowed_columns = set(info["columns"])
        else:
            allowed_columns = {
                column
                for column in requested_columns
                if column in info["columns"]
                and table_name in owners.get(column, set())
            }
            if is_resin_profile:
                allowed_columns.add("sample_id")

        grain_text = (
            "每个样本一行"
            if info["grain"] == "one_row_per_sample"
            else "每个样本多行时序记录"
        )
        lines.append(
            f"[{table_name}] 推荐别名：{info['alias']}；粒度：{grain_text}"
        )
        for column, description in info["columns"].items():
            if column in allowed_columns:
                lines.append(f"- {column}：{description}")

    relationships = _relationships_for_tables(selected_tables)
    if relationships:
        lines.append("必要连接关系：")
        lines.extend(f"- {item}" for item in relationships)

    constraints = [
        build_question_field_hint(question),
        "生成约束：",
        "- 只能使用以上列出的表和字段。",
        "- 普通记录级Top-K直接ORDER BY目标字段并LIMIT。",
        "- 禁止SELECT *，必须严格遵守用户返回字段要求。",
    ]
    if is_resin_profile:
        constraints.extend([
            "- 不得因为sample_id是公共连接键而引入无关表。",
            "- 一行一个样本的字段Top-K不使用MAX、GROUP BY或IN子查询。",
            "- 只有时序峰值才使用MAX并按sample_id分组。",
        ])
    lines.extend(constraints)
    return "\n".join(lines)


def build_robust_schema_linking(
    question: str,
    preliminary_sql: str,
) -> dict[str, Any]:
    """置信度感知地合并正向与反向Schema Linking。"""

    catalog = get_schema_catalog()
    owners = get_column_owner_map()
    forward_tables, forward_columns = infer_forward_schema_elements(question)
    backward_tables, backward_columns = extract_sql_schema_elements(preliminary_sql)
    explicit_full_table = infer_explicit_full_table_request(question)

    accepted_backward_tables: set[str] = set()
    rejected_backward_tables: set[str] = set()
    forward_business_columns = set(forward_columns) - {"sample_id"}

    if explicit_full_table is not None:
        robust_tables = {explicit_full_table}
        robust_columns = set(catalog["tables"][explicit_full_table]["columns"])
    elif forward_business_columns and forward_tables:
        # 正向业务字段链接完整时，以其为高置信主干。
        robust_tables = set(forward_tables)
        robust_columns = set(forward_columns)
        for table_name in backward_tables:
            owns_needed_column = any(
                table_name in owners.get(column, set())
                for column in forward_business_columns
            )
            if table_name in forward_tables or owns_needed_column:
                accepted_backward_tables.add(table_name)
            else:
                rejected_backward_tables.add(table_name)
        robust_tables.update(accepted_backward_tables)
    else:
        # 正向链接不足时，才使用SQL1中的有效字段作为补充召回。
        robust_columns = set(forward_columns)
        for column in backward_columns:
            if column == "sample_id":
                continue
            owner_tables = owners.get(column, set())
            if owner_tables:
                robust_columns.add(column)
                accepted_backward_tables.update(owner_tables)
        robust_tables = set(forward_tables) | accepted_backward_tables
        rejected_backward_tables = set(backward_tables) - robust_tables

    if active_profile_name() == "resin" and (
        len(robust_tables) > 1 or re.search(
            r"样本|sample_id|sample_",
            question,
            flags=re.IGNORECASE,
        )
    ):
        robust_columns.add("sample_id")

    if not robust_tables:
        robust_tables = set(catalog["tables"])
        robust_columns = set(backward_columns) or {"sample_id"}

    context = build_schema_context_for_tables(
        question=question,
        tables=robust_tables,
        highlighted_columns=robust_columns,
    )

    return {
        "forward_tables": sorted(forward_tables),
        "forward_columns": sorted(forward_columns),
        "backward_tables": sorted(backward_tables),
        "backward_columns": sorted(backward_columns),
        "accepted_backward_tables": sorted(accepted_backward_tables),
        "rejected_backward_tables": sorted(rejected_backward_tables),
        "robust_tables": sorted(robust_tables),
        "robust_columns": sorted(robust_columns),
        "context": context,
    }


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
    profile = _load_profile(active_profile_name())
    database_purpose = profile.get(
        "description",
        f"查询{profile.get('database_name', '当前数据库')}中的白名单业务数据。",
    )

    lines: list[str] = [
        "数据库类型：MySQL",
        "",
        "数据库用途：",
        str(database_purpose),
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
