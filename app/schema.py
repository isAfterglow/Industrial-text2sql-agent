from app.config import get_settings


def build_schema_context() -> str:
    """构造提供给 LLM 的人工维护 Schema 说明。"""

    settings = get_settings()

    static_table = settings.RESIN_TABLE_STATIC
    property_table = settings.RESIN_TABLE_MATERIAL_THERMAL_PROPERTY
    response_table = settings.RESIN_TABLE_THERMAL_RESPONSE

    return f"""
数据库类型：MySQL

数据库用途：
存储树脂基防热材料样本的静态参数、热物性参数和时序热响应。

==================================================
表一：{static_table}
==================================================

作用：
每行表示一个材料样本的静态材料参数。

字段：
- sample_id：样本唯一编号，主键
- rhov_i：原始材料密度
- rhoc_i：碳化材料密度
- porosity_v：原始材料孔隙率
- porosity_c：碳化材料孔隙率
- permeability_v：原始材料渗透率
- permeability_c：碳化材料渗透率

==================================================
表二：{property_table}
==================================================

作用：
每行表示一个样本的热物性和热解参数。

注意：
这张表不是外部热流、压力或加热时间等试验工况表。

字段：
- sample_id：样本编号，主键，同时关联 {static_table}.sample_id
- kv_list：原始材料热导率参数
- kc_list：碳化材料热导率参数
- cpv_list：原始材料比热容参数
- cpc_list：碳化材料比热容参数
- pyrolysis_heat：热解热
- surface_emissivity：表面发射率

==================================================
表三：{response_table}
==================================================

作用：
存储每个样本的时序热响应。
每个样本通常包含 3000 个响应点。

字段：
- sample_id：样本编号，关联 {static_table}.sample_id
- point_index：序列点编号，范围通常为 0 到 2999
- surface_temperature：该序列点的表面温度
- back_temperature：该序列点的背面温度
- mass：该序列点的质量

重要语义：
- 最大表面温度：MAX(surface_temperature)
- 最大背面温度：MAX(back_temperature)
- 初始响应点：point_index = 0
- 最终响应点：point_index = 2999
- point_index 是序列点编号，不能直接称为秒或物理时间
- 查询峰值时，应按 sample_id 分组
- 查询完整响应曲线时，必须限制 sample_id，并按 point_index 排序

==================================================
表关系
==================================================

1. {property_table}.sample_id = {static_table}.sample_id
   关系：1 对 1

2. {response_table}.sample_id = {static_table}.sample_id
   关系：1 对多

正确 JOIN 示例：

FROM {static_table} AS ms
JOIN {property_table} AS mtp
    ON ms.sample_id = mtp.sample_id

FROM {static_table} AS ms
JOIN {response_table} AS tr
    ON ms.sample_id = tr.sample_id

==================================================
SQL 生成规则
==================================================

1. 只生成一条 MySQL SELECT 查询。
2. 禁止 INSERT、UPDATE、DELETE、DROP、ALTER、CREATE。
3. 只能访问以上三张表。
4. 必须使用真实字段名，不允许猜测字段。
5. 禁止 SELECT *，应明确写出需要的字段。
6. 默认最多返回 200 行。
7. 查询 thermal_response 明细时，必须尽量限制 sample_id 或 point_index。
8. 不知道字段单位时，不得在结果中自行假设单位。
9. 不要使用数据库名前缀。
10. 只输出 SQL，不输出 Markdown、解释或代码围栏。

==================================================
GROUP BY 使用注意
==================================================

只有以下情况才使用 GROUP BY：

1. 对 thermal_response 按 sample_id 计算峰值、均值、最小值；
2. 用户明确要求按照某字段分组统计；
3. 一个样本的多条 thermal_response 需要聚合成一行。

如果查询只涉及一行一个样本的 material_static 和
material_thermal_property，不要使用 GROUP BY。

==================================================
使用最少必要表
==================================================

只连接回答问题所必需的数据表。

例如：

查询每个样本的峰值背温时，只需要 thermal_response，
不要连接 material_static 或 material_thermal_property。

只有用户要求同时展示其他表中的字段时，才添加 JOIN。
查样本原始密度时，不要连接 thermal_response，除非结果里要显示它的字段
查某个样本峰值背温时，不要连接 material_thermal_property，除非结果里要显示它的字段

==================================================
一对一表不需要聚合
==================================================

material_static 每个 sample_id 只有一行。

material_thermal_property 每个 sample_id 也只有一行。

查询某个样本的静态参数、热物性参数，
或按照这些参数排序时，不要使用 MIN、MAX、GROUP BY 去重。

==================================================
时序范围查询规则
==================================================

用户查询 point_index 范围时，必须使用 WHERE 条件限制范围。

例如“从0到20”表示：

point_index BETWEEN 0 AND 20

该范围包含21个点。

LIMIT 不能替代 point_index 的范围条件。

查询单个样本的响应明细时，必须同时：

1. 使用 sample_id 过滤；
2. 使用 point_index 过滤或合理 LIMIT；
3. 按 point_index ASC 排序。

==================================================
示例
==================================================

问题：
查询原始材料密度最低的5个样本。

SQL：
SELECT
    sample_id,
    rhov_i
FROM material_static
ORDER BY rhov_i ASC
LIMIT 5;

解释：静态参数排序，不聚合

问题：
查询表面发射率最高的5个样本，同时返回原始密度。

SQL：
SELECT
    ms.sample_id,
    ms.rhov_i,
    mtp.surface_emissivity
FROM material_static AS ms
JOIN material_thermal_property AS mtp
    ON ms.sample_id = mtp.sample_id
ORDER BY mtp.surface_emissivity DESC
LIMIT 5;

解释：一对一 JOIN，不分组

问题：
查询样本100从point_index 0到20的表面温度、背面温度和质量。

SQL：
SELECT
    sample_id,
    point_index,
    surface_temperature,
    back_temperature,
    mass
FROM thermal_response
WHERE sample_id = 'sample_000100'
  AND point_index BETWEEN 0 AND 20
ORDER BY point_index ASC
LIMIT 21;

解释：样本编号和点位范围

问题：
找出峰值背温最低的 5 个样本，并显示原始密度和表面发射率。

SQL：
SELECT
    ms.sample_id,
    ms.rhov_i,
    mtp.surface_emissivity,
    MAX(tr.back_temperature) AS peak_back_temperature
FROM {static_table} AS ms
JOIN {property_table} AS mtp
    ON ms.sample_id = mtp.sample_id
JOIN {response_table} AS tr
    ON ms.sample_id = tr.sample_id
GROUP BY
    ms.sample_id,
    ms.rhov_i,
    mtp.surface_emissivity
ORDER BY peak_back_temperature ASC
LIMIT 5;
""".strip()