"""Small memory-specific regression set; cases target state leakage, not trivia."""

SUITE = {
    "name": "memory_governance_regression",
    "version": "1.0.0",
    "profile": "resin",
    "cases": [
        {"id": "memory_reg_001", "category": "memory_conflict", "tags": ["independent_scope"], "turns": [
            {"question": "查询原始密度最高的3个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 3", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "查询样本32的碳化密度。", "gold_sql": "SELECT rhoc_i FROM material_static WHERE sample_id = 'sample_000032'", "expected_status": "success", "expected_columns": ["rhoc_i"], "ordered": False},
        ]},
        {"id": "memory_reg_002", "category": "memory_scope", "tags": ["previous_result_set"], "turns": [
            {"question": "查询原始密度最高的5个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 5", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "这些样本中只返回原始孔隙率。", "gold_sql": "SELECT porosity_v FROM material_static WHERE sample_id IN ({{previous_ids}})", "expected_status": "success", "expected_columns": ["porosity_v"], "ordered": False},
        ]},
        {"id": "memory_reg_003", "category": "memory_single_anchor", "tags": ["singular_reference"], "turns": [
            {"question": "查询原始密度最高的1个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 1", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "这个样本只返回碳化密度。", "gold_sql": "SELECT rhoc_i FROM material_static WHERE sample_id IN ({{previous_ids}})", "expected_status": "success", "expected_columns": ["rhoc_i"], "ordered": False},
        ]},
        {"id": "memory_reg_004", "category": "memory_scope_append_filter", "tags": ["delta", "filter"], "turns": [
            {"question": "查询原始密度最高的5个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 5", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "这些样本中原始孔隙率大于0.1的，返回样本编号。", "gold_sql": "SELECT sample_id FROM material_static WHERE sample_id IN ({{previous_ids}}) AND porosity_v > 0.1", "expected_status": "success", "expected_columns": ["sample_id"], "ordered": False},
        ]},
        {"id": "memory_reg_005", "category": "memory_reset", "tags": ["independent_query", "no_leak"], "turns": [
            {"question": "查询原始密度最高的3个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 3", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "重新查询原始孔隙率最低的2个样本。", "gold_sql": "SELECT sample_id, porosity_v FROM material_static ORDER BY porosity_v ASC LIMIT 2", "expected_status": "success", "expected_columns": ["sample_id", "porosity_v"], "ordered": True},
        ]},
        {"id": "memory_reg_006", "category": "memory_truncated_anchor", "tags": ["truncated", "clarification"], "turns": [
            {"question": "查询全部样本。", "gold_sql": "SELECT sample_id FROM material_static", "expected_status": "success", "expected_columns": ["sample_id"], "ordered": False},
            {"question": "这些样本只返回原始密度。", "gold_sql": "", "expected_status": "clarification_required", "expected_columns": [], "ordered": False},
        ]},
        {"id": "memory_reg_007", "category": "memory_display_projection", "tags": ["internal_anchor_column"], "turns": [
            {"question": "查询原始密度最高的3个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 3", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "这些样本只返回原始孔隙率。", "gold_sql": "SELECT porosity_v FROM material_static WHERE sample_id IN ({{previous_ids}})", "expected_status": "success", "expected_columns": ["porosity_v"], "ordered": False},
        ]},
        {"id": "memory_reg_008", "category": "memory_profile_boundary", "tags": ["profile_isolation"], "turns": [
            {"question": "查询原始密度最高的3个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 3", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "查询这些样本的耗电量。", "gold_sql": "", "expected_status": "clarification_required", "expected_columns": [], "ordered": False},
        ]},
        {"id": "memory_reg_009", "category": "memory_parent_anchor", "tags": ["parent_scope"], "turns": [
            {"question": "查询原始密度最高的5个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 5", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "这些样本中原始孔隙率最高的2个。", "gold_sql": "SELECT sample_id, porosity_v FROM material_static WHERE sample_id IN ({{previous_ids}}) ORDER BY porosity_v DESC LIMIT 2", "expected_status": "success", "expected_columns": ["sample_id", "porosity_v"], "ordered": True},
            {"question": "这两个样本只返回碳化孔隙率。", "gold_sql": "SELECT porosity_c FROM material_static WHERE sample_id IN ({{previous_ids}})", "expected_status": "success", "expected_columns": ["porosity_c"], "ordered": False},
        ]},
        {"id": "memory_reg_010", "category": "memory_conflict_clarification", "tags": ["conflict", "clarification"], "turns": [
            {"question": "查询原始密度最高的3个样本。", "gold_sql": "SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 3", "expected_status": "success", "expected_columns": ["sample_id", "rhov_i"], "ordered": True},
            {"question": "这些样本，并且重新查询样本32的碳化密度。", "gold_sql": "SELECT rhoc_i FROM material_static WHERE sample_id = 'sample_000032'", "expected_status": "success", "expected_columns": ["rhoc_i"], "ordered": False},
        ]},
    ],
}
