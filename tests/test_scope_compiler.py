from app.query_enhancement import compile_extended_query_sql
from app.schema import compile_query_spec_sql, get_schema_hash, set_active_profile


def _scope(**overrides):
    value = {
        "anchor_id": "anchor-test",
        "entity_type": "sample",
        "entity_key": "sample_id",
        "status": "active",
        "profile": "resin",
        "schema_hash": get_schema_hash(),
        "ordered_sample_ids": ["sample_000001", "sample_000002"],
    }
    value.update(overrides)
    return value


def test_query_spec_compiler_consumes_typed_scope():
    set_active_profile("resin")
    sql = compile_query_spec_sql({
        "eligible": True,
        "query_type": "response_detail",
        "table": "material_static",
        "select_columns": ["porosity_v"],
        "scope": _scope(),
    })
    assert "sample_000001" in sql and "sample_000002" in sql
    assert "sample_id IN" in sql


def test_scope_schema_mismatch_fails_closed():
    set_active_profile("resin")
    sql = compile_query_spec_sql({
        "eligible": True,
        "query_type": "response_detail",
        "table": "material_static",
        "select_columns": ["porosity_v"],
        "scope": _scope(schema_hash="stale-schema"),
    })
    assert sql == ""


def test_extended_compiler_consumes_typed_scope():
    set_active_profile("resin")
    sql = compile_extended_query_sql({
        "mode": "deterministic_extended",
        "output_columns": ["sample_id", "rhov_i"],
        "scope": _scope(ordered_sample_ids=["sample_000001"]),
    })
    assert "ms.sample_id = 'sample_000001'" in sql
