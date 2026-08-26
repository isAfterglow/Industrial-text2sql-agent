from app.memory import new_short_term_memory, update_short_term_memory


def test_successful_result_creates_typed_anchor():
    memory = new_short_term_memory("s-anchor")
    updated = update_short_term_memory(
        memory,
        question="查前三个样本",
        resolved_question="查原始密度最高的三个样本",
        query_spec={"order_by": {"column": "rhov_i", "direction": "DESC"}},
        validated_sql="SELECT sample_id, rhov_i FROM material_static ORDER BY rhov_i DESC LIMIT 3",
        columns=["sample_id", "rhov_i"],
        rows=[["sample_000001", 3], ["sample_000002", 2]],
        row_count=2,
        truncated=False,
        final_status="first_pass_success",
        turn_type="new_query",
    )
    scope = updated["last_result_scope"]
    assert scope["anchor_id"].startswith("anchor-")
    assert scope["entity_type"] == "sample"
    assert scope["entity_key"] == "sample_id"
    assert scope["status"] == "active"


def test_truncated_anchor_is_explicit():
    memory = new_short_term_memory("s-anchor")
    updated = update_short_term_memory(
        memory,
        question="查样本",
        resolved_question="查样本",
        query_spec={}, validated_sql="SELECT sample_id FROM material_static",
        columns=["sample_id"], rows=[[f"sample_{i:06d}"] for i in range(2)],
        row_count=200, truncated=True, final_status="first_pass_success",
        turn_type="new_query",
    )
    assert updated["last_result_scope"]["status"] == "truncated"
    assert updated["last_result_scope"]["storage_mode"] == "deferred_result_set"
    assert updated["last_result_scope"]["resolution_required"] is True
    assert updated["last_result_scope"]["entity_count"] == 200
