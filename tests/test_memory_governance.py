import os
from pathlib import Path

from app.long_term_memory.config import LongTermMemorySettings
from app.long_term_memory.service import LongTermMemoryService


def test_candidate_memory_starts_pending_review(tmp_path: Path):
    settings = LongTermMemorySettings(
        enabled=True, namespace="test", db_path=tmp_path / "memory.sqlite3",
        embedding_model="", embedding_device="cpu", allow_model_download=False,
        semantic_top_k=3, episodic_top_k=3, procedural_top_k=2,
        semantic_min_score=.48, episodic_min_score=.2, procedural_min_score=.25,
        episodic_candidate_k=20, episodic_max_examples=2,
        episodic_structural_min_score=.45, episodic_final_min_score=.48,
        episodic_mmr_lambda=.8, auto_save=False, max_prompt_chars=6000,
    )
    service = LongTermMemoryService(settings)
    record = service.remember_candidate_case(
        question="查询总耗电量", resolved_question="查询总耗电量",
        query_spec={"query_type": "aggregate", "eligible": True},
        sql="SELECT SUM(usage_kwh) FROM energy_readings",
        source="test",
    ).record
    assert record.metadata["lifecycle"] == "pending_review"
    assert record.metadata["promotion_status"] == "candidate"

