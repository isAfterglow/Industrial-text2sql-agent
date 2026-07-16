from __future__ import annotations

import tempfile
from pathlib import Path

from app.long_term_memory.config import LongTermMemorySettings
from app.long_term_memory.service import LongTermMemoryService


def build_test_settings(temp_dir: Path) -> LongTermMemorySettings:
    return LongTermMemorySettings(
        enabled=True,
        namespace="resin_text2sql_test",
        db_path=temp_dir / "memory.sqlite3",
        # 使用不存在的路径，明确测试Embedding不可用时的词法回退。
        embedding_model=str(temp_dir / "missing-bge-m3"),
        embedding_device="cpu",
        allow_model_download=False,
        semantic_top_k=3,
        episodic_top_k=3,
        procedural_top_k=2,
        semantic_min_score=0.48,
        episodic_min_score=0.28,
        procedural_min_score=0.25,
        auto_save=True,
        max_prompt_chars=6000,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="resin-ltm-test-") as directory:
        temp_dir = Path(directory)
        service = LongTermMemoryService(build_test_settings(temp_dir))
        service.ensure_default_memories()

        semantic = service.remember_semantic("生料热导率 -> kv_list")
        semantic_matches = service.retrieve_semantic(
            "查询生料热导率最高的5个样本"
        )
        augmented, _, applied_ids = service.apply_semantic_memories(
            "查询生料热导率最高的5个样本",
            semantic_matches,
        )
        assert "原始材料热导率" in augmented
        assert semantic.record.memory_id in applied_ids

        query_spec = {
            "eligible": False,
            "mode": "rsl",
            "query_type": "multi_table_topk",
            "select_columns": ["sample_id", "rhoc_i", "kv_list"],
            "filters": [],
            "order_by": {
                "kind": "column",
                "column": "rhoc_i",
                "direction": "DESC",
            },
            "limit": 7,
            "scalar_tables": [
                "material_static",
                "material_thermal_property",
            ],
        }
        service.remember_case(
            question="查询碳化密度最高7个，返回原始热导率",
            resolved_question="查询碳化密度最高7个，返回原始热导率",
            query_spec=query_spec,
            sql=(
                "SELECT ms.sample_id, ms.rhoc_i, mtp.kv_list "
                "FROM material_static AS ms "
                "JOIN material_thermal_property AS mtp "
                "ON ms.sample_id = mtp.sample_id "
                "ORDER BY ms.rhoc_i DESC LIMIT 7"
            ),
            source="test",
        )

        cases = service.retrieve_episodic(
            "查询碳化密度最高的5个并返回原始热导率",
            query_spec,
        )
        assert cases, "情节记忆没有召回相似案例。"
        assert "案例1" in service.build_few_shot_context(cases)

        procedures = service.retrieve_procedural(
            "IN子查询中使用LIMIT不支持"
        )
        assert procedures, "程序性记忆没有召回修复经验。"

        success, _ = service.forget(semantic.record.memory_id[:12])
        assert success, "长期记忆停用失败。"

        print("V0.8.1长期记忆基础测试通过。")
        print(service.status_summary())


if __name__ == "__main__":
    main()