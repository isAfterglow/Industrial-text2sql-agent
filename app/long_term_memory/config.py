from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LongTermMemorySettings:
    """持久化长期记忆和结构感知Few-shot配置。"""

    enabled: bool
    namespace: str
    db_path: Path
    embedding_model: str
    embedding_device: str
    allow_model_download: bool

    semantic_top_k: int
    episodic_top_k: int
    procedural_top_k: int
    semantic_min_score: float
    episodic_min_score: float
    procedural_min_score: float

    # QuerySpec-aware Hybrid Demonstration Retrieval
    episodic_candidate_k: int
    episodic_max_examples: int
    episodic_structural_min_score: float
    episodic_final_min_score: float
    episodic_mmr_lambda: float

    auto_save: bool
    max_prompt_chars: int


def get_long_term_memory_settings() -> LongTermMemorySettings:
    project_root = Path.cwd()
    db_path = Path(
        os.getenv(
            "LTM_DB_PATH",
            str(project_root / "data" / "long_term_memory.sqlite3"),
        )
    )

    return LongTermMemorySettings(
        enabled=_env_bool("LTM_ENABLED", True),
        namespace=os.getenv("LTM_NAMESPACE", "resin_text2sql"),
        db_path=db_path,
        embedding_model=os.getenv(
            "LTM_EMBEDDING_MODEL",
            str(project_root / "models" / "bge-m3"),
        ),
        embedding_device=os.getenv("LTM_EMBEDDING_DEVICE", "cpu"),
        allow_model_download=_env_bool("LTM_ALLOW_MODEL_DOWNLOAD", False),
        semantic_top_k=_env_int("LTM_SEMANTIC_TOP_K", 3),
        episodic_top_k=_env_int("LTM_EPISODIC_TOP_K", 3),
        procedural_top_k=_env_int("LTM_PROCEDURAL_TOP_K", 2),
        semantic_min_score=_env_float("LTM_SEMANTIC_MIN_SCORE", 0.48),
        episodic_min_score=_env_float("LTM_EPISODIC_MIN_SCORE", 0.20),
        procedural_min_score=_env_float("LTM_PROCEDURAL_MIN_SCORE", 0.25),
        episodic_candidate_k=_env_int("LTM_EPISODIC_CANDIDATE_K", 20),
        episodic_max_examples=_env_int("LTM_EPISODIC_MAX_EXAMPLES", 2),
        episodic_structural_min_score=_env_float(
            "LTM_EPISODIC_STRUCTURAL_MIN_SCORE", 0.45
        ),
        episodic_final_min_score=_env_float(
            "LTM_EPISODIC_FINAL_MIN_SCORE", 0.48
        ),
        episodic_mmr_lambda=_env_float("LTM_EPISODIC_MMR_LAMBDA", 0.80),
        auto_save=_env_bool("LTM_AUTO_SAVE", True),
        max_prompt_chars=_env_int("LTM_MAX_PROMPT_CHARS", 6000),
    )