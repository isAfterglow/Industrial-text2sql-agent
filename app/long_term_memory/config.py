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
    """V0.8.1长期记忆配置。

    所有配置均可通过环境变量覆盖，不要求修改现有app/config.py。
    """

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
        episodic_min_score=_env_float("LTM_EPISODIC_MIN_SCORE", 0.28),
        procedural_min_score=_env_float("LTM_PROCEDURAL_MIN_SCORE", 0.25),
        auto_save=_env_bool("LTM_AUTO_SAVE", True),
        max_prompt_chars=_env_int("LTM_MAX_PROMPT_CHARS", 6000),
    )