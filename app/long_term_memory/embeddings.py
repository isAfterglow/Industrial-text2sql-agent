from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np

from .config import LongTermMemorySettings
from .models import EmbeddingStatus


class EmbeddingProvider:
    """懒加载BGE-M3。

    本地模型不可用时不阻塞主查询，自动退化为词法检索。
    """

    def __init__(self, settings: LongTermMemorySettings) -> None:
        self.settings = settings
        self._model = None
        self._load_attempted = False
        self._load_error = ""
        self._lock = Lock()

    def status(self) -> EmbeddingStatus:
        if self._model is not None:
            return EmbeddingStatus(True, self.settings.embedding_model)
        if self._load_attempted:
            return EmbeddingStatus(
                False,
                self.settings.embedding_model,
                self._load_error,
            )
        return EmbeddingStatus(
            False,
            self.settings.embedding_model,
            "模型尚未按需加载。",
        )

    def _resolve_model_name(self) -> str | None:
        configured = self.settings.embedding_model.strip()
        if not configured:
            return None

        path = Path(configured).expanduser()
        if path.exists():
            return str(path.resolve())

        if self.settings.allow_model_download:
            return configured

        return None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_attempted:
                return None

            self._load_attempted = True
            model_name = self._resolve_model_name()
            if model_name is None:
                self._load_error = (
                    "未找到本地Embedding模型。请设置LTM_EMBEDDING_MODEL，"
                    "或将BGE-M3放在models/bge-m3。"
                )
                return None

            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    model_name,
                    device=self.settings.embedding_device,
                )
            except Exception as exc:  # pragma: no cover - 依赖具体运行环境
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._model = None

        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        values = [str(text).strip() for text in texts]
        if not values:
            return []

        model = self._ensure_model()
        if model is None:
            return []

        vectors = model.encode(
            values,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=min(16, len(values)),
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return [row.tolist() for row in array]

    def encode_one(self, text: str) -> list[float] | None:
        vectors = self.encode([text])
        return vectors[0] if vectors else None