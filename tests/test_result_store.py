from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.result_store import ResultAnchorStore


def test_result_anchor_store_round_trip():
    with TemporaryDirectory() as directory:
        class Settings:
            REDIS_URL = ""
            SESSION_MEMORY_TTL_SECONDS = 3600
            SESSION_MEMORY_DB_PATH = str(Path(directory) / "session.sqlite3")
        with patch("app.result_store.get_settings", return_value=Settings()):
            store = ResultAnchorStore()
            scope = {"anchor_id": "anchor-x", "sample_ids": ["sample_1"], "entity_count": 1, "profile": "resin", "schema_hash": "hash", "status": "truncated"}
            assert store.save(scope)["saved"] is True
            assert store.load("anchor-x")["sample_ids"] == ["sample_1"]
