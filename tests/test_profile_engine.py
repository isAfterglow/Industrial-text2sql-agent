from __future__ import annotations

from contextlib import contextmanager
import unittest
from unittest.mock import patch

from app.db import execute_readonly_query
from app.schema import set_active_profile


class _Result:
    def keys(self):
        return ["value"]

    def fetchmany(self, _limit):
        return [(1,)]


class _Connection:
    def execute(self, _statement):
        return _Result()


class _Engine:
    @contextmanager
    def connect(self):
        yield _Connection()


class ProfileEngineTests(unittest.TestCase):
    def test_execution_uses_explicit_active_profile_cache_key(self) -> None:
        set_active_profile("steel_industry")
        with patch("app.db.get_engine", return_value=_Engine()) as get_engine:
            result = execute_readonly_query("SELECT 1", 10)
        self.assertEqual(result["rows"], [[1]])
        get_engine.assert_called_once_with("steel_industry")


if __name__ == "__main__":
    unittest.main()
