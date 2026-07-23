"""Tests for the project cache: memoization, safe deletion, regenerability."""

import tempfile
import unittest
from pathlib import Path

from aion.workspace.cache import ProjectCache


class CacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = ProjectCache(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.get("stats", "abc"))

    def test_get_or_compute_runs_once_then_hits(self):
        calls = []

        def compute():
            calls.append(1)
            return {"n": 42}

        first = self.cache.get_or_compute("stats", "abc", compute)
        second = self.cache.get_or_compute("stats", "abc", compute)
        self.assertEqual(first, {"n": 42})
        self.assertEqual(second, {"n": 42})
        self.assertEqual(len(calls), 1)  # computed only on the miss

    def test_different_fingerprint_recomputes(self):
        self.cache.put("stats", "v1", {"n": 1})
        self.assertIsNone(self.cache.get("stats", "v2"))

    def test_delete_entry_is_safe_and_recomputable(self):
        self.cache.get_or_compute("stats", "abc", lambda: {"n": 1})
        self.cache.delete("stats", "abc")
        self.assertIsNone(self.cache.get("stats", "abc"))
        # Nothing lost: it recomputes from source.
        again = self.cache.get_or_compute("stats", "abc", lambda: {"n": 1})
        self.assertEqual(again, {"n": 1})

    def test_delete_namespace_and_clear(self):
        self.cache.put("stats", "a", {"n": 1})
        self.cache.put("quality", "b", {"n": 2})
        self.cache.delete("stats")
        self.assertIsNone(self.cache.get("stats", "a"))
        self.assertIsNotNone(self.cache.get("quality", "b"))
        self.cache.clear()
        self.assertIsNone(self.cache.get("quality", "b"))

    def test_envelope_records_provenance(self):
        self.cache.put("stats", "a", {"n": 1})
        path = Path(self._tmp.name) / "stats" / "a.json"
        import json
        env = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(env["fingerprint"], "a")
        self.assertIn("produced_by", env)
        self.assertEqual(env["value"], {"n": 1})


if __name__ == "__main__":
    unittest.main()
