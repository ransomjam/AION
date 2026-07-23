"""Tests for manifest.clear_default and ProjectStore.clear_default."""

import tempfile
import unittest
from pathlib import Path

from aion.workspace import manifest as manifest_mod
from aion.workspace.store import ProjectStore


class TestClearDefault(unittest.TestCase):
    def test_clears_matching_entry(self):
        m = {"defaults": {"tokenizer": "tok-abc", "embedding": None}}
        changed = manifest_mod.clear_default(m, "tokenizer", "tok-abc")
        self.assertTrue(changed)
        self.assertIsNone(m["defaults"]["tokenizer"])

    def test_no_change_when_different_id(self):
        m = {"defaults": {"tokenizer": "tok-xyz"}}
        changed = manifest_mod.clear_default(m, "tokenizer", "tok-abc")
        self.assertFalse(changed)
        self.assertEqual(m["defaults"]["tokenizer"], "tok-xyz")

    def test_no_change_when_already_none(self):
        m = {"defaults": {"tokenizer": None}}
        changed = manifest_mod.clear_default(m, "tokenizer", "tok-abc")
        self.assertFalse(changed)

    def test_embedding_key_cleared(self):
        m = {"defaults": {"embedding": "emb-001", "tokenizer": "tok-1"}}
        manifest_mod.clear_default(m, "embedding", "emb-001")
        self.assertIsNone(m["defaults"]["embedding"])
        self.assertEqual(m["defaults"]["tokenizer"], "tok-1")

    def test_new_manifest_has_embedding_default(self):
        m = manifest_mod.new_manifest("x", "X")
        self.assertIn("embedding", m["defaults"])
        self.assertIsNone(m["defaults"]["embedding"])

    def test_normalize_adds_embedding_to_old_manifest(self):
        # Simulate an old manifest that predates the embedding field
        old = {"id": "x", "name": "X", "defaults": {"tokenizer": None, "dataset": None, "model": None}}
        normalized = manifest_mod.normalize(old)
        self.assertIn("embedding", normalized["defaults"])


class TestProjectStoreClearDefault(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self._tmp.name))
        self.store.create("Alpha")

    def tearDown(self):
        self._tmp.cleanup()

    def test_clear_default_nulls_matching_entry(self):
        self.store.update("alpha", defaults={"tokenizer": "tok-1", "embedding": None,
                                             "dataset": None, "model": None})
        self.store.clear_default("alpha", "tokenizer", "tok-1")
        project = self.store.open("alpha")
        self.assertIsNone(project.manifest["defaults"]["tokenizer"])

    def test_clear_default_no_op_when_different(self):
        self.store.update("alpha", defaults={"tokenizer": "tok-2", "embedding": None,
                                             "dataset": None, "model": None})
        self.store.clear_default("alpha", "tokenizer", "tok-1")
        project = self.store.open("alpha")
        self.assertEqual(project.manifest["defaults"]["tokenizer"], "tok-2")

    def test_clear_default_persisted_to_disk(self):
        self.store.update("alpha", defaults={"tokenizer": "tok-x", "embedding": None,
                                             "dataset": None, "model": None})
        self.store.clear_default("alpha", "tokenizer", "tok-x")
        # Re-open from disk
        reopened = ProjectStore(Path(self._tmp.name)).open("alpha")
        self.assertIsNone(reopened.manifest["defaults"]["tokenizer"])


if __name__ == "__main__":
    unittest.main()
