"""Tests for ProjectStore, Project layout, and the manifest."""

import json
import tempfile
import unittest
from pathlib import Path

from aion.workspace import manifest as manifest_mod
from aion.workspace.project import SUBDIRS
from aion.workspace.store import ProjectExists, ProjectNotFound, ProjectStore


class ProjectStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ProjectStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_makes_manifest_and_all_subdirs(self):
        p = self.store.create("TinyGPT", description="tiny language model")
        self.assertEqual(p.id, "tinygpt")
        self.assertTrue(p.manifest_path().is_file())
        for sub in SUBDIRS:
            self.assertTrue((p.root / sub).is_dir(), f"missing subdir {sub}")
        self.assertEqual(p.manifest["description"], "tiny language model")
        self.assertEqual(p.status, "active")

    def test_duplicate_id_rejected(self):
        self.store.create("TinyGPT")
        with self.assertRaises(ProjectExists):
            self.store.create("TINYGPT")  # same slug: tinygpt

    def test_open_unknown_raises(self):
        with self.assertRaises(ProjectNotFound):
            self.store.open("nope")

    def test_list_orders_by_updated_desc(self):
        self.store.create("Alpha")
        self.store.create("Beta")
        self.store.update("alpha", description="touched last")
        ids = [p.id for p in self.store.list()]
        self.assertEqual(ids[0], "alpha")
        self.assertIn("beta", ids)

    def test_update_rejects_unknown_field(self):
        self.store.create("Alpha")
        with self.assertRaises(ValueError):
            self.store.update("alpha", not_a_field=1)

    def test_archive_sets_status(self):
        self.store.create("Alpha")
        self.assertEqual(self.store.archive("alpha").status, "archived")

    def test_reopen_reads_persisted_manifest(self):
        self.store.create("Alpha", language="en")
        reopened = ProjectStore(Path(self._tmp.name)).open("alpha")
        self.assertEqual(reopened.manifest["language"], "en")


class ManifestTest(unittest.TestCase):
    def test_normalize_fills_missing_known_fields(self):
        m = manifest_mod.normalize({"id": "x", "name": "X"})
        self.assertEqual(m["status"], "active")
        self.assertIn("defaults", m)
        self.assertEqual(m["defaults"]["tokenizer"], None)

    def test_normalize_preserves_unknown_fields(self):
        # Forward compatibility: newer sections written by future code survive.
        m = manifest_mod.normalize({"id": "x", "future_section": {"k": 1}})
        self.assertEqual(m["future_section"], {"k": 1})

    def test_new_manifest_has_timestamps(self):
        m = manifest_mod.new_manifest("x", "X")
        self.assertTrue(m["created_at"])
        self.assertEqual(m["created_at"], m["updated_at"])


if __name__ == "__main__":
    unittest.main()
