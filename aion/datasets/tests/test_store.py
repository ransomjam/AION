"""Tests for dataset storage: document CRUD, stable ids, import, fingerprint."""

import tempfile
import unittest
from pathlib import Path

from aion.datasets.store import DatasetExists, DatasetNotFound, DatasetStore


class DatasetStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = DatasetStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_and_add_documents(self):
        ds = self.store.create("Corpus One", source="manual")
        self.assertEqual(ds.id, "corpus-one")
        a = ds.add_document("hello world")
        b = ds.add_document("second doc")
        self.assertEqual([a, b], [1, 2])
        self.assertEqual(ds.meta["document_count"], 2)
        self.assertEqual(ds.get_document(a), "hello world")

    def test_duplicate_dataset_rejected(self):
        self.store.create("Corpus")
        with self.assertRaises(DatasetExists):
            self.store.create("corpus")

    def test_ids_are_stable_and_not_reused_after_delete(self):
        ds = self.store.create("C")
        d1 = ds.add_document("one")
        d2 = ds.add_document("two")
        ds.delete_document(d1)
        d3 = ds.add_document("three")
        self.assertEqual(d3, 3)                 # next_id advanced, gap left at 1
        self.assertEqual(ds.document_ids(), [d2, d3])
        with self.assertRaises(DatasetNotFound):
            ds.get_document(d1)

    def test_edit_document(self):
        ds = self.store.create("C")
        d = ds.add_document("before")
        ds.edit_document(d, "after")
        self.assertEqual(ds.get_document(d), "after")

    def test_stream_is_in_id_order(self):
        ds = self.store.create("C")
        ds.add_documents(["a", "b", "c"])
        self.assertEqual([t for _i, t in ds.stream()], ["a", "b", "c"])

    def test_fingerprint_changes_on_edit(self):
        ds = self.store.create("C")
        d = ds.add_document("hello")
        fp1 = ds.fingerprint()
        ds.edit_document(d, "hello world longer")
        self.assertNotEqual(fp1, ds.fingerprint())

    def test_import_file_and_folder(self):
        src = Path(self._tmp.name) / "src"
        src.mkdir()
        (src / "a.txt").write_text("alpha", encoding="utf-8")
        (src / "b.txt").write_text("beta", encoding="utf-8")
        ds = self.store.create("C")
        ids = ds.import_folder(src)
        self.assertEqual(len(ids), 2)
        self.assertEqual(sorted(t for _i, t in ds.stream()), ["alpha", "beta"])

    def test_reopen_and_delete_dataset(self):
        ds = self.store.create("C")
        ds.add_document("x")
        reopened = self.store.open("c")
        self.assertEqual(reopened.meta["document_count"], 1)
        self.store.delete("c")
        self.assertFalse(self.store.exists("c"))


if __name__ == "__main__":
    unittest.main()
