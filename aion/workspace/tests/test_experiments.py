"""Tests for ExperimentStore."""

import tempfile
import unittest
from pathlib import Path

from aion.workspace.experiments import ExperimentStore


class TestExperimentStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ExperimentStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_and_get(self):
        rec = self.store.record(
            experiment_type="train_tokenizer",
            dataset_id="ds1",
            dataset_fingerprint="fp1",
            params={"vocab_size": 500},
            artifact_id="tok-abc",
            metrics={"compression_ratio": 1.5, "training_time_s": 0.1},
        )
        self.assertIn("id", rec)
        self.assertEqual(rec["type"], "train_tokenizer")
        fetched = self.store.get(rec["id"])
        self.assertEqual(fetched["artifact_id"], "tok-abc")

    def test_list_returns_all(self):
        for i in range(3):
            self.store.record(
                experiment_type="train_tokenizer",
                dataset_id="ds1", dataset_fingerprint="fp",
                params={}, artifact_id=f"tok-{i}", metrics={},
            )
        self.assertEqual(len(self.store.list()), 3)

    def test_list_sorted_newest_first(self):
        ids = []
        for i in range(3):
            r = self.store.record(
                experiment_type="train_tokenizer",
                dataset_id="ds1", dataset_fingerprint="fp",
                params={}, artifact_id=f"tok-{i}", metrics={},
            )
            ids.append(r["id"])
        listed = self.store.list()
        listed_ids = [r["id"] for r in listed]
        # All records present; sorted descending by created_at (ties ok at 1s resolution)
        self.assertEqual(sorted(listed_ids), sorted(ids))
        # Verify the sort key is created_at descending
        dates = [r["created_at"] for r in listed]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.store.get("nonexistent"))

    def test_produced_by_present(self):
        rec = self.store.record(
            experiment_type="train_tokenizer",
            dataset_id="ds1", dataset_fingerprint="fp",
            params={}, artifact_id="tok-x", metrics={},
        )
        self.assertIn("produced_by", rec)
        self.assertIn("aion_version", rec["produced_by"])


if __name__ == "__main__":
    unittest.main()
