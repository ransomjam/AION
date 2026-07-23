"""Tests for ModelStore — save, list, open_manifest, delete, set_status."""

import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from aion.nn.layers import Linear, ReLU
from aion.nn.loss import MSELoss
from aion.nn.optim import SGD
from aion.nn.sequential import Sequential
from aion.nn.store import ModelNotFound, ModelStore
from aion.nn.tensor import Tensor
from aion.nn.trainer import Trainer, TrainingResult


def _train_small():
    rng = np.random.default_rng(0)
    model = Sequential(Linear(2, 4, rng=rng), ReLU(), Linear(4, 1, rng=rng))
    X = np.ones((4, 2))
    y = np.ones((4, 1))
    def batches():
        yield Tensor(X), y
    trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01))
    result = trainer.train(batches, epochs=2, seed=0)
    return model, result


class TestModelStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ModelStore(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_list(self):
        model, result = _train_small()
        manifest = self.store.save(model, result, name="test-model",
                                   architecture="sequential-v1")
        self.assertEqual(manifest["name"], "test-model")
        listed = self.store.list()
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["id"], manifest["id"])

    def test_open_manifest(self):
        model, result = _train_small()
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        reopened = self.store.open_manifest(m["id"])
        self.assertEqual(reopened["id"], m["id"])
        self.assertIn("param_count", reopened)

    def test_not_found_raises(self):
        with self.assertRaises(ModelNotFound):
            self.store.open_manifest("nonexistent")

    def test_weights_npz_written(self):
        model, result = _train_small()
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        weights_path = Path(self._tmp.name) / m["id"] / "model" / "weights.npz"
        self.assertTrue(weights_path.is_file())

    def test_weights_round_trip(self):
        model, result = _train_small()
        # Capture weights by name before saving.
        original = {(p.name or f"param_{i}"): p.data.copy()
                    for i, p in enumerate(model.parameters())}
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        weights = np.load(str(Path(self._tmp.name) / m["id"] / "model" / "weights.npz"))
        for key, orig in original.items():
            np.testing.assert_array_equal(weights[key], orig)

    def test_statistics_saved(self):
        model, result = _train_small()
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        stats = self.store.statistics(m["id"])
        self.assertIsNotNone(stats)
        self.assertIn("loss_history", stats)

    def test_loss_history_not_in_manifest_metrics(self):
        model, result = _train_small()
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        self.assertNotIn("loss_history", m.get("metrics", {}))

    def test_delete(self):
        model, result = _train_small()
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        self.store.delete(m["id"])
        self.assertFalse(self.store.exists(m["id"]))

    def test_set_status(self):
        model, result = _train_small()
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        updated = self.store.set_status(m["id"], "default")
        self.assertEqual(updated["status"], "default")
        reopened = self.store.open_manifest(m["id"])
        self.assertEqual(reopened["status"], "default")

    def test_model_fingerprint_present(self):
        model, result = _train_small()
        m = self.store.save(model, result, name="m1", architecture="seq-v1")
        self.assertIn("model_fingerprint", m)
        self.assertTrue(len(m["model_fingerprint"]) > 0)

    def test_list_sorted_newest_first(self):
        model, result = _train_small()
        m1 = self.store.save(model, result, name="first", architecture="v1")
        time.sleep(1.1)  # ensure distinct created_at timestamps
        m2 = self.store.save(model, result, name="second", architecture="v1")
        listed = self.store.list()
        self.assertEqual(listed[0]["id"], m2["id"])


if __name__ == "__main__":
    unittest.main()
