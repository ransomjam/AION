import unittest
import tempfile
from pathlib import Path
import numpy as np
from aion.gpt.config import GPTConfig
from aion.gpt.model import GPTModel
from aion.gpt.store import GPTNotFound, GPTStore
from aion.gpt.trainer import GPTTrainingResult


def _small_model(seed=0):
    cfg = GPTConfig(
        vocab_size=16, d_model=8, n_heads=2, n_layers=1,
        d_ff=16, dropout=0.0, max_seq_len=8, tie_weights=True,
    )
    return GPTModel(cfg, rng=np.random.default_rng(seed))


def _fake_result():
    return GPTTrainingResult(metrics={
        "loss_history": [3.0, 2.5],
        "val_loss_history": [],
        "val_perplexity_history": [],
        "grad_norm_history": [1.0, 0.9],
        "tokens_per_sec": [500.0, 510.0],
        "final_loss": 2.5,
        "final_val_loss": None,
        "final_val_perplexity": None,
        "epochs": 2,
        "param_count": 100,
        "training_time_s": 1.2,
        "seed": 0,
        "tokens_processed": 1000,
    })


class TestGPTStore(unittest.TestCase):

    def test_save_returns_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            manifest = store.save(model, _fake_result(), name="test-gpt")
            self.assertEqual(manifest["architecture"], "gpt-v1")
            self.assertIn("id", manifest)

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            manifest = store.save(model, _fake_result(), name="test-gpt")
            self.assertTrue(store.exists(manifest["id"]))
            self.assertFalse(store.exists("nonexistent"))

    def test_open_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            manifest = store.save(model, _fake_result(), name="my-gpt")
            loaded = store.open_manifest(manifest["id"])
            self.assertEqual(loaded["name"], "my-gpt")

    def test_open_manifest_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            with self.assertRaises(GPTNotFound):
                store.open_manifest("does-not-exist")

    def test_load_restores_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            original = [p.data.copy() for p in model.parameters()]
            manifest = store.save(model, _fake_result(), name="test-gpt")

            loaded = store.load(manifest["id"])
            for orig, p in zip(original, loaded.parameters()):
                np.testing.assert_allclose(p.data, orig, atol=1e-10)

    def test_load_weight_tying_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            manifest = store.save(model, _fake_result(), name="tied-gpt")
            loaded = store.load(manifest["id"])
            # After load, head.W and embedding.table should be the same object
            self.assertIs(loaded.head.W, loaded.embedding.table)

    def test_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            manifest = store.save(model, _fake_result(), name="test-gpt")
            stats = store.statistics(manifest["id"])
            self.assertIn("loss_history", stats)

    def test_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            for i in range(3):
                store.save(_small_model(i), _fake_result(), name=f"gpt-{i}")
            listing = store.list()
            self.assertEqual(len(listing), 3)
            self.assertTrue(all(m["architecture"] == "gpt-v1" for m in listing))

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            manifest = store.save(model, _fake_result(), name="test-gpt")
            store.delete(manifest["id"])
            self.assertFalse(store.exists(manifest["id"]))

    def test_set_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GPTStore(Path(tmp))
            model = _small_model()
            manifest = store.save(model, _fake_result(), name="test-gpt")
            updated = store.set_status(manifest["id"], "production")
            self.assertEqual(updated["status"], "production")


if __name__ == "__main__":
    unittest.main()
