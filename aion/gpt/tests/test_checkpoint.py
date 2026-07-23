import unittest
import tempfile
from pathlib import Path
import numpy as np
from aion.gpt.checkpoint import GPTCheckpoint
from aion.gpt.config import GPTConfig
from aion.gpt.model import GPTModel


def _small_model(seed=0):
    cfg = GPTConfig(
        vocab_size=16, d_model=8, n_heads=2, n_layers=1,
        d_ff=16, dropout=0.0, max_seq_len=8,
    )
    return GPTModel(cfg, rng=np.random.default_rng(seed))


class TestGPTCheckpoint(unittest.TestCase):

    def test_save_and_load_restores_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _small_model()
            ckpt = GPTCheckpoint(Path(tmp), "run1")
            original = [p.data.copy() for p in model.parameters()]
            ckpt.save(model, epoch=1, metrics={"loss": 2.5})

            # Corrupt weights
            for p in model.parameters():
                p.data[:] = 0.0

            ckpt.load(model, epoch=1)
            for orig, p in zip(original, model.parameters()):
                np.testing.assert_allclose(p.data, orig, atol=1e-10)

    def test_load_returns_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _small_model()
            ckpt = GPTCheckpoint(Path(tmp), "run1")
            ckpt.save(model, epoch=2, metrics={"loss": 1.8, "val_loss": 2.0})
            meta = ckpt.load(model, epoch=2)
            self.assertEqual(meta["epoch"], 2)
            self.assertAlmostEqual(meta["metrics"]["loss"], 1.8)

    def test_latest_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _small_model()
            ckpt = GPTCheckpoint(Path(tmp), "run1")
            self.assertIsNone(ckpt.latest_epoch())
            ckpt.save(model, epoch=1, metrics={})
            ckpt.save(model, epoch=3, metrics={})
            self.assertEqual(ckpt.latest_epoch(), 3)

    def test_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _small_model()
            ckpt = GPTCheckpoint(Path(tmp), "run1")
            ckpt.save(model, epoch=1, metrics={"loss": 3.0})
            ckpt.save(model, epoch=2, metrics={"loss": 2.5})
            listing = ckpt.list()
            self.assertEqual(len(listing), 2)
            self.assertEqual(listing[0]["epoch"], 1)
            self.assertEqual(listing[1]["epoch"], 2)

    def test_load_missing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _small_model()
            ckpt = GPTCheckpoint(Path(tmp), "run1")
            with self.assertRaises(FileNotFoundError):
                ckpt.load(model, epoch=99)


if __name__ == "__main__":
    unittest.main()
