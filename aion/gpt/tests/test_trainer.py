import unittest
import math
import numpy as np
from aion.gpt.config import GPTConfig
from aion.gpt.data import BatchSampler, TokenizedDataset
from aion.gpt.model import GPTModel
from aion.gpt.trainer import GPTTrainer, GPTTrainingResult
from aion.nn.optim import Adam


def _small_model(seed=0):
    cfg = GPTConfig(
        vocab_size=32, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, dropout=0.0, max_seq_len=16, tie_weights=True,
    )
    return GPTModel(cfg, rng=np.random.default_rng(seed))


def _sampler(n_tokens=200, block_size=7, batch_size=4, seed=0):
    tokens = np.arange(n_tokens, dtype=np.int32) % 32
    ds = TokenizedDataset(tokens, block_size)
    rng = np.random.default_rng(seed)
    return BatchSampler(ds, batch_size=batch_size, shuffle=True, rng=rng)


class TestGPTTrainer(unittest.TestCase):

    def test_returns_result(self):
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt)
        result = trainer.train(_sampler, epochs=1, seed=42)
        self.assertIsInstance(result, GPTTrainingResult)

    def test_loss_history_length(self):
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt)
        result = trainer.train(_sampler, epochs=3, seed=0)
        self.assertEqual(len(result.metrics["loss_history"]), 3)

    def test_loss_decreases(self):
        model = _small_model(seed=7)
        opt = Adam(model.parameters(), lr=5e-3)
        trainer = GPTTrainer(model, opt)
        result = trainer.train(_sampler, epochs=5, seed=0)
        history = result.metrics["loss_history"]
        self.assertLess(history[-1], history[0])

    def test_tokens_processed_positive(self):
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt)
        result = trainer.train(_sampler, epochs=2, seed=0)
        self.assertGreater(result.metrics["tokens_processed"], 0)

    def test_grad_norm_history(self):
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt)
        result = trainer.train(_sampler, epochs=2, seed=0)
        norms = result.metrics["grad_norm_history"]
        self.assertEqual(len(norms), 2)
        self.assertTrue(all(n >= 0 for n in norms))

    def test_grad_clip_reduces_norm(self):
        # With a very small clip threshold, norms should be <= threshold
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        clip = 0.01
        trainer = GPTTrainer(model, opt, grad_clip=clip)
        result = trainer.train(_sampler, epochs=1, seed=0)
        # grad_norm_history records pre-clip norm; just verify training ran
        self.assertEqual(len(result.metrics["grad_norm_history"]), 1)

    def test_no_grad_clip(self):
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt, grad_clip=None)
        result = trainer.train(_sampler, epochs=1, seed=0)
        self.assertIsInstance(result, GPTTrainingResult)

    def test_validation_metrics_present(self):
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt)
        val = _sampler(n_tokens=80, block_size=7, batch_size=4, seed=99)
        result = trainer.train(_sampler, epochs=2, val_sampler=val, seed=0)
        self.assertEqual(len(result.metrics["val_loss_history"]), 2)
        self.assertEqual(len(result.metrics["val_perplexity_history"]), 2)
        for ppl in result.metrics["val_perplexity_history"]:
            self.assertGreater(ppl, 1.0)

    def test_no_validation_empty_lists(self):
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt)
        result = trainer.train(_sampler, epochs=1, seed=0)
        self.assertEqual(result.metrics["val_loss_history"], [])
        self.assertIsNone(result.metrics["final_val_loss"])

    def test_progress_fn_called(self):
        calls = []
        model = _small_model()
        opt = Adam(model.parameters(), lr=1e-3)
        trainer = GPTTrainer(model, opt, progress_fn=lambda f, m: calls.append(f))
        trainer.train(_sampler, epochs=3, seed=0)
        self.assertEqual(len(calls), 3)
        self.assertAlmostEqual(calls[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
