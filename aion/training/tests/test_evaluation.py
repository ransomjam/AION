import unittest
import numpy as np
from aion.gpt.config import GPTConfig
from aion.gpt.data import BatchSampler, TokenizedDataset
from aion.gpt.model import GPTModel
from aion.training.evaluation import EvaluationResult, EvaluationRunner


def _small_model(seed=0):
    cfg = GPTConfig(
        vocab_size=32, d_model=16, n_heads=2, n_layers=1,
        d_ff=32, dropout=0.0, max_seq_len=16, tie_weights=True,
    )
    return GPTModel(cfg, rng=np.random.default_rng(seed))


def _sampler(n_tokens=200, block_size=7, batch_size=4):
    tokens = np.arange(n_tokens, dtype=np.int32) % 32
    ds = TokenizedDataset(tokens, block_size)
    return BatchSampler(ds, batch_size=batch_size, shuffle=False)


class TestEvaluationRunner(unittest.TestCase):

    def test_returns_result(self):
        model = _small_model()
        runner = EvaluationRunner(model)
        result = runner.evaluate(_sampler())
        self.assertIsInstance(result, EvaluationResult)

    def test_val_loss_positive(self):
        model = _small_model()
        runner = EvaluationRunner(model)
        result = runner.evaluate(_sampler())
        self.assertGreater(result.val_loss, 0.0)

    def test_perplexity_greater_than_one(self):
        model = _small_model()
        runner = EvaluationRunner(model)
        result = runner.evaluate(_sampler())
        self.assertGreater(result.perplexity, 1.0)

    def test_n_batches_positive(self):
        model = _small_model()
        runner = EvaluationRunner(model)
        result = runner.evaluate(_sampler())
        self.assertGreater(result.n_batches, 0)

    def test_max_batches_respected(self):
        model = _small_model()
        runner = EvaluationRunner(model, max_batches=1)
        result = runner.evaluate(_sampler())
        self.assertEqual(result.n_batches, 1)

    def test_elapsed_s_non_negative(self):
        model = _small_model()
        runner = EvaluationRunner(model)
        result = runner.evaluate(_sampler())
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_callable_sampler(self):
        model = _small_model()
        runner = EvaluationRunner(model)
        result = runner.evaluate(lambda: _sampler())
        self.assertIsInstance(result, EvaluationResult)

    def test_model_returns_to_train_mode(self):
        model = _small_model()
        model.train()
        runner = EvaluationRunner(model)
        runner.evaluate(_sampler())
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
