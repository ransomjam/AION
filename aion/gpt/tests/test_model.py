import unittest
import numpy as np
from aion.gpt.config import GPTConfig
from aion.gpt.model import GPTModel, LanguageModelHead
from aion.nn.parameter import Parameter


class TestLanguageModelHead(unittest.TestCase):

    def test_output_shape(self):
        head = LanguageModelHead(d_model=16, vocab_size=32)
        from aion.nn.tensor import Tensor
        x = Tensor(np.random.randn(2, 5, 16), requires_grad=True)
        out = head(x)
        self.assertEqual(out.data.shape, (2, 5, 32))

    def test_weight_tying(self):
        shared = Parameter(np.random.randn(32, 16), name="table")
        head = LanguageModelHead(d_model=16, vocab_size=32, weight=shared)
        self.assertIs(head.W, shared)

    def test_no_bias_parameter(self):
        head = LanguageModelHead(d_model=16, vocab_size=32)
        param_names = [p.name for p in head.parameters()]
        self.assertNotIn("b", param_names)


class TestGPTModel(unittest.TestCase):

    def _small_cfg(self, tie=True):
        return GPTConfig(
            vocab_size=32, d_model=16, n_heads=2, n_layers=2,
            d_ff=32, dropout=0.0, max_seq_len=64, tie_weights=tie,
        )

    def test_forward_shapes(self):
        rng = np.random.default_rng(0)
        model = GPTModel(self._small_cfg(), rng=rng)
        token_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
        logits, attn_weights = model(token_ids)
        self.assertEqual(logits.data.shape, (2, 5, 32))
        self.assertEqual(len(attn_weights), 2)  # n_layers

    def test_weight_tying_shares_parameter(self):
        rng = np.random.default_rng(1)
        model = GPTModel(self._small_cfg(tie=True), rng=rng)
        self.assertIs(model.head.W, model.embedding.table)

    def test_no_weight_tying(self):
        rng = np.random.default_rng(2)
        model = GPTModel(self._small_cfg(tie=False), rng=rng)
        self.assertIsNot(model.head.W, model.embedding.table)

    def test_tied_model_has_fewer_params(self):
        rng1 = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        tied = GPTModel(self._small_cfg(tie=True), rng=rng1)
        untied = GPTModel(self._small_cfg(tie=False), rng=rng2)
        # Tied model shares embedding/head weight → fewer unique parameters
        self.assertLess(tied.param_count(), untied.param_count())

    def test_causal_mask_applied(self):
        # Verify that future tokens don't influence past positions by checking
        # that changing a future token doesn't change the logit at position 0.
        rng = np.random.default_rng(4)
        model = GPTModel(self._small_cfg(), rng=rng)
        ids1 = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
        ids2 = np.array([[1, 2, 3, 4, 9]], dtype=np.int32)  # last token differs
        logits1, _ = model(ids1)
        logits2, _ = model(ids2)
        # Position 0 logits must be identical (causal mask blocks future)
        np.testing.assert_allclose(
            logits1.data[0, 0], logits2.data[0, 0], atol=1e-6
        )

    def test_repr(self):
        model = GPTModel(self._small_cfg())
        r = repr(model)
        self.assertIn("GPTModel", r)
        self.assertIn("vocab=32", r)


if __name__ == "__main__":
    unittest.main()
