"""Tests for LayerNorm and Dropout modules."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.transformer.norm import LayerNorm
from aion.transformer.dropout import Dropout


class TestLayerNormModule(unittest.TestCase):

    def test_output_shape(self):
        ln = LayerNorm(16)
        x = Tensor(np.random.default_rng(0).standard_normal((2, 5, 16)))
        out = ln(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_has_two_parameters(self):
        ln = LayerNorm(8)
        params = ln.parameters()
        self.assertEqual(len(params), 2)

    def test_gamma_ones_beta_zeros_init(self):
        ln = LayerNorm(8)
        np.testing.assert_array_equal(ln.gamma.data, np.ones(8))
        np.testing.assert_array_equal(ln.beta.data, np.zeros(8))

    def test_normalized_output(self):
        ln = LayerNorm(16)
        x = Tensor(np.random.default_rng(1).standard_normal((3, 4, 16)))
        out = ln(x)
        np.testing.assert_allclose(out.data.mean(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.data.std(axis=-1), 1.0, atol=1e-4)

    def test_grad_flows_to_gamma_beta(self):
        ln = LayerNorm(8)
        x = Tensor(np.random.default_rng(2).standard_normal((2, 3, 8)))
        t_sum(ln(x)).backward()
        self.assertIsNotNone(ln.gamma.grad)
        self.assertIsNotNone(ln.beta.grad)

    def test_repr(self):
        self.assertIn("16", repr(LayerNorm(16)))


class TestDropout(unittest.TestCase):

    def test_identity_at_eval(self):
        drop = Dropout(p=0.5)
        drop.eval()
        x = Tensor(np.ones((4, 4)))
        out = drop(x)
        np.testing.assert_array_equal(out.data, x.data)

    def test_identity_when_p_zero(self):
        drop = Dropout(p=0.0)
        x = Tensor(np.ones((4, 4)))
        out = drop(x)
        np.testing.assert_array_equal(out.data, x.data)

    def test_zeros_some_during_training(self):
        drop = Dropout(p=0.9)
        drop.train()
        x = Tensor(np.ones((100,)))
        out = drop(x)
        self.assertLess(float((out.data != 0).sum()), 50)

    def test_invalid_p_raises(self):
        with self.assertRaises(ValueError):
            Dropout(p=1.0)
        with self.assertRaises(ValueError):
            Dropout(p=-0.1)

    def test_no_parameters(self):
        self.assertEqual(len(Dropout(0.1).parameters()), 0)

    def test_train_eval_propagates_to_children(self):
        """train()/eval() must propagate through _modules."""
        from aion.transformer.ffn import FeedForward
        ffn = FeedForward(8, 32, dropout=0.5)
        ffn.eval()
        self.assertFalse(ffn.drop.training)
        ffn.train()
        self.assertTrue(ffn.drop.training)

    def test_repr(self):
        self.assertIn("0.1", repr(Dropout(0.1)))


if __name__ == "__main__":
    unittest.main()
