"""Tests for FeedForward module."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.transformer.ffn import FeedForward


class TestFeedForward(unittest.TestCase):

    def _x(self, batch=2, seq=5, d=16, seed=0):
        return Tensor(np.random.default_rng(seed).standard_normal((batch, seq, d)),
                      requires_grad=True)

    def test_output_shape_relu(self):
        ffn = FeedForward(16, 64, activation="relu")
        out = ffn(self._x())
        self.assertEqual(out.shape, (2, 5, 16))

    def test_output_shape_gelu(self):
        ffn = FeedForward(16, 64, activation="gelu")
        out = ffn(self._x())
        self.assertEqual(out.shape, (2, 5, 16))

    def test_four_parameters(self):
        ffn = FeedForward(16, 64)
        self.assertEqual(len(ffn.parameters()), 4)  # W1, b1, W2, b2

    def test_backward_runs(self):
        ffn = FeedForward(8, 32)
        x = self._x(d=8)
        t_sum(ffn(x)).backward()
        for p in ffn.parameters():
            self.assertIsNotNone(p.grad)

    def test_invalid_activation_raises(self):
        with self.assertRaises(ValueError):
            FeedForward(8, 32, activation="swish")

    def test_dropout_identity_at_eval(self):
        ffn = FeedForward(8, 32, dropout=0.9)
        ffn.eval()
        x = self._x(d=8)
        out1 = ffn(x)
        out2 = ffn(x)
        np.testing.assert_array_equal(out1.data, out2.data)

    def test_repr(self):
        r = repr(FeedForward(16, 64, activation="gelu"))
        self.assertIn("gelu", r)
        self.assertIn("16", r)


if __name__ == "__main__":
    unittest.main()
