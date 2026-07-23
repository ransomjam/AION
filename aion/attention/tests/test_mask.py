"""Tests for attention mask abstractions."""

import unittest
import numpy as np

from aion.attention.mask import CausalMask, PaddingMask, _NEG_INF


class TestCausalMask(unittest.TestCase):

    def test_shape(self):
        m = CausalMask()
        b = m.bias(4, 4)
        self.assertEqual(b.shape, (1, 1, 4, 4))

    def test_lower_triangle_zero(self):
        m = CausalMask()
        b = m.bias(4, 4)[0, 0]
        for i in range(4):
            for j in range(i + 1):
                self.assertEqual(b[i, j], 0.0, f"b[{i},{j}] should be 0")

    def test_upper_triangle_neg_inf(self):
        m = CausalMask()
        b = m.bias(4, 4)[0, 0]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertEqual(b[i, j], _NEG_INF, f"b[{i},{j}] should be NEG_INF")

    def test_non_square(self):
        m = CausalMask()
        b = m.bias(3, 5)
        self.assertEqual(b.shape, (1, 1, 3, 5))

    def test_batch_size_ignored(self):
        """CausalMask shape is independent of batch_size."""
        m = CausalMask()
        b1 = m.bias(4, 4, batch_size=1)
        b2 = m.bias(4, 4, batch_size=8)
        np.testing.assert_array_equal(b1, b2)

    def test_broadcasts_over_scores(self):
        """Bias must broadcast over [batch, n_heads, seq_q, seq_k]."""
        m = CausalMask()
        b = m.bias(4, 4)
        scores = np.zeros((2, 8, 4, 4))
        result = scores + b   # should not raise
        self.assertEqual(result.shape, (2, 8, 4, 4))


class TestPaddingMask(unittest.TestCase):

    def test_shape(self):
        valid = np.array([[True, True, False, False]])
        m = PaddingMask(valid)
        b = m.bias(4, 4, batch_size=1)
        self.assertEqual(b.shape, (1, 1, 1, 4))

    def test_valid_positions_zero(self):
        valid = np.array([[True, True, False]])
        m = PaddingMask(valid)
        b = m.bias(3, 3)[0, 0, 0]
        self.assertEqual(b[0], 0.0)
        self.assertEqual(b[1], 0.0)

    def test_padding_positions_neg_inf(self):
        valid = np.array([[True, True, False]])
        m = PaddingMask(valid)
        b = m.bias(3, 3)[0, 0, 0]
        self.assertEqual(b[2], _NEG_INF)

    def test_batch_varies(self):
        valid = np.array([
            [True, True, False],
            [True, False, False],
        ])
        m = PaddingMask(valid)
        b = m.bias(3, 3)
        self.assertEqual(b.shape, (2, 1, 1, 3))
        self.assertEqual(b[0, 0, 0, 2], _NEG_INF)
        self.assertEqual(b[1, 0, 0, 1], _NEG_INF)
        self.assertEqual(b[1, 0, 0, 0], 0.0)

    def test_broadcasts_over_scores(self):
        valid = np.array([[True, True, False, False],
                          [True, False, False, False]])
        m = PaddingMask(valid)
        b = m.bias(4, 4, batch_size=2)
        scores = np.zeros((2, 8, 4, 4))
        result = scores + b
        self.assertEqual(result.shape, (2, 8, 4, 4))


if __name__ == "__main__":
    unittest.main()
