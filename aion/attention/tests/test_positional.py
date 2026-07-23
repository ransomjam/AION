"""Tests for positional encoding modules."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.attention.positional import SinusoidalPE, LearnedPE, RotaryPE


class TestSinusoidalPE(unittest.TestCase):

    def test_output_shape(self):
        pe = SinusoidalPE(d_model=16)
        x = Tensor(np.zeros((2, 5, 16)), requires_grad=True)
        out = pe(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_adds_encoding(self):
        pe = SinusoidalPE(d_model=8)
        x = Tensor(np.zeros((1, 4, 8)))
        out = pe(x)
        # Output should equal the PE table slice (since x is zeros).
        np.testing.assert_allclose(out.data[0], pe._table[:4], atol=1e-12)

    def test_no_parameters(self):
        pe = SinusoidalPE(d_model=16)
        self.assertEqual(pe.parameters(), [])

    def test_sin_cos_pattern(self):
        """Even dims use sin, odd dims use cos."""
        pe = SinusoidalPE(d_model=4)
        table = pe._table
        # Position 0: sin(0)=0, cos(0)=1 alternating.
        self.assertAlmostEqual(table[0, 0], 0.0, places=10)  # sin
        self.assertAlmostEqual(table[0, 1], 1.0, places=10)  # cos

    def test_different_positions_differ(self):
        pe = SinusoidalPE(d_model=16)
        self.assertFalse(np.allclose(pe._table[0], pe._table[1]))

    def test_grad_flows_through_x(self):
        """Gradient flows through x (PE is a constant addition)."""
        pe = SinusoidalPE(d_model=4)
        x = Tensor(np.ones((1, 3, 4)), requires_grad=True)
        out = pe(x)
        from aion.nn.ops import t_sum
        loss = t_sum(out)
        loss.backward()
        np.testing.assert_allclose(x.grad, np.ones((1, 3, 4)), atol=1e-12)

    def test_repr(self):
        pe = SinusoidalPE(d_model=32, max_len=512)
        self.assertIn("32", repr(pe))
        self.assertIn("512", repr(pe))


class TestLearnedPE(unittest.TestCase):

    def test_output_shape(self):
        pe = LearnedPE(d_model=16)
        x = Tensor(np.zeros((2, 5, 16)), requires_grad=True)
        out = pe(x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_has_one_parameter(self):
        pe = LearnedPE(d_model=16)
        params = pe.parameters()
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0].shape, (4096, 16))

    def test_initialised_from_sinusoidal(self):
        """LearnedPE starts from sinusoidal values."""
        from aion.attention.positional import _sinusoidal_table
        pe = LearnedPE(d_model=8, max_len=16)
        expected = _sinusoidal_table(16, 8)
        np.testing.assert_allclose(pe.pe.data, expected, atol=1e-12)

    def test_grad_flows_through_pe(self):
        """Gradient flows through the learned PE parameter."""
        pe = LearnedPE(d_model=4, max_len=8)
        x = Tensor(np.zeros((1, 3, 4)))
        out = pe(x)
        from aion.nn.ops import t_sum
        loss = t_sum(out)
        loss.backward()
        self.assertIsNotNone(pe.pe.grad)

    def test_repr(self):
        pe = LearnedPE(d_model=64, max_len=256)
        self.assertIn("64", repr(pe))
        self.assertIn("256", repr(pe))


class TestRotaryPE(unittest.TestCase):

    def test_forward_raises(self):
        rpe = RotaryPE(d_head=16)
        x = Tensor(np.zeros((1, 4, 16)))
        with self.assertRaises(NotImplementedError):
            rpe(x)

    def test_no_parameters(self):
        rpe = RotaryPE(d_head=16)
        self.assertEqual(rpe.parameters(), [])

    def test_repr(self):
        rpe = RotaryPE(d_head=32, max_len=512)
        self.assertIn("32", repr(rpe))


if __name__ == "__main__":
    unittest.main()
