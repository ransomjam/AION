"""Tests for gelu, layer_norm, and dropout_mask ops added in Milestone 7."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.parameter import Parameter
from aion.nn.ops import gelu, layer_norm, dropout_mask, t_sum


def numerical_grad(f, x_data, eps=1e-5):
    grad = np.zeros_like(x_data)
    it = np.nditer(x_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x_data[idx]
        x_data[idx] = orig + eps
        fp = f(x_data)
        x_data[idx] = orig - eps
        fm = f(x_data)
        grad[idx] = (fp - fm) / (2 * eps)
        x_data[idx] = orig
        it.iternext()
    return grad


class TestGELU(unittest.TestCase):

    def test_forward_positive_input(self):
        """Large positive x → GELU(x) ≈ x."""
        x = Tensor(np.array([10.0]))
        out = gelu(x)
        self.assertAlmostEqual(float(out.data[0]), 10.0, places=3)

    def test_forward_negative_input(self):
        """Large negative x → GELU(x) ≈ 0."""
        x = Tensor(np.array([-10.0]))
        out = gelu(x)
        self.assertAlmostEqual(float(out.data[0]), 0.0, places=3)

    def test_forward_zero(self):
        """GELU(0) = 0."""
        x = Tensor(np.array([0.0]))
        out = gelu(x)
        self.assertAlmostEqual(float(out.data[0]), 0.0, places=10)

    def test_requires_grad_propagated(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        out = gelu(x)
        self.assertTrue(out.requires_grad)

    def test_no_grad_when_not_required(self):
        x = Tensor(np.array([1.0, 2.0]), requires_grad=False)
        out = gelu(x)
        self.assertFalse(out.requires_grad)

    def test_backward_numerical(self):
        rng = np.random.default_rng(0)
        x_data = rng.standard_normal((3, 4))

        def f(xd):
            t = Tensor(xd.copy(), requires_grad=True)
            return float(t_sum(gelu(t)).data.flat[0])

        num = numerical_grad(f, x_data.copy())
        x = Tensor(x_data.copy(), requires_grad=True)
        t_sum(gelu(x)).backward()
        np.testing.assert_allclose(x.grad, num, atol=1e-5)

    def test_output_shape(self):
        x = Tensor(np.ones((2, 3, 4)), requires_grad=True)
        out = gelu(x)
        self.assertEqual(out.shape, (2, 3, 4))


class TestLayerNorm(unittest.TestCase):

    def _make(self, shape, seed=0):
        rng = np.random.default_rng(seed)
        d = shape[-1]
        x = Tensor(rng.standard_normal(shape), requires_grad=True)
        gamma = Parameter(np.ones(d), name="gamma")
        beta = Parameter(np.zeros(d), name="beta")
        return x, gamma, beta

    def test_output_shape(self):
        x, g, b = self._make((2, 5, 8))
        out = layer_norm(x, g, b)
        self.assertEqual(out.shape, (2, 5, 8))

    def test_normalized_mean_near_zero(self):
        """With gamma=1, beta=0, output mean over last axis ≈ 0."""
        x, g, b = self._make((4, 6, 16))
        out = layer_norm(x, g, b)
        means = out.data.mean(axis=-1)
        np.testing.assert_allclose(means, 0.0, atol=1e-6)

    def test_normalized_std_near_one(self):
        """With gamma=1, beta=0, output std over last axis ≈ 1."""
        x, g, b = self._make((4, 6, 16))
        out = layer_norm(x, g, b)
        stds = out.data.std(axis=-1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-4)

    def test_beta_shifts_output(self):
        x, g, b = self._make((2, 3, 8))
        b_val = np.full(8, 5.0)
        b2 = Parameter(b_val, name="beta")
        out = layer_norm(x, g, b2)
        np.testing.assert_allclose(out.data.mean(axis=-1), 5.0, atol=1e-6)

    def test_gamma_scales_output(self):
        x, g, b = self._make((2, 3, 8))
        g2 = Parameter(np.full(8, 2.0), name="gamma")
        out = layer_norm(x, g2, b)
        stds = out.data.std(axis=-1)
        np.testing.assert_allclose(stds, 2.0, atol=1e-4)

    def test_requires_grad_propagated(self):
        x, g, b = self._make((2, 4, 8))
        out = layer_norm(x, g, b)
        self.assertTrue(out.requires_grad)

    def test_backward_x_numerical(self):
        rng = np.random.default_rng(1)
        x_data = rng.standard_normal((2, 3, 8))
        g_data = np.ones(8)
        b_data = np.zeros(8)

        def f(xd):
            t = Tensor(xd.copy(), requires_grad=True)
            g = Parameter(g_data.copy(), name="gamma")
            bv = Parameter(b_data.copy(), name="beta")
            return float(t_sum(layer_norm(t, g, bv)).data.flat[0])

        num = numerical_grad(f, x_data.copy())
        x = Tensor(x_data.copy(), requires_grad=True)
        g = Parameter(g_data.copy(), name="gamma")
        bv = Parameter(b_data.copy(), name="beta")
        t_sum(layer_norm(x, g, bv)).backward()
        np.testing.assert_allclose(x.grad, num, atol=1e-5)

    def test_backward_gamma_numerical(self):
        rng = np.random.default_rng(2)
        x_data = rng.standard_normal((2, 3, 8))
        g_data = rng.uniform(0.5, 1.5, 8)
        b_data = np.zeros(8)

        def f(gd):
            t = Tensor(x_data.copy())
            g = Parameter(gd.copy(), name="gamma")
            bv = Parameter(b_data.copy(), name="beta")
            return float(t_sum(layer_norm(t, g, bv)).data.flat[0])

        num = numerical_grad(f, g_data.copy())
        x = Tensor(x_data.copy())
        g = Parameter(g_data.copy(), name="gamma")
        bv = Parameter(b_data.copy(), name="beta")
        t_sum(layer_norm(x, g, bv)).backward()
        np.testing.assert_allclose(g.grad, num, atol=1e-5)

    def test_backward_beta_numerical(self):
        rng = np.random.default_rng(3)
        x_data = rng.standard_normal((2, 3, 8))
        g_data = np.ones(8)
        b_data = rng.standard_normal(8)

        def f(bd):
            t = Tensor(x_data.copy())
            g = Parameter(g_data.copy(), name="gamma")
            bv = Parameter(bd.copy(), name="beta")
            return float(t_sum(layer_norm(t, g, bv)).data.flat[0])

        num = numerical_grad(f, b_data.copy())
        x = Tensor(x_data.copy())
        g = Parameter(g_data.copy(), name="gamma")
        bv = Parameter(b_data.copy(), name="beta")
        t_sum(layer_norm(x, g, bv)).backward()
        np.testing.assert_allclose(bv.grad, num, atol=1e-5)


class TestDropoutMask(unittest.TestCase):

    def test_keeps_all_when_mask_all_true(self):
        x = Tensor(np.ones((3, 4)), requires_grad=True)
        mask = np.ones((3, 4), dtype=bool)
        out = dropout_mask(x, mask, keep_prob=0.5)
        np.testing.assert_allclose(out.data, np.full((3, 4), 2.0))

    def test_zeros_masked_positions(self):
        x = Tensor(np.ones((4,)), requires_grad=True)
        mask = np.array([True, False, True, False])
        out = dropout_mask(x, mask, keep_prob=0.5)
        np.testing.assert_allclose(out.data, [2.0, 0.0, 2.0, 0.0])

    def test_backward_passes_through_kept(self):
        x = Tensor(np.ones((4,)), requires_grad=True)
        mask = np.array([True, False, True, False])
        out = dropout_mask(x, mask, keep_prob=0.5)
        t_sum(out).backward()
        np.testing.assert_allclose(x.grad, [2.0, 0.0, 2.0, 0.0])

    def test_requires_grad_propagated(self):
        x = Tensor(np.ones((3,)), requires_grad=True)
        mask = np.ones(3, dtype=bool)
        out = dropout_mask(x, mask, keep_prob=1.0)
        self.assertTrue(out.requires_grad)


if __name__ == "__main__":
    unittest.main()
