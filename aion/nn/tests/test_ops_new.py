"""Tests for softmax and transpose ops added in Milestone 6."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor, get_default_dtype, set_default_dtype
from aion.nn.ops import softmax, transpose

# Finite-difference gradient checks need float64 precision; opt in for this
# module and restore the framework default (float32) afterwards.
_SAVED_DTYPE = None


def setUpModule():
    global _SAVED_DTYPE
    _SAVED_DTYPE = get_default_dtype()
    set_default_dtype(np.float64)


def tearDownModule():
    set_default_dtype(_SAVED_DTYPE)


def numerical_grad(f, x_data, eps=1e-5):
    """Finite-difference gradient of scalar f w.r.t. x_data."""
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


class TestSoftmax(unittest.TestCase):

    def test_forward_sums_to_one(self):
        a = Tensor(np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]]))
        out = softmax(a, axis=-1)
        np.testing.assert_allclose(out.data.sum(axis=-1), [1.0, 1.0], atol=1e-12)

    def test_forward_values(self):
        a = Tensor(np.array([[0.0, 0.0, 0.0]]))
        out = softmax(a, axis=-1)
        np.testing.assert_allclose(out.data, [[1/3, 1/3, 1/3]], atol=1e-12)

    def test_numerically_stable_large_inputs(self):
        a = Tensor(np.array([[1000.0, 1001.0, 1002.0]]))
        out = softmax(a, axis=-1)
        self.assertTrue(np.all(np.isfinite(out.data)))
        np.testing.assert_allclose(out.data.sum(axis=-1), [1.0], atol=1e-12)

    def test_requires_grad_propagated(self):
        a = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        out = softmax(a, axis=-1)
        self.assertTrue(out.requires_grad)

    def test_no_grad_when_not_required(self):
        a = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=False)
        out = softmax(a, axis=-1)
        self.assertFalse(out.requires_grad)

    def test_backward_numerical(self):
        rng = np.random.default_rng(0)
        x_data = rng.standard_normal((2, 4))

        def f(xd):
            t = Tensor(xd.copy(), requires_grad=True)
            out = softmax(t, axis=-1)
            s = out.data.sum()
            return float(s)

        num = numerical_grad(f, x_data.copy())

        x = Tensor(x_data.copy(), requires_grad=True)
        out = softmax(x, axis=-1)
        # sum all outputs → scalar backward
        from aion.nn.ops import t_sum
        loss = t_sum(out)
        loss.backward()
        np.testing.assert_allclose(x.grad, num, atol=1e-6)

    def test_backward_jacobian(self):
        """Gradient of a single output w.r.t. all inputs."""
        x_data = np.array([[1.0, 2.0, 3.0]])

        def f(xd):
            t = Tensor(xd.copy(), requires_grad=True)
            out = softmax(t, axis=-1)
            return float(out.data[0, 1])   # second output

        num = numerical_grad(f, x_data.copy())

        x = Tensor(x_data.copy(), requires_grad=True)
        out = softmax(x, axis=-1)
        # Seed grad to pick out the second output.
        out.grad = np.array([[0.0, 1.0, 0.0]])
        out._backward()
        np.testing.assert_allclose(x.grad, num, atol=1e-6)


class TestTranspose(unittest.TestCase):

    def test_forward_2d(self):
        a = Tensor(np.arange(6.0).reshape(2, 3))
        out = transpose(a, (1, 0))
        self.assertEqual(out.shape, (3, 2))
        np.testing.assert_array_equal(out.data, a.data.T)

    def test_forward_3d(self):
        a = Tensor(np.arange(24.0).reshape(2, 3, 4))
        out = transpose(a, (0, 2, 1))
        self.assertEqual(out.shape, (2, 4, 3))

    def test_requires_grad_propagated(self):
        a = Tensor(np.ones((2, 3)), requires_grad=True)
        out = transpose(a, (1, 0))
        self.assertTrue(out.requires_grad)

    def test_no_grad_when_not_required(self):
        a = Tensor(np.ones((2, 3)), requires_grad=False)
        out = transpose(a, (1, 0))
        self.assertFalse(out.requires_grad)

    def test_backward_2d(self):
        a = Tensor(np.arange(6.0).reshape(2, 3), requires_grad=True)
        out = transpose(a, (1, 0))
        out.grad = np.ones((3, 2))
        out._backward()
        np.testing.assert_array_equal(a.grad, np.ones((2, 3)))

    def test_backward_numerical(self):
        rng = np.random.default_rng(1)
        x_data = rng.standard_normal((2, 3, 4))

        def f(xd):
            t = Tensor(xd.copy(), requires_grad=True)
            out = transpose(t, (0, 2, 1))
            return float(out.data.sum())

        num = numerical_grad(f, x_data.copy())

        x = Tensor(x_data.copy(), requires_grad=True)
        out = transpose(x, (0, 2, 1))
        from aion.nn.ops import t_sum
        loss = t_sum(out)
        loss.backward()
        np.testing.assert_allclose(x.grad, num, atol=1e-6)

    def test_double_transpose_identity(self):
        rng = np.random.default_rng(2)
        x_data = rng.standard_normal((3, 4))
        a = Tensor(x_data, requires_grad=True)
        out = transpose(transpose(a, (1, 0)), (1, 0))
        np.testing.assert_array_equal(out.data, x_data)


if __name__ == "__main__":
    unittest.main()
