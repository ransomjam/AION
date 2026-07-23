"""Tests for differentiable operations — gradient correctness via finite differences.

Each test verifies that the analytical gradient produced by the backward pass
matches the numerical gradient computed by finite differences.
"""

import unittest
import numpy as np
from aion.nn.tensor import Tensor
from aion.nn.ops import (
    add, mul, matmul, relu, tanh, sigmoid, log, exp,
    sum as t_sum, mean as t_mean, reshape, embedding_lookup, log_softmax,
)


def numerical_grad(fn, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute numerical gradient of scalar fn w.r.t. x via central differences.

    ``fn`` receives a plain ndarray and must return a scalar Tensor.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = float(fn(x.copy()).data.flat[0])
        x[idx] = orig - eps
        fm = float(fn(x.copy()).data.flat[0])
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


class TestAddGrad(unittest.TestCase):
    def test_add_grad(self):
        a_data = np.array([1.0, 2.0, 3.0])
        b_data = np.array([4.0, 5.0, 6.0])
        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        t_sum(add(a, b)).backward()
        np.testing.assert_allclose(a.grad, np.ones(3))
        np.testing.assert_allclose(b.grad, np.ones(3))

    def test_add_broadcast_grad(self):
        a = Tensor(np.ones((3, 4)), requires_grad=True)
        b = Tensor(np.ones((4,)), requires_grad=True)
        t_sum(add(a, b)).backward()
        self.assertEqual(a.grad.shape, (3, 4))
        self.assertEqual(b.grad.shape, (4,))
        np.testing.assert_allclose(b.grad, np.full((4,), 3.0))


class TestMulGrad(unittest.TestCase):
    def test_mul_grad(self):
        a_data = np.array([2.0, 3.0])
        b_data = np.array([4.0, 5.0])
        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        t_sum(mul(a, b)).backward()
        np.testing.assert_allclose(a.grad, b_data)
        np.testing.assert_allclose(b.grad, a_data)


class TestMatmulGrad(unittest.TestCase):
    def _check(self, a_shape, b_shape):
        rng = np.random.default_rng(0)
        a_data = rng.standard_normal(a_shape)
        b_data = rng.standard_normal(b_shape)

        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        t_sum(matmul(a, b)).backward()

        def fn_a(x): return t_sum(matmul(Tensor(x, requires_grad=True), Tensor(b_data)))
        def fn_b(x): return t_sum(matmul(Tensor(a_data), Tensor(x, requires_grad=True)))

        np.testing.assert_allclose(a.grad, numerical_grad(fn_a, a_data.copy()), atol=1e-6)
        np.testing.assert_allclose(b.grad, numerical_grad(fn_b, b_data.copy()), atol=1e-6)

    def test_2d(self):
        self._check((3, 4), (4, 5))

    def test_square(self):
        self._check((4, 4), (4, 4))


class TestActivationGrads(unittest.TestCase):
    def _check_unary(self, op, x_data):
        x = Tensor(x_data.copy(), requires_grad=True)
        t_sum(op(x)).backward()
        def fn(v): return t_sum(op(Tensor(v, requires_grad=True)))
        np.testing.assert_allclose(x.grad, numerical_grad(fn, x_data.copy()), atol=1e-6)

    def test_relu(self):
        self._check_unary(relu, np.array([-1.0, 0.5, 2.0]))

    def test_tanh(self):
        self._check_unary(tanh, np.array([-1.0, 0.0, 1.0]))

    def test_sigmoid(self):
        self._check_unary(sigmoid, np.array([-2.0, 0.0, 2.0]))

    def test_log(self):
        self._check_unary(log, np.array([0.5, 1.0, 2.0]))

    def test_exp(self):
        self._check_unary(exp, np.array([-1.0, 0.0, 1.0]))


class TestSumMeanGrad(unittest.TestCase):
    def test_sum_all(self):
        x = Tensor(np.ones((3, 4)), requires_grad=True)
        t_sum(x).backward()
        np.testing.assert_allclose(x.grad, np.ones((3, 4)))

    def test_sum_axis(self):
        x = Tensor(np.ones((3, 4)), requires_grad=True)
        t_sum(t_sum(x, axis=1)).backward()
        np.testing.assert_allclose(x.grad, np.ones((3, 4)))

    def test_mean_all(self):
        x = Tensor(np.ones((2, 3)), requires_grad=True)
        t_mean(x).backward()
        np.testing.assert_allclose(x.grad, np.full((2, 3), 1.0 / 6))


class TestReshapeGrad(unittest.TestCase):
    def test_reshape_grad(self):
        x = Tensor(np.arange(6.0), requires_grad=True)
        t_sum(reshape(x, (2, 3))).backward()
        np.testing.assert_allclose(x.grad, np.ones(6))


class TestEmbeddingLookupGrad(unittest.TestCase):
    def test_scatter_add(self):
        table = Tensor(np.ones((5, 3)), requires_grad=True)
        ids = np.array([0, 2, 0])
        t_sum(embedding_lookup(table, ids)).backward()
        expected = np.zeros((5, 3))
        expected[0] = 2.0
        expected[2] = 1.0
        np.testing.assert_allclose(table.grad, expected)


class TestLogSoftmaxGrad(unittest.TestCase):
    def test_log_softmax_grad(self):
        rng = np.random.default_rng(1)
        x_data = rng.standard_normal((4, 5))
        x = Tensor(x_data.copy(), requires_grad=True)
        t_sum(log_softmax(x, axis=-1)).backward()
        def fn(v): return t_sum(log_softmax(Tensor(v, requires_grad=True), axis=-1))
        np.testing.assert_allclose(x.grad, numerical_grad(fn, x_data.copy()), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
