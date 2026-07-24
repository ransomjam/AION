"""Tests for Tensor: backward pass, graph release, requires_grad semantics."""

import unittest
import numpy as np
from aion.nn.tensor import Tensor, tensor, get_default_dtype
from aion.nn.ops import add, matmul, mul, relu, sum as t_sum


class TestTensorBasics(unittest.TestCase):
    def test_data_is_ndarray(self):
        t = Tensor([1.0, 2.0])
        self.assertIsInstance(t.data, np.ndarray)

    def test_default_dtype_is_float32(self):
        # Training default is float32 to keep the autograd graph's memory bounded.
        self.assertEqual(Tensor([1.0, 2.0]).data.dtype, np.float32)
        self.assertEqual(get_default_dtype(), np.float32)

    def test_explicit_dtype_override(self):
        self.assertEqual(Tensor([1.0], dtype=np.float64).data.dtype, np.float64)

    def test_shape_and_ndim(self):
        t = Tensor(np.ones((3, 4)))
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(t.ndim, 2)

    def test_item(self):
        # Exact scalar value requires float64; opt in via the explicit dtype.
        t = Tensor(np.array(3.14), dtype=np.float64)
        self.assertAlmostEqual(t.item(), 3.14)

    def test_grad_none_initially(self):
        t = Tensor([1.0], requires_grad=True)
        self.assertIsNone(t.grad)

    def test_requires_grad_false_by_default(self):
        t = Tensor([1.0])
        self.assertFalse(t.requires_grad)


class TestBackward(unittest.TestCase):
    def test_scalar_add_backward(self):
        a = Tensor(np.array(3.0), requires_grad=True)
        b = Tensor(np.array(4.0), requires_grad=True)
        c = add(a, b)
        c.backward()
        self.assertAlmostEqual(float(a.grad), 1.0)
        self.assertAlmostEqual(float(b.grad), 1.0)

    def test_mul_backward(self):
        a = Tensor(np.array(3.0), requires_grad=True)
        b = Tensor(np.array(4.0), requires_grad=True)
        c = mul(a, b)
        c.backward()
        self.assertAlmostEqual(float(a.grad), 4.0)
        self.assertAlmostEqual(float(b.grad), 3.0)

    def test_chain_rule(self):
        # f(x) = (x * 2) + 1  → df/dx = 2
        x = Tensor(np.array(5.0), requires_grad=True)
        two = Tensor(np.array(2.0))
        one = Tensor(np.array(1.0))
        y = add(mul(x, two), one)
        y.backward()
        self.assertAlmostEqual(float(x.grad), 2.0)

    def test_matmul_backward(self):
        # y = x @ W, dy/dW = x.T @ grad_y
        x = Tensor(np.ones((2, 3)), requires_grad=False)
        W = Tensor(np.ones((3, 4)), requires_grad=True)
        y = matmul(x, W)
        loss = t_sum(y)
        loss.backward()
        # grad_W = x.T @ ones(2,4) = ones(3,2) @ ones(2,4) = 2*ones(3,4)
        np.testing.assert_allclose(W.grad, np.full((3, 4), 2.0))

    def test_grad_accumulates(self):
        # Two paths to the same parameter — gradients should sum.
        x = Tensor(np.array(2.0), requires_grad=True)
        y = add(x, x)   # dy/dx = 2
        y.backward()
        self.assertAlmostEqual(float(x.grad), 2.0)

    def test_no_grad_for_requires_grad_false(self):
        a = Tensor(np.array(3.0), requires_grad=False)
        b = Tensor(np.array(4.0), requires_grad=True)
        c = add(a, b)
        c.backward()
        self.assertIsNone(a.grad)
        self.assertAlmostEqual(float(b.grad), 1.0)


class TestGraphRelease(unittest.TestCase):
    def test_graph_released_after_backward(self):
        x = Tensor(np.array(1.0), requires_grad=True)
        one = Tensor(np.array(1.0), requires_grad=True)
        two = Tensor(np.array(2.0), requires_grad=True)
        y = add(x, one)
        z = mul(y, two)
        z.backward()
        # After backward, _inputs cleared on all visited nodes.
        self.assertEqual(z._inputs, ())
        self.assertEqual(y._inputs, ())

    def test_second_backward_is_noop(self):
        # After graph release, calling backward again does not corrupt state.
        x = Tensor(np.array(2.0), requires_grad=True)
        two = Tensor(np.array(3.0), requires_grad=True)
        y = mul(x, two)
        y.backward()
        grad_after_first = float(x.grad)
        y.backward()
        self.assertAlmostEqual(float(x.grad), grad_after_first)


class TestRequiresGrad(unittest.TestCase):
    def test_frozen_parameter_not_in_graph(self):
        frozen = Tensor(np.array(5.0), requires_grad=False)
        active = Tensor(np.array(2.0), requires_grad=True)
        out = mul(frozen, active)
        out.backward()
        self.assertIsNone(frozen.grad)
        self.assertAlmostEqual(float(active.grad), 5.0)

    def test_all_frozen_produces_no_grad(self):
        a = Tensor(np.array(1.0), requires_grad=False)
        b = Tensor(np.array(2.0), requires_grad=False)
        c = add(a, b)
        c.backward()
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)

    def test_requires_grad_propagates_through_ops(self):
        x = Tensor(np.array(2.0), requires_grad=True)
        y = mul(x, Tensor(np.array(3.0)))
        # y is an intermediate — it must have requires_grad=True so it
        # appears in downstream _inputs and the graph is traversable.
        self.assertTrue(y.requires_grad)


if __name__ == "__main__":
    unittest.main()
