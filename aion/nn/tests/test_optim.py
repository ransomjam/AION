"""Tests for optimizers — SGD, SGD+momentum, Adam."""

import unittest
import numpy as np
from aion.nn.parameter import Parameter
from aion.nn.optim import SGD, Adam


def _param_with_grad(data, grad):
    p = Parameter(np.array(data, dtype=np.float64))
    p.grad = np.array(grad, dtype=np.float64)
    return p


class TestSGD(unittest.TestCase):
    def test_basic_update(self):
        p = _param_with_grad([1.0, 2.0], [0.1, 0.2])
        opt = SGD([p], lr=1.0)
        opt.step()
        np.testing.assert_allclose(p.data, [0.9, 1.8])

    def test_lr_scaling(self):
        p = _param_with_grad([1.0], [1.0])
        opt = SGD([p], lr=0.5)
        opt.step()
        np.testing.assert_allclose(p.data, [0.5])

    def test_zero_grad(self):
        p = _param_with_grad([1.0], [1.0])
        opt = SGD([p], lr=1.0)
        opt.zero_grad()
        self.assertIsNone(p.grad)

    def test_skips_none_grad(self):
        p = Parameter(np.array([1.0]))
        p.grad = None
        opt = SGD([p], lr=1.0)
        opt.step()
        np.testing.assert_allclose(p.data, [1.0])  # unchanged

    def test_skips_frozen(self):
        p = Parameter(np.array([1.0]), requires_grad=False)
        p.grad = np.array([1.0])
        opt = SGD([p], lr=1.0)
        opt.step()
        np.testing.assert_allclose(p.data, [1.0])  # unchanged

    def test_momentum(self):
        p = _param_with_grad([0.0], [1.0])
        opt = SGD([p], lr=1.0, momentum=0.9)
        opt.step()
        # velocity = 0.9*0 + 1.0 = 1.0; data = 0 - 1.0*1.0 = -1.0
        np.testing.assert_allclose(p.data, [-1.0])
        p.grad = np.array([1.0])
        opt.step()
        # velocity = 0.9*1.0 + 1.0 = 1.9; data = -1.0 - 1.0*1.9 = -2.9
        np.testing.assert_allclose(p.data, [-2.9])


class TestAdam(unittest.TestCase):
    def test_update_reduces_param(self):
        p = _param_with_grad([1.0], [1.0])
        opt = Adam([p], lr=0.1)
        opt.step()
        self.assertLess(float(p.data[0]), 1.0)

    def test_multiple_steps(self):
        p = _param_with_grad([1.0], [1.0])
        opt = Adam([p], lr=0.1)
        for _ in range(10):
            p.grad = np.array([1.0])
            opt.step()
        self.assertLess(float(p.data[0]), 0.5)

    def test_skips_none_grad(self):
        p = Parameter(np.array([1.0]))
        p.grad = None
        opt = Adam([p], lr=0.1)
        opt.step()
        np.testing.assert_allclose(p.data, [1.0])

    def test_skips_frozen(self):
        p = Parameter(np.array([1.0]), requires_grad=False)
        p.grad = np.array([1.0])
        opt = Adam([p], lr=0.1)
        opt.step()
        np.testing.assert_allclose(p.data, [1.0])

    def test_step_counter_increments(self):
        p = _param_with_grad([1.0], [1.0])
        opt = Adam([p], lr=0.1)
        opt.step()
        self.assertEqual(opt._t, 1)
        opt.step()
        self.assertEqual(opt._t, 2)


if __name__ == "__main__":
    unittest.main()
