"""Tests for loss functions — forward values and gradient correctness."""

import unittest
import numpy as np
from aion.nn.tensor import Tensor
from aion.nn.loss import BCELoss, CrossEntropyLoss, MSELoss
from aion.nn.ops import t_sum as _sum


def numerical_grad_loss(loss_fn, pred_data, targets, eps=1e-5):
    grad = np.zeros_like(pred_data)
    it = np.nditer(pred_data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = pred_data[idx]
        pred_data[idx] = orig + eps
        fp = float(loss_fn(Tensor(pred_data.copy()), targets).data.flat[0])
        pred_data[idx] = orig - eps
        fm = float(loss_fn(Tensor(pred_data.copy()), targets).data.flat[0])
        pred_data[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


class TestCrossEntropyLoss(unittest.TestCase):
    def test_forward_value(self):
        # Uniform logits → loss = log(num_classes)
        logits = Tensor(np.zeros((4, 5)))
        targets = np.array([0, 1, 2, 3])
        loss = CrossEntropyLoss()(logits, targets)
        self.assertAlmostEqual(float(loss.data), np.log(5), places=5)

    def test_perfect_prediction_low_loss(self):
        logits = Tensor(np.array([[10.0, -10.0], [-10.0, 10.0]]))
        targets = np.array([0, 1])
        loss = CrossEntropyLoss()(logits, targets)
        self.assertLess(float(loss.data), 0.01)

    def test_backward_grad(self):
        rng = np.random.default_rng(0)
        pred_data = rng.standard_normal((3, 4))
        targets = np.array([0, 2, 1])
        pred = Tensor(pred_data.copy(), requires_grad=True)
        loss = CrossEntropyLoss()(pred, targets)
        loss.backward()
        num = numerical_grad_loss(CrossEntropyLoss(), pred_data.copy(), targets)
        np.testing.assert_allclose(pred.grad, num, atol=1e-6)

    def test_returns_scalar(self):
        logits = Tensor(np.ones((2, 3)))
        loss = CrossEntropyLoss()(logits, np.array([0, 1]))
        self.assertEqual(loss.shape, ())


class TestMSELoss(unittest.TestCase):
    def test_forward_value(self):
        pred = Tensor(np.array([1.0, 2.0, 3.0]))
        targets = np.array([1.0, 2.0, 3.0])
        loss = MSELoss()(pred, targets)
        self.assertAlmostEqual(float(loss.data), 0.0)

    def test_forward_nonzero(self):
        pred = Tensor(np.array([0.0, 0.0]))
        targets = np.array([1.0, 1.0])
        loss = MSELoss()(pred, targets)
        self.assertAlmostEqual(float(loss.data), 1.0)

    def test_backward_grad(self):
        rng = np.random.default_rng(1)
        pred_data = rng.standard_normal((4,))
        targets = rng.standard_normal((4,))
        pred = Tensor(pred_data.copy(), requires_grad=True)
        MSELoss()(pred, targets).backward()
        num = numerical_grad_loss(MSELoss(), pred_data.copy(), targets)
        np.testing.assert_allclose(pred.grad, num, atol=1e-6)


class TestBCELoss(unittest.TestCase):
    def test_forward_value(self):
        pred = Tensor(np.array([0.5, 0.5]))
        targets = np.array([1.0, 0.0])
        loss = BCELoss()(pred, targets)
        expected = -np.mean([np.log(0.5), np.log(0.5)])
        self.assertAlmostEqual(float(loss.data), expected, places=5)

    def test_backward_grad(self):
        rng = np.random.default_rng(2)
        pred_data = rng.uniform(0.1, 0.9, (5,))
        targets = rng.integers(0, 2, (5,)).astype(float)
        pred = Tensor(pred_data.copy(), requires_grad=True)
        BCELoss()(pred, targets).backward()
        num = numerical_grad_loss(BCELoss(), pred_data.copy(), targets)
        np.testing.assert_allclose(pred.grad, num, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
