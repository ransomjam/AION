import unittest
import numpy as np
from aion.gpt.loss import CausalLanguageModelLoss
from aion.nn.tensor import Tensor


class TestCausalLanguageModelLoss(unittest.TestCase):

    def _make_logits(self, batch=2, seq=4, vocab=8, seed=0):
        rng = np.random.default_rng(seed)
        data = rng.standard_normal((batch, seq, vocab))
        return Tensor(data, requires_grad=True)

    def test_output_is_scalar(self):
        loss_fn = CausalLanguageModelLoss()
        logits = self._make_logits()
        targets = np.array([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=np.intp)
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.data.shape, ())

    def test_loss_positive(self):
        loss_fn = CausalLanguageModelLoss()
        logits = self._make_logits()
        targets = np.zeros((2, 4), dtype=np.intp)
        loss = loss_fn(logits, targets)
        self.assertGreater(float(loss.data), 0.0)

    def test_backward_runs(self):
        loss_fn = CausalLanguageModelLoss()
        logits = self._make_logits()
        targets = np.array([[1, 2, 3, 4], [5, 6, 7, 0]], dtype=np.intp)
        loss = loss_fn(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.data.shape)

    def test_perfect_prediction_low_loss(self):
        # Logits heavily favour the correct class → low loss
        loss_fn = CausalLanguageModelLoss()
        logits_data = np.full((1, 3, 4), -10.0)
        targets = np.array([[2, 1, 0]], dtype=np.intp)
        for t in range(3):
            logits_data[0, t, targets[0, t]] = 100.0
        logits = Tensor(logits_data, requires_grad=True)
        loss = loss_fn(logits, targets)
        self.assertLess(float(loss.data), 0.01)

    def test_uniform_logits_near_log_vocab(self):
        import math
        vocab = 8
        loss_fn = CausalLanguageModelLoss()
        logits = Tensor(np.zeros((1, 4, vocab)), requires_grad=True)
        targets = np.zeros((1, 4), dtype=np.intp)
        loss = loss_fn(logits, targets)
        self.assertAlmostEqual(float(loss.data), math.log(vocab), places=4)


if __name__ == "__main__":
    unittest.main()
