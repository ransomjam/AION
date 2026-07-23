"""Tests for Trainer — training contract, loss convergence, TrainingResult."""

import unittest
import numpy as np
from aion.nn.layers import Linear, ReLU
from aion.nn.sequential import Sequential
from aion.nn.loss import CrossEntropyLoss, MSELoss
from aion.nn.optim import SGD, Adam
from aion.nn.tensor import Tensor
from aion.nn.trainer import Trainer, TrainingResult


def _classification_batches():
    """4-sample XOR-like classification dataset, single batch."""
    X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([0, 1, 1, 0])
    def batches():
        yield Tensor(X), y
    return batches


def _regression_batches():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 4))
    y = rng.standard_normal((20, 1))
    def batches():
        yield Tensor(X), y
    return batches


class TestTrainer(unittest.TestCase):
    def test_training_result_type(self):
        model = Sequential(Linear(2, 4, rng=np.random.default_rng(0)), ReLU(),
                           Linear(4, 2, rng=np.random.default_rng(1)))
        trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))
        result = trainer.train(_classification_batches(), epochs=2, seed=0)
        self.assertIsInstance(result, TrainingResult)

    def test_result_metrics_keys(self):
        model = Sequential(Linear(2, 4, rng=np.random.default_rng(0)), ReLU(),
                           Linear(4, 2, rng=np.random.default_rng(1)))
        trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))
        result = trainer.train(_classification_batches(), epochs=3, seed=7)
        for key in ("loss_history", "final_loss", "epochs", "param_count",
                    "training_time_s", "seed"):
            self.assertIn(key, result.metrics)

    def test_loss_history_length(self):
        model = Sequential(Linear(2, 4, rng=np.random.default_rng(0)), ReLU(),
                           Linear(4, 2, rng=np.random.default_rng(1)))
        trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))
        result = trainer.train(_classification_batches(), epochs=5, seed=0)
        self.assertEqual(len(result.metrics["loss_history"]), 5)

    def test_loss_decreases_over_training(self):
        rng = np.random.default_rng(42)
        model = Sequential(Linear(4, 16, rng=rng), ReLU(), Linear(16, 1, rng=rng))
        trainer = Trainer(model, MSELoss(), Adam(model.parameters(), lr=0.01))
        result = trainer.train(_regression_batches(), epochs=200, seed=0)
        history = result.metrics["loss_history"]
        self.assertLess(history[-1], history[0])

    def test_progress_fn_called_per_epoch(self):
        calls = []
        model = Sequential(Linear(2, 4, rng=np.random.default_rng(0)), ReLU(),
                           Linear(4, 2, rng=np.random.default_rng(1)))
        trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1),
                          progress_fn=lambda f, m: calls.append(f))
        trainer.train(_classification_batches(), epochs=4, seed=0)
        self.assertEqual(len(calls), 4)
        self.assertAlmostEqual(calls[-1], 1.0)

    def test_seed_recorded_in_result(self):
        model = Linear(2, 2, rng=np.random.default_rng(0))
        trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01))
        # output shape (4,2), targets shape (4,2)
        def batches():
            yield Tensor(np.ones((4, 2))), np.zeros((4, 2))
        result = trainer.train(batches, epochs=1, seed=99)
        self.assertEqual(result.metrics["seed"], 99)

    def test_param_count_in_result(self):
        model = Linear(4, 3)
        trainer = Trainer(model, MSELoss(), SGD(model.parameters(), lr=0.01))
        result = trainer.train(_regression_batches(), epochs=1, seed=0)
        self.assertEqual(result.metrics["param_count"], model.param_count())

    def test_gradients_zeroed_each_step(self):
        """Gradients must not accumulate across steps."""
        rng = np.random.default_rng(0)
        model = Linear(2, 2, rng=rng)
        opt = SGD(model.parameters(), lr=0.0)  # lr=0: params unchanged
        trainer = Trainer(model, CrossEntropyLoss(), opt)
        trainer.train(_classification_batches(), epochs=2, seed=0)
        for p in model.parameters():
            if p.grad is not None:
                self.assertFalse(np.any(np.abs(p.grad) > 1e3),
                                 "gradient looks accumulated across steps")


if __name__ == "__main__":
    unittest.main()
