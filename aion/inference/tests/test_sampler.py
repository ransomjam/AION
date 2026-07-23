import unittest
import numpy as np
from aion.inference.sampler import GreedySampler, StochasticSampler


class TestGreedySampler(unittest.TestCase):

    def test_selects_argmax(self):
        sampler = GreedySampler()
        logits = np.array([1.0, 5.0, 2.0, 3.0])
        token_id, probs = sampler.sample(logits)
        self.assertEqual(token_id, 1)

    def test_probs_sum_to_one(self):
        sampler = GreedySampler()
        logits = np.array([1.0, 2.0, 3.0])
        _, probs = sampler.sample(logits)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_probs_shape(self):
        sampler = GreedySampler()
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        _, probs = sampler.sample(logits)
        self.assertEqual(probs.shape, (4,))

    def test_deterministic(self):
        sampler = GreedySampler()
        logits = np.array([0.1, 0.9, 0.5])
        t1, _ = sampler.sample(logits)
        t2, _ = sampler.sample(logits)
        self.assertEqual(t1, t2)

    def test_handles_neg_inf(self):
        sampler = GreedySampler()
        logits = np.array([-1e9, -1e9, 3.0, -1e9])
        token_id, _ = sampler.sample(logits)
        self.assertEqual(token_id, 2)


class TestStochasticSampler(unittest.TestCase):

    def test_returns_valid_token_id(self):
        rng = np.random.default_rng(0)
        sampler = StochasticSampler(rng=rng)
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        token_id, probs = sampler.sample(logits)
        self.assertGreaterEqual(token_id, 0)
        self.assertLess(token_id, 4)

    def test_probs_sum_to_one(self):
        rng = np.random.default_rng(1)
        sampler = StochasticSampler(rng=rng)
        logits = np.array([1.0, 2.0, 3.0])
        _, probs = sampler.sample(logits)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=6)

    def test_seeded_reproducibility(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s1 = StochasticSampler(rng=np.random.default_rng(42))
        s2 = StochasticSampler(rng=np.random.default_rng(42))
        ids1 = [s1.sample(logits)[0] for _ in range(10)]
        ids2 = [s2.sample(logits)[0] for _ in range(10)]
        self.assertEqual(ids1, ids2)

    def test_different_seeds_differ(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s1 = StochasticSampler(rng=np.random.default_rng(0))
        s2 = StochasticSampler(rng=np.random.default_rng(99))
        ids1 = [s1.sample(logits)[0] for _ in range(20)]
        ids2 = [s2.sample(logits)[0] for _ in range(20)]
        self.assertNotEqual(ids1, ids2)

    def test_peaked_distribution_mostly_top(self):
        # Very peaked logits → sampler should almost always pick token 3
        rng = np.random.default_rng(7)
        sampler = StochasticSampler(rng=rng)
        logits = np.array([0.0, 0.0, 0.0, 100.0])
        counts = [0, 0, 0, 0]
        for _ in range(50):
            tid, _ = sampler.sample(logits)
            counts[tid] += 1
        self.assertEqual(counts[3], 50)


if __name__ == "__main__":
    unittest.main()
