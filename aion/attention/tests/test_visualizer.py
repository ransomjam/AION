"""Tests for AttentionVisualizer."""

import unittest
import numpy as np

from aion.attention.visualizer import AttentionVisualizer


class TestHeatmap(unittest.TestCase):

    def setUp(self):
        self.viz = AttentionVisualizer()
        self.w = np.array([[0.5, 0.3, 0.2],
                           [0.1, 0.7, 0.2]])
        self.tq = ["a", "b"]
        self.tk = ["x", "y", "z"]

    def test_keys_present(self):
        h = self.viz.heatmap(self.w, self.tq, self.tk)
        self.assertIn("tokens_q", h)
        self.assertIn("tokens_k", h)
        self.assertIn("matrix", h)

    def test_tokens_preserved(self):
        h = self.viz.heatmap(self.w, self.tq, self.tk)
        self.assertEqual(h["tokens_q"], self.tq)
        self.assertEqual(h["tokens_k"], self.tk)

    def test_matrix_shape(self):
        h = self.viz.heatmap(self.w, self.tq, self.tk)
        self.assertEqual(len(h["matrix"]), 2)
        self.assertEqual(len(h["matrix"][0]), 3)

    def test_matrix_values(self):
        h = self.viz.heatmap(self.w, self.tq, self.tk)
        self.assertAlmostEqual(h["matrix"][0][0], 0.5, places=5)
        self.assertAlmostEqual(h["matrix"][1][1], 0.7, places=5)

    def test_output_is_plain_python(self):
        h = self.viz.heatmap(self.w, self.tq, self.tk)
        for row in h["matrix"]:
            for v in row:
                self.assertIsInstance(v, float)


class TestMultiHeadHeatmaps(unittest.TestCase):

    def setUp(self):
        self.viz = AttentionVisualizer()
        rng = np.random.default_rng(0)
        raw = rng.dirichlet(np.ones(4), size=(3, 4))   # [n_heads, seq_q, seq_k]
        self.w = raw.reshape(3, 4, 4)
        self.tq = ["a", "b", "c", "d"]
        self.tk = ["a", "b", "c", "d"]

    def test_returns_one_per_head(self):
        heatmaps = self.viz.multi_head_heatmaps(self.w, self.tq, self.tk)
        self.assertEqual(len(heatmaps), 3)

    def test_head_index_present(self):
        heatmaps = self.viz.multi_head_heatmaps(self.w, self.tq, self.tk)
        for i, h in enumerate(heatmaps):
            self.assertEqual(h["head"], i)

    def test_each_has_matrix(self):
        heatmaps = self.viz.multi_head_heatmaps(self.w, self.tq, self.tk)
        for h in heatmaps:
            self.assertIn("matrix", h)
            self.assertEqual(len(h["matrix"]), 4)


class TestHeadComparison(unittest.TestCase):

    def setUp(self):
        self.viz = AttentionVisualizer()
        rng = np.random.default_rng(1)
        # [n_heads, seq_q, seq_k] — uniform distribution
        raw = np.ones((4, 5, 5)) / 5.0
        self.w_uniform = raw
        # Peaked distribution: each query attends only to position 0.
        peaked = np.zeros((4, 5, 5))
        peaked[:, :, 0] = 1.0
        self.w_peaked = peaked

    def test_keys_present(self):
        r = self.viz.head_comparison(self.w_uniform)
        self.assertIn("n_heads", r)
        self.assertIn("heads", r)
        self.assertIn("entropy_variance", r)

    def test_n_heads(self):
        r = self.viz.head_comparison(self.w_uniform)
        self.assertEqual(r["n_heads"], 4)

    def test_uniform_high_entropy(self):
        r = self.viz.head_comparison(self.w_uniform)
        for h in r["heads"]:
            self.assertGreater(h["entropy"], 1.0)

    def test_peaked_low_entropy(self):
        r = self.viz.head_comparison(self.w_peaked)
        for h in r["heads"]:
            self.assertLess(h["entropy"], 0.01)

    def test_peaked_high_max_attn(self):
        r = self.viz.head_comparison(self.w_peaked)
        for h in r["heads"]:
            self.assertAlmostEqual(h["max_attn"], 1.0, places=5)

    def test_4d_input_averaged(self):
        """4-D input [batch, n_heads, seq_q, seq_k] is averaged over batch."""
        rng = np.random.default_rng(2)
        w4d = np.abs(rng.standard_normal((3, 4, 5, 5)))
        w4d /= w4d.sum(axis=-1, keepdims=True)
        r = self.viz.head_comparison(w4d)
        self.assertEqual(r["n_heads"], 4)

    def test_per_head_fields(self):
        r = self.viz.head_comparison(self.w_uniform)
        for h in r["heads"]:
            self.assertIn("head", h)
            self.assertIn("entropy", h)
            self.assertIn("max_attn", h)
            self.assertIn("sparsity", h)


class TestQueryKeySimilarity(unittest.TestCase):

    def setUp(self):
        self.viz = AttentionVisualizer()

    def test_shape(self):
        Q = np.random.default_rng(0).standard_normal((3, 8))
        K = np.random.default_rng(1).standard_normal((5, 8))
        r = self.viz.query_key_similarity(Q, K, ["a","b","c"], ["x","y","z","p","q"])
        self.assertEqual(len(r["matrix"]), 3)
        self.assertEqual(len(r["matrix"][0]), 5)

    def test_self_similarity_is_one(self):
        """A vector's cosine similarity with itself is 1."""
        v = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0]])
        r = self.viz.query_key_similarity(v, v, ["a","b"], ["a","b"])
        self.assertAlmostEqual(r["matrix"][0][0], 1.0, places=5)
        self.assertAlmostEqual(r["matrix"][1][1], 1.0, places=5)

    def test_orthogonal_similarity_is_zero(self):
        v = np.array([[1.0, 0.0], [0.0, 1.0]])
        r = self.viz.query_key_similarity(v, v, ["a","b"], ["a","b"])
        self.assertAlmostEqual(r["matrix"][0][1], 0.0, places=5)
        self.assertAlmostEqual(r["matrix"][1][0], 0.0, places=5)

    def test_output_is_plain_python(self):
        Q = np.ones((2, 4))
        K = np.ones((2, 4))
        r = self.viz.query_key_similarity(Q, K, ["a","b"], ["a","b"])
        for row in r["matrix"]:
            for v in row:
                self.assertIsInstance(v, float)

    def test_tokens_preserved(self):
        Q = np.ones((2, 4))
        K = np.ones((3, 4))
        r = self.viz.query_key_similarity(Q, K, ["a","b"], ["x","y","z"])
        self.assertEqual(r["tokens_q"], ["a","b"])
        self.assertEqual(r["tokens_k"], ["x","y","z"])


if __name__ == "__main__":
    unittest.main()
