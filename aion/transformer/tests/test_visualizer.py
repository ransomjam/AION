"""Tests for TransformerVisualizer."""

import unittest
import numpy as np

from aion.transformer.visualizer import TransformerVisualizer


class TestTransformerVisualizer(unittest.TestCase):

    def setUp(self):
        self.viz = TransformerVisualizer()
        rng = np.random.default_rng(0)
        self.outputs = [rng.standard_normal((2, 5, 16)) for _ in range(3)]
        self.weights = [
            np.abs(rng.standard_normal((2, 4, 5, 5))) for _ in range(3)
        ]
        for w in self.weights:
            w /= w.sum(axis=-1, keepdims=True)

    def test_layer_outputs_keys(self):
        r = self.viz.layer_outputs(self.outputs)
        self.assertIn("n_layers", r)
        self.assertIn("mean_magnitude", r)

    def test_layer_outputs_count(self):
        r = self.viz.layer_outputs(self.outputs)
        self.assertEqual(r["n_layers"], 3)
        self.assertEqual(len(r["mean_magnitude"]), 3)

    def test_layer_outputs_plain_python(self):
        r = self.viz.layer_outputs(self.outputs)
        for v in r["mean_magnitude"]:
            self.assertIsInstance(v, float)

    def test_residual_norms_keys(self):
        inputs = [np.zeros_like(o) for o in self.outputs]
        r = self.viz.residual_norms(inputs, self.outputs)
        self.assertIn("residual_norm", r)
        self.assertEqual(len(r["residual_norm"]), 3)

    def test_layernorm_stats_keys(self):
        r = self.viz.layernorm_stats(self.outputs)
        self.assertIn("layers", r)
        for layer in r["layers"]:
            self.assertIn("mean", layer)
            self.assertIn("variance", layer)

    def test_ffn_activations_relu(self):
        pre = [np.random.default_rng(i).standard_normal((2, 5, 64)) for i in range(3)]
        r = self.viz.ffn_activations(pre, activation="relu")
        self.assertEqual(r["activation"], "relu")
        for layer in r["layers"]:
            self.assertIn("dead_fraction", layer)
            self.assertGreaterEqual(layer["dead_fraction"], 0.0)
            self.assertLessEqual(layer["dead_fraction"], 1.0)

    def test_ffn_activations_gelu(self):
        pre = [np.random.default_rng(i).standard_normal((2, 5, 64)) for i in range(2)]
        r = self.viz.ffn_activations(pre, activation="gelu")
        self.assertEqual(r["activation"], "gelu")
        for layer in r["layers"]:
            self.assertIn("mean", layer)
            self.assertIn("std", layer)

    def test_cross_attention_heatmap_shape(self):
        rng = np.random.default_rng(0)
        cw = np.abs(rng.standard_normal((2, 4, 5, 7)))
        cw /= cw.sum(axis=-1, keepdims=True)
        r = self.viz.cross_attention_heatmap(
            cw, src_tokens=["a"]*7, tgt_tokens=["b"]*5
        )
        self.assertEqual(len(r["matrix"]), 5)
        self.assertEqual(len(r["matrix"][0]), 7)

    def test_cross_attention_heatmap_plain_python(self):
        rng = np.random.default_rng(1)
        cw = np.abs(rng.standard_normal((1, 2, 3, 4)))
        cw /= cw.sum(axis=-1, keepdims=True)
        r = self.viz.cross_attention_heatmap(
            cw, src_tokens=["x"]*4, tgt_tokens=["y"]*3
        )
        for row in r["matrix"]:
            for v in row:
                self.assertIsInstance(v, float)

    def test_depth_comparison_keys(self):
        r = self.viz.depth_comparison(self.weights)
        self.assertIn("n_layers", r)
        self.assertIn("layers", r)
        self.assertEqual(r["n_layers"], 3)

    def test_depth_comparison_fields(self):
        r = self.viz.depth_comparison(self.weights)
        for layer in r["layers"]:
            self.assertIn("mean_entropy", layer)
            self.assertIn("mean_sparsity", layer)

    def test_depth_comparison_plain_python(self):
        r = self.viz.depth_comparison(self.weights)
        for layer in r["layers"]:
            self.assertIsInstance(layer["mean_entropy"], float)
            self.assertIsInstance(layer["mean_sparsity"], float)


if __name__ == "__main__":
    unittest.main()
