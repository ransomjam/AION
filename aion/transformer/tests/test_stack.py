"""Tests for TransformerStack and TransformerEncoder."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.attention.mask import CausalMask
from aion.transformer.stack import TransformerStack
from aion.transformer.encoder import TransformerEncoder
from aion.transformer.config import EncoderConfig, TransformerConfig


def _t(shape, seed=0):
    return Tensor(np.random.default_rng(seed).standard_normal(shape),
                  requires_grad=True)


class TestTransformerStack(unittest.TestCase):

    def _stack(self, d=16, h=4, n=2, d_ff=64, **kw):
        return TransformerStack(d, h, n, d_ff, rng=np.random.default_rng(0), **kw)

    def test_output_shape(self):
        s = self._stack()
        out, weights = s(_t((2, 5, 16)))
        self.assertEqual(out.shape, (2, 5, 16))

    def test_block_weights_count(self):
        s = self._stack(n=3)
        _, weights = s(_t((1, 4, 16)))
        self.assertEqual(len(weights), 3)

    def test_block_weights_shape(self):
        s = self._stack(n=2)
        _, weights = s(_t((2, 5, 16)))
        for w in weights:
            self.assertEqual(w.shape, (2, 4, 5, 5))

    def test_block_weights_detached(self):
        s = self._stack()
        _, weights = s(_t((1, 4, 16)))
        for w in weights:
            self.assertIsInstance(w, np.ndarray)

    def test_backward_runs(self):
        s = self._stack()
        x = _t((2, 5, 16))
        out, _ = s(x)
        t_sum(out).backward()
        self.assertIsNotNone(x.grad)
        for p in s.parameters():
            self.assertIsNotNone(p.grad)

    def test_with_causal_mask(self):
        s = self._stack()
        out, weights = s(_t((1, 4, 16)), mask=CausalMask())
        self.assertEqual(out.shape, (1, 4, 16))
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(float(weights[0][0, 0, i, j]), 0.0, places=4)

    def test_train_eval_propagates(self):
        s = self._stack()
        s.eval()
        self.assertFalse(s.training)
        for p in s._modules.values():
            self.assertFalse(p.training)
        s.train()
        self.assertTrue(s.training)

    def test_repr(self):
        r = repr(self._stack())
        self.assertIn("TransformerStack", r)
        self.assertIn("16", r)


class TestTransformerEncoder(unittest.TestCase):

    def _enc(self, d=16, h=4, n=2, d_ff=64, **kw):
        return TransformerEncoder(d, h, n, d_ff, rng=np.random.default_rng(0), **kw)

    def test_is_transformer_stack(self):
        self.assertIsInstance(self._enc(), TransformerStack)

    def test_output_shape(self):
        enc = self._enc()
        out, _ = enc(_t((2, 5, 16)))
        self.assertEqual(out.shape, (2, 5, 16))

    def test_from_encoder_config(self):
        cfg = EncoderConfig(d_model=16, n_heads=4, n_layers=3, d_ff=64)
        enc = TransformerEncoder.from_config(cfg, rng=np.random.default_rng(0))
        self.assertEqual(enc.n_layers, 3)
        out, weights = enc(_t((1, 4, 16)))
        self.assertEqual(out.shape, (1, 4, 16))
        self.assertEqual(len(weights), 3)

    def test_from_transformer_config(self):
        cfg = TransformerConfig(d_model=16, n_heads=4, n_encoder_layers=2, d_ff=64)
        enc = TransformerEncoder.from_config(cfg, rng=np.random.default_rng(0))
        self.assertEqual(enc.n_layers, 2)

    def test_repr(self):
        r = repr(self._enc())
        self.assertIn("TransformerEncoder", r)


if __name__ == "__main__":
    unittest.main()
