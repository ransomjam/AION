"""Tests for TransformerDecoder stack."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.attention.mask import CausalMask
from aion.transformer.decoder import TransformerDecoder
from aion.transformer.config import DecoderConfig, TransformerConfig


def _t(shape, seed=0, requires_grad=True):
    return Tensor(np.random.default_rng(seed).standard_normal(shape),
                  requires_grad=requires_grad)


class TestTransformerDecoder(unittest.TestCase):

    def _dec(self, d=16, h=4, n=2, d_ff=64, **kw):
        return TransformerDecoder(d, h, n, d_ff, rng=np.random.default_rng(0), **kw)

    def test_output_shape(self):
        dec = self._dec()
        x = _t((2, 5, 16))
        enc = _t((2, 7, 16), seed=1, requires_grad=False)
        out, self_ws, cross_ws = dec(x, enc)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_self_weights_count(self):
        dec = self._dec(n=3)
        x = _t((1, 4, 16))
        enc = _t((1, 6, 16), seed=1, requires_grad=False)
        _, self_ws, _ = dec(x, enc)
        self.assertEqual(len(self_ws), 3)

    def test_cross_weights_count(self):
        dec = self._dec(n=3)
        x = _t((1, 4, 16))
        enc = _t((1, 6, 16), seed=1, requires_grad=False)
        _, _, cross_ws = dec(x, enc)
        self.assertEqual(len(cross_ws), 3)

    def test_self_weights_shape(self):
        dec = self._dec()
        x = _t((2, 5, 16))
        enc = _t((2, 7, 16), seed=1, requires_grad=False)
        _, self_ws, _ = dec(x, enc)
        for w in self_ws:
            self.assertEqual(w.shape, (2, 4, 5, 5))

    def test_cross_weights_shape(self):
        dec = self._dec()
        x = _t((2, 5, 16))
        enc = _t((2, 7, 16), seed=1, requires_grad=False)
        _, _, cross_ws = dec(x, enc)
        for w in cross_ws:
            self.assertEqual(w.shape, (2, 4, 5, 7))

    def test_weights_detached(self):
        dec = self._dec()
        x = _t((1, 3, 16))
        enc = _t((1, 4, 16), seed=1, requires_grad=False)
        _, self_ws, cross_ws = dec(x, enc)
        for w in self_ws + cross_ws:
            self.assertIsInstance(w, np.ndarray)

    def test_backward_runs(self):
        dec = self._dec()
        x = _t((2, 5, 16))
        enc = _t((2, 7, 16), seed=1, requires_grad=False)
        out, _, _ = dec(x, enc)
        t_sum(out).backward()
        self.assertIsNotNone(x.grad)
        for p in dec.parameters():
            self.assertIsNotNone(p.grad)

    def test_causal_self_mask(self):
        dec = self._dec()
        x = _t((1, 4, 16))
        enc = _t((1, 4, 16), seed=1, requires_grad=False)
        _, self_ws, _ = dec(x, enc, self_mask=CausalMask())
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(float(self_ws[0][0, 0, i, j]), 0.0, places=4)

    def test_from_decoder_config(self):
        cfg = DecoderConfig(d_model=16, n_heads=4, n_layers=3, d_ff=64)
        dec = TransformerDecoder.from_config(cfg, rng=np.random.default_rng(0))
        self.assertEqual(dec.n_layers, 3)
        x = _t((1, 4, 16))
        enc = _t((1, 4, 16), seed=1, requires_grad=False)
        out, self_ws, cross_ws = dec(x, enc)
        self.assertEqual(out.shape, (1, 4, 16))
        self.assertEqual(len(self_ws), 3)

    def test_from_transformer_config(self):
        cfg = TransformerConfig(d_model=16, n_heads=4, n_decoder_layers=2, d_ff=64)
        dec = TransformerDecoder.from_config(cfg, rng=np.random.default_rng(0))
        self.assertEqual(dec.n_layers, 2)

    def test_train_eval_propagates(self):
        dec = self._dec()
        dec.eval()
        self.assertFalse(dec.training)
        dec.train()
        self.assertTrue(dec.training)

    def test_repr(self):
        r = repr(self._dec())
        self.assertIn("TransformerDecoder", r)
        self.assertIn("16", r)


if __name__ == "__main__":
    unittest.main()
