"""Tests for the full Transformer encoder-decoder model."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.nn.parameter import Parameter
from aion.attention.mask import CausalMask, PaddingMask
from aion.transformer.model import Transformer
from aion.transformer.config import TransformerConfig


def _t(shape, seed=0, requires_grad=True):
    return Tensor(np.random.default_rng(seed).standard_normal(shape),
                  requires_grad=requires_grad)


def _cfg(**kw):
    defaults = dict(d_model=16, n_heads=4, n_encoder_layers=2,
                    n_decoder_layers=2, d_ff=64)
    defaults.update(kw)
    return TransformerConfig(**defaults)


class TestTransformer(unittest.TestCase):

    def _model(self, **kw):
        return Transformer(_cfg(**kw), rng=np.random.default_rng(0))

    def test_output_shape(self):
        m = self._model()
        src = _t((2, 6, 16))
        tgt = _t((2, 5, 16), seed=1)
        out, _ = m(src, tgt)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_viz_keys(self):
        m = self._model()
        _, viz = m(_t((1, 4, 16)), _t((1, 3, 16), seed=1))
        self.assertIn("encoder_weights", viz)
        self.assertIn("decoder_self_weights", viz)
        self.assertIn("decoder_cross_weights", viz)

    def test_viz_encoder_weights_count(self):
        m = self._model(n_encoder_layers=3)
        _, viz = m(_t((1, 4, 16)), _t((1, 3, 16), seed=1))
        self.assertEqual(len(viz["encoder_weights"]), 3)

    def test_viz_decoder_weights_count(self):
        m = self._model(n_decoder_layers=3)
        _, viz = m(_t((1, 4, 16)), _t((1, 3, 16), seed=1))
        self.assertEqual(len(viz["decoder_self_weights"]), 3)
        self.assertEqual(len(viz["decoder_cross_weights"]), 3)

    def test_viz_weights_are_ndarrays(self):
        m = self._model()
        _, viz = m(_t((1, 4, 16)), _t((1, 3, 16), seed=1))
        for w in viz["encoder_weights"] + viz["decoder_self_weights"] + viz["decoder_cross_weights"]:
            self.assertIsInstance(w, np.ndarray)

    def test_backward_runs(self):
        m = self._model()
        src = _t((2, 6, 16))
        tgt = _t((2, 5, 16), seed=1)
        out, _ = m(src, tgt)
        t_sum(out).backward()
        self.assertIsNotNone(src.grad)
        self.assertIsNotNone(tgt.grad)
        for p in m.parameters():
            self.assertIsNotNone(p.grad)

    def test_with_masks(self):
        m = self._model()
        src = _t((2, 6, 16))
        tgt = _t((2, 5, 16), seed=1)
        valid = np.ones((2, 6), dtype=bool)
        out, _ = m(src, tgt,
                   tgt_mask=CausalMask(),
                   cross_mask=PaddingMask(valid))
        self.assertEqual(out.shape, (2, 5, 16))

    def test_train_eval_propagates(self):
        m = self._model()
        m.eval()
        self.assertFalse(m.training)
        self.assertFalse(m.encoder.training)
        self.assertFalse(m.decoder.training)
        m.train()
        self.assertTrue(m.encoder.training)

    def test_validate_called_on_bad_config(self):
        with self.assertRaises(ValueError):
            Transformer(TransformerConfig(d_model=10, n_heads=3))

    def test_repr(self):
        r = repr(self._model())
        self.assertIn("Transformer", r)
        self.assertIn("16", r)

    # ── weight tying ──────────────────────────────────────────────────────────

    def test_weights_tied_false_by_default(self):
        m = self._model()
        self.assertFalse(m.weights_tied)

    def test_tie_weights_sets_flag(self):
        m = self._model()
        emb = Parameter(np.zeros((100, 16)), name="table")
        proj = Parameter(np.zeros((16, 100)), name="proj")
        m.tie_weights(emb, proj)
        self.assertTrue(m.weights_tied)

    def test_tie_weights_stores_parameters(self):
        m = self._model()
        emb = Parameter(np.zeros((100, 16)), name="table")
        proj = emb  # same object — actual tying
        m.tie_weights(emb, proj)
        tied = object.__getattribute__(m, "_tied_weights")
        self.assertIs(tied["embedding"], emb)
        self.assertIs(tied["projection"], proj)


if __name__ == "__main__":
    unittest.main()
