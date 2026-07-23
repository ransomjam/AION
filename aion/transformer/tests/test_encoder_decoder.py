"""Tests for TransformerEncoder, TransformerDecoder, and Transformer."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.attention.mask import CausalMask
from aion.transformer.config import DecoderConfig, EncoderConfig, TransformerConfig
from aion.transformer.encoder import TransformerEncoder
from aion.transformer.decoder import TransformerDecoder
from aion.transformer.model import Transformer


def _t(shape, seed=0, requires_grad=True):
    return Tensor(np.random.default_rng(seed).standard_normal(shape),
                  requires_grad=requires_grad)


class TestTransformerEncoder(unittest.TestCase):

    def _enc(self, n=2, d=16, h=4, d_ff=64):
        return TransformerEncoder(d, h, n, d_ff, rng=np.random.default_rng(0))

    def test_output_shape(self):
        out, ws = self._enc()(_t((2, 5, 16)))
        self.assertEqual(out.shape, (2, 5, 16))

    def test_block_weights_count(self):
        _, ws = self._enc(n=3)(_t((2, 5, 16)))
        self.assertEqual(len(ws), 3)

    def test_block_weights_shape(self):
        _, ws = self._enc(n=2)(_t((2, 5, 16)))
        for w in ws:
            self.assertEqual(w.shape, (2, 4, 5, 5))

    def test_block_weights_detached(self):
        _, ws = self._enc()(_t((2, 5, 16)))
        for w in ws:
            self.assertIsInstance(w, np.ndarray)

    def test_backward_runs(self):
        enc = self._enc()
        x = _t((2, 5, 16))
        out, _ = enc(x)
        t_sum(out).backward()
        self.assertIsNotNone(x.grad)
        for p in enc.parameters():
            self.assertIsNotNone(p.grad)

    def test_from_encoder_config(self):
        cfg = EncoderConfig(d_model=16, n_heads=4, n_layers=2, d_ff=64)
        enc = TransformerEncoder.from_config(cfg, rng=np.random.default_rng(0))
        out, _ = enc(_t((1, 4, 16)))
        self.assertEqual(out.shape, (1, 4, 16))

    def test_from_transformer_config(self):
        cfg = TransformerConfig(d_model=16, n_heads=4, n_encoder_layers=2, d_ff=64)
        enc = TransformerEncoder.from_config(cfg, rng=np.random.default_rng(0))
        out, _ = enc(_t((1, 4, 16)))
        self.assertEqual(out.shape, (1, 4, 16))

    def test_train_eval_propagates(self):
        enc = self._enc()
        enc.eval()
        for blk in enc._blocks:
            self.assertFalse(blk.drop.training)
        enc.train()
        for blk in enc._blocks:
            self.assertTrue(blk.drop.training)


class TestTransformerDecoder(unittest.TestCase):

    def _dec(self, n=2, d=16, h=4, d_ff=64):
        return TransformerDecoder(d, h, n, d_ff, rng=np.random.default_rng(0))

    def _enc_out(self, batch=2, src=7):
        return _t((batch, src, 16), seed=1, requires_grad=False)

    def test_output_shape(self):
        out, _, _ = self._dec()(_t((2, 5, 16)), self._enc_out())
        self.assertEqual(out.shape, (2, 5, 16))

    def test_self_weights_count(self):
        _, sws, _ = self._dec(n=3)(_t((2, 5, 16)), self._enc_out())
        self.assertEqual(len(sws), 3)

    def test_cross_weights_count(self):
        _, _, cws = self._dec(n=3)(_t((2, 5, 16)), self._enc_out())
        self.assertEqual(len(cws), 3)

    def test_cross_weights_shape(self):
        _, _, cws = self._dec()(_t((2, 5, 16)), self._enc_out())
        for cw in cws:
            self.assertEqual(cw.shape, (2, 4, 5, 7))

    def test_backward_runs(self):
        dec = self._dec()
        x = _t((2, 5, 16))
        out, _, _ = dec(x, self._enc_out())
        t_sum(out).backward()
        self.assertIsNotNone(x.grad)

    def test_from_decoder_config(self):
        cfg = DecoderConfig(d_model=16, n_heads=4, n_layers=2, d_ff=64)
        dec = TransformerDecoder.from_config(cfg, rng=np.random.default_rng(0))
        out, _, _ = dec(_t((1, 4, 16)), self._enc_out(batch=1))
        self.assertEqual(out.shape, (1, 4, 16))


class TestTransformer(unittest.TestCase):

    def _cfg(self, enc=2, dec=2):
        return TransformerConfig(d_model=16, n_heads=4,
                                 n_encoder_layers=enc, n_decoder_layers=dec,
                                 d_ff=64)

    def test_output_shape(self):
        model = Transformer(self._cfg(), rng=np.random.default_rng(0))
        src = _t((2, 6, 16))
        tgt = _t((2, 5, 16), seed=1)
        out, viz = model(src, tgt)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_viz_keys(self):
        model = Transformer(self._cfg(), rng=np.random.default_rng(0))
        _, viz = model(_t((1, 4, 16)), _t((1, 3, 16), seed=1))
        self.assertIn("encoder_weights", viz)
        self.assertIn("decoder_self_weights", viz)
        self.assertIn("decoder_cross_weights", viz)

    def test_viz_lengths(self):
        model = Transformer(self._cfg(enc=3, dec=2), rng=np.random.default_rng(0))
        _, viz = model(_t((1, 4, 16)), _t((1, 3, 16), seed=1))
        self.assertEqual(len(viz["encoder_weights"]), 3)
        self.assertEqual(len(viz["decoder_self_weights"]), 2)
        self.assertEqual(len(viz["decoder_cross_weights"]), 2)

    def test_backward_runs(self):
        model = Transformer(self._cfg(), rng=np.random.default_rng(0))
        src = _t((2, 6, 16))
        tgt = _t((2, 5, 16), seed=1)
        out, _ = model(src, tgt)
        t_sum(out).backward()
        self.assertIsNotNone(src.grad)
        self.assertIsNotNone(tgt.grad)

    def test_causal_mask_on_decoder(self):
        model = Transformer(self._cfg(), rng=np.random.default_rng(0))
        src = _t((1, 4, 16))
        tgt = _t((1, 4, 16), seed=1)
        _, viz = model(src, tgt, tgt_mask=CausalMask())
        for sw in viz["decoder_self_weights"]:
            for i in range(4):
                for j in range(i + 1, 4):
                    self.assertAlmostEqual(float(sw[0, 0, i, j]), 0.0, places=4)

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            Transformer(TransformerConfig(d_model=10, n_heads=3))

    def test_repr(self):
        r = repr(Transformer(self._cfg()))
        self.assertIn("16", r)

    def test_train_eval_propagates(self):
        model = Transformer(self._cfg(), rng=np.random.default_rng(0))
        model.eval()
        self.assertFalse(model.encoder._blocks[0].drop.training)
        model.train()
        self.assertTrue(model.encoder._blocks[0].drop.training)


if __name__ == "__main__":
    unittest.main()
