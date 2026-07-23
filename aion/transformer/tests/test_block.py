"""Tests for TransformerEncoderBlock and TransformerDecoderBlock."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.attention.mask import CausalMask
from aion.transformer.block import TransformerEncoderBlock, TransformerDecoderBlock


def _t(shape, seed=0, requires_grad=True):
    return Tensor(np.random.default_rng(seed).standard_normal(shape),
                  requires_grad=requires_grad)


class TestTransformerEncoderBlock(unittest.TestCase):

    def _block(self, d=16, h=4, d_ff=64, **kw):
        return TransformerEncoderBlock(d, h, d_ff, rng=np.random.default_rng(0), **kw)

    def test_output_shape(self):
        out, _ = self._block()(_t((2, 5, 16)))
        self.assertEqual(out.shape, (2, 5, 16))

    def test_weights_shape(self):
        _, w = self._block()(_t((2, 5, 16)))
        self.assertEqual(w.shape, (2, 4, 5, 5))

    def test_weights_detached(self):
        _, w = self._block()(_t((1, 4, 16)))
        self.assertIsInstance(w, np.ndarray)

    def test_backward_runs(self):
        blk = self._block()
        x = _t((2, 5, 16))
        t_sum(blk(x)[0]).backward()
        self.assertIsNotNone(x.grad)
        for p in blk.parameters():
            self.assertIsNotNone(p.grad)

    def test_with_causal_mask(self):
        _, w = self._block()(_t((1, 4, 16)), mask=CausalMask())
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(float(w[0, 0, i, j]), 0.0, places=4)

    def test_post_norm(self):
        out, _ = self._block(pre_norm=False)(_t((2, 5, 16)))
        self.assertEqual(out.shape, (2, 5, 16))

    def test_parameter_count(self):
        # 4 MHA + 4 FFN + 2*LN(2 each) = 12
        self.assertEqual(len(self._block().parameters()), 12)

    def test_repr(self):
        self.assertIn("16", repr(self._block()))


class TestTransformerDecoderBlock(unittest.TestCase):

    def _block(self, d=16, h=4, d_ff=64, **kw):
        return TransformerDecoderBlock(d, h, d_ff, rng=np.random.default_rng(0), **kw)

    def _enc(self, batch=2, src=7):
        return _t((batch, src, 16), seed=1, requires_grad=False)

    def test_output_shape(self):
        out, _, _ = self._block()(_t((2, 5, 16)), self._enc())
        self.assertEqual(out.shape, (2, 5, 16))

    def test_self_weights_shape(self):
        _, sw, _ = self._block()(_t((2, 5, 16)), self._enc())
        self.assertEqual(sw.shape, (2, 4, 5, 5))

    def test_cross_weights_shape(self):
        _, _, cw = self._block()(_t((2, 5, 16)), self._enc())
        self.assertEqual(cw.shape, (2, 4, 5, 7))

    def test_weights_detached(self):
        _, sw, cw = self._block()(_t((1, 3, 16)), self._enc(batch=1))
        self.assertIsInstance(sw, np.ndarray)
        self.assertIsInstance(cw, np.ndarray)

    def test_backward_runs(self):
        blk = self._block()
        x = _t((2, 5, 16))
        t_sum(blk(x, self._enc())[0]).backward()
        self.assertIsNotNone(x.grad)

    def test_causal_self_mask(self):
        _, sw, _ = self._block()(_t((1, 4, 16)), self._enc(batch=1, src=4),
                                 self_mask=CausalMask())
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(float(sw[0, 0, i, j]), 0.0, places=4)

    def test_parameter_count(self):
        # 2 MHA (4 each) + FFN (4) + 3 LN (2 each) = 18
        self.assertEqual(len(self._block().parameters()), 18)


if __name__ == "__main__":
    unittest.main()
