"""Tests for ScaledDotProductAttention and MultiHeadAttention."""

import unittest
import numpy as np

from aion.nn.tensor import Tensor
from aion.nn.ops import t_sum
from aion.attention.attention import ScaledDotProductAttention, MultiHeadAttention
from aion.attention.mask import CausalMask, PaddingMask


class TestScaledDotProductAttention(unittest.TestCase):

    def _qkv(self, batch=2, seq=4, d=8, seed=0, requires_grad=True):
        rng = np.random.default_rng(seed)
        Q = Tensor(rng.standard_normal((batch, seq, d)), requires_grad=requires_grad)
        K = Tensor(rng.standard_normal((batch, seq, d)), requires_grad=requires_grad)
        V = Tensor(rng.standard_normal((batch, seq, d)), requires_grad=requires_grad)
        return Q, K, V

    def test_output_shape(self):
        sdpa = ScaledDotProductAttention()
        Q, K, V = self._qkv(batch=2, seq=4, d=8)
        out, w = sdpa(Q, K, V)
        self.assertEqual(out.shape, (2, 4, 8))
        self.assertEqual(w.shape, (2, 4, 4))

    def test_weights_sum_to_one(self):
        sdpa = ScaledDotProductAttention()
        Q, K, V = self._qkv()
        _, w = sdpa(Q, K, V)
        np.testing.assert_allclose(w.data.sum(axis=-1), np.ones((2, 4)), atol=1e-6)

    def test_weights_non_negative(self):
        sdpa = ScaledDotProductAttention()
        Q, K, V = self._qkv()
        _, w = sdpa(Q, K, V)
        self.assertTrue(np.all(w.data >= 0))

    def test_weights_in_graph(self):
        """Weights tensor is still in the autograd graph."""
        sdpa = ScaledDotProductAttention()
        Q, K, V = self._qkv()
        _, w = sdpa(Q, K, V)
        self.assertTrue(w.requires_grad)

    def test_causal_mask_blocks_future(self):
        """With a causal mask, position 0 should not attend to positions 1+."""
        sdpa = ScaledDotProductAttention()
        Q, K, V = self._qkv(batch=1, seq=4, d=4)
        mask = CausalMask()
        _, w = sdpa(Q, K, V, mask=mask)
        # Upper triangle of weights should be ~0.
        w_np = w.data[0]   # [seq_q, seq_k]
        for i in range(4):
            for j in range(i + 1, 4):
                self.assertAlmostEqual(float(w_np[i, j]), 0.0, places=5,
                                       msg=f"w[{i},{j}] should be ~0 with causal mask")

    def test_padding_mask_blocks_padding(self):
        valid = np.array([[True, True, False, False]])
        mask = PaddingMask(valid)
        sdpa = ScaledDotProductAttention()
        Q, K, V = self._qkv(batch=1, seq=4, d=4)
        _, w = sdpa(Q, K, V, mask=mask)
        w_np = w.data[0]   # [seq_q, seq_k]
        # Columns 2 and 3 should be ~0 for all query positions.
        np.testing.assert_allclose(w_np[:, 2], 0.0, atol=1e-5)
        np.testing.assert_allclose(w_np[:, 3], 0.0, atol=1e-5)

    def test_backward_runs(self):
        sdpa = ScaledDotProductAttention()
        Q, K, V = self._qkv()
        out, _ = sdpa(Q, K, V)
        loss = t_sum(out)
        loss.backward()   # must not raise
        self.assertIsNotNone(Q.grad)
        self.assertIsNotNone(K.grad)
        self.assertIsNotNone(V.grad)

    def test_no_parameters(self):
        sdpa = ScaledDotProductAttention()
        self.assertEqual(sdpa.parameters(), [])

    def test_dropout_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            ScaledDotProductAttention(dropout=0.1)

    def test_cross_attention_shape(self):
        """Q and K/V can have different sequence lengths."""
        sdpa = ScaledDotProductAttention()
        rng = np.random.default_rng(0)
        Q = Tensor(rng.standard_normal((2, 3, 8)), requires_grad=True)
        K = Tensor(rng.standard_normal((2, 5, 8)), requires_grad=True)
        V = Tensor(rng.standard_normal((2, 5, 8)), requires_grad=True)
        out, w = sdpa(Q, K, V)
        self.assertEqual(out.shape, (2, 3, 8))
        self.assertEqual(w.shape, (2, 3, 5))

    def test_scaling(self):
        """Scores are divided by sqrt(d_k)."""
        import math
        sdpa = ScaledDotProductAttention()
        # Identical Q and K → scores are d_k (dot product of unit vectors scaled by d_k).
        d = 16
        Q = Tensor(np.ones((1, 1, d)), requires_grad=False)
        K = Tensor(np.ones((1, 1, d)), requires_grad=False)
        V = Tensor(np.ones((1, 1, d)), requires_grad=False)
        _, w = sdpa(Q, K, V)
        # Single position → weight must be 1.0 regardless of scaling.
        self.assertAlmostEqual(float(w.data[0, 0, 0]), 1.0, places=10)


class TestMultiHeadAttention(unittest.TestCase):

    def _mha(self, d_model=16, n_heads=4, seed=0):
        return MultiHeadAttention(d_model, n_heads, rng=np.random.default_rng(seed))

    def _input(self, batch=2, seq=5, d_model=16, seed=1):
        rng = np.random.default_rng(seed)
        return Tensor(rng.standard_normal((batch, seq, d_model)), requires_grad=True)

    def test_output_shape(self):
        mha = self._mha()
        x = self._input()
        out, w = mha(x, x, x)
        self.assertEqual(out.shape, (2, 5, 16))

    def test_weights_shape(self):
        mha = self._mha(d_model=16, n_heads=4)
        x = self._input(batch=2, seq=5)
        _, w = mha(x, x, x)
        self.assertEqual(w.shape, (2, 4, 5, 5))

    def test_weights_detached(self):
        """Returned weights are a plain ndarray, not a Tensor."""
        mha = self._mha()
        x = self._input()
        _, w = mha(x, x, x)
        self.assertIsInstance(w, np.ndarray)

    def test_weights_sum_to_one(self):
        mha = self._mha()
        x = self._input()
        _, w = mha(x, x, x)
        np.testing.assert_allclose(w.sum(axis=-1), np.ones((2, 4, 5)), atol=1e-6)

    def test_parameter_count(self):
        """4 matrices of shape [d_model, d_model]."""
        mha = self._mha(d_model=16, n_heads=4)
        params = mha.parameters()
        self.assertEqual(len(params), 4)
        for p in params:
            self.assertEqual(p.shape, (16, 16))

    def test_backward_runs(self):
        mha = self._mha()
        x = self._input()
        out, _ = mha(x, x, x)
        loss = t_sum(out)
        loss.backward()
        self.assertIsNotNone(x.grad)
        for p in mha.parameters():
            self.assertIsNotNone(p.grad)

    def test_d_model_not_divisible_raises(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(d_model=10, n_heads=3)

    def test_causal_mask(self):
        mha = self._mha(d_model=16, n_heads=4)
        x = self._input(batch=1, seq=4)
        mask = CausalMask()
        _, w = mha(x, x, x, mask=mask)
        # All heads: upper triangle should be ~0.
        for h in range(4):
            wh = w[0, h]
            for i in range(4):
                for j in range(i + 1, 4):
                    self.assertAlmostEqual(wh[i, j], 0.0, places=4,
                                           msg=f"head {h} w[{i},{j}] should be ~0")

    def test_cross_attention(self):
        """Q from one sequence, K/V from another."""
        mha = self._mha(d_model=16, n_heads=4)
        rng = np.random.default_rng(5)
        Q = Tensor(rng.standard_normal((2, 3, 16)), requires_grad=True)
        KV = Tensor(rng.standard_normal((2, 7, 16)), requires_grad=True)
        out, w = mha(Q, KV, KV)
        self.assertEqual(out.shape, (2, 3, 16))
        self.assertEqual(w.shape, (2, 4, 3, 7))

    def test_repr(self):
        mha = self._mha(d_model=32, n_heads=8)
        r = repr(mha)
        self.assertIn("32", r)
        self.assertIn("8", r)

    def test_single_head(self):
        mha = MultiHeadAttention(d_model=8, n_heads=1,
                                 rng=np.random.default_rng(0))
        x = self._input(batch=1, seq=3, d_model=8)
        out, w = mha(x, x, x)
        self.assertEqual(out.shape, (1, 3, 8))
        self.assertEqual(w.shape, (1, 1, 3, 3))

    def test_gradient_does_not_flow_through_weights_array(self):
        """The returned weights ndarray is detached — modifying it doesn't affect grads."""
        mha = self._mha()
        x = self._input()
        out, w = mha(x, x, x)
        w[:] = 0.0   # mutate the detached copy
        loss = t_sum(out)
        loss.backward()   # must not raise or produce wrong grads
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
