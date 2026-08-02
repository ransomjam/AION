"""Causality contracts for the GPT stack.

These exist because a real bug shipped past the previous tests.  The head-merge
in ``MultiHeadAttention`` reshaped ``[batch*H, seq, D]`` straight to
``[batch, seq, d_model]`` without the transpose that inverts the split.  That
reinterprets the (head, position) axes as (position, head), so output position
i read head 0 at positions i..i+H-1 — future positions, past the causal mask.

The model therefore scored well under teacher forcing (it could see part of the
answer) and generated nonsense at inference (the answer is not there yet).

Why the old tests passed
------------------------
``test_causal_mask_blocks_future`` exercised ``ScaledDotProductAttention``
directly and never touched the merge.  ``test_causal_mask_applied`` used
``n_heads=2``, perturbed position 4, and checked position 0 — but under the
scramble position 0 only reads positions 0..n_heads-1, so a perturbation at
position 4 could not reach it.  It passed by coincidence of index choice.

The tests below are written so that coincidence cannot save them: they perturb
EVERY position and check EVERY earlier one, and they use ``n_heads`` values
where the scramble reaches across the positions being checked.
"""

from __future__ import annotations

import unittest

import numpy as np

from aion import backend
from aion.gpt.config import GPTConfig
from aion.gpt.model import GPTModel


def _logits(model, ids) -> np.ndarray:
    out, _ = model(np.asarray(ids, dtype=np.int32))
    return backend.to_host(out.data)


class TestCausality(unittest.TestCase):
    """No information may flow from a position to any earlier position."""

    def _model(self, n_heads=4, n_layers=2, seq=12):
        cfg = GPTConfig(vocab_size=64, d_model=32, n_heads=n_heads,
                        n_layers=n_layers, d_ff=64, max_seq_len=seq, dropout=0.0)
        model = GPTModel(cfg, rng=np.random.default_rng(0))
        model.eval()
        return model

    def test_no_future_information_reaches_any_earlier_position(self):
        """Perturb every position; every strictly earlier position must be unmoved.

        The exhaustive sweep is the point.  Checking one (perturbed, observed)
        pair is what let the original bug through.
        """
        seq = 10
        model = self._model(seq=seq)
        base = np.arange(seq, dtype=np.int32).reshape(1, seq)
        reference = _logits(model, base)

        for j in range(1, seq):
            altered = base.copy()
            altered[0, j] = 63                     # a token not otherwise present
            changed = _logits(model, altered)
            delta = np.abs(reference[0, :j] - changed[0, :j]).max()
            self.assertEqual(
                delta, 0.0,
                f"editing position {j} moved the logits at positions 0..{j-1} "
                f"by {delta:.3e}; the causal mask is leaking",
            )

    def test_prefix_consistency(self):
        """logits(x[:k]) must equal logits(x)[:k] for every k.

        The sharpest causality check available: if a shorter input changes the
        answer for tokens both inputs share, the model was reading ahead.  It is
        also exactly the property generation depends on, since generation only
        ever has the prefix.
        """
        seq = 10
        model = self._model(seq=seq)
        base = np.arange(seq, dtype=np.int32).reshape(1, seq)
        full = _logits(model, base)

        for k in range(1, seq):
            prefix = _logits(model, base[:, :k])
            delta = np.abs(full[0, :k] - prefix[0]).max()
            # float32 reassociation in the shorter reduction, nothing more.
            self.assertLess(
                delta, 1e-5,
                f"logits for the first {k} tokens changed by {delta:.3e} when the "
                f"rest of the sequence was removed",
            )

    def test_causality_holds_for_many_head_counts(self):
        """The scramble's reach depended on n_heads, so sweep n_heads."""
        seq = 12
        for n_heads in (1, 2, 4, 8, 16):   # every divisor of d_model=32 up to 16
            model = self._model(n_heads=n_heads, seq=seq)
            base = np.arange(seq, dtype=np.int32).reshape(1, seq)
            reference = _logits(model, base)
            altered = base.copy()
            altered[0, 1] = 63                     # position 1: inside every scramble
            changed = _logits(model, altered)
            delta = np.abs(reference[0, 0] - changed[0, 0]).max()
            self.assertEqual(delta, 0.0,
                             f"n_heads={n_heads}: position 0 saw position 1")


class TestHeadMergeInvertsSplit(unittest.TestCase):
    """The merge must be the exact inverse of the split."""

    def test_merge_round_trips_split(self):
        from aion.attention.attention import MultiHeadAttention
        from aion.nn.ops import reshape, transpose
        from aion.nn.tensor import Tensor

        batch, H, seq, D = 2, 4, 6, 3
        d_model = H * D
        # Tag every element with its (batch, seq, feature) identity, split it,
        # then merge it back.  Anything but the identity means axes were mixed.
        original = np.arange(batch * seq * d_model, dtype=np.float32)
        original = original.reshape(batch, seq, d_model)

        split = MultiHeadAttention._split_heads(Tensor(original), batch, seq, H, D)
        flat = reshape(split, (batch * H, seq, D))
        unflat = reshape(flat, (batch, H, seq, D))
        merged = reshape(transpose(unflat, (0, 2, 1, 3)), (batch, seq, d_model))

        np.testing.assert_array_equal(backend.to_host(merged.data), original)

    def test_naive_reshape_is_not_the_inverse(self):
        """Pin the actual defect, so nobody 'simplifies' the transpose away."""
        batch, H, seq, D = 2, 4, 6, 3
        src = np.arange(batch * H * seq * D, dtype=np.float32).reshape(batch * H, seq, D)
        naive = src.reshape(batch, seq, H * D)
        correct = src.reshape(batch, H, seq, D).transpose(0, 2, 1, 3).reshape(batch, seq, H * D)
        self.assertFalse(np.array_equal(naive, correct),
                         "if these ever match, this test has stopped testing anything")


if __name__ == "__main__":
    unittest.main()
