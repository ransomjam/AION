"""Attention mask abstractions.

Masks are applied inside ``ScaledDotProductAttention`` as additive biases
*before* the softmax.  Adding a large negative value (-1e9) to a position
drives its softmax weight to zero without breaking the gradient graph.

Why additive, not multiplicative?
----------------------------------
Zeroing weights *after* softmax (multiplicative) produces an incorrect
probability distribution — the remaining weights no longer sum to 1 — and
breaks gradient flow through masked positions.  Additive masking is
numerically correct and fully differentiable.

Masks return plain ``np.ndarray`` bias tensors, not ``Tensor`` objects.
``ScaledDotProductAttention`` wraps them in a ``Tensor(requires_grad=False)``
before adding to the scores.  Mask positions are not learned; they must not
appear in the gradient graph.

Shape convention
----------------
All bias arrays have shape ``[1, 1, seq_q, seq_k]`` so they broadcast
correctly over ``[batch, n_heads, seq_q, seq_k]`` score tensors.
``PaddingMask`` produces ``[batch, 1, 1, seq_k]`` to vary per sample.
"""

from __future__ import annotations

import numpy as np

_NEG_INF = -1e9


class AttentionMask:
    """Base class for attention masks.

    Subclasses implement ``bias(seq_q, seq_k, batch_size)`` and return a
    ``np.ndarray`` that is added to the raw attention scores before softmax.
    """

    def bias(self, seq_q: int, seq_k: int, batch_size: int = 1) -> np.ndarray:
        """Return an additive bias array for the given sequence lengths.

        Returns
        -------
        np.ndarray
            Shape broadcastable to ``[batch, n_heads, seq_q, seq_k]``.
            Zero where attention is allowed; ``_NEG_INF`` where it is blocked.
        """
        raise NotImplementedError


class CausalMask(AttentionMask):
    """Lower-triangular causal mask for autoregressive (decoder) attention.

    Position ``i`` may only attend to positions ``j <= i``.  The upper
    triangle is filled with ``_NEG_INF``; the lower triangle (including the
    diagonal) is zero.

    Shape: ``[1, 1, seq_q, seq_k]`` — identical for every batch element and
    every head.

    Example (seq=4)::

        [[0,   -inf, -inf, -inf],
         [0,   0,    -inf, -inf],
         [0,   0,    0,    -inf],
         [0,   0,    0,    0   ]]
    """

    def bias(self, seq_q: int, seq_k: int, batch_size: int = 1) -> np.ndarray:
        mask = np.triu(np.full((seq_q, seq_k), _NEG_INF), k=1)
        return mask[np.newaxis, np.newaxis, :, :]   # [1, 1, seq_q, seq_k]


class PaddingMask(AttentionMask):
    """Per-batch padding mask for variable-length sequences.

    Parameters
    ----------
    valid:
        Boolean array of shape ``[batch, seq_k]``.  ``True`` means the
        position is a real token; ``False`` means it is padding and should
        receive ``_NEG_INF``.

    Shape: ``[batch, 1, 1, seq_k]`` — varies per batch element, broadcasts
    over all heads and all query positions.
    """

    def __init__(self, valid: np.ndarray) -> None:
        self._valid = np.asarray(valid, dtype=bool)   # [batch, seq_k]

    def bias(self, seq_q: int, seq_k: int, batch_size: int = 1) -> np.ndarray:
        # [batch, seq_k] → [batch, 1, 1, seq_k]
        b = np.where(self._valid, 0.0, _NEG_INF)
        return b[:, np.newaxis, np.newaxis, :]
