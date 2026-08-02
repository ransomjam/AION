"""Scaled dot-product attention and multi-head attention.

ScaledDotProductAttention
    The atomic attention operation.  No trainable parameters.  Composes
    directly with any mask.  Returns ``(output, weights)`` so callers can
    inspect the attention distribution without re-running the forward pass.

MultiHeadAttention
    Wraps ``ScaledDotProductAttention`` with four projection matrices
    (W_Q, W_K, W_V, W_O).  Splits ``d_model`` into ``n_heads`` independent
    heads, runs attention on each, concatenates, and projects back.

    Supports both self-attention (Q = K = V = same input) and cross-attention
    (Q from one sequence, K/V from another) — the module does not distinguish
    them; the caller passes the appropriate tensors.

Shape convention
----------------
All inputs and outputs use ``[batch, seq_len, d_model]``.  Internally,
multi-head attention reshapes to ``[batch, n_heads, seq_len, d_head]`` for
the per-head computation, then reshapes back.

Attention weights
-----------------
``ScaledDotProductAttention.forward`` returns the softmax weight tensor
(still in the computation graph).  ``MultiHeadAttention.forward`` returns
all head weights stacked as a plain ``np.ndarray`` of shape
``[batch, n_heads, seq_q, seq_k]`` — detached from the graph via
``.data.copy()``.  Visualization code operates on this array; it never
touches the autograd graph.
"""

from __future__ import annotations

import math

import numpy as np

from aion.nn.init import xavier_uniform
from aion.nn.module import Module
from aion.nn.ops import add, matmul, reshape, softmax, transpose
from aion.nn.parameter import Parameter
from aion.nn.tensor import Tensor

from .mask import AttentionMask


class _PrecomputedMask(AttentionMask):
    """Internal adapter wrapping a pre-expanded bias array.

    Used by ``MultiHeadAttention`` to pass an already-broadcast
    ``[batch*heads, seq_q, seq_k]`` bias into ``ScaledDotProductAttention``
    without re-computing it.
    """

    def __init__(self, bias: np.ndarray) -> None:
        self._bias = bias   # [batch*heads, seq_q, seq_k]

    def bias(self, seq_q: int, seq_k: int, batch_size: int = 1) -> np.ndarray:
        return self._bias


class ScaledDotProductAttention(Module):
    """Scaled dot-product attention (Vaswani et al., 2017).

    Computes::

        scores  = Q @ K^T / sqrt(d_k)          [batch, seq_q, seq_k]
        scores += mask.bias(...)                (if mask is provided)
        weights = softmax(scores, axis=-1)      [batch, seq_q, seq_k]
        output  = weights @ V                   [batch, seq_q, d_v]

    No trainable parameters.  Projection is the caller's responsibility
    (``MultiHeadAttention`` provides it).

    Parameters
    ----------
    dropout:
        Reserved for future use.  Not implemented; present for interface
        compatibility with future Transformer milestones.
    """

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        if dropout != 0.0:
            raise NotImplementedError("Attention dropout is not yet implemented.")

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: AttentionMask | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute scaled dot-product attention.

        Parameters
        ----------
        Q:
            Query tensor, shape ``[batch, seq_q, d_k]``.
        K:
            Key tensor, shape ``[batch, seq_k, d_k]``.
        V:
            Value tensor, shape ``[batch, seq_k, d_v]``.
        mask:
            Optional ``AttentionMask``.  Its ``bias`` array is added to the
            raw scores before softmax.

        Returns
        -------
        output:
            Shape ``[batch, seq_q, d_v]``.
        weights:
            Softmax attention weights, shape ``[batch, seq_q, seq_k]``.
            Still in the computation graph — callers that only need the
            values for visualization should use ``.data.copy()``.
        """
        d_k = Q.shape[-1]
        scale = 1.0 / math.sqrt(d_k)

        # scores: [batch, seq_q, seq_k]
        # K^T via transpose of last two axes.
        ndim = K.ndim
        perm = tuple(range(ndim - 2)) + (ndim - 1, ndim - 2)
        K_t = transpose(K, perm)
        scores_raw = matmul(Q, K_t)

        # Scale: multiply by a non-grad constant tensor.
        scale_t = Tensor(np.array(scale), requires_grad=False)
        from aion.nn.ops import mul
        scores = mul(scores_raw, scale_t)

        # Additive mask.
        if mask is not None:
            batch, seq_q = Q.shape[0], Q.shape[1]
            seq_k = K.shape[1]
            bias_np = mask.bias(seq_q, seq_k, batch_size=batch)
            # Reshape bias to match scores ndim by removing leading size-1 axes.
            # scores: [batch, seq_q, seq_k] (3-D)
            # CausalMask:  [1, 1, seq_q, seq_k] → squeeze axis 0 → [1, seq_q, seq_k]
            # PaddingMask: [batch, 1, 1, seq_k] → squeeze axis 1 → [batch, 1, seq_k]
            while bias_np.ndim > scores.ndim:
                # Find the first axis of size 1 and remove it.
                ax = next(i for i, s in enumerate(bias_np.shape) if s == 1)
                bias_np = np.squeeze(bias_np, axis=ax)
            bias_t = Tensor(bias_np, requires_grad=False)
            scores = add(scores, bias_t)

        weights = softmax(scores, axis=-1)
        output = matmul(weights, V)
        return output, weights

    def __repr__(self) -> str:
        return "ScaledDotProductAttention()"


class MultiHeadAttention(Module):
    """Multi-head attention with learned Q, K, V, and output projections.

    Splits ``d_model`` into ``n_heads`` independent heads of size
    ``d_head = d_model // n_heads``, runs ``ScaledDotProductAttention`` on
    each head in parallel (via batched matmul), concatenates, and projects
    back to ``d_model``.

    Supports self-attention and cross-attention transparently:

    - Self-attention:  ``forward(x, x, x)``
    - Cross-attention: ``forward(query_seq, key_value_seq, key_value_seq)``

    Parameters
    ----------
    d_model:
        Model dimensionality.  Must be divisible by ``n_heads``.
    n_heads:
        Number of attention heads.
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        rng = rng or np.random.default_rng()
        self.W_Q = Parameter(xavier_uniform(d_model, d_model, rng=rng), name="W_Q")
        self.W_K = Parameter(xavier_uniform(d_model, d_model, rng=rng), name="W_K")
        self.W_V = Parameter(xavier_uniform(d_model, d_model, rng=rng), name="W_V")
        self.W_O = Parameter(xavier_uniform(d_model, d_model, rng=rng), name="W_O")
        self._sdpa = ScaledDotProductAttention()

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: AttentionMask | None = None,
    ) -> tuple[Tensor, np.ndarray]:
        """Compute multi-head attention.

        Parameters
        ----------
        Q:
            Query input, shape ``[batch, seq_q, d_model]``.
        K:
            Key input, shape ``[batch, seq_k, d_model]``.
        V:
            Value input, shape ``[batch, seq_k, d_model]``.
        mask:
            Optional ``AttentionMask`` applied inside each head.

        Returns
        -------
        output:
            Shape ``[batch, seq_q, d_model]``.
        all_weights:
            Plain ``np.ndarray`` of shape ``[batch, n_heads, seq_q, seq_k]``.
            Detached from the autograd graph — safe for visualization.
        """
        batch, seq_q, _ = Q.shape
        seq_k = K.shape[1]
        H, D = self.n_heads, self.d_head

        # Linear projections: [batch, seq, d_model]
        Q_proj = matmul(Q, self.W_Q)
        K_proj = matmul(K, self.W_K)
        V_proj = matmul(V, self.W_V)

        # Split into heads: [batch, seq, d_model] → [batch, n_heads, seq, d_head]
        Q_h = self._split_heads(Q_proj, batch, seq_q, H, D)
        K_h = self._split_heads(K_proj, batch, seq_k, H, D)
        V_h = self._split_heads(V_proj, batch, seq_k, H, D)

        # Merge batch and head dims for a single batched SDPA call:
        # [batch, n_heads, seq, d_head] → [batch*n_heads, seq, d_head]
        Q_flat = reshape(Q_h, (batch * H, seq_q, D))
        K_flat = reshape(K_h, (batch * H, seq_k, D))
        V_flat = reshape(V_h, (batch * H, seq_k, D))

        # Expand the mask bias so it broadcasts over the flattened batch*head
        # axis.  A [1, seq_q, seq_k] bias broadcasts over [batch*H, ...] on its
        # own, so a mask that is identical for every batch element and head — a
        # causal mask, the only kind a decoder uses — needs no expansion at all.
        #
        # Materialising it was expensive out of proportion to its content: it is
        # a constant triangle, but `broadcast_to(...).reshape(...)` forces a real
        # [batch*H, seq_q, seq_k] array, which is then copied host-to-device once
        # per layer per step.  At batch 64, seq 512, 6 layers that is 3.2 GB of
        # PCIe traffic every step — more wall-clock than the arithmetic it feeds.
        flat_mask = None
        if mask is not None:
            bias_4d = mask.bias(seq_q, seq_k, batch_size=batch)  # [b,1,1,sk] or [1,1,sq,sk]
            if bias_4d.shape[0] == 1:
                # Same for every batch element (causal): drop the leading axes and
                # let broadcasting do the rest.  No copy, no per-layer transfer.
                bias_flat = bias_4d.reshape(1, bias_4d.shape[2], bias_4d.shape[3])
            else:
                # Per-sample (padding mask): genuinely varies, so expand it.
                bias_flat = np.broadcast_to(
                    bias_4d, (batch, H, seq_q, seq_k)
                ).reshape(batch * H, seq_q, seq_k)
            flat_mask = _PrecomputedMask(bias_flat)

        out_flat, w_flat = self._sdpa(Q_flat, K_flat, V_flat, flat_mask)
        # out_flat: [batch*n_heads, seq_q, d_head]
        # w_flat:   [batch*n_heads, seq_q, seq_k]

        # Collect weights (detached) before reshaping.
        all_weights = w_flat.data.copy().reshape(batch, H, seq_q, seq_k)

        # Merge heads: [batch*n_heads, seq_q, d_head] → [batch, seq_q, d_model].
        #
        # This MUST mirror _split_heads exactly, transpose included.  A direct
        # reshape from [batch*H, seq_q, D] to [batch, seq_q, d_model] is not the
        # inverse of the split: it reinterprets the (head, position) axes as
        # (position, head), so output position i ends up reading head 0 at
        # positions i..i+H-1.  That silently destroys the head structure AND
        # leaks future positions past the causal mask, which makes the model
        # score well during teacher-forced training and generate nonsense at
        # inference.  See test_causal_integrity.py.
        out_h = reshape(out_flat, (batch, H, seq_q, D))       # unflatten heads
        out_t = transpose(out_h, (0, 2, 1, 3))                # [batch, seq_q, H, D]
        out_merged = reshape(out_t, (batch, seq_q, self.d_model))

        # Output projection.
        output = matmul(out_merged, self.W_O)
        return output, all_weights

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _split_heads(x: Tensor, batch: int, seq: int, H: int, D: int) -> Tensor:
        """Reshape ``[batch, seq, d_model]`` → ``[batch, H, seq, D]``."""
        # [batch, seq, H, D]
        x_r = reshape(x, (batch, seq, H, D))
        # [batch, H, seq, D]
        return transpose(x_r, (0, 2, 1, 3))

    def __repr__(self) -> str:
        return (f"MultiHeadAttention(d_model={self.d_model}, "
                f"n_heads={self.n_heads}, d_head={self.d_head})")
