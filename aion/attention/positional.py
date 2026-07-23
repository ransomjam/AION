"""Positional encoding modules.

Positional encodings inject sequence-order information into token embeddings.
Without them, attention is permutation-invariant and cannot distinguish
"the cat sat" from "sat the cat".

Three strategies are provided:

SinusoidalPE (no parameters)
    Fixed sinusoidal encoding from Vaswani et al. (2017).  Precomputes a
    ``[max_len, d_model]`` table at construction.  ``forward(x)`` adds the
    slice ``PE[:seq_len]`` to ``x``.  No gradient flows through PE — it is a
    constant addition.  Appropriate for Transformer encoder/decoder.

LearnedPE (has parameters)
    A ``Parameter`` of shape ``[max_len, d_model]``, initialised from
    sinusoidal values so training starts from a sensible prior.  ``forward(x)``
    adds ``PE[:seq_len]`` to ``x``.  Gradient flows through PE — positions are
    learned.  Used by BERT-style models.

RotaryPE (stub — implementation deferred)
    Rotary Position Embedding (Su et al., 2021).  RoPE encodes position by
    rotating Q and K vectors rather than adding to embeddings; it interacts
    with the attention score computation directly and requires a ``rotate_half``
    op.  The interface is defined here so the Transformer milestone can
    implement it without architectural changes.  Calling ``forward`` raises
    ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np

from aion.nn.module import Module
from aion.nn.ops import add
from aion.nn.parameter import Parameter
from aion.nn.tensor import Tensor


def _sinusoidal_table(max_len: int, d_model: int) -> np.ndarray:
    """Compute the sinusoidal PE table of shape ``[max_len, d_model]``."""
    pos = np.arange(max_len)[:, np.newaxis]          # [max_len, 1]
    i = np.arange(0, d_model, 2)[np.newaxis, :]      # [1, d_model/2]
    angles = pos / np.power(10000.0, i / d_model)    # [max_len, d_model/2]
    table = np.zeros((max_len, d_model))
    table[:, 0::2] = np.sin(angles)
    table[:, 1::2] = np.cos(angles[:, : d_model // 2])
    return table


class SinusoidalPE(Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    No trainable parameters.  The encoding table is precomputed at
    construction and never updated.

    Parameters
    ----------
    d_model:
        Embedding dimensionality.  Must match the model's ``d_model``.
    max_len:
        Maximum sequence length the table is precomputed for.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Store as plain ndarray — not a Parameter; not differentiable.
        object.__setattr__(self, "_table", _sinusoidal_table(max_len, d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to ``x``.

        Parameters
        ----------
        x:
            Input of shape ``[batch, seq_len, d_model]``.
        """
        seq_len = x.shape[1]
        pe = Tensor(self._table[:seq_len], requires_grad=False)
        return add(x, pe)

    def __repr__(self) -> str:
        return f"SinusoidalPE(d_model={self.d_model}, max_len={self.max_len})"


class LearnedPE(Module):
    """Learned positional encoding.

    A trainable ``Parameter`` of shape ``[max_len, d_model]``, initialised
    from sinusoidal values so training starts from a sensible prior.

    Parameters
    ----------
    d_model:
        Embedding dimensionality.
    max_len:
        Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = Parameter(_sinusoidal_table(max_len, d_model), name="pe")

    def forward(self, x: Tensor) -> Tensor:
        """Add learned positional encoding to ``x``.

        Parameters
        ----------
        x:
            Input of shape ``[batch, seq_len, d_model]``.
        """
        from aion.nn.ops import embedding_lookup
        seq_len = x.shape[1]
        ids = np.arange(seq_len, dtype=np.intp)
        pe_slice = embedding_lookup(self.pe, ids)   # [seq_len, d_model]
        return add(x, pe_slice)

    def __repr__(self) -> str:
        return f"LearnedPE(d_model={self.d_model}, max_len={self.max_len})"


class RotaryPE(Module):
    """Rotary Position Embedding — interface stub (implementation deferred).

    RoPE (Su et al., 2021) encodes position by rotating Q and K vectors in
    the attention score computation rather than adding to token embeddings.
    It requires a ``rotate_half`` op and couples to ``ScaledDotProductAttention``
    directly.  The interface is defined here; implementation lands at the
    Transformer milestone when it is actually needed.

    Parameters
    ----------
    d_head:
        Per-head dimensionality (``d_model // n_heads``).
    max_len:
        Maximum sequence length.
    """

    def __init__(self, d_head: int, max_len: int = 4096) -> None:
        super().__init__()
        self.d_head = d_head
        self.max_len = max_len

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError(
            "RotaryPE is a stub.  Implementation is deferred to the Transformer milestone."
        )

    def __repr__(self) -> str:
        return f"RotaryPE(d_head={self.d_head}, max_len={self.max_len})"
