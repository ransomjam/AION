"""Transformer block abstractions.

Hierarchy
---------
TransformerBlock (abstract base)
    TransformerEncoderBlock  — self-attention + FFN
    TransformerDecoderBlock  — causal self-attention + cross-attention + FFN

Both blocks support pre-norm (default) and post-norm via ``pre_norm``.

Pre-norm structure (encoder)::

    x_n = norm1(x)
    x   = x + dropout(attn(x_n, x_n, x_n, mask))
    x   = x + dropout(ffn(norm2(x)))

Post-norm structure (encoder)::

    x = norm1(x + dropout(attn(x, x, x, mask)))
    x = norm2(x + dropout(ffn(x)))

Both blocks return ``(output, attn_weights)`` so stacks can collect per-block
attention arrays for visualization without re-running the forward pass.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from aion.attention.attention import MultiHeadAttention
from aion.attention.mask import AttentionMask
from aion.nn.module import Module
from aion.nn.ops import add
from aion.nn.tensor import Tensor

from .dropout import Dropout
from .ffn import FeedForward
from .norm import LayerNorm


class TransformerBlock(Module):
    """Abstract base for all transformer block variants.

    Establishes the shared constructor signature and the ``d_model`` /
    ``n_heads`` introspection properties used by stacks and the store.
    Subclasses implement ``forward``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        pre_norm: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self._d_model = d_model
        self._n_heads = n_heads
        self.pre_norm = pre_norm

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_heads(self) -> int:
        return self._n_heads

    @abstractmethod
    def forward(self, x: Tensor, *args, **kwargs):
        ...


class TransformerEncoderBlock(TransformerBlock):
    """One encoder layer: self-attention + FFN with residuals and layer norm.

    Parameters
    ----------
    d_model:
        Model dimensionality.
    n_heads:
        Number of attention heads.
    d_ff:
        Feed-forward inner dimensionality.
    dropout:
        Dropout probability.
    activation:
        FFN activation: ``"relu"`` or ``"gelu"``.
    pre_norm:
        If ``True`` (default), apply LayerNorm before each sub-layer.
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        pre_norm: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(d_model, n_heads, d_ff, dropout, activation, pre_norm, rng)
        self.norm1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, activation=activation,
                               dropout=dropout, rng=rng)
        self.drop = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: AttentionMask | None = None,
    ) -> tuple[Tensor, np.ndarray]:
        """Apply one encoder block.

        Parameters
        ----------
        x:
            Input of shape ``[batch, seq, d_model]``.
        mask:
            Optional ``AttentionMask`` for self-attention.

        Returns
        -------
        output:
            Shape ``[batch, seq, d_model]``.
        attn_weights:
            Detached ``np.ndarray`` of shape ``[batch, n_heads, seq, seq]``.
        """
        if self.pre_norm:
            x_n = self.norm1(x)
            attn_out, weights = self.attn(x_n, x_n, x_n, mask)
            x = add(x, self.drop(attn_out))
            x = add(x, self.drop(self.ffn(self.norm2(x))))
        else:
            attn_out, weights = self.attn(x, x, x, mask)
            x = self.norm1(add(x, self.drop(attn_out)))
            x = self.norm2(add(x, self.drop(self.ffn(x))))
        return x, weights

    def __repr__(self) -> str:
        return (f"TransformerEncoderBlock(d_model={self._d_model}, "
                f"n_heads={self._n_heads}, pre_norm={self.pre_norm})")


class TransformerDecoderBlock(TransformerBlock):
    """One decoder layer: causal self-attention + cross-attention + FFN.

    Parameters
    ----------
    d_model:
        Model dimensionality.
    n_heads:
        Number of attention heads.
    d_ff:
        Feed-forward inner dimensionality.
    dropout:
        Dropout probability.
    activation:
        FFN activation: ``"relu"`` or ``"gelu"``.
    pre_norm:
        If ``True`` (default), apply LayerNorm before each sub-layer.
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        pre_norm: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(d_model, n_heads, d_ff, dropout, activation, pre_norm, rng)
        self.norm1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.norm2 = LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, rng=rng)
        self.norm3 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, activation=activation,
                               dropout=dropout, rng=rng)
        self.drop = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        self_mask: AttentionMask | None = None,
        cross_mask: AttentionMask | None = None,
    ) -> tuple[Tensor, np.ndarray, np.ndarray]:
        """Apply one decoder block.

        Parameters
        ----------
        x:
            Decoder input, shape ``[batch, tgt_seq, d_model]``.
        encoder_output:
            Encoder output, shape ``[batch, src_seq, d_model]``.
        self_mask:
            Mask for causal self-attention (typically ``CausalMask``).
        cross_mask:
            Mask for cross-attention (typically ``PaddingMask`` on source).

        Returns
        -------
        output:
            Shape ``[batch, tgt_seq, d_model]``.
        self_weights:
            Detached ``np.ndarray``, shape ``[batch, n_heads, tgt_seq, tgt_seq]``.
        cross_weights:
            Detached ``np.ndarray``, shape ``[batch, n_heads, tgt_seq, src_seq]``.
        """
        if self.pre_norm:
            x_n = self.norm1(x)
            sa_out, self_w = self.self_attn(x_n, x_n, x_n, self_mask)
            x = add(x, self.drop(sa_out))

            x_n = self.norm2(x)
            ca_out, cross_w = self.cross_attn(x_n, encoder_output, encoder_output, cross_mask)
            x = add(x, self.drop(ca_out))

            x = add(x, self.drop(self.ffn(self.norm3(x))))
        else:
            sa_out, self_w = self.self_attn(x, x, x, self_mask)
            x = self.norm1(add(x, self.drop(sa_out)))

            ca_out, cross_w = self.cross_attn(x, encoder_output, encoder_output, cross_mask)
            x = self.norm2(add(x, self.drop(ca_out)))

            x = self.norm3(add(x, self.drop(self.ffn(x))))
        return x, self_w, cross_w

    def __repr__(self) -> str:
        return (f"TransformerDecoderBlock(d_model={self._d_model}, "
                f"n_heads={self._n_heads}, pre_norm={self.pre_norm})")
