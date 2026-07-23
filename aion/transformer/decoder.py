"""TransformerDecoder — cross-attention decoder stack.

``TransformerDecoder`` stacks ``N`` ``TransformerDecoderBlock`` instances
(causal self-attention + cross-attention + FFN) and applies a final
``LayerNorm``.  It is the decoder half of an encoder-decoder model.

This is structurally distinct from ``TransformerStack``: each block has three
sub-layers instead of two, and ``forward`` requires an ``encoder_output``
argument.  It cannot share the ``TransformerStack`` implementation.

For decoder-only models (GPT-style), use ``TransformerStack`` with a
``CausalMask`` — there is no encoder output and no cross-attention.
"""

from __future__ import annotations

import numpy as np

from aion.attention.mask import AttentionMask
from aion.nn.module import Module
from aion.nn.tensor import Tensor

from .block import TransformerDecoderBlock
from .config import DecoderConfig, TransformerConfig
from .norm import LayerNorm


class TransformerDecoder(Module):
    """Cross-attention decoder stack (seq2seq decoder half).

    Parameters
    ----------
    d_model:
        Model dimensionality.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of decoder blocks.
    d_ff:
        Feed-forward inner dimensionality.
    dropout:
        Dropout probability.
    activation:
        FFN activation: ``"relu"`` or ``"gelu"``.
    pre_norm:
        Pre-norm (``True``) or post-norm (``False``).
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "relu",
        pre_norm: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        rng = rng or np.random.default_rng()
        for i in range(n_layers):
            setattr(self, f"block_{i}", TransformerDecoderBlock(
                d_model, n_heads, d_ff,
                dropout=dropout, activation=activation,
                pre_norm=pre_norm, rng=rng,
            ))
        object.__setattr__(self, "_blocks", [
            getattr(self, f"block_{i}") for i in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)

    @classmethod
    def from_config(
        cls,
        cfg: TransformerConfig | DecoderConfig,
        rng: np.random.Generator | None = None,
    ) -> "TransformerDecoder":
        """Construct from a config object."""
        n = cfg.n_layers if isinstance(cfg, DecoderConfig) else cfg.n_decoder_layers
        return cls(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=n,
            d_ff=cfg.d_ff, dropout=cfg.dropout, activation=cfg.activation,
            pre_norm=cfg.pre_norm, rng=rng,
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        self_mask: AttentionMask | None = None,
        cross_mask: AttentionMask | None = None,
    ) -> tuple[Tensor, list[np.ndarray], list[np.ndarray]]:
        """Run the decoder stack.

        Parameters
        ----------
        x:
            Decoder input, shape ``[batch, tgt_seq, d_model]``.
        encoder_output:
            Encoder output, shape ``[batch, src_seq, d_model]``.
        self_mask:
            Mask for causal self-attention.
        cross_mask:
            Mask for cross-attention over encoder output.

        Returns
        -------
        output:
            Shape ``[batch, tgt_seq, d_model]``.
        self_weights:
            List of ``N`` self-attention weight arrays,
            each ``[batch, n_heads, tgt_seq, tgt_seq]``.
        cross_weights:
            List of ``N`` cross-attention weight arrays,
            each ``[batch, n_heads, tgt_seq, src_seq]``.
        """
        self_ws: list[np.ndarray] = []
        cross_ws: list[np.ndarray] = []
        for block in self._blocks:
            x, sw, cw = block(x, encoder_output, self_mask, cross_mask)
            self_ws.append(sw)
            cross_ws.append(cw)
        return self.norm(x), self_ws, cross_ws

    def __repr__(self) -> str:
        return f"TransformerDecoder(d_model={self.d_model}, n_layers={self.n_layers})"
