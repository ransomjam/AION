"""TransformerStack — a reusable stack of N transformer blocks.

``TransformerStack`` is the core reusable abstraction.  It stacks ``N``
``TransformerEncoderBlock`` instances (self-attention + FFN) and applies a
final ``LayerNorm``.  It is the building block for both encoder-only models
(BERT-style) and decoder-only models (GPT-style, used with a ``CausalMask``).

Hierarchy
---------
TransformerStack
    TransformerEncoder  — encoder-only stack (BERT-style)
    TransformerDecoder  — cross-attention decoder stack (seq2seq)

``TransformerEncoder`` and ``TransformerDecoder`` are typed wrappers that
carry semantic meaning and config-factory methods.  The underlying stack
logic lives here and is not duplicated.

Why a separate stack class?
---------------------------
A decoder-only model (GPT) is a ``TransformerStack`` with a ``CausalMask``.
An encoder-only model (BERT) is a ``TransformerStack`` without a mask.
Both use ``TransformerEncoderBlock`` internally — there is no cross-attention.
Naming the shared abstraction ``TransformerStack`` avoids the confusion of
calling a GPT backbone a ``TransformerEncoder``.
"""

from __future__ import annotations

import numpy as np

from aion.attention.mask import AttentionMask
from aion.nn.module import Module
from aion.nn.tensor import Tensor

from .block import TransformerEncoderBlock
from .norm import LayerNorm


class TransformerStack(Module):
    """Stack of ``N`` encoder blocks with a final layer normalization.

    The canonical reusable transformer backbone.  Used directly for
    decoder-only models (with ``CausalMask``) and as the base for
    ``TransformerEncoder``.

    Parameters
    ----------
    d_model:
        Model dimensionality.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of blocks.
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
            setattr(self, f"block_{i}", TransformerEncoderBlock(
                d_model, n_heads, d_ff,
                dropout=dropout, activation=activation,
                pre_norm=pre_norm, rng=rng,
            ))
        object.__setattr__(self, "_blocks", [
            getattr(self, f"block_{i}") for i in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        mask: AttentionMask | None = None,
    ) -> tuple[Tensor, list[np.ndarray]]:
        """Run the stack.

        Parameters
        ----------
        x:
            Input of shape ``[batch, seq, d_model]``.
        mask:
            Optional ``AttentionMask`` passed to every block.

        Returns
        -------
        output:
            Shape ``[batch, seq, d_model]``.
        block_weights:
            List of ``N`` attention weight arrays, each
            ``[batch, n_heads, seq, seq]``.  Detached from the graph.
        """
        block_weights: list[np.ndarray] = []
        for block in self._blocks:
            x, w = block(x, mask)
            block_weights.append(w)
        return self.norm(x), block_weights

    def __repr__(self) -> str:
        return f"TransformerStack(d_model={self.d_model}, n_layers={self.n_layers})"
