"""TransformerEncoder — encoder-only transformer stack.

``TransformerEncoder`` is a ``TransformerStack`` with semantic meaning: it is
the encoder half of an encoder-decoder model, or a standalone encoder-only
model (BERT-style).  It adds config-factory methods and a typed ``__repr__``.

For decoder-only models (GPT-style), use ``TransformerStack`` directly with a
``CausalMask`` — there is no cross-attention and no separate encoder output.
"""

from __future__ import annotations

import numpy as np

from aion.attention.mask import AttentionMask
from aion.nn.tensor import Tensor

from .config import EncoderConfig, TransformerConfig
from .stack import TransformerStack


class TransformerEncoder(TransformerStack):
    """Encoder-only transformer stack (BERT-style or encoder half of seq2seq).

    Identical to ``TransformerStack`` in behaviour.  The subclass exists to
    carry semantic meaning and to provide ``from_config`` factory methods that
    accept ``EncoderConfig`` or the encoder half of ``TransformerConfig``.

    Parameters
    ----------
    d_model:
        Model dimensionality.
    n_heads:
        Number of attention heads.
    n_layers:
        Number of encoder blocks.
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

    @classmethod
    def from_config(
        cls,
        cfg: TransformerConfig | EncoderConfig,
        rng: np.random.Generator | None = None,
    ) -> "TransformerEncoder":
        """Construct from a config object."""
        n = cfg.n_layers if isinstance(cfg, EncoderConfig) else cfg.n_encoder_layers
        return cls(
            d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=n,
            d_ff=cfg.d_ff, dropout=cfg.dropout, activation=cfg.activation,
            pre_norm=cfg.pre_norm, rng=rng,
        )

    def forward(
        self,
        x: Tensor,
        mask: AttentionMask | None = None,
    ) -> tuple[Tensor, list[np.ndarray]]:
        return super().forward(x, mask)

    def __repr__(self) -> str:
        return f"TransformerEncoder(d_model={self.d_model}, n_layers={self.n_layers})"
