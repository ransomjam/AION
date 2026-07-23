"""GPT model components.

LanguageModelHead
    Linear projection from d_model → vocab_size with no bias.
    When weight tying is enabled, W is the same Parameter object as the
    embedding table (shape [vocab_size, d_model]); the forward computes
    x @ W.T so no separate parameter is needed.

GPTModel
    Composes: EmbeddingLayer + LearnedPE + TransformerStack(CausalMask) +
    LanguageModelHead.  Forward takes integer token_ids (np.ndarray) and
    returns (logits [batch, seq, vocab_size], attn_weights list).
"""

from __future__ import annotations

import numpy as np

from aion.attention.mask import CausalMask
from aion.nn.layers import EmbeddingLayer
from aion.nn.module import Module
from aion.nn.ops import matmul
from aion.nn.parameter import Parameter
from aion.nn.tensor import Tensor
from aion.attention.positional import LearnedPE
from aion.transformer.stack import TransformerStack

from .config import GPTConfig


class LanguageModelHead(Module):
    """Project hidden states to vocabulary logits.

    Parameters
    ----------
    d_model:
        Input dimensionality.
    vocab_size:
        Output vocabulary size.
    weight:
        If provided, this Parameter is used as W (weight tying).  The
        forward computes ``x @ weight.T`` so weight shape is
        ``[vocab_size, d_model]``.  If None, a new Parameter is created
        with shape ``[vocab_size, d_model]``.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        weight: Parameter | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        if weight is not None:
            self.W = weight
        else:
            scale = (d_model ** -0.5)
            self.W = Parameter(
                np.random.default_rng().uniform(-scale, scale, (vocab_size, d_model)),
                name="W",
            )

    def forward(self, x: Tensor) -> Tensor:
        """x: [batch, seq, d_model] → logits: [batch, seq, vocab_size]."""
        from aion.nn.ops import transpose
        # W is [vocab_size, d_model]; transpose to [d_model, vocab_size]
        W_t = transpose(self.W, (1, 0))
        return matmul(x, W_t)

    def __repr__(self) -> str:
        return f"LanguageModelHead(d_model={self.d_model}, vocab_size={self.vocab_size})"


class GPTModel(Module):
    """Decoder-only language model (GPT-style).

    Architecture: token embedding + learned positional encoding +
    TransformerStack with CausalMask + LanguageModelHead.

    Parameters
    ----------
    cfg:
        ``GPTConfig`` specifying all hyperparameters.
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        cfg: GPTConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        rng = rng or np.random.default_rng()

        self.embedding = EmbeddingLayer(cfg.vocab_size, cfg.d_model, rng=rng)
        self.pos_encoding = LearnedPE(cfg.d_model, cfg.max_seq_len)
        self.stack = TransformerStack(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            activation=cfg.activation,
            pre_norm=cfg.pre_norm,
            rng=rng,
        )
        # Weight tying: head.W is the same Parameter as embedding.table
        head_weight = self.embedding.table if cfg.tie_weights else None
        self.head = LanguageModelHead(cfg.d_model, cfg.vocab_size, weight=head_weight)
        object.__setattr__(self, "_causal_mask", CausalMask())

    def forward(
        self,
        token_ids: np.ndarray,
    ) -> tuple[Tensor, list[np.ndarray]]:
        """Forward pass.

        Parameters
        ----------
        token_ids:
            Integer array of shape ``[batch, seq_len]``.

        Returns
        -------
        logits:
            Shape ``[batch, seq_len, vocab_size]``.
        attn_weights:
            List of ``n_layers`` attention weight arrays, each
            ``[batch, n_heads, seq_len, seq_len]``.
        """
        mask = object.__getattribute__(self, "_causal_mask")
        x = self.embedding(token_ids)           # [batch, seq, d_model]
        x = self.pos_encoding(x)                # [batch, seq, d_model]
        x, attn_weights = self.stack(x, mask)   # [batch, seq, d_model]
        logits = self.head(x)                   # [batch, seq, vocab_size]
        return logits, attn_weights

    def __repr__(self) -> str:
        c = self.cfg
        return (
            f"GPTModel(vocab={c.vocab_size}, d_model={c.d_model}, "
            f"n_heads={c.n_heads}, n_layers={c.n_layers}, "
            f"tie_weights={c.tie_weights})"
        )
