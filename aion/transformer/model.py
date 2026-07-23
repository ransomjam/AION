"""Full encoder-decoder Transformer.

``Transformer`` composes ``TransformerEncoder`` and ``TransformerDecoder``.
It is the entry point for sequence-to-sequence tasks.

Architecture variants
---------------------
Encoder-only (BERT-style):
    Use ``TransformerEncoder`` directly.  No cross-attention.

Decoder-only (GPT-style):
    Use ``TransformerStack`` directly with a ``CausalMask``.  No encoder
    output, no cross-attention.

Full encoder-decoder (seq2seq):
    Use ``Transformer``.  Both encoder and decoder are required.

Weight tying
------------
Language models commonly tie the input embedding matrix to the output
projection (the logit head) to reduce parameter count and improve
generalization (Press & Wolf, 2017).  ``Transformer`` exposes
``tie_weights(embedding, projection)`` to register this relationship
explicitly.  The actual tying (sharing the same ``Parameter`` object) is
performed by the caller when constructing the model head — this method
records the intent so the store can verify and restore it on load.

The full implementation of the logit head and weight tying belongs in the
GPT/BERT milestones.  The framework supports it without precluding it.
"""

from __future__ import annotations

import numpy as np

from aion.attention.mask import AttentionMask
from aion.nn.module import Module
from aion.nn.parameter import Parameter
from aion.nn.tensor import Tensor

from .config import TransformerConfig
from .decoder import TransformerDecoder
from .encoder import TransformerEncoder


class Transformer(Module):
    """Full encoder-decoder Transformer.

    Parameters
    ----------
    cfg:
        ``TransformerConfig`` specifying all hyperparameters.
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        cfg: TransformerConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        rng = rng or np.random.default_rng()
        self.encoder = TransformerEncoder.from_config(cfg, rng=rng)
        self.decoder = TransformerDecoder.from_config(cfg, rng=rng)
        # Weight-tying registry: maps role → Parameter.
        # Populated by tie_weights(); read by TransformerStore on save.
        object.__setattr__(self, "_tied_weights", {})

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: AttentionMask | None = None,
        tgt_mask: AttentionMask | None = None,
        cross_mask: AttentionMask | None = None,
    ) -> tuple[Tensor, dict]:
        """Run the full encoder-decoder forward pass.

        Parameters
        ----------
        src:
            Source sequence, shape ``[batch, src_seq, d_model]``.
        tgt:
            Target sequence, shape ``[batch, tgt_seq, d_model]``.
        src_mask:
            Mask for encoder self-attention (e.g. ``PaddingMask``).
        tgt_mask:
            Mask for decoder self-attention (e.g. ``CausalMask``).
        cross_mask:
            Mask for decoder cross-attention over encoder output.

        Returns
        -------
        output:
            Shape ``[batch, tgt_seq, d_model]``.
        viz:
            Dict of detached attention arrays for visualization::

                {
                    "encoder_weights":        list[np.ndarray],
                    "decoder_self_weights":   list[np.ndarray],
                    "decoder_cross_weights":  list[np.ndarray],
                }
        """
        enc_out, enc_weights = self.encoder(src, src_mask)
        dec_out, dec_self_w, dec_cross_w = self.decoder(
            tgt, enc_out, tgt_mask, cross_mask
        )
        viz = {
            "encoder_weights": enc_weights,
            "decoder_self_weights": dec_self_w,
            "decoder_cross_weights": dec_cross_w,
        }
        return dec_out, viz

    def tie_weights(self, embedding: Parameter, projection: Parameter) -> None:
        """Register an embedding/projection weight-tying relationship.

        Records that ``embedding`` and ``projection`` are (or should be) the
        same ``Parameter`` object.  The caller is responsible for the actual
        sharing — passing the same ``Parameter`` instance to both the
        embedding layer and the output projection.

        This method exists so the framework explicitly supports weight tying
        without implementing the logit head (which belongs in GPT/BERT).
        ``TransformerStore`` reads ``_tied_weights`` on save to record the
        relationship in ``architecture.json``.

        Parameters
        ----------
        embedding:
            The token embedding parameter (shape ``[vocab_size, d_model]``).
        projection:
            The output projection parameter (shape ``[d_model, vocab_size]``
            or ``[vocab_size, d_model]`` if transposed at forward time).
        """
        tied = object.__getattribute__(self, "_tied_weights")
        tied["embedding"] = embedding
        tied["projection"] = projection

    @property
    def weights_tied(self) -> bool:
        """True if ``tie_weights`` has been called."""
        return bool(object.__getattribute__(self, "_tied_weights"))

    def __repr__(self) -> str:
        c = self.cfg
        return (f"Transformer(d_model={c.d_model}, n_heads={c.n_heads}, "
                f"enc={c.n_encoder_layers}, dec={c.n_decoder_layers})")
