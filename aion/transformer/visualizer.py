"""TransformerVisualizer — pure analysis tool for transformer internals.

All methods accept plain ``np.ndarray`` inputs (tensors detached from the
autograd graph) and return plain Python dicts/lists.  No ``Tensor`` objects,
no NumPy arrays in the output.  The frontend renders the returned dicts
directly.

This mirrors the pattern in ``AttentionVisualizer``, ``stats.py``, and
``quality.py``: the backend computes; the frontend renders.
"""

from __future__ import annotations

import numpy as np


class TransformerVisualizer:
    """Produce JSON-serializable visualization data from transformer internals.

    All methods are stateless and accept plain ``np.ndarray`` inputs.
    """

    # ── layer output statistics ───────────────────────────────────────────────

    def layer_outputs(self, outputs_per_block: list[np.ndarray]) -> dict:
        """Mean activation magnitude per layer.

        Parameters
        ----------
        outputs_per_block:
            List of ``N`` arrays, each ``[batch, seq, d_model]``.

        Returns
        -------
        dict with keys:
            ``n_layers``, ``mean_magnitude`` (list of floats, one per layer).
        """
        magnitudes = [
            round(float(np.abs(out).mean()), 6)
            for out in outputs_per_block
        ]
        return {"n_layers": len(magnitudes), "mean_magnitude": magnitudes}

    # ── residual norms ────────────────────────────────────────────────────────

    def residual_norms(
        self,
        inputs_per_block: list[np.ndarray],
        outputs_per_block: list[np.ndarray],
    ) -> dict:
        """L2 norm of the residual (output - input) per layer.

        Shows how much each block changes the representation.  A very small
        residual norm means the block is contributing little; a very large
        one may indicate instability.

        Parameters
        ----------
        inputs_per_block:
            List of ``N`` input arrays, each ``[batch, seq, d_model]``.
        outputs_per_block:
            List of ``N`` output arrays, same shapes.

        Returns
        -------
        dict with keys:
            ``n_layers``, ``residual_norm`` (list of floats).
        """
        norms = [
            round(float(np.linalg.norm(out - inp)), 6)
            for inp, out in zip(inputs_per_block, outputs_per_block)
        ]
        return {"n_layers": len(norms), "residual_norm": norms}

    # ── layer norm statistics ─────────────────────────────────────────────────

    def layernorm_stats(self, inputs_per_norm: list[np.ndarray]) -> dict:
        """Mean and variance of inputs to each LayerNorm.

        Parameters
        ----------
        inputs_per_norm:
            List of arrays, each ``[batch, seq, d_model]``.

        Returns
        -------
        dict with keys:
            ``n_layers``, ``layers`` (list of dicts with ``mean``, ``variance``).
        """
        layers = []
        for i, x in enumerate(inputs_per_norm):
            layers.append({
                "layer": i,
                "mean": round(float(x.mean()), 6),
                "variance": round(float(x.var()), 6),
            })
        return {"n_layers": len(layers), "layers": layers}

    # ── FFN activation statistics ─────────────────────────────────────────────

    def ffn_activations(
        self,
        pre_act_per_block: list[np.ndarray],
        activation: str = "relu",
    ) -> dict:
        """Activation statistics for each FFN layer.

        For ReLU: reports the fraction of dead neurons (activations <= 0).
        For GELU: reports mean and std of the pre-activation values.

        Parameters
        ----------
        pre_act_per_block:
            List of ``N`` pre-activation arrays, each ``[batch, seq, d_ff]``.
        activation:
            ``"relu"`` or ``"gelu"``.

        Returns
        -------
        dict with keys:
            ``n_layers``, ``activation``, ``layers`` (list of per-layer dicts).
        """
        layers = []
        for i, x in enumerate(pre_act_per_block):
            if activation == "relu":
                dead = round(float((x <= 0).mean()), 6)
                layers.append({"layer": i, "dead_fraction": dead})
            else:
                layers.append({
                    "layer": i,
                    "mean": round(float(x.mean()), 6),
                    "std": round(float(x.std()), 6),
                })
        return {"n_layers": len(layers), "activation": activation, "layers": layers}

    # ── cross-attention heatmap ───────────────────────────────────────────────

    def cross_attention_heatmap(
        self,
        cross_weights: np.ndarray,
        src_tokens: list[str],
        tgt_tokens: list[str],
        head: int = 0,
        layer: int = -1,
    ) -> dict:
        """Heatmap of encoder-decoder cross-attention.

        Shows which source positions the decoder attends to when generating
        each target position.

        Parameters
        ----------
        cross_weights:
            Shape ``[n_layers, batch, n_heads, tgt_seq, src_seq]`` or
            ``[batch, n_heads, tgt_seq, src_seq]`` for a single layer.
        src_tokens:
            Source token strings (column labels).
        tgt_tokens:
            Target token strings (row labels).
        head:
            Which attention head to visualize.
        layer:
            Which layer to visualize (``-1`` = last).

        Returns
        -------
        dict with keys:
            ``src_tokens``, ``tgt_tokens``, ``matrix``, ``head``, ``layer``.
        """
        w = np.asarray(cross_weights, dtype=float)
        if w.ndim == 5:
            w = w[layer]          # [batch, n_heads, tgt, src]
        w = w[0, head]            # [tgt_seq, src_seq] — first batch element
        return {
            "src_tokens": list(src_tokens),
            "tgt_tokens": list(tgt_tokens),
            "matrix": [[round(float(v), 6) for v in row] for row in w],
            "head": head,
            "layer": layer,
        }

    # ── depth comparison ──────────────────────────────────────────────────────

    def depth_comparison(self, weights_per_block: list[np.ndarray]) -> dict:
        """Attention entropy and sparsity per block.

        Shows how attention patterns evolve with depth — early layers often
        attend broadly; later layers tend to be more focused.

        Parameters
        ----------
        weights_per_block:
            List of ``N`` arrays, each ``[batch, n_heads, seq, seq]``.

        Returns
        -------
        dict with keys:
            ``n_layers``, ``layers`` (list of dicts with ``mean_entropy``,
            ``mean_sparsity``).
        """
        eps = 1e-9
        layers = []
        for i, w in enumerate(weights_per_block):
            w = np.asarray(w, dtype=float)
            entropy = float(-np.mean(np.sum(w * np.log(w + eps), axis=-1)))
            sparsity = float(np.mean(w < 0.01))
            layers.append({
                "layer": i,
                "mean_entropy": round(entropy, 6),
                "mean_sparsity": round(sparsity, 6),
            })
        return {"n_layers": len(layers), "layers": layers}
