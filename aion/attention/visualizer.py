"""AttentionVisualizer — pure analysis tool for attention weight arrays.

All methods accept plain ``np.ndarray`` inputs (attention weights detached
from the autograd graph via ``.data.copy()``) and return plain Python
dicts/lists.  No ``Tensor`` objects, no NumPy arrays in the output.

The frontend renders the returned dicts directly.  No visualization logic
lives in the frontend.

This mirrors the pattern in ``aion/datasets/stats.py`` and ``quality.py``:
the backend computes; the frontend renders.
"""

from __future__ import annotations

import numpy as np


class AttentionVisualizer:
    """Produce JSON-serializable visualization data from attention weights.

    All methods are stateless and accept plain ``np.ndarray`` inputs.
    """

    # ── single-head heatmap ───────────────────────────────────────────────────

    def heatmap(
        self,
        weights: np.ndarray,
        tokens_q: list[str],
        tokens_k: list[str],
    ) -> dict:
        """Heatmap data for a single attention head.

        Parameters
        ----------
        weights:
            Shape ``[seq_q, seq_k]``.  Values should be in ``[0, 1]``
            (post-softmax).
        tokens_q:
            Query token strings (row labels).
        tokens_k:
            Key token strings (column labels).

        Returns
        -------
        dict with keys:
            ``tokens_q``, ``tokens_k``, ``matrix`` (list of lists of floats).
        """
        w = np.asarray(weights, dtype=float)
        return {
            "tokens_q": list(tokens_q),
            "tokens_k": list(tokens_k),
            "matrix": [[round(float(v), 6) for v in row] for row in w],
        }

    # ── multi-head heatmaps ───────────────────────────────────────────────────

    def multi_head_heatmaps(
        self,
        weights: np.ndarray,
        tokens_q: list[str],
        tokens_k: list[str],
    ) -> list[dict]:
        """One heatmap dict per head.

        Parameters
        ----------
        weights:
            Shape ``[n_heads, seq_q, seq_k]``.
        tokens_q:
            Query token strings.
        tokens_k:
            Key token strings.

        Returns
        -------
        list of heatmap dicts, one per head.
        """
        w = np.asarray(weights, dtype=float)
        return [
            {"head": h, **self.heatmap(w[h], tokens_q, tokens_k)}
            for h in range(w.shape[0])
        ]

    # ── head comparison ───────────────────────────────────────────────────────

    def head_comparison(self, weights: np.ndarray) -> dict:
        """Per-head summary statistics for comparing head behaviour.

        Parameters
        ----------
        weights:
            Shape ``[n_heads, seq_q, seq_k]`` or ``[batch, n_heads, seq_q, seq_k]``.
            If 4-D, statistics are averaged over the batch dimension.

        Returns
        -------
        dict with keys:
            ``n_heads``, ``heads`` (list of per-head dicts with ``entropy``,
            ``max_attn``, ``sparsity``), ``entropy_variance``.

        Metrics
        -------
        entropy:
            Mean per-query Shannon entropy: ``-sum(w * log(w + eps))``.
            Low = focused; high = diffuse.
        max_attn:
            Mean maximum attention weight per query position.
            High = the head consistently attends to one dominant token.
        sparsity:
            Fraction of weights below 0.01 (effectively zero).
            High = the head is sparse / selective.
        """
        w = np.asarray(weights, dtype=float)
        if w.ndim == 4:
            # Average over batch.
            w = w.mean(axis=0)   # [n_heads, seq_q, seq_k]

        eps = 1e-9
        heads = []
        for h in range(w.shape[0]):
            wh = w[h]   # [seq_q, seq_k]
            entropy = float(-np.mean(np.sum(wh * np.log(wh + eps), axis=-1)))
            max_attn = float(np.mean(wh.max(axis=-1)))
            sparsity = float(np.mean(wh < 0.01))
            heads.append({
                "head": h,
                "entropy": round(entropy, 6),
                "max_attn": round(max_attn, 6),
                "sparsity": round(sparsity, 6),
            })

        entropies = [h["entropy"] for h in heads]
        entropy_variance = round(float(np.var(entropies)), 6)

        return {
            "n_heads": w.shape[0],
            "heads": heads,
            "entropy_variance": entropy_variance,
        }

    # ── query/key similarity ──────────────────────────────────────────────────

    def query_key_similarity(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        tokens_q: list[str],
        tokens_k: list[str],
    ) -> dict:
        """Cosine similarity matrix between query and key vectors.

        Shows what the model "sees" before scaling and softmax.

        Parameters
        ----------
        Q:
            Query vectors, shape ``[seq_q, d_k]``.
        K:
            Key vectors, shape ``[seq_k, d_k]``.
        tokens_q:
            Query token strings.
        tokens_k:
            Key token strings.

        Returns
        -------
        dict with keys:
            ``tokens_q``, ``tokens_k``, ``matrix`` (cosine similarities).
        """
        Q = np.asarray(Q, dtype=float)
        K = np.asarray(K, dtype=float)
        eps = 1e-9
        Q_norm = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + eps)
        K_norm = K / (np.linalg.norm(K, axis=-1, keepdims=True) + eps)
        sim = Q_norm @ K_norm.T   # [seq_q, seq_k]
        return {
            "tokens_q": list(tokens_q),
            "tokens_k": list(tokens_k),
            "matrix": [[round(float(v), 6) for v in row] for row in sim],
        }
