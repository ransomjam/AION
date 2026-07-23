"""CBOW Word2Vec embedding — trained from first principles.

Algorithm (Mikolov et al., 2013 — Continuous Bag of Words with negative sampling):

1. Encode the corpus to token-id sequences using the supplied tokenizer.
2. Slide a context window of half-width ``window`` over each sequence.
   For each centre token, the context is the up-to-``window`` tokens on each side.
3. Maintain two matrices:
     W_in  [V, D]  input (context) embeddings — the final artifact
     W_out [V, D]  output (target) embeddings — discarded after training
   Both initialised with small uniform random values in [-0.5/D, 0.5/D],
   seeded deterministically so identical params → identical matrices.
4. For each training example:
     a. Average the context vectors from W_in  → context_vec
     b. Positive sample: sigmoid(dot(context_vec, W_out[centre]))
     c. Negative samples: k token ids drawn by unigram^(3/4) distribution
     d. Binary cross-entropy loss; backpropagate with SGD
     e. Update W_in rows (context tokens) and W_out rows (centre + negatives)
5. Learning rate decays linearly from ``lr`` to ``lr * 0.0001`` over all epochs.
6. After training, W_in is the embedding matrix.

Determinism guarantee: same seed + same corpus + same params → same vectors.json.
Correctness and readability over speed.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .base import Embedding, EmbeddingResult
from .linalg import (
    add_, cosine_similarity, dot, mat_add_outer_,
    mean_vector, norm, scale_,
)

__all__ = ["CBOWEmbedding"]

_SCHEMA_VERSION = 1


class CBOWEmbedding(Embedding):
    """CBOW Word2Vec embedding trained from first principles."""

    algorithm = "cbow-v1"

    def __init__(self) -> None:
        self._W: list[list[float]] = []   # W_in — the embedding matrix
        self._vocab_size: int = 0
        self._dims: int = 0

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        corpus: Iterable[str],
        *,
        tokenizer,
        dims: int = 64,
        epochs: int = 5,
        window: int = 2,
        neg_samples: int = 5,
        seed: int = 42,
        lr: float = 0.025,
        progress_fn=None,
        dataset_fingerprint: str = "",
        tokenizer_fingerprint: str = "",
        **kwargs,
    ) -> EmbeddingResult:
        t0 = time.monotonic()
        rng = random.Random(seed)

        V = tokenizer.vocab_size

        # ── 1. Encode corpus to id sequences ──────────────────────────────────
        sequences: list[list[int]] = []
        token_counts: Counter = Counter()
        for doc in corpus:
            ids = tokenizer.encode(doc)
            if ids:
                sequences.append(ids)
                token_counts.update(ids)

        total_tokens = sum(len(s) for s in sequences)

        # ── 2. Negative-sampling distribution: unigram^(3/4) ─────────────────
        # Build a sampling table of size _TABLE_SIZE for O(1) negative sampling.
        _TABLE_SIZE = 100_000
        freq_smoothed = [0.0] * V
        for tid, cnt in token_counts.items():
            if 0 <= tid < V:
                freq_smoothed[tid] = cnt ** 0.75
        total_smooth = sum(freq_smoothed) or 1.0
        # Fill table proportionally
        neg_table: list[int] = []
        cumulative = 0.0
        tid_iter = 0
        for slot in range(_TABLE_SIZE):
            while tid_iter < V and cumulative < (slot + 1) * total_smooth / _TABLE_SIZE:
                cumulative += freq_smoothed[tid_iter]
                tid_iter += 1
            neg_table.append(max(0, tid_iter - 1))

        # ── 3. Initialise weight matrices ─────────────────────────────────────
        init_range = 0.5 / dims
        W_in  = [[rng.uniform(-init_range, init_range) for _ in range(dims)] for _ in range(V)]
        W_out = [[rng.uniform(-init_range, init_range) for _ in range(dims)] for _ in range(V)]

        # ── 4. Training loop ──────────────────────────────────────────────────
        loss_history: list[float] = []
        total_steps = epochs * total_tokens
        steps_done = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            for seq in sequences:
                for pos, centre in enumerate(seq):
                    # Context window (variable at sequence boundaries)
                    ctx_start = max(0, pos - window)
                    ctx_end   = min(len(seq), pos + window + 1)
                    ctx_ids = [seq[i] for i in range(ctx_start, ctx_end) if i != pos]
                    if not ctx_ids:
                        continue

                    # Linear LR decay
                    progress_frac = steps_done / max(1, total_steps)
                    current_lr = lr * max(0.0001, 1.0 - progress_frac)

                    # Context vector: mean of W_in rows for context tokens
                    ctx_vecs = [W_in[c] for c in ctx_ids]
                    ctx_mean = mean_vector(ctx_vecs)

                    # Gradient accumulator for the context mean
                    grad_ctx = [0.0] * dims

                    # Positive + negative samples
                    targets = [(centre, 1)]
                    neg_drawn = set()
                    attempts = 0
                    while len(neg_drawn) < neg_samples and attempts < neg_samples * 4:
                        neg = neg_table[rng.randrange(_TABLE_SIZE)]
                        if neg != centre:
                            neg_drawn.add(neg)
                        attempts += 1
                    for neg in neg_drawn:
                        targets.append((neg, 0))

                    for target_id, label in targets:
                        score = dot(ctx_mean, W_out[target_id])
                        # Clamp to avoid overflow in exp
                        score = max(-10.0, min(10.0, score))
                        sigmoid = 1.0 / (1.0 + math.exp(-score))
                        err = label - sigmoid          # gradient of BCE
                        loss = -(label * math.log(sigmoid + 1e-10)
                                 + (1 - label) * math.log(1 - sigmoid + 1e-10))
                        epoch_loss += loss

                        # Update W_out[target_id]
                        mat_add_outer_(W_out, target_id, ctx_mean, current_lr * err)
                        # Accumulate gradient for context
                        add_(grad_ctx, W_out[target_id], err)

                    # Update each W_in context row
                    n_ctx = len(ctx_ids)
                    for c in ctx_ids:
                        mat_add_outer_(W_in, c, grad_ctx, current_lr / n_ctx)

                    steps_done += 1
                    epoch_steps += 1

            avg_loss = epoch_loss / max(1, epoch_steps)
            loss_history.append(round(avg_loss, 6))

            if progress_fn:
                progress_fn(
                    (epoch + 1) / epochs,
                    f"epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}",
                )

        training_time = time.monotonic() - t0

        # ── 5. Commit trained state ───────────────────────────────────────────
        self._W = W_in
        self._vocab_size = V
        self._dims = dims

        # ── 6. Evaluation metrics ─────────────────────────────────────────────
        coverage = len(token_counts) / V if V else 0.0

        # Nearest-neighbour coherence: avg cosine sim to top-5 neighbours
        # for the top-50 most frequent tokens (cheap intrinsic quality signal)
        top_ids = [tid for tid, _ in token_counts.most_common(50) if 0 <= tid < V]
        coherence = _nn_coherence(W_in, top_ids, n=5)

        metrics = {
            "vocab_size": V,
            "dims": dims,
            "epochs": epochs,
            "window_size": window,
            "neg_samples": neg_samples,
            "seed": seed,
            "lr": lr,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
            "tokens_processed": total_tokens * epochs,
            "training_time_s": round(training_time, 3),
            "nn_coherence": round(coherence, 4),
            "coverage": round(coverage, 4),
            "dataset_fingerprint": dataset_fingerprint,
            "tokenizer_fingerprint": tokenizer_fingerprint,
        }
        return EmbeddingResult(metrics=metrics)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, token_id: int) -> list[float]:
        if not (0 <= token_id < self._vocab_size):
            return [0.0] * self._dims
        return list(self._W[token_id])

    def embed_tokens(self, token_ids: list[int]) -> list[float]:
        """Mean-pool the vectors for ``token_ids``."""
        vecs = [self._W[i] for i in token_ids if 0 <= i < self._vocab_size]
        if not vecs:
            return [0.0] * self._dims
        return mean_vector(vecs)

    def most_similar(self, token_id: int, n: int = 10) -> list[tuple[int, float]]:
        if not (0 <= token_id < self._vocab_size) or not self._W:
            return []
        query = self._W[token_id]
        sims = []
        for i, vec in enumerate(self._W):
            if i == token_id:
                continue
            sims.append((i, cosine_similarity(query, vec)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:n]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "vocab_size": self._vocab_size,
            "dims": self._dims,
            "vectors": self._W,
        }
        (directory / "vectors.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "CBOWEmbedding":
        directory = Path(directory)
        data = json.loads((directory / "vectors.json").read_text(encoding="utf-8"))
        emb = cls()
        emb._W = data["vectors"]
        emb._vocab_size = data["vocab_size"]
        emb._dims = data["dims"]
        return emb

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def dims(self) -> int:
        return self._dims


# ── Internal helpers ──────────────────────────────────────────────────────────

def _nn_coherence(W: list[list[float]], token_ids: list[int], n: int) -> float:
    """Average cosine similarity to top-n neighbours for the given token ids."""
    if not token_ids or not W:
        return 0.0
    total = 0.0
    count = 0
    for tid in token_ids:
        query = W[tid]
        sims = sorted(
            (cosine_similarity(query, W[j]) for j in range(len(W)) if j != tid),
            reverse=True,
        )
        if sims:
            total += sum(sims[:n]) / min(n, len(sims))
            count += 1
    return total / count if count else 0.0
