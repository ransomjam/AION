"""Embedding abstraction — the generic interface every algorithm implements.

Every embedding in AION is a subclass of ``Embedding``.  The contract is
intentionally minimal so that CBOW, Skip-gram, GloVe, and future algorithms
all plug into the same store, API, and lab without any downstream changes.

    class CBOWEmbedding(Embedding):     algorithm = "cbow-v1"
    class SkipGramEmbedding(Embedding): algorithm = "skipgram-v1"

Responsibility split
--------------------
Subclasses implement **token-level behavior only**:

- ``encode(token_id)``        — vector for a single token id
- ``embed_tokens(token_ids)`` — aggregate vector for a sequence of ids
- ``most_similar(token_id)``  — nearest neighbours by cosine similarity
- ``train`` / ``save`` / ``load``

The base class provides one concrete shared method:

- ``embed_text(text, tokenizer)`` — convenience composition that calls
  ``tokenizer.encode(text)`` then ``embed_tokens``.  This is the only place
  in the embedding framework that touches a tokenizer.

``EmbeddingResult`` is the value returned by ``Embedding.train``.  It carries
only metrics — the matrix itself is the artifact, written to disk by ``save``.
This mirrors ``TrainingResult`` from the tokenizer framework exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbeddingResult:
    """Everything produced by a training run, kept separate from the artifact.

    The embedding matrix is the persistent artifact (written by ``save``).
    ``metrics`` is derived data used for visualization and experiment recording.
    """
    metrics: dict = field(default_factory=dict)
    # Expected metrics keys (all algorithms should populate where applicable):
    #   vocab_size, dims, epochs, window_size, neg_samples, seed,
    #   final_loss, loss_history (list[float], one per epoch),
    #   tokens_processed, training_time_s,
    #   nn_coherence, coverage,
    #   dataset_fingerprint, tokenizer_fingerprint


class Embedding(ABC):
    """Abstract base for all AION embeddings.

    Subclasses implement token-level behavior; text-level convenience is
    provided here.  Persistence routing, experiment recording, and
    default-setting are handled by ``EmbeddingStore`` and the API layer.
    """

    #: Short algorithm identifier written into manifest.json.
    #: Subclasses must set this as a class attribute.
    algorithm: str = ""

    @abstractmethod
    def train(
        self,
        corpus,           # Iterable[str] — raw text documents
        *,
        tokenizer,        # aion.tokenizers.base.Tokenizer
        dims: int,
        epochs: int,
        window: int,
        neg_samples: int,
        seed: int,
        progress_fn=None,
        **kwargs,
    ) -> EmbeddingResult:
        """Train on ``corpus`` using ``tokenizer`` and return an ``EmbeddingResult``.

        After this call the embedding is ready to encode.
        ``progress_fn(fraction, message)`` is called periodically if provided.
        """

    @abstractmethod
    def encode(self, token_id: int) -> list[float]:
        """Return the embedding vector for a single token id."""

    @abstractmethod
    def embed_tokens(self, token_ids: list[int]) -> list[float]:
        """Return a single aggregate vector for a sequence of token ids.

        The canonical implementation is mean-pooling over ``encode`` results,
        but subclasses may override (e.g. weighted pooling, max-pooling).
        Returns a zero vector of length ``self.dims`` for an empty sequence.
        """

    @abstractmethod
    def most_similar(self, token_id: int, n: int = 10) -> list[tuple[int, float]]:
        """Return the ``n`` most similar token ids and their cosine similarities."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Write the model file (vectors.json) into ``directory``."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "Embedding":
        """Load an embedding from ``directory`` (the ``model/`` subdirectory)."""

    # ── shared convenience ────────────────────────────────────────────────────

    def embed_text(self, text: str, tokenizer) -> list[float]:
        """Return a single vector for ``text``.

        Delegates to ``tokenizer.encode`` then ``embed_tokens``.  This is the
        only place in the embedding framework that touches a tokenizer.
        Subclasses must not override this method.
        """
        return self.embed_tokens(tokenizer.encode(text))

    # ── introspection ─────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def dims(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"algorithm={self.algorithm!r}, "
                f"vocab_size={self.vocab_size}, dims={self.dims})")
