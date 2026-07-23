"""Tokenizer abstraction — the generic interface every algorithm implements.

Every tokenizer in AION is a subclass of ``Tokenizer``.  The contract is
intentionally minimal so that BPE, WordPiece, Unigram, and future algorithms
all plug into the same store, API, and lab without any downstream changes.

    class ByteLevelBPETokenizer(Tokenizer): ...
    class WordPieceTokenizer(Tokenizer):    ...

``TrainingResult`` is the value returned by ``Tokenizer.train``.  It carries
the merge history (for visualization) and the evaluation metrics (for the
experiment record) separately from the tokenizer artifact itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MergeStep:
    """One BPE merge iteration — the unit of the merge history."""
    rank: int                  # 0-based merge index
    pair: tuple[str, str]      # the two symbols that were merged
    new_token: str             # the resulting symbol
    pair_freq: int             # weighted frequency of this pair in the corpus
    vocab_size: int            # vocabulary size after this merge
    corpus_tokens: int         # total symbol count in the corpus after this merge


@dataclass
class TrainingResult:
    """Everything produced by a training run, kept separate from the artifact.

    The tokenizer itself (merges + vocabulary) is the persistent artifact.
    ``merge_history`` and ``metrics`` are derived data used for visualization
    and experiment recording; they live in ``statistics.json``.
    """
    merge_history: list[MergeStep] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    # metrics keys (all algorithms should populate these where applicable):
    #   base_vocab_size, final_vocab_size, merge_count,
    #   compression_ratio, avg_tokens_per_word, fertility,
    #   training_time_s, dataset_fingerprint


class Tokenizer(ABC):
    """Abstract base for all AION tokenizers.

    Subclasses implement the four abstract methods; everything else
    (persistence routing, experiment recording, default-setting) is handled
    by ``TokenizerStore`` and the API layer.
    """

    #: Short algorithm identifier written into manifest.json.
    #: Subclasses must set this as a class attribute.
    algorithm: str = ""

    @abstractmethod
    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int,
        progress_fn=None,
        **kwargs,
    ) -> TrainingResult:
        """Train on ``corpus`` and return a ``TrainingResult``.

        After this call the tokenizer is ready to encode/decode.
        ``progress_fn(fraction, message)`` is called periodically if provided.
        """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to a list of token ids."""

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string."""

    @abstractmethod
    def save(self, directory: Path) -> None:
        """Write the model files (merges, vocabulary) into ``directory``."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> "Tokenizer":
        """Load a tokenizer from ``directory`` (the ``model/`` subdirectory)."""

    # ── shared helpers ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (including specials)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(algorithm={self.algorithm!r}, vocab_size={self.vocab_size})"
