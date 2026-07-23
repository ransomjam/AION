"""Byte-Level BPE tokenizer — trained from first principles.

Algorithm (Sennrich et al., 2016 — byte-level variant as in GPT-2):

1. Pre-tokenize the corpus with the existing ``normalize`` + ``tokenize``
   pipeline (one source of truth for text processing).
2. Encode every pre-token as a sequence of UTF-8 byte symbols
   (``\\x00``..``\\xff``).  The base vocabulary is exactly 256 byte symbols
   plus the four special tokens — no OOV is possible for any Unicode input.
3. Count ``word_freq: {word_tuple: count}`` across the corpus.
4. Iterate:
   a. Count adjacent-pair frequencies, weighted by word count.
   b. Select the most frequent pair; ties broken lexicographically
      (determinism guarantee — same input + params ⇒ same merges.json).
   c. Merge the pair everywhere in the corpus representation.
   d. Append the new symbol to the vocabulary; record a ``MergeStep``.
   e. Repeat until ``vocab_size`` is reached or no pairs remain.
5. Encode: apply merges in learned order (greedy, longest-match via the
   merge rank index).
6. Decode: map ids → byte symbols → concatenate → decode UTF-8
   (``errors='replace'`` so corrupt sequences never raise).

Correctness and readability over speed.  Pair counts are recomputed each
iteration.  An incremental index is a documented future optimization.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from aion.tokenization.text import normalize, tokenize
from .base import MergeStep, Tokenizer, TrainingResult

__all__ = ["ByteLevelBPETokenizer"]

# ── Special tokens (fixed ids 0-3) ────────────────────────────────────────────
_SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>")
_PAD, _UNK, _BOS, _EOS = _SPECIALS

# ── Byte symbol encoding ──────────────────────────────────────────────────────
# Each of the 256 possible byte values is represented as a printable symbol so
# that the vocabulary and merge rules are human-readable in JSON.
# We use the Latin-1 Supplement + a private-use mapping for control bytes,
# matching the GPT-2 convention (Radford et al., 2019).

def _build_byte_encoder() -> dict[int, str]:
    """Map each byte value 0-255 to a unique printable Unicode character."""
    # Bytes that are already printable ASCII or Latin-1 keep their character.
    printable = (
        list(range(ord("!"), ord("~") + 1))   # 33-126
        + list(range(ord("¡"), ord("¬") + 1)) # 161-172
        + list(range(ord("®"), ord("ÿ") + 1)) # 174-255
    )
    enc: dict[int, str] = {b: chr(b) for b in printable}
    # Remaining bytes (control chars, space, etc.) map to U+0100 onwards.
    extra = 0
    for b in range(256):
        if b not in enc:
            enc[b] = chr(256 + extra)
            extra += 1
    return enc


_BYTE_ENC: dict[int, str] = _build_byte_encoder()
_BYTE_DEC: dict[str, int] = {v: k for k, v in _BYTE_ENC.items()}


def _word_to_bytes(word: str) -> tuple[str, ...]:
    """Encode a pre-token string as a tuple of byte symbols."""
    return tuple(_BYTE_ENC[b] for b in word.encode("utf-8"))


def _symbols_to_str(symbols: tuple[str, ...]) -> str:
    """Decode a symbol tuple back to a UTF-8 string."""
    raw = bytes(_BYTE_DEC[s] for s in symbols if s in _BYTE_DEC)
    return raw.decode("utf-8", errors="replace")


# ── Core BPE helpers ──────────────────────────────────────────────────────────

def _count_pairs(word_freq: dict[tuple, int]) -> Counter:
    """Count adjacent-pair frequencies weighted by word count."""
    counts: Counter = Counter()
    for symbols, freq in word_freq.items():
        for a, b in zip(symbols, symbols[1:]):
            counts[(a, b)] += freq
    return counts


def _best_pair(counts: Counter) -> tuple[str, str] | None:
    """Return the most frequent pair; ties broken lexicographically."""
    if not counts:
        return None
    return max(counts, key=lambda p: (counts[p], (-ord(p[0][0]), -ord(p[1][0])) if p[0] and p[1] else (0, 0)))


def _merge_pair(
    word_freq: dict[tuple, int],
    pair: tuple[str, str],
    new_token: str,
) -> dict[tuple, int]:
    """Apply one merge rule to the entire corpus representation."""
    a, b = pair
    new_word_freq: dict[tuple, int] = {}
    for symbols, freq in word_freq.items():
        merged: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                merged.append(new_token)
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        new_word_freq[tuple(merged)] = freq
    return new_word_freq


def _corpus_token_count(word_freq: dict[tuple, int]) -> int:
    return sum(len(syms) * freq for syms, freq in word_freq.items())


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class ByteLevelBPETokenizer(Tokenizer):
    """Byte-Level BPE tokenizer trained from first principles."""

    algorithm = "bpe-byte-v1"

    def __init__(self) -> None:
        self._merges: list[tuple[str, str]] = []          # ordered merge rules
        self._vocab: list[str] = []                        # index = id
        self._token_to_id: dict[str, int] = {}
        self._merge_rank: dict[tuple[str, str], int] = {}  # pair -> rank (for encode)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        corpus: Iterable[str],
        *,
        vocab_size: int,
        progress_fn=None,
        dataset_fingerprint: str = "",
        **kwargs,
    ) -> TrainingResult:
        """Train BPE on ``corpus`` until ``vocab_size`` is reached."""
        if vocab_size < len(_SPECIALS) + 256:
            raise ValueError(
                f"vocab_size must be >= {len(_SPECIALS) + 256} "
                f"(4 specials + 256 base bytes); got {vocab_size}"
            )

        t0 = time.monotonic()

        # ── 1. Build word frequency table ─────────────────────────────────────
        raw_freq: Counter = Counter()
        for doc in corpus:
            for word in tokenize(normalize(doc), normalize_first=False):
                raw_freq[word] += 1

        # Represent each word as a byte-symbol tuple
        word_freq: dict[tuple, int] = {
            _word_to_bytes(word): freq for word, freq in raw_freq.items()
        }

        # ── 2. Base vocabulary ────────────────────────────────────────────────
        vocab: list[str] = list(_SPECIALS) + [_BYTE_ENC[b] for b in range(256)]
        merges: list[tuple[str, str]] = []

        n_merges_target = vocab_size - len(vocab)
        merge_history: list[MergeStep] = []

        def _progress(done: int, total: int, msg: str = "") -> None:
            if progress_fn:
                progress_fn(done / total if total else 1.0, msg)

        # ── 3. BPE merge loop ─────────────────────────────────────────────────
        for rank in range(n_merges_target):
            counts = _count_pairs(word_freq)
            pair = _best_pair(counts)
            if pair is None:
                break

            new_token = pair[0] + pair[1]
            word_freq = _merge_pair(word_freq, pair, new_token)
            merges.append(pair)
            vocab.append(new_token)

            corpus_tokens = _corpus_token_count(word_freq)
            merge_history.append(MergeStep(
                rank=rank,
                pair=pair,
                new_token=new_token,
                pair_freq=counts[pair],
                vocab_size=len(vocab),
                corpus_tokens=corpus_tokens,
            ))

            _progress(rank + 1, n_merges_target,
                      f"merge {rank + 1}/{n_merges_target}: {pair[0]!r}+{pair[1]!r}")

        training_time = time.monotonic() - t0

        # ── 4. Commit the trained state ───────────────────────────────────────
        self._merges = merges
        self._vocab = vocab
        self._token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self._merge_rank = {pair: rank for rank, pair in enumerate(merges)}

        # ── 5. Compute metrics ────────────────────────────────────────────────
        base_tokens = _corpus_token_count(
            {_word_to_bytes(w): f for w, f in raw_freq.items()}
        )
        final_tokens = _corpus_token_count(word_freq)
        compression = round(base_tokens / final_tokens, 4) if final_tokens else 1.0

        total_words = sum(raw_freq.values())
        avg_tokens_per_word = round(final_tokens / total_words, 4) if total_words else 0.0

        metrics = {
            "base_vocab_size": len(_SPECIALS) + 256,
            "final_vocab_size": len(vocab),
            "merge_count": len(merges),
            "compression_ratio": compression,
            "avg_tokens_per_word": avg_tokens_per_word,
            "fertility": avg_tokens_per_word,
            "training_time_s": round(training_time, 3),
            "dataset_fingerprint": dataset_fingerprint,
        }

        return TrainingResult(merge_history=merge_history, metrics=metrics)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to token ids using the learned merge rules."""
        unk_id = self._token_to_id.get(_UNK, 1)
        ids: list[int] = []
        for word in tokenize(normalize(text), normalize_first=False):
            symbols = list(_word_to_bytes(word))
            # Apply merges greedily in rank order
            while len(symbols) > 1:
                # Find the lowest-rank (earliest-learned) adjacent pair
                best_rank = len(self._merge_rank)
                best_i = -1
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    r = self._merge_rank.get(pair, len(self._merge_rank))
                    if r < best_rank:
                        best_rank = r
                        best_i = i
                if best_i == -1:
                    break
                new_tok = symbols[best_i] + symbols[best_i + 1]
                symbols = symbols[:best_i] + [new_tok] + symbols[best_i + 2:]
            ids.extend(self._token_to_id.get(s, unk_id) for s in symbols)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to a string."""
        special_ids = {self._token_to_id[s] for s in _SPECIALS if s in self._token_to_id}
        symbols: list[str] = []
        for i in ids:
            if not (0 <= i < len(self._vocab)) or i in special_ids:
                continue
            # A merged token is a concatenation of byte symbols; each byte
            # symbol is exactly one character in the _BYTE_ENC space.
            # Decompose by iterating characters — each char is one byte symbol.
            for ch in self._vocab[i]:
                symbols.append(ch)
        return _symbols_to_str(tuple(symbols))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Write ``merges.json`` and ``vocabulary.json`` into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        merges_payload = {
            "version": 1,
            "algorithm": self.algorithm,
            "merges": [list(pair) for pair in self._merges],
        }
        (directory / "merges.json").write_text(
            json.dumps(merges_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        vocab_payload = {
            "version": 1,
            "algorithm": self.algorithm,
            "tokens": self._vocab,
        }
        (directory / "vocabulary.json").write_text(
            json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "ByteLevelBPETokenizer":
        """Load a trained tokenizer from ``directory``."""
        directory = Path(directory)
        merges_data = json.loads((directory / "merges.json").read_text(encoding="utf-8"))
        vocab_data = json.loads((directory / "vocabulary.json").read_text(encoding="utf-8"))

        tok = cls()
        tok._merges = [tuple(pair) for pair in merges_data["merges"]]
        tok._vocab = vocab_data["tokens"]
        tok._token_to_id = {t: i for i, t in enumerate(tok._vocab)}
        tok._merge_rank = {tuple(pair): rank for rank, pair in enumerate(tok._merges)}
        return tok

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
