"""Vocabulary construction from first principles.

A model does not consume strings; it consumes integers. A **vocabulary** is the
bidirectional map between tokens (strings) and ids (integers) that makes that
possible, plus the policy for the two hard questions every vocabulary must
answer:

    1. Which tokens are worth keeping?  (rare ones are dropped)
    2. What happens to a token we have never seen?  (it becomes <unk>)

This implementation is intentionally small and explicit so the mechanics are
visible. It depends only on :mod:`aion.tokenization.text` and the standard
library.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .text import tokenize

__all__ = ["Vocabulary"]

# Special tokens. These occupy the first, fixed ids so that a model's embedding
# table has stable, reserved slots for them regardless of the corpus.
PAD = "<pad>"  # padding, to make variable-length sequences rectangular
UNK = "<unk>"  # any token not in the vocabulary maps here
BOS = "<bos>"  # beginning of sequence
EOS = "<eos>"  # end of sequence
DEFAULT_SPECIALS = (PAD, UNK, BOS, EOS)


class Vocabulary:
    """A frozen token<->id mapping built from a corpus.

    Construct with :meth:`build`; do not populate an instance by hand. Once built,
    a vocabulary is treated as immutable — encoding must be deterministic and
    reproducible, and a mapping that shifts under you silently corrupts every id
    a model has already learned.
    """

    def __init__(self, id_to_token: list[str], frequencies: dict[str, int],
                 specials: tuple[str, ...]):
        self._id_to_token = list(id_to_token)
        self._token_to_id = {tok: i for i, tok in enumerate(id_to_token)}
        self._frequencies = dict(frequencies)
        self._specials = tuple(specials)
        if UNK not in self._token_to_id:
            # Encoding out-of-vocabulary tokens is impossible without <unk>; refuse
            # to build a vocabulary that cannot represent unseen input.
            raise ValueError(f"vocabulary must contain the {UNK!r} token")

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        corpus: Iterable[str],
        *,
        min_freq: int = 1,
        max_size: int | None = None,
        specials: tuple[str, ...] = DEFAULT_SPECIALS,
    ) -> "Vocabulary":
        """Build a vocabulary from an iterable of raw text documents.

        Each document is tokenized with the shared :func:`tokenize`, so the
        vocabulary and every downstream consumer see exactly the same tokens.

        - ``min_freq``: tokens appearing fewer than this many times across the
          whole corpus are dropped (they become ``<unk>`` at encode time). Rare
          tokens are mostly noise and typos, and they inflate the embedding table.
        - ``max_size``: optional cap on the total vocabulary size *including*
          specials. When the surviving tokens exceed the cap, the least frequent
          are dropped first.
        - ``specials``: reserved tokens placed at ids ``0..len(specials)-1``.

        **Ordering is deterministic** — the heart of reproducibility. Specials
        come first in the given order; corpus tokens follow, sorted by descending
        frequency and then alphabetically to break ties. Given the same corpus and
        options, :meth:`build` always produces byte-for-byte the same vocabulary.
        """
        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")

        counts: Counter[str] = Counter()
        for document in corpus:
            counts.update(tokenize(document))

        # Specials must never be shadowed by a real corpus token that happens to
        # collide with them, and they are not "frequencies" — remove them here.
        for special in specials:
            counts.pop(special, None)

        # Deterministic tie-break: frequency desc, then token asc.
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = [(tok, freq) for tok, freq in ranked if freq >= min_freq]

        if max_size is not None:
            room = max_size - len(specials)
            if room < 0:
                raise ValueError("max_size is smaller than the number of specials")
            kept = kept[:room]

        id_to_token = list(specials) + [tok for tok, _ in kept]
        frequencies = {tok: freq for tok, freq in kept}
        return cls(id_to_token, frequencies, specials)

    # ── Encoding / decoding ────────────────────────────────────────────────────

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Tokenize ``text`` and map each token to its id.

        Unknown tokens map to the ``<unk>`` id rather than raising — real input
        always contains words the training corpus never had. Optionally wrap the
        sequence in ``<bos>``/``<eos>`` markers, which sequence models use to know
        where an example begins and ends.
        """
        unk_id = self._token_to_id[UNK]
        ids = [self._token_to_id.get(tok, unk_id) for tok in tokenize(text)]
        if add_bos:
            ids.insert(0, self._token_to_id[BOS])
        if add_eos:
            ids.append(self._token_to_id[EOS])
        return ids

    def decode(self, ids: Iterable[int], *, skip_specials: bool = True) -> list[str]:
        """Map ids back to their tokens.

        This is not a perfect inverse of :func:`tokenize` — spacing and any tokens
        that were replaced by ``<unk>`` are not recoverable — so it returns the
        list of tokens rather than a re-joined string, keeping the lossiness
        honest and visible. By default the special tokens are omitted.
        """
        special_ids = {self._token_to_id[s] for s in self._specials}
        out: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self._id_to_token):
                raise KeyError(f"id {i} is out of range for a vocabulary of size {len(self)}")
            if skip_specials and i in special_ids:
                continue
            out.append(self._id_to_token[i])
        return out

    # ── Introspection ──────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._id_to_token)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def token_to_id(self, token: str) -> int:
        """Return the id for ``token``, or the ``<unk>`` id if it is not present."""
        return self._token_to_id.get(token, self._token_to_id[UNK])

    def id_to_token(self, i: int) -> str:
        return self._id_to_token[i]

    def frequency(self, token: str) -> int:
        """Corpus frequency of ``token`` (0 for specials and unknown tokens)."""
        return self._frequencies.get(token, 0)

    @property
    def specials(self) -> tuple[str, ...]:
        return self._specials

    # ── Persistence ────────────────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize to a stable JSON string.

        The ordered ``tokens`` list *is* the id mapping (index = id), so we store
        it as a list rather than a dict — that keeps ids explicit and the file
        diff-friendly across versions.
        """
        payload = {
            "version": 1,
            "specials": list(self._specials),
            "tokens": self._id_to_token,
            "frequencies": self._frequencies,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "Vocabulary":
        payload = json.loads(data)
        return cls(payload["tokens"], payload.get("frequencies", {}),
                   tuple(payload["specials"]))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
