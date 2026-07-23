"""DatasetFingerprint — stable identity for a specific corpus build.

Captures everything that affects the token sequence: dataset content,
tokenizer identity, EOS token, and split fraction.  The ``combined`` field
is the cache key for packed token arrays.

Two training runs with the same ``combined`` value are guaranteed to have
seen the same token sequence in the same order (given the same seed).
"""

from __future__ import annotations

from dataclasses import dataclass

from aion.util import fingerprint as _fp


@dataclass
class DatasetFingerprint:
    dataset_ids: list[str]
    dataset_fingerprints: list[str]   # per-dataset content fingerprints
    tokenizer_id: str
    tokenizer_fingerprint: str
    eos_id: int
    split: float
    max_documents: int | None
    shuffle_seed: int | None = None   # document-shuffle seed (None = source order)
    combined: str = ""                # computed; set by build()

    @classmethod
    def build(
        cls,
        dataset_ids: list[str],
        dataset_fingerprints: list[str],
        tokenizer_id: str,
        tokenizer_fingerprint: str,
        eos_id: int,
        split: float,
        max_documents: int | None,
        shuffle_seed: int | None = None,
    ) -> "DatasetFingerprint":
        hash_args = [
            dataset_ids,
            dataset_fingerprints,
            tokenizer_id,
            tokenizer_fingerprint,
            eos_id,
            split,
            max_documents,
        ]
        # Only fold the shuffle seed into the hash when it is set, so unshuffled
        # builds keep the exact fingerprint they had before this field existed
        # (backward compatibility with pre-existing caches and fingerprints).
        if shuffle_seed is not None:
            hash_args.append(("shuffle_seed", shuffle_seed))
        combined = _fp(*hash_args)
        return cls(
            dataset_ids=dataset_ids,
            dataset_fingerprints=dataset_fingerprints,
            tokenizer_id=tokenizer_id,
            tokenizer_fingerprint=tokenizer_fingerprint,
            eos_id=eos_id,
            split=split,
            max_documents=max_documents,
            shuffle_seed=shuffle_seed,
            combined=combined,
        )

    def to_dict(self) -> dict:
        return {
            "dataset_ids": self.dataset_ids,
            "dataset_fingerprints": self.dataset_fingerprints,
            "tokenizer_id": self.tokenizer_id,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "eos_id": self.eos_id,
            "split": self.split,
            "max_documents": self.max_documents,
            "shuffle_seed": self.shuffle_seed,
            "combined": self.combined,
        }
