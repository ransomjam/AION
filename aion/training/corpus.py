"""CorpusManager — assembles the training corpus from project datasets.

Reads from DatasetStore, tokenizes documents, packs into flat token arrays,
splits into train/val by token count, and caches the result under
project/cache/corpus/<fingerprint>/.

The cache is keyed by DatasetFingerprint.combined.  A new tokenizer or
changed dataset produces a new fingerprint and therefore a new cache entry.
Old entries are never automatically invalidated — project/cache is entirely
disposable and may be safely cleared by the user.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aion.datasets.store import DatasetStore
from aion.util import fingerprint as _fp

from .fingerprint import DatasetFingerprint

logger = logging.getLogger("aion.training.corpus")


@dataclass
class CorpusStats:
    n_documents: int
    n_train_tokens: int
    n_val_tokens: int
    n_total_tokens: int
    vocab_coverage: float
    mean_doc_length: float
    median_doc_length: float
    max_doc_length: int
    min_doc_length: int

    def to_dict(self) -> dict:
        return {
            "n_documents": self.n_documents,
            "n_train_tokens": self.n_train_tokens,
            "n_val_tokens": self.n_val_tokens,
            "n_total_tokens": self.n_total_tokens,
            "vocab_coverage": self.vocab_coverage,
            "mean_doc_length": self.mean_doc_length,
            "median_doc_length": self.median_doc_length,
            "max_doc_length": self.max_doc_length,
            "min_doc_length": self.min_doc_length,
        }


@dataclass
class CorpusResult:
    train_tokens: np.ndarray
    val_tokens: np.ndarray
    fingerprint: DatasetFingerprint
    stats: CorpusStats


class CorpusManager:
    """Assemble and cache a training corpus from project datasets.

    Parameters
    ----------
    data_dir:
        The project's ``data/`` directory (``project.dir('data')``).
    cache_dir:
        The project's ``cache/`` directory (``project.dir('cache')``).
    """

    def __init__(self, data_dir: Path, cache_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self._store = DatasetStore(self.data_dir)

    def build(
        self,
        dataset_ids: list[str],
        tokenizer,
        *,
        tokenizer_id: str = "",
        tokenizer_fingerprint: str = "",
        eos_id: int,
        split: float = 0.9,
        max_documents: int | None = None,
        shuffle_seed: int | None = None,
        min_documents: int = 1,
        progress_fn=None,
    ) -> CorpusResult:
        """Build (or load from cache) the packed corpus.

        Parameters
        ----------
        dataset_ids:
            List of dataset ids to include.
        tokenizer:
            A ``Tokenizer`` instance with an ``encode(text) -> list[int]`` method.
        tokenizer_id / tokenizer_fingerprint:
            Identity of the tokenizer for fingerprinting.
        eos_id:
            Token id used as document separator.
        split:
            Fraction of tokens used for training (remainder for validation).
        max_documents:
            Cap on total documents across all datasets.  None = no cap.
        shuffle_seed:
            If set, documents are shuffled deterministically with this seed
            before packing, so the train/val split does not simply take the
            final source in dataset order.  ``None`` preserves source order.
        min_documents:
            Minimum number of documents required; fewer raises ``ValueError``.
        progress_fn:
            Optional ``progress_fn(fraction, message)`` callback.
        """
        # Compute per-dataset fingerprints
        logger.info("Loading corpus: datasets=%s split=%.2f", dataset_ids, split)
        ds_fps = []
        for ds_id in dataset_ids:
            ds = self._store.open(ds_id)
            ds_fps.append(ds.fingerprint())

        fp = DatasetFingerprint.build(
            dataset_ids=dataset_ids,
            dataset_fingerprints=ds_fps,
            tokenizer_id=tokenizer_id,
            tokenizer_fingerprint=tokenizer_fingerprint,
            eos_id=eos_id,
            split=split,
            max_documents=max_documents,
            shuffle_seed=shuffle_seed,
        )

        # Check cache
        t_cache = time.monotonic()
        cached = self._load_cache(fp.combined)
        if cached is not None:
            logger.info(
                "Corpus cache HIT (%s) in %.2fs: %d train / %d val tokens",
                fp.combined[:12], time.monotonic() - t_cache,
                len(cached.train_tokens), len(cached.val_tokens),
            )
            return cached
        logger.info("Corpus cache MISS (%s) — building from documents", fp.combined[:12])

        # Collect documents (source order), then optionally shuffle at the
        # document level.  Shuffling happens BEFORE tokenization/packing so the
        # validation tail is a representative mix of sources, not the last one.
        t_load = time.monotonic()
        doc_texts: list[str] = []
        for ds_id in dataset_ids:
            ds = self._store.open(ds_id)
            for _, text in ds.stream():
                if max_documents is not None and len(doc_texts) >= max_documents:
                    break
                doc_texts.append(text)

        if len(doc_texts) < max(1, min_documents):
            raise ValueError(
                f"corpus has too few documents: found {len(doc_texts)}, "
                f"need at least {max(1, min_documents)}. Prepare more data before training."
            )
        total_chars = sum(len(t) for t in doc_texts)
        logger.info(
            "Loaded %d documents (%d chars) in %.2fs",
            len(doc_texts), total_chars, time.monotonic() - t_load,
        )

        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            order = rng.permutation(len(doc_texts))
            doc_texts = [doc_texts[i] for i in order]
            logger.info("Shuffled documents (seed=%d)", shuffle_seed)

        # Tokenize documents
        logger.info("Tokenizing %d documents...", len(doc_texts))
        t_tok = time.monotonic()
        all_doc_tokens: list[list[int]] = []
        for text in doc_texts:
            all_doc_tokens.append(tokenizer.encode(text))
            n_done = len(all_doc_tokens)
            if n_done % 25 == 0 or n_done == len(doc_texts):
                elapsed = time.monotonic() - t_tok
                rate = n_done / elapsed if elapsed > 0 else 0.0
                logger.info(
                    "  tokenized %d/%d documents (%.1fs, %.1f docs/s)",
                    n_done, len(doc_texts), elapsed, rate,
                )
                if progress_fn:
                    progress_fn(n_done / len(doc_texts), f"tokenized {n_done} documents")

        if not all_doc_tokens:
            raise ValueError("corpus is empty — no documents found in the given datasets")
        n_doc_tokens = sum(len(d) for d in all_doc_tokens)
        logger.info(
            "Tokenized %d documents -> %d tokens in %.2fs",
            len(all_doc_tokens), n_doc_tokens, time.monotonic() - t_tok,
        )

        # Pack into flat array with EOS separators
        logger.info("Packing sequences (EOS id=%d)...", eos_id)
        t_pack = time.monotonic()
        flat: list[int] = []
        for doc in all_doc_tokens:
            flat.extend(doc)
            flat.append(eos_id)
        tokens = np.array(flat, dtype=np.int32)

        # Split by token count
        split_idx = int(len(tokens) * split)
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        logger.info(
            "Packed %d tokens in %.2fs -> %d train / %d val",
            len(tokens), time.monotonic() - t_pack, len(train_tokens), len(val_tokens),
        )

        # Corpus statistics
        doc_lengths = [len(d) for d in all_doc_tokens]
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size and vocab_size > 0:
            seen = len(set(int(t) for t in tokens if t != eos_id))
            vocab_coverage = round(seen / vocab_size, 4)
        else:
            vocab_coverage = 0.0

        stats = CorpusStats(
            n_documents=len(all_doc_tokens),
            n_train_tokens=int(len(train_tokens)),
            n_val_tokens=int(len(val_tokens)),
            n_total_tokens=int(len(tokens)),
            vocab_coverage=vocab_coverage,
            mean_doc_length=round(float(np.mean(doc_lengths)), 2),
            median_doc_length=float(statistics.median(doc_lengths)),
            max_doc_length=int(max(doc_lengths)),
            min_doc_length=int(min(doc_lengths)),
        )

        result = CorpusResult(
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            fingerprint=fp,
            stats=stats,
        )
        t_save = time.monotonic()
        self._save_cache(fp.combined, result)
        logger.info("Cached corpus to disk in %.2fs", time.monotonic() - t_save)
        return result

    # ── cache helpers ─────────────────────────────────────────────────────────

    def _cache_dir(self, combined: str) -> Path:
        return self.cache_dir / "corpus" / combined

    def _load_cache(self, combined: str) -> CorpusResult | None:
        d = self._cache_dir(combined)
        if not (d / "train.npy").is_file():
            return None
        try:
            train_tokens = np.load(str(d / "train.npy"))
            val_tokens = np.load(str(d / "val.npy"))
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            fp_data = meta["fingerprint"]
            fp = DatasetFingerprint(
                dataset_ids=fp_data["dataset_ids"],
                dataset_fingerprints=fp_data["dataset_fingerprints"],
                tokenizer_id=fp_data["tokenizer_id"],
                tokenizer_fingerprint=fp_data["tokenizer_fingerprint"],
                eos_id=fp_data["eos_id"],
                split=fp_data["split"],
                max_documents=fp_data["max_documents"],
                shuffle_seed=fp_data.get("shuffle_seed"),
                combined=fp_data["combined"],
            )
            stats_d = meta["stats"]
            stats = CorpusStats(**stats_d)
            return CorpusResult(
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                fingerprint=fp,
                stats=stats,
            )
        except Exception:
            return None

    def _save_cache(self, combined: str, result: CorpusResult) -> None:
        d = self._cache_dir(combined)
        d.mkdir(parents=True, exist_ok=True)
        np.save(str(d / "train.npy"), result.train_tokens)
        np.save(str(d / "val.npy"), result.val_tokens)
        meta = {
            "fingerprint": result.fingerprint.to_dict(),
            "stats": result.stats.to_dict(),
        }
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
