"""TokenizerStore — save, list, open, delete, and set-default tokenizers.

Mirrors DatasetStore in structure.  Each tokenizer lives under:

    projects/<p>/tokenizers/<tokenizer_id>/
        manifest.json
        model/
            merges.json
            vocabulary.json
        training/
            statistics.json

The store is algorithm-agnostic: it dispatches ``load`` via the ``algorithm``
field in ``manifest.json`` using the ``_REGISTRY`` dict.  Adding a new
algorithm is one line in that dict.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from aion.util import code_version, fingerprint, now_iso, slugify

SCHEMA_VERSION = 1


class TokenizerNotFound(Exception):
    pass


class TokenizerExists(Exception):
    pass


# ── Algorithm registry ────────────────────────────────────────────────────────
# Maps algorithm id -> Tokenizer subclass.  Import lazily to avoid circular deps.

def _registry() -> dict:
    from .bpe import ByteLevelBPETokenizer
    return {
        "bpe-byte-v1": ByteLevelBPETokenizer,
    }


# ── Store ─────────────────────────────────────────────────────────────────────

class TokenizerStore:
    """Create, list, open, delete, and manage tokenizers within one project."""

    def __init__(self, tokenizers_dir: Path):
        self.tokenizers_dir = Path(tokenizers_dir)

    # ── internal paths ────────────────────────────────────────────────────────

    def _root(self, tokenizer_id: str) -> Path:
        return self.tokenizers_dir / tokenizer_id

    def _manifest_path(self, tokenizer_id: str) -> Path:
        return self._root(tokenizer_id) / "manifest.json"

    def _model_dir(self, tokenizer_id: str) -> Path:
        return self._root(tokenizer_id) / "model"

    def _training_dir(self, tokenizer_id: str) -> Path:
        return self._root(tokenizer_id) / "training"

    # ── save (called after training) ──────────────────────────────────────────

    def save(
        self,
        tokenizer,
        training_result,
        *,
        name: str,
        description: str = "",
        dataset_id: str = "",
        dataset_fingerprint: str = "",
        params: dict | None = None,
    ) -> dict:
        """Persist a trained tokenizer and return its manifest."""
        tokenizer_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
        root = self._root(tokenizer_id)
        model_dir = self._model_dir(tokenizer_id)
        training_dir = self._training_dir(tokenizer_id)

        model_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)

        # Write model files
        tokenizer.save(model_dir)

        # Write statistics
        stats = {
            "merge_history": [
                {
                    "rank": s.rank,
                    "pair": list(s.pair),
                    "new_token": s.new_token,
                    "pair_freq": s.pair_freq,
                    "vocab_size": s.vocab_size,
                    "corpus_tokens": s.corpus_tokens,
                }
                for s in training_result.merge_history
            ],
            **training_result.metrics,
        }
        (training_dir / "statistics.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Compute vocabulary fingerprint from the saved vocabulary file
        vocab_text = (model_dir / "vocabulary.json").read_text(encoding="utf-8")
        vocab_fp = fingerprint(vocab_text)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "id": tokenizer_id,
            "name": name,
            "description": description,
            "algorithm": tokenizer.algorithm,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "vocab_size": tokenizer.vocab_size,
            "merge_count": len(training_result.merge_history),
            "special_tokens": list(("<pad>", "<unk>", "<bos>", "<eos>")),
            "params": params or {},
            "status": "experimental",
            "created_at": now_iso(),
            "vocabulary_fingerprint": vocab_fp,
            "metrics": training_result.metrics,
            "produced_by": code_version(),
        }
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    # ── read ──────────────────────────────────────────────────────────────────

    def exists(self, tokenizer_id: str) -> bool:
        return self._manifest_path(tokenizer_id).is_file()

    def open_manifest(self, tokenizer_id: str) -> dict:
        path = self._manifest_path(tokenizer_id)
        if not path.is_file():
            raise TokenizerNotFound(f"no tokenizer {tokenizer_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, tokenizer_id: str):
        """Load and return a trained Tokenizer instance."""
        manifest = self.open_manifest(tokenizer_id)
        algorithm = manifest["algorithm"]
        reg = _registry()
        if algorithm not in reg:
            raise ValueError(f"unknown algorithm {algorithm!r}")
        return reg[algorithm].load(self._model_dir(tokenizer_id))

    def statistics(self, tokenizer_id: str) -> dict | None:
        path = self._training_dir(tokenizer_id) / "statistics.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        if not self.tokenizers_dir.is_dir():
            return []
        manifests = []
        for child in sorted(self.tokenizers_dir.iterdir()):
            mp = child / "manifest.json"
            if mp.is_file():
                try:
                    manifests.append(json.loads(mp.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass
        manifests.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return manifests

    def delete(self, tokenizer_id: str) -> None:
        root = self._root(tokenizer_id)
        if root.is_dir():
            shutil.rmtree(root)

    def set_status(self, tokenizer_id: str, status: str) -> dict:
        """Update the status field in the manifest (e.g. 'default')."""
        manifest = self.open_manifest(tokenizer_id)
        manifest["status"] = status
        self._manifest_path(tokenizer_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
