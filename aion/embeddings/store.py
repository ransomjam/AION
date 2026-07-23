"""EmbeddingStore — save, list, open, delete, and set-default embeddings.

Mirrors TokenizerStore in structure.  Each embedding lives under:

    projects/<p>/embeddings/<embedding_id>/
        manifest.json
        model/
            vectors.json
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


class EmbeddingNotFound(Exception):
    pass


def _registry() -> dict:
    from .cbow import CBOWEmbedding
    return {
        "cbow-v1": CBOWEmbedding,
    }


class EmbeddingStore:
    """Create, list, open, delete, and manage embeddings within one project."""

    def __init__(self, embeddings_dir: Path):
        self.embeddings_dir = Path(embeddings_dir)

    # ── internal paths ────────────────────────────────────────────────────────

    def _root(self, embedding_id: str) -> Path:
        return self.embeddings_dir / embedding_id

    def _manifest_path(self, embedding_id: str) -> Path:
        return self._root(embedding_id) / "manifest.json"

    def _model_dir(self, embedding_id: str) -> Path:
        return self._root(embedding_id) / "model"

    def _training_dir(self, embedding_id: str) -> Path:
        return self._root(embedding_id) / "training"

    # ── save (called after training) ──────────────────────────────────────────

    def save(
        self,
        embedding,
        result,
        *,
        name: str,
        description: str = "",
        tokenizer_id: str = "",
        tokenizer_fingerprint: str = "",
        dataset_id: str = "",
        dataset_fingerprint: str = "",
        params: dict | None = None,
    ) -> dict:
        """Persist a trained embedding and return its manifest."""
        embedding_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
        model_dir = self._model_dir(embedding_id)
        training_dir = self._training_dir(embedding_id)

        model_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)

        embedding.save(model_dir)

        stats = {"loss_history": result.metrics.get("loss_history", []), **result.metrics}
        (training_dir / "statistics.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        vectors_text = (model_dir / "vectors.json").read_text(encoding="utf-8")
        emb_fp = fingerprint(vectors_text)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "id": embedding_id,
            "name": name,
            "description": description,
            "algorithm": embedding.algorithm,
            "tokenizer_id": tokenizer_id,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "vocab_size": embedding.vocab_size,
            "dims": embedding.dims,
            "params": params or {},
            "status": "experimental",
            "created_at": now_iso(),
            "embedding_fingerprint": emb_fp,
            "metrics": {k: v for k, v in result.metrics.items() if k != "loss_history"},
            "produced_by": code_version(),
        }
        self._manifest_path(embedding_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    # ── read ──────────────────────────────────────────────────────────────────

    def exists(self, embedding_id: str) -> bool:
        return self._manifest_path(embedding_id).is_file()

    def open_manifest(self, embedding_id: str) -> dict:
        path = self._manifest_path(embedding_id)
        if not path.is_file():
            raise EmbeddingNotFound(f"no embedding {embedding_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, embedding_id: str):
        """Load and return a trained Embedding instance."""
        manifest = self.open_manifest(embedding_id)
        algorithm = manifest["algorithm"]
        reg = _registry()
        if algorithm not in reg:
            raise ValueError(f"unknown algorithm {algorithm!r}")
        return reg[algorithm].load(self._model_dir(embedding_id))

    def statistics(self, embedding_id: str) -> dict | None:
        path = self._training_dir(embedding_id) / "statistics.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        if not self.embeddings_dir.is_dir():
            return []
        manifests = []
        for child in sorted(self.embeddings_dir.iterdir()):
            mp = child / "manifest.json"
            if mp.is_file():
                try:
                    manifests.append(json.loads(mp.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass
        manifests.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return manifests

    def delete(self, embedding_id: str) -> None:
        root = self._root(embedding_id)
        if root.is_dir():
            shutil.rmtree(root)

    def set_status(self, embedding_id: str, status: str) -> dict:
        manifest = self.open_manifest(embedding_id)
        manifest["status"] = status
        self._manifest_path(embedding_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
