"""ModelStore — save, list, open, delete, and set-default neural network models.

Mirrors ``TokenizerStore`` and ``EmbeddingStore`` in structure.  Each model
lives under:

    projects/<p>/models/<model_id>/
        manifest.json
        model/
            weights.npz         all parameter arrays (numpy compressed binary)
            architecture.json   layer names, shapes, hyperparams
        training/
            statistics.json     loss_history + full metrics

``weights.npz`` is the correct format for parameter arrays: portable,
compressed, exact float64 representation, zero-dependency within NumPy.
JSON cannot represent float arrays without precision loss.

The store is architecture-agnostic: ``load`` reconstructs the model from
``architecture.json`` using the ``_registry`` dict, then loads weights from
``weights.npz`` by parameter name.  Adding a new architecture is one entry in
``_registry``.
"""

from __future__ import annotations

import io
import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from aion import backend
from aion.util import code_version, fingerprint, now_iso, slugify

SCHEMA_VERSION = 1


class ModelNotFound(Exception):
    pass


def _registry() -> dict:
    """Maps architecture name → Module subclass.  Import lazily."""
    # Populated as model architectures are added.
    return {}


class ModelStore:
    """Create, list, open, delete, and manage models within one project."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)

    # ── internal paths ────────────────────────────────────────────────────────

    def _root(self, model_id: str) -> Path:
        return self.models_dir / model_id

    def _manifest_path(self, model_id: str) -> Path:
        return self._root(model_id) / "manifest.json"

    def _model_dir(self, model_id: str) -> Path:
        return self._root(model_id) / "model"

    def _training_dir(self, model_id: str) -> Path:
        return self._root(model_id) / "training"

    # ── save ──────────────────────────────────────────────────────────────────

    def save(
        self,
        model,
        result,
        *,
        name: str,
        architecture: str,
        description: str = "",
        dataset_id: str = "",
        dataset_fingerprint: str = "",
        tokenizer_id: str = "",
        params: dict | None = None,
    ) -> dict:
        """Persist a trained model and return its manifest.

        Parameters
        ----------
        model:
            A trained ``Module`` instance.
        result:
            A ``TrainingResult`` from ``Trainer.train``.
        architecture:
            Short string identifying the model class (e.g. ``"mlp-v1"``).
        """
        model_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
        model_dir = self._model_dir(model_id)
        training_dir = self._training_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)

        # ── weights.npz ───────────────────────────────────────────────────────
        # Host copies: a saved model is device-neutral, loadable wherever.
        weight_arrays = {
            p.name or f"param_{i}": backend.to_host(p.data)
            for i, p in enumerate(model.parameters())
        }
        weights_path = model_dir / "weights.npz"
        np.savez_compressed(str(weights_path), **weight_arrays)

        # ── architecture.json ─────────────────────────────────────────────────
        arch = {
            "architecture": architecture,
            "param_count": model.param_count(),
            "param_shapes": {
                (p.name or f"param_{i}"): list(p.shape)
                for i, p in enumerate(model.parameters())
            },
            "params": params or {},
        }
        (model_dir / "architecture.json").write_text(
            json.dumps(arch, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ── statistics.json ───────────────────────────────────────────────────
        (training_dir / "statistics.json").write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ── model fingerprint ─────────────────────────────────────────────────
        buf = io.BytesIO()
        np.savez_compressed(buf, **weight_arrays)
        model_fp = fingerprint(buf.getvalue())

        # ── manifest.json ─────────────────────────────────────────────────────
        metrics_summary = {
            k: v for k, v in result.metrics.items() if k != "loss_history"
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "id": model_id,
            "name": name,
            "description": description,
            "architecture": architecture,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "tokenizer_id": tokenizer_id,
            "param_count": model.param_count(),
            "params": params or {},
            "status": "experimental",
            "created_at": now_iso(),
            "model_fingerprint": model_fp,
            "metrics": metrics_summary,
            "produced_by": code_version(),
        }
        self._manifest_path(model_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    # ── read ──────────────────────────────────────────────────────────────────

    def exists(self, model_id: str) -> bool:
        return self._manifest_path(model_id).is_file()

    def open_manifest(self, model_id: str) -> dict:
        path = self._manifest_path(model_id)
        if not path.is_file():
            raise ModelNotFound(f"no model {model_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, model_id: str):
        """Reconstruct a model from disk and return it with weights loaded."""
        manifest = self.open_manifest(model_id)
        architecture = manifest["architecture"]
        reg = _registry()
        if architecture not in reg:
            raise ValueError(f"unknown architecture {architecture!r}")
        arch_data = json.loads(
            (self._model_dir(model_id) / "architecture.json").read_text(encoding="utf-8")
        )
        model = reg[architecture](arch_data["params"])
        weights = np.load(str(self._model_dir(model_id) / "weights.npz"))
        for p in model.parameters():
            key = p.name or f"param_{model.parameters().index(p)}"
            if key in weights:
                p.data = backend.asarray(weights[key], dtype=p.data.dtype)
        return model

    def statistics(self, model_id: str) -> dict | None:
        path = self._training_dir(model_id) / "statistics.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict]:
        if not self.models_dir.is_dir():
            return []
        manifests = []
        for child in sorted(self.models_dir.iterdir()):
            mp = child / "manifest.json"
            if mp.is_file():
                try:
                    manifests.append(json.loads(mp.read_text(encoding="utf-8")))
                except (json.JSONDecodeError, OSError):
                    pass
        manifests.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        return manifests

    def delete(self, model_id: str) -> None:
        root = self._root(model_id)
        if root.is_dir():
            shutil.rmtree(root)

    def set_status(self, model_id: str, status: str) -> dict:
        manifest = self.open_manifest(model_id)
        manifest["status"] = status
        self._manifest_path(model_id).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest
