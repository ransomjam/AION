"""GPTStore — save, list, open, delete GPT language models.

Standalone store: not a subclass of TransformerStore.  Reuses the same
index-prefixed parameter keying scheme (``p{i}__{name}``) for weights.npz.

Layout under ``projects/<p>/models/<model_id>/``:
    manifest.json
    model/
        weights.npz         all parameter arrays
        architecture.json   config + architecture string + tying info
    training/
        statistics.json     full training metrics

Architecture identifier: ``"gpt-v1"``
"""

from __future__ import annotations

import io
import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from aion.util import code_version, fingerprint, now_iso, slugify

ARCHITECTURE = "gpt-v1"
SCHEMA_VERSION = 1


class GPTNotFound(Exception):
    pass


def _param_key(i: int, p) -> str:
    return f"p{i}__{p.name or 'param'}"


class GPTStore:
    """Create, list, open, delete, and manage GPT models in a project."""

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
        description: str = "",
        dataset_id: str = "",
        dataset_fingerprint: str = "",
        tokenizer_id: str = "",
        tokenizer_fingerprint: str = "",
        params: dict | None = None,
    ) -> dict:
        """Persist a trained GPTModel and return its manifest.

        Parameters
        ----------
        model:
            A trained ``GPTModel`` instance.
        result:
            A ``GPTTrainingResult`` from ``GPTTrainer.train``.
        """
        model_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
        model_dir = self._model_dir(model_id)
        training_dir = self._training_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)

        # ── weights.npz ───────────────────────────────────────────────────────
        all_params = model.parameters()
        weight_arrays = {_param_key(i, p): p.data for i, p in enumerate(all_params)}
        np.savez_compressed(str(model_dir / "weights.npz"), **weight_arrays)

        # ── architecture.json ─────────────────────────────────────────────────
        cfg_dict = model.cfg.to_dict()
        arch_data = {
            "architecture": ARCHITECTURE,
            "config": cfg_dict,
            "param_count": model.param_count(),
            "param_keys": [_param_key(i, p) for i, p in enumerate(all_params)],
            "weights_tied": model.cfg.tie_weights,
            "extra_params": params or {},
        }
        (model_dir / "architecture.json").write_text(
            json.dumps(arch_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ── statistics.json ───────────────────────────────────────────────────
        (training_dir / "statistics.json").write_text(
            json.dumps(result.metrics, ensure_ascii=False, indent=2), encoding="utf-8"
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
            "architecture": ARCHITECTURE,
            "config": cfg_dict,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "tokenizer_id": tokenizer_id,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "param_count": model.param_count(),
            "weights_tied": model.cfg.tie_weights,
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
            raise GPTNotFound(f"no GPT model {model_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, model_id: str):
        """Reconstruct a GPTModel from disk with weights loaded.

        Handles weight tying: if ``cfg.tie_weights`` is True, the head's W
        is set to the same Parameter object as the embedding table after
        loading (the npz contains only one copy of the tied weight).
        """
        from .config import GPTConfig
        from .model import GPTModel

        manifest = self.open_manifest(model_id)
        arch_data = json.loads(
            (self._model_dir(model_id) / "architecture.json").read_text(encoding="utf-8")
        )
        cfg = GPTConfig.from_dict(arch_data["config"])
        model = GPTModel(cfg)

        weights = np.load(str(self._model_dir(model_id) / "weights.npz"))
        param_keys = arch_data.get("param_keys", [])
        for i, p in enumerate(model.parameters()):
            key = param_keys[i] if i < len(param_keys) else _param_key(i, p)
            if key in weights:
                p.data = weights[key].astype(p.data.dtype)

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
                    m = json.loads(mp.read_text(encoding="utf-8"))
                    if m.get("architecture") == ARCHITECTURE:
                        manifests.append(m)
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
