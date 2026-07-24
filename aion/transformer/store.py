"""TransformerStore — save, list, open, delete transformer models.

Follows the same architectural pattern as ``TokenizerStore`` and
``EmbeddingStore``.  Each model lives under:

    projects/<p>/models/<model_id>/
        manifest.json
        model/
            weights.npz         all parameter arrays (numpy compressed binary)
            architecture.json   config dict + architecture string + tying info
        training/
            statistics.json     loss_history + full metrics

Architecture strings
--------------------
``"encoder-v1"``     → ``TransformerEncoder``  (encoder-only / BERT-style)
``"decoder-v1"``     → ``TransformerStack``    (decoder-only / GPT-style)
``"transformer-v1"`` → ``Transformer``         (full encoder-decoder)

Weight tying
------------
If ``Transformer.tie_weights()`` was called before saving, the manifest
records ``"weights_tied": true`` and ``architecture.json`` records the
parameter names involved.  The logit head itself belongs in GPT/BERT.

Parameter keying
----------------
Parameters are stored in ``weights.npz`` as ``"p{i}__{name}"`` where ``i``
is the zero-based index in ``model.parameters()``.  The index prefix avoids
name collisions across blocks (every block has a ``W_Q``, ``gamma``, etc.).
The load path matches by index, not by name.
"""

from __future__ import annotations

import io
import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from aion.util import code_version, fingerprint, now_iso, slugify

SCHEMA_VERSION = 1


class TransformerNotFound(Exception):
    pass


def _registry() -> dict:
    """Maps architecture string → (config_cls, model_cls).  Imported lazily."""
    from .config import DecoderConfig, EncoderConfig, TransformerConfig
    from .encoder import TransformerEncoder
    from .model import Transformer
    from .stack import TransformerStack
    return {
        "encoder-v1":     (EncoderConfig,     TransformerEncoder),
        "decoder-v1":     (DecoderConfig,     TransformerStack),
        "transformer-v1": (TransformerConfig, Transformer),
    }


def _param_key(i: int, p) -> str:
    return f"p{i}__{p.name or 'param'}"


class TransformerStore:
    """Create, list, open, delete, and manage transformer models in a project."""

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
        """Persist a trained transformer model and return its manifest.

        Parameters
        ----------
        model:
            A trained ``Module`` instance (``TransformerEncoder``,
            ``TransformerStack``, or ``Transformer``).
        result:
            A ``TrainingResult`` from ``Trainer.train``.
        architecture:
            One of ``"encoder-v1"``, ``"decoder-v1"``, ``"transformer-v1"``.
        params:
            Optional extra hyperparameters to record in the manifest.
        """
        if architecture not in _registry():
            raise ValueError(
                f"unknown architecture {architecture!r}; "
                f"expected one of {sorted(_registry())}"
            )

        model_id = f"{slugify(name)}-{uuid.uuid4().hex[:8]}"
        model_dir = self._model_dir(model_id)
        training_dir = self._training_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        training_dir.mkdir(parents=True, exist_ok=True)

        # ── weights.npz ───────────────────────────────────────────────────────
        # Keyed by "p{i}__{name}" to avoid collisions across blocks.
        all_params = model.parameters()
        weight_arrays = {_param_key(i, p): p.data for i, p in enumerate(all_params)}
        np.savez_compressed(str(model_dir / "weights.npz"), **weight_arrays)

        # ── architecture.json ─────────────────────────────────────────────────
        cfg_dict = model.cfg.to_dict() if hasattr(model, "cfg") else {}
        weights_tied = getattr(model, "weights_tied", False)
        tied_names: dict = {}
        if weights_tied:
            tied = object.__getattribute__(model, "_tied_weights")
            tied_names = {role: (p.name or "") for role, p in tied.items()}
        arch_data = {
            "architecture": architecture,
            "config": cfg_dict,
            "param_count": model.param_count(),
            "param_keys": [_param_key(i, p) for i, p in enumerate(all_params)],
            "weights_tied": weights_tied,
            "tied_weight_names": tied_names,
            "extra_params": params or {},
        }
        (model_dir / "architecture.json").write_text(
            json.dumps(arch_data, ensure_ascii=False, indent=2), encoding="utf-8"
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
        metrics_summary = {k: v for k, v in result.metrics.items() if k != "loss_history"}
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "id": model_id,
            "name": name,
            "description": description,
            "architecture": architecture,
            "config": cfg_dict,
            "dataset_id": dataset_id,
            "dataset_fingerprint": dataset_fingerprint,
            "tokenizer_id": tokenizer_id,
            "param_count": model.param_count(),
            "weights_tied": weights_tied,
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
            raise TransformerNotFound(f"no model {model_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, model_id: str):
        """Reconstruct a model from disk and return it with weights loaded.

        Reads ``architecture.json`` to reconstruct the config, constructs the
        model, then loads weights from ``weights.npz`` by position index.
        """
        manifest = self.open_manifest(model_id)
        architecture = manifest["architecture"]
        reg = _registry()
        if architecture not in reg:
            raise ValueError(f"unknown architecture {architecture!r}")

        arch_data = json.loads(
            (self._model_dir(model_id) / "architecture.json").read_text(encoding="utf-8")
        )
        config_cls, model_cls = reg[architecture]
        cfg = config_cls.from_dict(arch_data["config"])

        # Construct the model.  Order matters: Transformer check must come
        # before TransformerStack because TransformerEncoder is a subclass of
        # TransformerStack but is not a Transformer.
        from .model import Transformer
        from .stack import TransformerStack
        if model_cls is Transformer:
            model = Transformer(cfg)
        elif issubclass(model_cls, TransformerStack):
            # decoder-v1 uses TransformerStack directly (no from_config).
            model = model_cls(
                d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                d_ff=cfg.d_ff, dropout=cfg.dropout,
                activation=cfg.activation, pre_norm=cfg.pre_norm,
            )
        else:
            model = model_cls.from_config(cfg)

        # Load weights by position index.
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
