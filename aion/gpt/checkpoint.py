"""GPTCheckpoint — save and restore mid-training state.

Layout under ``projects/<p>/checkpoints/<run_id>/``:
    checkpoint_<epoch>.npz      parameter arrays (index-prefixed keys)
    checkpoint_<epoch>.json     epoch metadata (loss, config snapshot)

The most recent checkpoint can be loaded to resume training or inspect
intermediate model quality without committing to GPTStore.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aion.util import now_iso


def _param_key(i: int, p) -> str:
    return f"p{i}__{p.name or 'param'}"


class GPTCheckpoint:
    """Save and load GPT training checkpoints.

    Parameters
    ----------
    checkpoints_dir:
        Root checkpoints directory for the project
        (``project.dir('checkpoints')``).
    run_id:
        Unique identifier for this training run.  All checkpoints for the
        run are stored under ``checkpoints_dir / run_id /``.
    """

    def __init__(self, checkpoints_dir: Path, run_id: str) -> None:
        self.run_dir = Path(checkpoints_dir) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _optim_path(self, epoch: int) -> Path:
        return self.run_dir / f"optim_{epoch}.npz"

    def save(self, model, epoch: int, metrics: dict, optimizer=None) -> Path:
        """Save model parameters and epoch metadata.

        Parameters
        ----------
        model:
            ``GPTModel`` instance.
        epoch:
            Current epoch number (1-based).
        metrics:
            Dict of scalar metrics to record (loss, val_loss, etc.).
        optimizer:
            Optional optimizer whose state (moment buffers + step counter) is
            saved to ``optim_<epoch>.npz`` and whose scalars are recorded in the
            metadata under ``"optimizer"`` so training can resume seamlessly.

        Returns
        -------
        Path to the saved ``.npz`` file.
        """
        params = model.parameters()
        arrays = {_param_key(i, p): p.data for i, p in enumerate(params)}
        npz_path = self.run_dir / f"checkpoint_{epoch}.npz"
        np.savez_compressed(str(npz_path), **arrays)

        optimizer_scalars = None
        if optimizer is not None:
            optimizer_scalars = optimizer.state_dict()
            moment_arrays = optimizer.moment_arrays()
            # np.savez requires at least one array; skip the file if there is no
            # buffered state yet (e.g. an optimizer that has not stepped).
            if moment_arrays:
                np.savez_compressed(str(self._optim_path(epoch)), **moment_arrays)

        meta = {
            "epoch": epoch,
            "param_keys": [_param_key(i, p) for i, p in enumerate(params)],
            "metrics": metrics,
            "optimizer": optimizer_scalars,
            "saved_at": now_iso(),
        }
        (self.run_dir / f"checkpoint_{epoch}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return npz_path

    def load(self, model, epoch: int, optimizer=None) -> dict:
        """Restore model parameters (and optionally optimizer state).

        Parameters
        ----------
        model:
            ``GPTModel`` instance with the same architecture as when saved.
        epoch:
            Epoch number to restore.
        optimizer:
            Optional optimizer to restore in place from the saved state.

        Returns
        -------
        The metadata dict saved alongside the checkpoint.
        """
        npz_path = self.run_dir / f"checkpoint_{epoch}.npz"
        meta_path = self.run_dir / f"checkpoint_{epoch}.json"
        if not npz_path.is_file():
            raise FileNotFoundError(f"no checkpoint for epoch {epoch} in {self.run_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        weights = np.load(str(npz_path))
        param_keys = meta.get("param_keys", [])
        for i, p in enumerate(model.parameters()):
            key = param_keys[i] if i < len(param_keys) else _param_key(i, p)
            if key in weights:
                p.data = weights[key].astype(np.float64)

        if optimizer is not None and meta.get("optimizer") is not None:
            optim_path = self._optim_path(epoch)
            moment_arrays = dict(np.load(str(optim_path))) if optim_path.is_file() else {}
            optimizer.load_state_dict(meta["optimizer"], moment_arrays)
        return meta

    def latest_epoch(self) -> int | None:
        """Return the highest saved epoch number, or None if no checkpoints."""
        epochs = []
        for f in self.run_dir.glob("checkpoint_*.json"):
            try:
                epochs.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
        return max(epochs) if epochs else None

    def list(self) -> list[dict]:
        """Return metadata for all checkpoints in this run, sorted by epoch."""
        metas = []
        for f in sorted(self.run_dir.glob("checkpoint_*.json")):
            try:
                metas.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
        return sorted(metas, key=lambda m: m.get("epoch", 0))
