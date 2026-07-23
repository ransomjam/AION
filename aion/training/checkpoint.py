"""CheckpointManager — checkpoint lifecycle management for a training run.

Wraps GPTCheckpoint and adds:
- Automatic pruning: keeps only the last N checkpoints.
- Best-checkpoint tracking: copies the checkpoint with the lowest val_loss
  to a ``best/`` subdirectory.
- Checkpoint manifest: checkpoints/<run_id>/manifest.json listing all saved
  checkpoints with their metrics.

The ``best/`` checkpoint is a file copy (not a symlink) for Windows compatibility.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from aion.gpt.checkpoint import GPTCheckpoint
from aion.gpt.trainer import TrainingCallback
from aion.util import now_iso


class CheckpointManager:
    """Manage checkpoints for a single training run.

    Parameters
    ----------
    checkpoints_dir:
        Project checkpoints directory (``project.dir('checkpoints')``).
    run_id:
        Unique identifier for this training run.
    keep_last_n:
        Maximum number of regular checkpoints to retain.
    """

    def __init__(
        self,
        checkpoints_dir: Path,
        run_id: str,
        keep_last_n: int = 3,
    ) -> None:
        self._ckpt = GPTCheckpoint(checkpoints_dir, run_id)
        self.run_dir = self._ckpt.run_dir
        self.keep_last_n = keep_last_n
        self._best_val_loss: float = float("inf")
        self._manifest_path = self.run_dir / "manifest.json"

    def save(self, model, epoch: int, metrics: dict, optimizer=None) -> Path:
        """Save a checkpoint and update the manifest.

        If ``optimizer`` is given, its state is checkpointed alongside the model
        so a resumed run continues from the exact optimizer state.  If
        ``metrics`` contains ``val_loss`` and it is the best seen so far, the
        checkpoint is also copied to ``best/``.
        """
        path = self._ckpt.save(model, epoch, metrics, optimizer=optimizer)
        self._update_manifest(epoch, metrics, str(path))
        self.prune()

        val_loss = metrics.get("val_loss")
        if val_loss is not None and val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._copy_best(epoch)

        return path

    def load_latest(self, model, optimizer=None) -> dict | None:
        """Load the most recent checkpoint into model.  Returns metadata or None.

        If ``optimizer`` is given, its state is restored too.
        """
        epoch = self._ckpt.latest_epoch()
        if epoch is None:
            return None
        return self._ckpt.load(model, epoch, optimizer=optimizer)

    def load_best(self, model) -> dict | None:
        """Load the best (lowest val_loss) checkpoint into model."""
        best_dir = self.run_dir / "best"
        meta_path = best_dir / "best.json"
        if not meta_path.is_file():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        epoch = meta.get("epoch")
        if epoch is None:
            return None
        # Temporarily load from best/ by copying back to run_dir
        best_npz = best_dir / f"checkpoint_{epoch}.npz"
        if not best_npz.is_file():
            return None
        tmp_npz = self.run_dir / f"checkpoint_{epoch}.npz"
        tmp_json = self.run_dir / f"checkpoint_{epoch}.json"
        shutil.copy2(best_npz, tmp_npz)
        shutil.copy2(best_dir / f"checkpoint_{epoch}.json", tmp_json)
        result = self._ckpt.load(model, epoch)
        return result

    def load_at_epoch(self, model, epoch: int) -> dict:
        return self._ckpt.load(model, epoch)

    def list(self) -> list[dict]:
        """Return all checkpoint entries from the manifest, sorted by epoch."""
        if not self._manifest_path.is_file():
            return []
        data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return sorted(data.get("checkpoints", []), key=lambda c: c.get("epoch", 0))

    def prune(self) -> None:
        """Delete oldest checkpoints beyond keep_last_n."""
        all_ckpts = self._ckpt.list()
        if len(all_ckpts) <= self.keep_last_n:
            return
        to_delete = all_ckpts[: len(all_ckpts) - self.keep_last_n]
        for meta in to_delete:
            epoch = meta.get("epoch")
            if epoch is None:
                continue
            for suffix in (".npz", ".json"):
                p = self.run_dir / f"checkpoint_{epoch}{suffix}"
                if p.is_file():
                    p.unlink()
            optim_p = self.run_dir / f"optim_{epoch}.npz"
            if optim_p.is_file():
                optim_p.unlink()

    # ── internal ──────────────────────────────────────────────────────────────

    def _update_manifest(self, epoch: int, metrics: dict, path: str) -> None:
        if self._manifest_path.is_file():
            data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        else:
            data = {"checkpoints": []}
        data["checkpoints"].append({
            "epoch": epoch,
            "path": path,
            "metrics": metrics,
            "saved_at": now_iso(),
        })
        self._manifest_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _copy_best(self, epoch: int) -> None:
        best_dir = self.run_dir / "best"
        best_dir.mkdir(exist_ok=True)
        for suffix in (".npz", ".json"):
            src = self.run_dir / f"checkpoint_{epoch}{suffix}"
            if src.is_file():
                shutil.copy2(src, best_dir / f"checkpoint_{epoch}{suffix}")
        optim_src = self.run_dir / f"optim_{epoch}.npz"
        if optim_src.is_file():
            shutil.copy2(optim_src, best_dir / f"optim_{epoch}.npz")
        meta_path = self.run_dir / f"checkpoint_{epoch}.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            (best_dir / "best.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )


class CheckpointCallback(TrainingCallback):
    """Saves a checkpoint at the end of every N epochs via CheckpointManager."""

    def __init__(
        self,
        manager: CheckpointManager,
        every_n_epochs: int = 1,
    ) -> None:
        self.manager = manager
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, context: dict) -> None:
        epoch = context.get("epoch", 0)
        if (epoch + 1) % self.every_n_epochs == 0:
            metrics = {
                k: context[k]
                for k in ("mean_loss", "val_loss", "val_perplexity", "global_step")
                if k in context
            }
            self.manager.save(
                trainer.model, epoch + 1, metrics,
                optimizer=getattr(trainer, "optimizer", None),
            )
