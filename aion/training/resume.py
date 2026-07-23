"""ResumeTraining — restore a training run from a checkpoint.

Restores model weights, optimizer state (moment buffers + step counter), and
RNG state from the latest checkpoint, and returns the epoch/step to resume
from.  RNG state is serialized to JSON alongside each checkpoint; together with
the restored optimizer state this lets a resumed run continue as if it had
never stopped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .checkpoint import CheckpointManager


class ResumeError(Exception):
    """Raised when ``--resume`` is requested but no resumable run exists."""


@dataclass
class ResumeState:
    """State restored from a checkpoint."""
    start_epoch: int                    # epoch to resume from (0-based)
    start_step: int                     # global step to resume from
    metrics_history: list[dict] = field(default_factory=list)
    rng_states: dict = field(default_factory=dict)


class ResumeTraining:
    """Restore a training run from its latest checkpoint.

    Parameters
    ----------
    checkpoint_manager:
        The ``CheckpointManager`` for the run being resumed.
    logs_dir:
        The run's log directory (``logs/<run_id>/``).
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        logs_dir: Path,
    ) -> None:
        self.manager = checkpoint_manager
        self.logs_dir = Path(logs_dir)

    def can_resume(self) -> bool:
        """Return True if at least one checkpoint exists for this run."""
        return bool(self.manager.list())

    def restore(self, model, optimizer=None) -> ResumeState:
        """Load the latest checkpoint into model and return resume state.

        Parameters
        ----------
        model:
            ``GPTModel`` instance to restore weights into.
        optimizer:
            Optional optimizer.  When given, its moment buffers and step
            counter are restored from the checkpoint so training continues
            with the exact optimizer state it had before stopping.
        """
        meta = self.manager.load_latest(model, optimizer=optimizer)
        if meta is None:
            return ResumeState(start_epoch=0, start_step=0)

        epoch = meta.get("epoch", 0)
        step = meta.get("metrics", {}).get("global_step", 0)

        # Load RNG states if saved
        rng_path = self.logs_dir / f"rng_state_epoch_{epoch}.json"
        rng_states: dict = {}
        if rng_path.is_file():
            try:
                rng_states = json.loads(rng_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        # Load metrics history from JSONL log
        metrics_history: list[dict] = []
        log_path = self.logs_dir / "metrics.jsonl"
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    if r.get("type") == "epoch" and r.get("epoch", 0) < epoch:
                        metrics_history.append(r)
                except (json.JSONDecodeError, ValueError):
                    pass

        return ResumeState(
            start_epoch=epoch,
            start_step=step,
            metrics_history=metrics_history,
            rng_states=rng_states,
        )

    @staticmethod
    def save_rng_state(
        logs_dir: Path,
        epoch: int,
        rng_model: np.random.Generator,
        rng_data: np.random.Generator,
        rng_sample: np.random.Generator,
    ) -> None:
        """Serialize RNG states to JSON for reproducible resume."""
        state = {
            "epoch": epoch,
            "rng_model": rng_model.bit_generator.state,
            "rng_data": rng_data.bit_generator.state,
            "rng_sample": rng_sample.bit_generator.state,
        }
        path = Path(logs_dir) / f"rng_state_epoch_{epoch}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def restore_rng_state(
        rng_states: dict,
        rng_model: np.random.Generator,
        rng_data: np.random.Generator,
        rng_sample: np.random.Generator,
    ) -> None:
        """Restore RNG states from a saved dict."""
        if "rng_model" in rng_states:
            rng_model.bit_generator.state = rng_states["rng_model"]
        if "rng_data" in rng_states:
            rng_data.bit_generator.state = rng_states["rng_data"]
        if "rng_sample" in rng_states:
            rng_sample.bit_generator.state = rng_states["rng_sample"]
