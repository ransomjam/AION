"""MetricsCollector — accumulates training metrics and writes JSONL logs.

Per-step metrics: step, epoch, train_loss, lr, grad_norm, tokens, elapsed_s
Per-epoch metrics: epoch, mean_loss, val_loss, perplexity, lr, elapsed_s, eta_s

JSONL format: one JSON object per line, appendable and streamable.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from aion.util import now_iso


@dataclass
class TrainingMetrics:
    """Snapshot of the latest training state."""
    epoch: int = 0
    global_step: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    perplexity: float | None = None
    lr: float = 0.0
    grad_norm: float = 0.0
    tokens_per_sec: float = 0.0
    elapsed_s: float = 0.0
    eta_s: float | None = None
    checkpoint_path: str = ""


class MetricsCollector:
    """Accumulate and persist training metrics.

    Parameters
    ----------
    log_path:
        Path to the JSONL file where metrics are appended.
        Parent directory is created on first write.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = Path(log_path)
        self._step_history: list[dict] = []
        self._epoch_history: list[dict] = []
        self._t_start = time.monotonic()

    def record_step(self, step: int, epoch: int, metrics: dict) -> None:
        """Record per-step metrics."""
        record = {"type": "step", "step": step, "epoch": epoch, **metrics}
        self._step_history.append(record)
        self._append(record)

    def record_epoch(self, epoch: int, metrics: dict) -> None:
        """Record per-epoch metrics."""
        record = {
            "type": "epoch",
            "epoch": epoch,
            "timestamp": now_iso(),
            **metrics,
        }
        self._epoch_history.append(record)
        self._append(record)

    def step_history(self) -> list[dict]:
        return list(self._step_history)

    def epoch_history(self) -> list[dict]:
        return list(self._epoch_history)

    def latest(self) -> dict:
        """Return the most recent epoch record, or empty dict."""
        return dict(self._epoch_history[-1]) if self._epoch_history else {}

    def elapsed_s(self) -> float:
        return time.monotonic() - self._t_start

    def _append(self, record: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
