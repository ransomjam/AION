"""TrainingDashboard — pure formatting layer over MetricsCollector state.

Transforms the in-memory state of a MetricsCollector into a serializable
dict suitable for the application layer to render.  Owns no storage and
reads no files — it only formats what the collector already holds.

DashboardData keys
------------------
status          str     "idle" | "running" | "complete" | "failed"
epoch           int     current epoch (1-based, 0 if not started)
epochs          int     total epochs
global_step     int
train_loss      float | None    latest epoch mean loss
val_loss        float | None    latest epoch val loss
perplexity      float | None    latest epoch perplexity
lr              float           latest learning rate
grad_norm       float           latest mean grad norm
tokens_per_sec  float           latest tokens/sec
elapsed_s       float           total elapsed seconds
eta_s           float | None    estimated seconds remaining
checkpoint_path str             path of latest checkpoint (empty if none)
loss_curve      list[dict]      [{epoch, train_loss, val_loss}]
lr_curve        list[dict]      [{step, lr}]  (sampled, max 500 points)
"""

from __future__ import annotations

from .metrics import MetricsCollector


class TrainingDashboard:
    """Format MetricsCollector state into a serializable dashboard snapshot.

    Parameters
    ----------
    collector:
        The ``MetricsCollector`` for the active run.
    epochs:
        Total number of epochs in the run (for ETA and progress).
    """

    def __init__(self, collector: MetricsCollector, epochs: int) -> None:
        self._collector = collector
        self._epochs = epochs

    def snapshot(self, *, status: str = "running") -> dict:
        """Return a serializable dashboard snapshot.

        Parameters
        ----------
        status:
            Current run status string.
        """
        latest = self._collector.latest()
        epoch_history = self._collector.epoch_history()
        step_history = self._collector.step_history()

        current_epoch = latest.get("epoch", 0)
        elapsed = self._collector.elapsed_s()

        # ETA: linear extrapolation from elapsed / epochs_done
        eta: float | None = None
        if current_epoch > 0 and current_epoch < self._epochs:
            eta = round(elapsed / current_epoch * (self._epochs - current_epoch), 1)

        # Loss curve: one point per epoch
        loss_curve = [
            {
                "epoch": r["epoch"] + 1,
                "train_loss": r.get("mean_loss"),
                "val_loss": r.get("val_loss"),
            }
            for r in epoch_history
        ]

        # LR curve: sampled from step history (max 500 points)
        lr_steps = [
            {"step": r["step"], "lr": r.get("lr", 0.0)}
            for r in step_history
            if "lr" in r
        ]
        if len(lr_steps) > 500:
            stride = len(lr_steps) // 500
            lr_steps = lr_steps[::stride]

        return {
            "status": status,
            "epoch": current_epoch + 1 if epoch_history else 0,
            "epochs": self._epochs,
            "global_step": latest.get("global_step", 0),
            "train_loss": latest.get("mean_loss"),
            "val_loss": latest.get("val_loss"),
            "perplexity": latest.get("perplexity"),
            "lr": latest.get("lr", 0.0),
            "grad_norm": latest.get("grad_norm", 0.0),
            "tokens_per_sec": latest.get("tokens_per_sec", 0.0),
            "elapsed_s": round(elapsed, 1),
            "eta_s": eta,
            "checkpoint_path": latest.get("checkpoint_path", ""),
            "loss_curve": loss_curve,
            "lr_curve": lr_steps,
        }
