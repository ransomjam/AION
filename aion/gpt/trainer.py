"""GPTTrainer — language-model training loop.

Differences from the generic ``Trainer``:
- Input batches are (token_ids_x, token_ids_y) integer arrays, not Tensors.
- GPTModel.forward returns (logits, attn_weights); only logits go to the loss.
- Gradient clipping (global L2 norm) applied before optimizer.step().
- Tokens-per-second tracked per epoch.
- Optional validation BatchSampler: val loss + perplexity computed after each
  training epoch.
- ``TrainingCallback`` hooks called at step and epoch boundaries so external
  components (checkpointing, sample generation, LR scheduling, experiment
  logging) can observe and react without modifying the loop.
- Optional ``LearningRateScheduler`` applied per step via the callback system.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from aion.nn.optim import Optimizer

from .loss import CausalLanguageModelLoss
from .model import GPTModel


# ── Callback protocol ─────────────────────────────────────────────────────────

class TrainingCallback:
    """Base training callback.  Override the methods you need.

    All methods are no-ops by default so subclasses only implement what they
    care about.  Multiple callbacks are supported via ``TrainingCallbackList``.

    Step context dict keys
    ----------------------
    epoch, step, global_step, loss, grad_norm, tokens, lr

    Epoch context dict keys
    -----------------------
    epoch, mean_loss, val_loss, val_perplexity, tokens_per_sec, elapsed_s
    """

    def on_train_begin(self, trainer: "GPTTrainer", context: dict) -> None:
        """Called once before the first epoch."""

    def on_epoch_begin(self, trainer: "GPTTrainer", context: dict) -> None:
        """Called at the start of each epoch."""

    def on_step_end(self, trainer: "GPTTrainer", context: dict) -> None:
        """Called after each optimizer step.

        The LR scheduler hook lives here: read ``context['global_step']`` and
        ``context['total_steps']``, then update ``trainer.optimizer.lr``.
        """

    def on_epoch_end(self, trainer: "GPTTrainer", context: dict) -> None:
        """Called at the end of each epoch, after validation."""

    def on_train_end(self, trainer: "GPTTrainer", context: dict) -> None:
        """Called once after the last epoch."""


class TrainingCallbackList:
    """Dispatches events to a list of ``TrainingCallback`` instances."""

    def __init__(self, callbacks: list[TrainingCallback] | None = None) -> None:
        self._callbacks: list[TrainingCallback] = callbacks or []

    def append(self, cb: TrainingCallback) -> None:
        self._callbacks.append(cb)

    def on_train_begin(self, trainer, ctx):
        for cb in self._callbacks: cb.on_train_begin(trainer, ctx)

    def on_epoch_begin(self, trainer, ctx):
        for cb in self._callbacks: cb.on_epoch_begin(trainer, ctx)

    def on_step_end(self, trainer, ctx):
        for cb in self._callbacks: cb.on_step_end(trainer, ctx)

    def on_epoch_end(self, trainer, ctx):
        for cb in self._callbacks: cb.on_epoch_end(trainer, ctx)

    def on_train_end(self, trainer, ctx):
        for cb in self._callbacks: cb.on_train_end(trainer, ctx)

    def __len__(self) -> int:
        return len(self._callbacks)


@dataclass
class GPTTrainingResult:
    """Outcome of a completed GPT training run."""
    metrics: dict = field(default_factory=dict)
    # Expected keys:
    #   loss_history          list[float]  — mean train loss per epoch
    #   val_loss_history      list[float]  — mean val loss per epoch ([] if no val)
    #   val_perplexity_history list[float] — perplexity per epoch ([] if no val)
    #   grad_norm_history     list[float]  — mean grad norm per epoch
    #   tokens_per_sec        list[float]  — tokens/sec per epoch
    #   final_loss            float
    #   final_val_loss        float | None
    #   final_val_perplexity  float | None
    #   epochs                int
    #   param_count           int
    #   training_time_s       float
    #   seed                  int
    #   tokens_processed      int


def _global_grad_norm(model: GPTModel) -> float:
    """Compute the global L2 norm of all parameter gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(np.sum(p.grad ** 2))
    return math.sqrt(total)


def _clip_gradients(model: GPTModel, max_norm: float) -> float:
    """Clip gradients in-place by global L2 norm.  Returns the pre-clip norm."""
    norm = _global_grad_norm(model)
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= scale
    return norm


class GPTTrainer:
    """Training loop for ``GPTModel``.

    Parameters
    ----------
    model:
        The ``GPTModel`` to train.
    optimizer:
        An ``Optimizer`` constructed with ``model.parameters()``.
    grad_clip:
        Maximum global gradient L2 norm.  Set to ``None`` or ``0`` to
        disable clipping.  Default is ``1.0``.
    callbacks:
        ``TrainingCallback`` instances (or a ``TrainingCallbackList``) called
        at step and epoch boundaries.  Use these for LR scheduling,
        checkpointing, sample generation, and experiment logging.
    progress_fn:
        Optional callback ``progress_fn(fraction, message)`` called once
        per epoch.  Matches the ``run_job`` progress contract.
    """

    def __init__(
        self,
        model: GPTModel,
        optimizer: Optimizer,
        *,
        grad_clip: float | None = 1.0,
        callbacks: list[TrainingCallback] | TrainingCallbackList | None = None,
        progress_fn=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip if grad_clip else None
        self.progress_fn = progress_fn
        self._loss_fn = CausalLanguageModelLoss()
        if isinstance(callbacks, TrainingCallbackList):
            self._callbacks = callbacks
        else:
            self._callbacks = TrainingCallbackList(callbacks)

    def train(
        self,
        train_sampler,
        *,
        epochs: int,
        val_sampler=None,
        seed: int = 0,
        start_epoch: int = 0,
        start_step: int = 0,
        total_epochs: int | None = None,
        history: dict | None = None,
    ) -> GPTTrainingResult:
        """Run the full training loop.

        Parameters
        ----------
        train_sampler:
            Iterable (or callable returning iterable) of ``(x, y)`` integer
            array pairs.  If callable, called once per epoch for reshuffling.
        epochs:
            Number of epochs to run in THIS call (the remaining epochs when
            resuming).
        val_sampler:
            Optional iterable of ``(x, y)`` pairs for validation.  Evaluated
            after each training epoch.  Not reshuffled between epochs.
        seed:
            Recorded in the result for reproducibility.
        start_epoch:
            0-based epoch offset for a resumed run.  Epoch labels, checkpoint
            numbers, and RNG-state files continue from here instead of restarting
            at 0.
        start_step:
            Initial global step for a resumed run so the LR schedule and step
            counter continue instead of restarting warmup.
        total_epochs:
            Total epochs across the whole run (used for the scheduler's
            ``total_steps`` denominator and progress).  Defaults to
            ``start_epoch + epochs``.
        history:
            Optional prior per-epoch metric lists to prepend so the returned
            history spans the full run, not just the resumed epochs.
        """
        t_total = time.monotonic()
        total_epochs = total_epochs if total_epochs is not None else start_epoch + epochs

        history = history or {}
        loss_history: list[float] = list(history.get("loss_history", []))
        val_loss_history: list[float] = list(history.get("val_loss_history", []))
        val_perplexity_history: list[float] = list(history.get("val_perplexity_history", []))
        grad_norm_history: list[float] = list(history.get("grad_norm_history", []))
        tokens_per_sec_history: list[float] = list(history.get("tokens_per_sec", []))
        total_tokens = 0
        global_step = start_step

        # Estimate total steps for scheduler callbacks (best-effort).
        # Samplers may not have __len__; fall back to 0 (unknown).
        try:
            steps_per_epoch = len(train_sampler) if not callable(train_sampler) \
                else len(train_sampler())
        except Exception:
            steps_per_epoch = 0
        total_steps = total_epochs * steps_per_epoch

        self._callbacks.on_train_begin(self, {
            "epochs": total_epochs, "total_steps": total_steps, "seed": seed,
        })

        for epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()
            t_epoch = time.monotonic()
            epoch_loss = 0.0
            epoch_norm = 0.0
            n_batches = 0
            epoch_tokens = 0

            self._callbacks.on_epoch_begin(self, {"epoch": epoch, "epochs": total_epochs})

            batches = train_sampler() if callable(train_sampler) else train_sampler
            for x_ids, y_ids in batches:
                # x_ids, y_ids: [batch, seq] integer arrays
                self.optimizer.zero_grad()
                logits, _ = self.model(x_ids)
                loss = self._loss_fn(logits, y_ids)
                loss.backward()

                if self.grad_clip:
                    norm = _clip_gradients(self.model, self.grad_clip)
                else:
                    norm = _global_grad_norm(self.model)

                self.optimizer.step()
                global_step += 1

                batch_tokens = int(np.prod(x_ids.shape))
                epoch_tokens += batch_tokens
                step_loss = float(loss.data.flat[0])
                epoch_loss += step_loss
                epoch_norm += norm
                n_batches += 1

                self._callbacks.on_step_end(self, {
                    "epoch": epoch,
                    "step": n_batches - 1,
                    "global_step": global_step,
                    "total_steps": total_steps,
                    "loss": step_loss,
                    "grad_norm": norm,
                    "tokens": batch_tokens,
                    "lr": self.optimizer.lr,
                })

            total_tokens += epoch_tokens
            elapsed = time.monotonic() - t_epoch
            tps = epoch_tokens / elapsed if elapsed > 0 else 0.0
            mean_loss = epoch_loss / max(1, n_batches)
            mean_norm = epoch_norm / max(1, n_batches)

            loss_history.append(round(mean_loss, 6))
            grad_norm_history.append(round(mean_norm, 6))
            tokens_per_sec_history.append(round(tps, 2))

            # ── validation ────────────────────────────────────────────────────
            val_loss = None
            val_ppl = None
            if val_sampler is not None:
                val_loss, val_ppl = self._evaluate(val_sampler)
                val_loss_history.append(round(val_loss, 6))
                val_perplexity_history.append(round(val_ppl, 4))

            self._callbacks.on_epoch_end(self, {
                "epoch": epoch,
                "epochs": total_epochs,
                "mean_loss": mean_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "tokens_per_sec": tps,
                "elapsed_s": elapsed,
                "global_step": global_step,
            })

            if self.progress_fn:
                msg = f"epoch {epoch + 1}/{total_epochs}  loss={mean_loss:.4f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.4f}  ppl={val_ppl:.2f}"
                self.progress_fn((epoch + 1) / total_epochs, msg)

        training_time = time.monotonic() - t_total
        final_val_loss = val_loss_history[-1] if val_loss_history else None
        final_val_ppl = val_perplexity_history[-1] if val_perplexity_history else None

        result = GPTTrainingResult(metrics={
            "loss_history": loss_history,
            "val_loss_history": val_loss_history,
            "val_perplexity_history": val_perplexity_history,
            "grad_norm_history": grad_norm_history,
            "tokens_per_sec": tokens_per_sec_history,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "final_val_loss": final_val_loss,
            "final_val_perplexity": final_val_ppl,
            "epochs": total_epochs,
            "param_count": self.model.param_count(),
            "training_time_s": round(training_time, 3),
            "seed": seed,
            "tokens_processed": total_tokens,
        })

        self._callbacks.on_train_end(self, {
            "epochs": total_epochs,
            "total_steps": global_step,
            "training_time_s": training_time,
        })
        return result

    def _evaluate(self, val_sampler) -> tuple[float, float]:
        """Compute mean validation loss and perplexity (no gradients)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        batches = val_sampler() if callable(val_sampler) else val_sampler
        for x_ids, y_ids in batches:
            logits, _ = self.model(x_ids)
            loss = self._loss_fn(logits, y_ids)
            total_loss += float(loss.data.flat[0])
            n_batches += 1
        mean_loss = total_loss / max(1, n_batches)
        perplexity = math.exp(min(mean_loss, 500.0))  # cap to avoid overflow
        self.model.train()
        return mean_loss, perplexity
