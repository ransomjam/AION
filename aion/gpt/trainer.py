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

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from aion import backend
from aion.backend import xp
from aion.nn.optim import Optimizer

from .loss import CausalLanguageModelLoss
from .model import GPTModel

logger = logging.getLogger("aion.gpt.trainer")


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


def _fmt_dur(seconds: float) -> str:
    """Human-readable duration, e.g. '2h 13m 05s' or '42.3s'."""
    if seconds is None or seconds != seconds or seconds < 0:  # None / NaN / negative
        return "?"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def _rng_state(gen):
    """Snapshot a NumPy Generator's bit-generator state (JSON-serializable)."""
    return gen.bit_generator.state if gen is not None else None


def _global_grad_norm(model: GPTModel) -> float:
    """Compute the global L2 norm of all parameter gradients.

    Each parameter's squared sum is reduced on the compute device, then the
    whole set is brought back in ONE transfer.  Reading them one at a time
    would work, but on a GPU every ``float()`` is a synchronisation point, and
    this runs on every step for every parameter — enough stalls to show up in
    tokens/sec.  The per-parameter reduction dtype and the host-side
    accumulation order are unchanged, so the value is identical to reading
    them individually.
    """
    squares = [xp.sum(p.grad ** 2) for p in model.parameters()
               if p.grad is not None]
    if not squares:
        return 0.0
    total = 0.0
    for s in backend.to_host(xp.stack(squares)):
        total += float(s)
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
        log_every_n_steps: int = 25,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.grad_clip = grad_clip if grad_clip else None
        self.progress_fn = progress_fn
        # Emit an INFO progress line every N training steps (plus the first few
        # steps individually).  Logging only; does not affect the training loop.
        self.log_every_n_steps = log_every_n_steps
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
        data_rng=None,
        start_batch_in_epoch: int = 0,
        checkpoint_fn=None,
        checkpoint_every_n_steps: int = 0,
        checkpoint_every_minutes: float = 0.0,
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
        data_rng:
            The NumPy Generator used by ``train_sampler`` to shuffle each epoch.
            Its state is snapshotted at the start of every epoch so a mid-epoch
            checkpoint can replay the exact same batch order on resume.
        start_batch_in_epoch:
            When resuming mid-epoch, the number of batches already completed in
            the resumed epoch; they are skipped (no forward/backward) so the run
            continues from the exact step it stopped at.
        checkpoint_fn:
            Optional ``checkpoint_fn(reason, trainer_state)`` invoked to persist
            a full checkpoint every ``checkpoint_every_n_steps`` steps and/or
            ``checkpoint_every_minutes`` minutes, and once more on
            ``KeyboardInterrupt`` (reason ``"interrupt"``) before re-raising.
        checkpoint_every_n_steps / checkpoint_every_minutes:
            Step- and wall-clock-based checkpoint cadences (0 disables each).
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

        # ── checkpoint bookkeeping ────────────────────────────────────────────
        run_t0 = time.monotonic()
        last_ckpt_t = run_t0
        steps_this_run = 0
        tokens_this_run = 0
        data_rng_epoch_state = _rng_state(data_rng)  # snapshot before epoch loop

        def _do_checkpoint(reason: str, epoch: int, batch_in_epoch: int,
                           data_state, tokens_seen: int | None = None) -> None:
            nonlocal last_ckpt_t
            if checkpoint_fn is None:
                return
            checkpoint_fn(reason, {
                "reason": reason,
                "global_step": global_step,
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "total_epochs": total_epochs,
                # RNG state at the START of the epoch the resume will begin, so
                # the shuffle (and thus batch order) replays identically.
                "data_rng_epoch_state": data_state,
                # ``total_tokens`` only advances at the epoch boundary, so a
                # mid-epoch checkpoint that read it alone would always record 0.
                # Callers inside the loop pass the running total explicitly.
                "tokens_processed": total_tokens if tokens_seen is None else tokens_seen,
                "history": {
                    "loss_history": list(loss_history),
                    "val_loss_history": list(val_loss_history),
                    "val_perplexity_history": list(val_perplexity_history),
                    "grad_norm_history": list(grad_norm_history),
                    "tokens_per_sec": list(tokens_per_sec_history),
                },
            })
            last_ckpt_t = time.monotonic()

        epoch = start_epoch
        n_batches = 0
        # Absolute position within the current epoch, in the epoch's own batch
        # numbering.  Distinct from ``n_batches`` (which counts only what THIS
        # process ran) and the only value that is correct to checkpoint.
        batch_in_epoch = start_batch_in_epoch
        # Initialised here too, so the KeyboardInterrupt handler can always read
        # them even if the interrupt lands before the first epoch body runs.
        epoch_tokens = 0
        try:
          for epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()
            t_epoch = time.monotonic()
            epoch_loss = 0.0
            epoch_norm = 0.0
            n_batches = 0
            epoch_tokens = 0

            # Snapshot the data RNG at the START of this epoch so a mid-epoch
            # checkpoint can replay the identical shuffle on resume.
            data_rng_epoch_state = _rng_state(data_rng)
            # On the first (resumed) epoch, skip batches already completed.
            skip = start_batch_in_epoch if epoch == start_epoch else 0
            batch_in_epoch = skip

            self._callbacks.on_epoch_begin(self, {"epoch": epoch, "epochs": total_epochs})
            logger.info(
                "Epoch %d/%d starting (~%s steps)%s...",
                epoch + 1, total_epochs, steps_per_epoch or "?",
                f", resuming after {skip} batches" if skip else "",
            )

            batches = train_sampler() if callable(train_sampler) else train_sampler
            for batch_i, (x_ids, y_ids) in enumerate(batches):
                if batch_i < skip:
                    continue  # already done before the interruption
                # x_ids, y_ids: [batch, seq] integer arrays
                step_t0 = time.monotonic()
                first_step = steps_this_run == 0  # first step of this run

                self.optimizer.zero_grad()

                if first_step:
                    logger.info("First forward pass...")
                _t = time.monotonic()
                logits, _ = self.model(x_ids)
                if first_step:
                    logger.info("First forward pass done in %.2fs", time.monotonic() - _t)

                loss = self._loss_fn(logits, y_ids)

                if first_step:
                    logger.info("First backward pass...")
                _t = time.monotonic()
                loss.backward()
                if first_step:
                    logger.info("First backward pass done in %.2fs", time.monotonic() - _t)

                if self.grad_clip:
                    norm = _clip_gradients(self.model, self.grad_clip)
                else:
                    norm = _global_grad_norm(self.model)

                if first_step:
                    logger.info("First optimizer step...")
                _t = time.monotonic()
                self.optimizer.step()
                if first_step:
                    logger.info("First optimizer step done in %.2fs", time.monotonic() - _t)
                global_step += 1

                batch_in_epoch = batch_i + 1
                batch_tokens = int(np.prod(x_ids.shape))
                epoch_tokens += batch_tokens
                step_loss = loss.item()
                epoch_loss += step_loss
                epoch_norm += norm
                n_batches += 1
                steps_this_run += 1
                tokens_this_run += batch_tokens

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

                # ── per-step progress + ETA log (observability only) ───────────
                if steps_this_run <= 5 or (
                    self.log_every_n_steps and steps_this_run % self.log_every_n_steps == 0
                ):
                    run_elapsed = time.monotonic() - run_t0
                    avg_step = run_elapsed / max(1, steps_this_run)
                    steps_per_sec = 1.0 / avg_step if avg_step > 0 else 0.0
                    run_tps = tokens_this_run / run_elapsed if run_elapsed > 0 else 0.0
                    eta_epoch = (steps_per_epoch - (batch_i + 1)) * avg_step \
                        if steps_per_epoch else None
                    eta_train = (total_steps - global_step) * avg_step \
                        if total_steps else None
                    logger.info(
                        "step %d | epoch %d/%d | loss=%.4f | lr=%.2e | %.0f tok/s | "
                        "%.2f steps/s | elapsed %s | ETA epoch %s | ETA train %s",
                        global_step, epoch + 1, total_epochs, step_loss,
                        self.optimizer.lr, run_tps, steps_per_sec,
                        _fmt_dur(run_elapsed), _fmt_dur(eta_epoch), _fmt_dur(eta_train),
                    )

                # ── step / wall-clock checkpoint triggers ──────────────────────
                # Mid-epoch: resume replays THIS epoch, so save its start state.
                #
                # ``batch_in_epoch`` must be the ABSOLUTE position in the epoch
                # (batch_i + 1), not ``n_batches``.  ``n_batches`` resets to 0
                # each epoch and counts only the batches this PROCESS executed,
                # so on a resumed run it is short by exactly ``skip`` — and the
                # next resume would then replay those ``skip`` batches, training
                # on them twice and breaking the "resumes identically to an
                # uninterrupted run" guarantee.  The two are equal only on a
                # fresh run, which is why this survived until a second resume.
                if checkpoint_every_n_steps and global_step % checkpoint_every_n_steps == 0:
                    _do_checkpoint("step_interval", epoch, batch_in_epoch,
                                   data_rng_epoch_state, total_tokens + epoch_tokens)
                elif (checkpoint_every_minutes
                      and (time.monotonic() - last_ckpt_t) >= checkpoint_every_minutes * 60):
                    _do_checkpoint("minutes", epoch, batch_in_epoch,
                                   data_rng_epoch_state, total_tokens + epoch_tokens)

            total_tokens += epoch_tokens
            elapsed = time.monotonic() - t_epoch
            tps = epoch_tokens / elapsed if elapsed > 0 else 0.0
            mean_loss = epoch_loss / max(1, n_batches)
            mean_norm = epoch_norm / max(1, n_batches)

            loss_history.append(round(mean_loss, 6))
            grad_norm_history.append(round(mean_norm, 6))
            tokens_per_sec_history.append(round(tps, 2))

            logger.info(
                "Epoch %d/%d done in %.1fs | train_loss=%.4f | %.0f tok/s",
                epoch + 1, total_epochs, elapsed, mean_loss, tps,
            )

            # ── validation ────────────────────────────────────────────────────
            val_loss = None
            val_ppl = None
            if val_sampler is not None:
                logger.info("Running validation...")
                _t = time.monotonic()
                val_loss, val_ppl = self._evaluate(val_sampler)
                val_loss_history.append(round(val_loss, 6))
                val_perplexity_history.append(round(val_ppl, 4))
                logger.info(
                    "Validation: val_loss=%.4f perplexity=%.2f in %.1fs",
                    val_loss, val_ppl, time.monotonic() - _t,
                )

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

            # Always checkpoint at the epoch boundary when step-checkpointing is
            # active, so ``latest/`` reflects a completed epoch.  batch_in_epoch=0
            # + epoch+1 means resume starts the NEXT epoch cleanly, so save the
            # CURRENT data-RNG state (== that next epoch's start).
            if checkpoint_fn is not None and (
                checkpoint_every_n_steps or checkpoint_every_minutes
            ):
                _do_checkpoint("epoch_end", epoch + 1, 0, _rng_state(data_rng))
        except KeyboardInterrupt:
            # Never lose hours of work: persist a final checkpoint, then re-raise
            # so the caller can report and exit cleanly.  Mid-epoch => replay this
            # epoch, so save its start state.
            logger.warning("KeyboardInterrupt received — saving final checkpoint...")
            _do_checkpoint("interrupt", epoch, batch_in_epoch, data_rng_epoch_state,
                           total_tokens + epoch_tokens)
            raise

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
            total_loss += loss.item()
            n_batches += 1
        mean_loss = total_loss / max(1, n_batches)
        perplexity = math.exp(min(mean_loss, 500.0))  # cap to avoid overflow
        self.model.train()
        return mean_loss, perplexity
