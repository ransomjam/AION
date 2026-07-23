"""Learning rate schedulers for the training loop.

LearningRateScheduler
    Protocol: get_lr(step, total_steps) -> float

ConstantScheduler
    Returns base_lr at every step.

LinearWarmupScheduler
    Linearly increases LR from 0 to base_lr over warmup_steps, then constant.

CosineDecayScheduler
    Linear warmup then cosine decay from base_lr to min_lr.

SchedulerCallback
    TrainingCallback that applies a scheduler by updating optimizer.lr
    at each step.  This is the approved integration point: the scheduler
    lives outside GPTTrainer and is injected via the callback system.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from aion.gpt.trainer import TrainingCallback


class LearningRateScheduler(ABC):
    """Abstract base for all LR schedulers."""

    @abstractmethod
    def get_lr(self, step: int, total_steps: int) -> float:
        """Return the learning rate for the given global step.

        Parameters
        ----------
        step:
            0-based global step count.
        total_steps:
            Total number of steps in the run (epochs × steps_per_epoch).
            May be 0 if unknown.
        """


class ConstantScheduler(LearningRateScheduler):
    """Returns ``base_lr`` at every step."""

    def __init__(self, base_lr: float) -> None:
        self.base_lr = base_lr

    def get_lr(self, step: int, total_steps: int) -> float:
        return self.base_lr


class LinearWarmupScheduler(LearningRateScheduler):
    """Linear warmup from 0 to ``base_lr`` over ``warmup_steps``, then constant."""

    def __init__(self, base_lr: float, warmup_steps: int) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int, total_steps: int) -> float:
        if self.warmup_steps == 0 or step >= self.warmup_steps:
            return self.base_lr
        return self.base_lr * (step + 1) / self.warmup_steps


class CosineDecayScheduler(LearningRateScheduler):
    """Linear warmup then cosine decay from ``base_lr`` to ``min_lr``.

    After warmup, the LR follows:
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
    where progress = (step - warmup_steps) / max(1, total_steps - warmup_steps).
    """

    def __init__(
        self,
        base_lr: float,
        min_lr: float = 1e-5,
        warmup_steps: int = 0,
    ) -> None:
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

    def get_lr(self, step: int, total_steps: int) -> float:
        # Warmup phase
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps

        # Cosine decay phase
        decay_steps = max(1, total_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


def build_scheduler(
    name: str,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr: float = 1e-5,
) -> LearningRateScheduler:
    """Construct a scheduler by name from a ``TrainingConfig``."""
    if name == "constant":
        return ConstantScheduler(base_lr)
    if name == "linear_warmup":
        return LinearWarmupScheduler(base_lr, warmup_steps)
    if name == "cosine":
        return CosineDecayScheduler(base_lr, min_lr=min_lr, warmup_steps=warmup_steps)
    raise ValueError(f"unknown scheduler {name!r}; expected 'constant', 'linear_warmup', 'cosine'")


class SchedulerCallback(TrainingCallback):
    """Applies a ``LearningRateScheduler`` by updating ``optimizer.lr`` each step.

    This is the approved integration point: the scheduler is injected into
    GPTTrainer via the callback system rather than embedded in the loop.
    """

    def __init__(self, scheduler: LearningRateScheduler) -> None:
        self.scheduler = scheduler
        self._lr_history: list[float] = []

    def on_step_end(self, trainer, context: dict) -> None:
        step = context.get("global_step", 0)
        total = context.get("total_steps", 0)
        lr = self.scheduler.get_lr(step, total)
        trainer.optimizer.lr = lr
        self._lr_history.append(lr)

    @property
    def lr_history(self) -> list[float]:
        return list(self._lr_history)
