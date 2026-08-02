"""Trainer — the training loop for AION neural network models.

The ``Trainer`` owns the training loop and enforces the correct gradient
accumulation contract:

    1. optimizer.zero_grad()   — clear gradients from the previous step
    2. predictions = model(x)  — forward pass; builds the computation graph
    3. loss = loss_fn(pred, y) — compute scalar loss; extends the graph
    4. loss.backward()         — reverse pass; accumulates gradients; releases graph
    5. optimizer.step()        — update parameters

This contract is documented here and enforced by the loop.  Callers must not
call ``backward()`` or ``step()`` outside of this sequence.

``Trainer`` does not own the model, the data, or the optimizer configuration.
It calls ``progress_fn(fraction, message)`` once per epoch, matching the
existing ``run_job`` contract so training can be wrapped as a tracked job.

``TrainingResult`` mirrors ``EmbeddingResult``: a metrics dict consumed by
``ExperimentStore.record`` unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .loss import CrossEntropyLoss, MSELoss, BCELoss
from .module import Module
from .optim import Optimizer
from .tensor import Tensor


@dataclass
class TrainingResult:
    """Outcome of a completed training run.

    Mirrors ``EmbeddingResult`` so ``ExperimentStore.record`` can consume it
    without any changes.
    """
    metrics: dict = field(default_factory=dict)
    # Expected keys:
    #   loss_history  list[float] — mean loss per epoch
    #   final_loss    float
    #   epochs        int
    #   param_count   int
    #   training_time_s float
    #   seed          int


class Trainer:
    """Runs the training loop for a ``Module`` model.

    Parameters
    ----------
    model:
        The model to train.
    loss_fn:
        A loss ``Module`` (``CrossEntropyLoss``, ``MSELoss``, ``BCELoss``).
    optimizer:
        An ``Optimizer`` already constructed with ``model.parameters()``.
    progress_fn:
        Optional callback ``progress_fn(fraction: float, message: str)``
        called once per epoch.  Matches the ``run_job`` progress contract.
    """

    def __init__(
        self,
        model: Module,
        loss_fn: Module,
        optimizer: Optimizer,
        *,
        progress_fn=None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.progress_fn = progress_fn

    def train(
        self,
        batches_fn,
        *,
        epochs: int,
        seed: int = 0,
    ) -> TrainingResult:
        """Run the full training loop and return a ``TrainingResult``.

        Parameters
        ----------
        batches_fn:
            Zero-argument callable that returns an iterable of ``(x, y)``
            pairs for one epoch.  Called once per epoch so the data can be
            reshuffled between epochs.  ``x`` must be a ``Tensor`` with
            ``requires_grad=False``; ``y`` must be a plain NumPy array or
            Python sequence.
        epochs:
            Number of full passes over the data.
        seed:
            Random seed recorded in the result for reproducibility.  The
            caller is responsible for seeding NumPy before calling ``train``
            if determinism is required.
        """
        t0 = time.monotonic()
        loss_history: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for x, y in batches_fn():
                # ── training contract ─────────────────────────────────────────
                self.optimizer.zero_grad()          # 1. clear gradients
                predictions = self.model(x)         # 2. forward pass
                loss = self.loss_fn(predictions, y) # 3. compute loss
                loss.backward()                     # 4. backward + graph release
                self.optimizer.step()               # 5. update parameters
                # ─────────────────────────────────────────────────────────────

                epoch_loss += loss.item()
                n_batches += 1

            mean_loss = epoch_loss / max(1, n_batches)
            loss_history.append(round(mean_loss, 6))

            if self.progress_fn:
                self.progress_fn(
                    (epoch + 1) / epochs,
                    f"epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}",
                )

        training_time = time.monotonic() - t0
        return TrainingResult(metrics={
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "epochs": epochs,
            "param_count": self.model.param_count(),
            "training_time_s": round(training_time, 3),
            "seed": seed,
        })
