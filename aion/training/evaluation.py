"""EvaluationRunner — compute validation loss and perplexity.

Thin wrapper around GPTTrainer._evaluate that can be called independently
of the training loop.  Used by TrainingProject to run evaluation on demand
and by the CheckpointCallback to record val_loss for best-checkpoint tracking.

EvaluationResult
    val_loss        float
    perplexity      float
    n_batches       int
    elapsed_s       float
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from aion.gpt.loss import CausalLanguageModelLoss


@dataclass
class EvaluationResult:
    val_loss: float
    perplexity: float
    n_batches: int
    elapsed_s: float


class EvaluationRunner:
    """Evaluate a GPTModel on a validation sampler.

    Parameters
    ----------
    model:
        A ``GPTModel`` instance.
    max_batches:
        Cap on the number of batches evaluated.  ``None`` = full set.
    """

    def __init__(self, model, *, max_batches: int | None = None) -> None:
        self.model = model
        self.max_batches = max_batches
        self._loss_fn = CausalLanguageModelLoss()

    def evaluate(self, val_sampler) -> EvaluationResult:
        """Run evaluation and return an ``EvaluationResult``.

        Parameters
        ----------
        val_sampler:
            Iterable (or callable returning iterable) of ``(x, y)`` integer
            array pairs.  Not reshuffled.
        """
        self.model.eval()
        t0 = time.monotonic()
        total_loss = 0.0
        n_batches = 0

        batches = val_sampler() if callable(val_sampler) else val_sampler
        for x_ids, y_ids in batches:
            if self.max_batches is not None and n_batches >= self.max_batches:
                break
            logits, _ = self.model(x_ids)
            loss = self._loss_fn(logits, y_ids)
            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        mean_loss = total_loss / max(1, n_batches)
        perplexity = math.exp(min(mean_loss, 500.0))
        return EvaluationResult(
            val_loss=round(mean_loss, 6),
            perplexity=round(perplexity, 4),
            n_batches=n_batches,
            elapsed_s=round(time.monotonic() - t0, 3),
        )
