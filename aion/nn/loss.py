"""Loss functions for AION neural network training.

Each loss function takes ``predictions`` (a ``Tensor``) and ``targets`` (a
plain NumPy array or Python sequence) and returns a **scalar ``Tensor``**.
The caller calls ``.backward()`` on the returned tensor to propagate gradients.

Loss functions are ``Module`` subclasses so they participate in the same
``forward()`` contract, but they have no trainable parameters.

Implemented losses
------------------
CrossEntropyLoss  — numerically stable log-softmax + NLL; for classification
MSELoss           — mean squared error; for regression
BCELoss           — binary cross-entropy; for binary / multi-label tasks
"""

from __future__ import annotations

import numpy as np

from aion import backend
from aion.backend import xp

from .module import Module
from .ops import log_softmax
from .tensor import Tensor


class CrossEntropyLoss(Module):
    """Numerically stable cross-entropy for multi-class classification.

    Combines log-softmax and negative log-likelihood in a single operation to
    avoid computing softmax and log separately (which is numerically unstable).

    Parameters
    ----------
    predictions:
        Logits of shape ``[batch, num_classes]``.
    targets:
        Integer class indices of shape ``[batch]``.  Not a ``Tensor``.
    """

    def forward(self, predictions: Tensor, targets) -> Tensor:  # type: ignore[override]
        batch = np.asarray(targets, dtype=np.intp).shape[0]
        lsm = log_softmax(predictions, axis=-1)
        # Row and column indices must live wherever the logits live: one pair
        # of index arrays, built once and reused by the backward closure.
        rows = backend.as_index(np.arange(batch))
        cols = backend.as_index(targets)
        nll_data = -lsm.data[rows, cols]
        loss_val = nll_data.mean()

        out = Tensor(loss_val)
        if lsm.requires_grad:
            out.requires_grad = True
            def _backward() -> None:
                g = xp.zeros_like(lsm.data)
                g[rows, cols] = -1.0 / batch
                if lsm.grad is None:
                    lsm.grad = xp.zeros_like(lsm.data)
                lsm.grad += out.grad * g
            out._backward = _backward
            out._inputs = (lsm,)
        return out


class MSELoss(Module):
    """Mean squared error: mean((predictions - targets)^2).

    Parameters
    ----------
    predictions:
        Predicted values, any shape.
    targets:
        Target values, same shape as ``predictions``.  Not a ``Tensor``.
    """

    def forward(self, predictions: Tensor, targets) -> Tensor:  # type: ignore[override]
        targets = backend.asarray(targets, dtype=np.float64)
        diff_data = predictions.data - targets
        loss_val = xp.mean(diff_data ** 2)
        n = predictions.data.size

        out = Tensor(loss_val)
        if predictions.requires_grad:
            out.requires_grad = True
            def _backward() -> None:
                g = out.grad * (2.0 / n) * diff_data
                if predictions.grad is None:
                    predictions.grad = xp.zeros_like(predictions.data)
                predictions.grad += g
            out._backward = _backward
            out._inputs = (predictions,)
        return out


class BCELoss(Module):
    """Binary cross-entropy: -mean(y*log(p) + (1-y)*log(1-p)).

    Expects ``predictions`` to be probabilities in (0, 1) — apply ``Sigmoid``
    before this loss.

    Parameters
    ----------
    predictions:
        Predicted probabilities, any shape.
    targets:
        Binary targets (0 or 1), same shape.  Not a ``Tensor``.
    """

    _EPS = 1e-12

    def forward(self, predictions: Tensor, targets) -> Tensor:  # type: ignore[override]
        targets = backend.asarray(targets, dtype=np.float64)
        p = xp.clip(predictions.data, self._EPS, 1.0 - self._EPS)
        loss_val = -xp.mean(
            targets * xp.log(p) + (1.0 - targets) * xp.log(1.0 - p)
        )
        n = predictions.data.size

        out = Tensor(loss_val)
        if predictions.requires_grad:
            out.requires_grad = True
            def _backward() -> None:
                g = out.grad * (-(targets / p) + (1.0 - targets) / (1.0 - p)) / n
                if predictions.grad is None:
                    predictions.grad = xp.zeros_like(predictions.data)
                predictions.grad += g
            out._backward = _backward
            out._inputs = (predictions,)
        return out
