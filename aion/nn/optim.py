"""Optimizers for AION neural network training.

Each optimizer receives a list of ``Parameter`` objects at construction.  On
``step()`` it reads ``param.data`` and ``param.grad`` and updates ``param.data``
in place.  It never touches the module, the computation graph, or any tensor
that is not a registered parameter.

Parameters with ``requires_grad=False`` or ``grad is None`` are silently
skipped.  This means frozen layers and parameters that were not reached during
the forward pass are never updated.

``zero_grad()`` sets ``param.grad = None`` on all owned parameters.  ``None``
is the correct signal for "no gradient computed yet" — distinct from a gradient
that happens to be zero.

Training contract (enforced by ``Trainer``)
-------------------------------------------
1. optimizer.zero_grad()
2. predictions = model(x)
3. loss = loss_fn(predictions, y)
4. loss.backward()
5. optimizer.step()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .parameter import Parameter


class Optimizer(ABC):
    """Abstract base for all AION optimizers."""

    def __init__(self, parameters: list[Parameter], lr: float) -> None:
        self.parameters = parameters
        self.lr = lr

    @abstractmethod
    def step(self) -> None:
        """Update parameter values using accumulated gradients."""

    def zero_grad(self) -> None:
        """Set all parameter gradients to ``None``."""
        for p in self.parameters:
            p.zero_grad()


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum.

    Update rule (no momentum):
        param.data -= lr * param.grad

    Update rule (with momentum):
        velocity = momentum * velocity + param.grad
        param.data -= lr * velocity

    Parameters
    ----------
    parameters:
        Parameters to optimise.
    lr:
        Learning rate.
    momentum:
        Momentum coefficient.  0.0 disables momentum.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        # Per-parameter velocity buffers, initialised lazily.
        self._velocity: dict[int, np.ndarray] = {}

    def step(self) -> None:
        for p in self.parameters:
            if not p.requires_grad or p.grad is None:
                continue
            if self.momentum != 0.0:
                v = self._velocity.get(id(p))
                if v is None:
                    v = np.zeros_like(p.data)
                v = self.momentum * v + p.grad
                self._velocity[id(p)] = v
                p.data -= self.lr * v
            else:
                p.data -= self.lr * p.grad


class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2015).

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param.data -= lr * m_hat / (sqrt(v_hat) + eps)

    Parameters
    ----------
    parameters:
        Parameters to optimise.
    lr:
        Learning rate.  Default 1e-3 is the value from the paper.
    beta1:
        Exponential decay rate for the first moment.
    beta2:
        Exponential decay rate for the second moment.
    eps:
        Small constant for numerical stability.
    """

    def __init__(
        self,
        parameters: list[Parameter],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m: dict[int, np.ndarray] = {}   # first moment
        self._v: dict[int, np.ndarray] = {}   # second moment
        self._t: int = 0                       # step counter

    def step(self) -> None:
        self._t += 1
        for p in self.parameters:
            if not p.requires_grad or p.grad is None:
                continue
            pid = id(p)
            m = self._m.get(pid, np.zeros_like(p.data))
            v = self._v.get(pid, np.zeros_like(p.data))

            m = self.beta1 * m + (1.0 - self.beta1) * p.grad
            v = self.beta2 * v + (1.0 - self.beta2) * (p.grad ** 2)

            self._m[pid] = m
            self._v[pid] = v

            m_hat = m / (1.0 - self.beta1 ** self._t)
            v_hat = v / (1.0 - self.beta2 ** self._t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
