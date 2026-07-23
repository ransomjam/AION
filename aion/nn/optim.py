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

    # ── checkpointing ──────────────────────────────────────────────────────────
    # State is split into JSON-serializable scalars (``state_dict``) and NumPy
    # buffers (``moment_arrays``) so a checkpoint can persist scalars in its
    # metadata and buffers in an ``.npz`` alongside the model weights.  Buffers
    # are keyed by PARAMETER INDEX (not ``id(p)``, which is not stable across
    # processes) so they re-bind correctly to a freshly constructed optimizer.

    def state_dict(self) -> dict:
        """Return JSON-serializable optimizer scalars."""
        return {"type": "optimizer", "lr": self.lr}

    def moment_arrays(self) -> dict[str, np.ndarray]:
        """Return per-parameter NumPy state buffers, keyed for ``.npz`` storage."""
        return {}

    def load_state_dict(self, scalars: dict, arrays: dict) -> None:
        """Restore optimizer state from ``state_dict`` scalars and buffers."""
        if "lr" in scalars:
            self.lr = scalars["lr"]


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

    def state_dict(self) -> dict:
        return {"type": "sgd", "lr": self.lr, "momentum": self.momentum}

    def moment_arrays(self) -> dict[str, np.ndarray]:
        idx = {id(p): i for i, p in enumerate(self.parameters)}
        return {f"vel_{idx[pid]}": v for pid, v in self._velocity.items()}

    def load_state_dict(self, scalars: dict, arrays: dict) -> None:
        if "lr" in scalars:
            self.lr = scalars["lr"]
        if "momentum" in scalars:
            self.momentum = scalars["momentum"]
        self._velocity = {}
        for i, p in enumerate(self.parameters):
            key = f"vel_{i}"
            if key in arrays:
                self._velocity[id(p)] = np.asarray(arrays[key])

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

    def state_dict(self) -> dict:
        return {
            "type": "adam", "lr": self.lr, "beta1": self.beta1,
            "beta2": self.beta2, "eps": self.eps, "t": self._t,
        }

    def moment_arrays(self) -> dict[str, np.ndarray]:
        idx = {id(p): i for i, p in enumerate(self.parameters)}
        arrays: dict[str, np.ndarray] = {}
        for pid, m in self._m.items():
            arrays[f"m_{idx[pid]}"] = m
        for pid, v in self._v.items():
            arrays[f"v_{idx[pid]}"] = v
        return arrays

    def load_state_dict(self, scalars: dict, arrays: dict) -> None:
        if "lr" in scalars:
            self.lr = scalars["lr"]
        self.beta1 = scalars.get("beta1", self.beta1)
        self.beta2 = scalars.get("beta2", self.beta2)
        self.eps = scalars.get("eps", self.eps)
        self._t = int(scalars.get("t", 0))
        self._m = {}
        self._v = {}
        for i, p in enumerate(self.parameters):
            mk, vk = f"m_{i}", f"v_{i}"
            if mk in arrays:
                self._m[id(p)] = np.asarray(arrays[mk])
            if vk in arrays:
                self._v[id(p)] = np.asarray(arrays[vk])

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
