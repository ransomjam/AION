"""Weight initializers for AION neural network layers.

All functions return ``np.ndarray`` values suitable for wrapping in a
``Parameter``.  Seeded via a ``numpy.random.Generator`` for reproducibility.
"""

from __future__ import annotations

import numpy as np


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Xavier / Glorot uniform initialisation (Glorot & Bengio, 2010).

    Draws from Uniform[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
    Appropriate for tanh and sigmoid activations.
    """
    rng = rng or np.random.default_rng()
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def he_normal(
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """He / Kaiming normal initialisation (He et al., 2015).

    Draws from Normal(0, sqrt(2 / fan_in)).
    Appropriate for ReLU activations.
    """
    rng = rng or np.random.default_rng()
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def uniform(
    low: float,
    high: float,
    shape: tuple,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform initialisation over [low, high)."""
    rng = rng or np.random.default_rng()
    return rng.uniform(low, high, size=shape)


def zeros(shape: tuple | int) -> np.ndarray:
    """Zero initialisation."""
    return np.zeros(shape, dtype=np.float64)


def ones(shape: tuple | int) -> np.ndarray:
    """Ones initialisation."""
    return np.ones(shape, dtype=np.float64)
