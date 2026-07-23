"""Dropout module.

``Dropout`` applies inverted dropout during training and is an identity at
inference.  It checks ``self.training`` (set by ``Module.train()`` /
``Module.eval()``) to determine which mode it is in.

The dropout mask is a plain ``np.ndarray`` — not a ``Tensor`` and not
differentiable.  The backward passes the gradient through kept positions
only, scaled by ``1 / keep_prob``, via the ``dropout_mask`` op.
"""

from __future__ import annotations

import numpy as np

from aion.nn.module import Module
from aion.nn.ops import dropout_mask
from aion.nn.tensor import Tensor


class Dropout(Module):
    """Inverted dropout: zero a random fraction of activations during training.

    During training, each element is independently zeroed with probability
    ``p`` and the remaining elements are scaled by ``1 / (1 - p)`` so the
    expected value is preserved.

    During evaluation (``self.training == False``), the input is returned
    unchanged.

    Parameters
    ----------
    p:
        Probability of zeroing an element.  ``0.0`` disables dropout.
    """

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = np.random.default_rng().random(x.shape) >= self.p
        return dropout_mask(x, mask, keep_prob)

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
