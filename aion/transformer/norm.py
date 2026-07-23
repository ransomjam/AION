"""Layer normalization module.

``LayerNorm`` wraps the fused ``layer_norm`` op from ``aion.nn.ops``.  The op
handles the full backward in a single closure; this module owns the learned
``gamma`` (scale) and ``beta`` (shift) parameters.
"""

from __future__ import annotations

import numpy as np

from aion.nn.module import Module
from aion.nn.ops import layer_norm
from aion.nn.parameter import Parameter
from aion.nn.tensor import Tensor


class LayerNorm(Module):
    """Layer normalization over the last axis (Ba et al., 2016).

    Normalizes each token's feature vector to zero mean and unit variance,
    then applies a learned affine transformation::

        y = gamma * (x - mean) / sqrt(var + eps) + beta

    Parameters
    ----------
    d_model:
        Size of the last dimension (the feature dimension).
    eps:
        Small constant added to the variance for numerical stability.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = Parameter(np.ones(d_model), name="gamma")
        self.beta = Parameter(np.zeros(d_model), name="beta")

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.gamma, self.beta, self.eps)

    def __repr__(self) -> str:
        return f"LayerNorm(d_model={self.d_model}, eps={self.eps})"
