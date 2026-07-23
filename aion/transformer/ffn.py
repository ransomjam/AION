"""Position-wise Feed-Forward Network.

``FeedForward`` applies two linear transformations with an activation in
between::

    FFN(x) = Linear2(activation(Linear1(x)))

The inner dimension ``d_ff`` is typically ``4 * d_model``.  Both ``relu``
and ``gelu`` are supported; ``gelu`` is the modern default for transformer
language models.
"""

from __future__ import annotations

import numpy as np

from aion.nn.init import zeros, xavier_uniform
from aion.nn.module import Module
from aion.nn.ops import add, gelu, matmul, relu
from aion.nn.parameter import Parameter
from aion.nn.tensor import Tensor

from .dropout import Dropout


class FeedForward(Module):
    """Position-wise two-layer feed-forward network.

    Parameters
    ----------
    d_model:
        Input and output dimensionality.
    d_ff:
        Inner (hidden) dimensionality.  Typically ``4 * d_model``.
    activation:
        ``"relu"`` or ``"gelu"``.
    dropout:
        Dropout probability applied after the first linear + activation.
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu",
        dropout: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if activation not in ("relu", "gelu"):
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation!r}")
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation

        rng = rng or np.random.default_rng()
        self.W1 = Parameter(xavier_uniform(d_model, d_ff, rng=rng), name="W1")
        self.b1 = Parameter(zeros(d_ff), name="b1")
        self.W2 = Parameter(xavier_uniform(d_ff, d_model, rng=rng), name="W2")
        self.b2 = Parameter(zeros(d_model), name="b2")
        self.drop = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = add(matmul(x, self.W1), self.b1)
        h = gelu(h) if self.activation == "gelu" else relu(h)
        h = self.drop(h)
        return add(matmul(h, self.W2), self.b2)

    def __repr__(self) -> str:
        return (f"FeedForward(d_model={self.d_model}, d_ff={self.d_ff}, "
                f"activation={self.activation!r})")
