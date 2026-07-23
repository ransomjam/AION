"""Neural network layers for AION.

Each layer is a ``Module`` subclass.  Layers only implement ``forward()``; the
autograd engine handles backward automatically.

Layers implemented here
-----------------------
Linear          — affine transformation: y = x @ W + b
EmbeddingLayer  — differentiable token embedding table
ReLU            — stateless activation
Tanh            — stateless activation
Sigmoid         — stateless activation
"""

from __future__ import annotations

import numpy as np

from .init import he_normal, uniform, xavier_uniform, zeros
from .module import Module
from .ops import add, embedding_lookup, matmul, relu, sigmoid, tanh
from .parameter import Parameter
from .tensor import Tensor


class Linear(Module):
    """Affine transformation: y = x @ W + b.

    Parameters
    ----------
    in_features:
        Size of each input sample.
    out_features:
        Size of each output sample.
    bias:
        If ``False``, no bias term is added.
    rng:
        NumPy random generator for weight initialisation.  Pass an explicit
        seeded generator for reproducibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Xavier uniform: appropriate for tanh/sigmoid; acceptable for ReLU.
        self.W = Parameter(
            xavier_uniform(in_features, out_features, rng=rng), name="W"
        )
        self.b = (
            Parameter(zeros(out_features), name="b") if bias else None
        )

    def forward(self, x: Tensor) -> Tensor:
        out = matmul(x, self.W)
        if self.b is not None:
            out = add(out, self.b)
        return out

    def __repr__(self) -> str:
        return (f"Linear(in={self.in_features}, out={self.out_features}, "
                f"bias={self.b is not None})")


class EmbeddingLayer(Module):
    """Differentiable token embedding table.

    Wraps a ``[vocab_size, dims]`` parameter matrix.  The forward pass is a
    row-gather (``table[ids]``); gradients scatter-add back into the selected
    rows via ``embedding_lookup``.

    ``token_ids`` passed to ``forward`` must be a plain ``np.ndarray`` of
    integer indices — not a ``Tensor``.  Integer indices are not differentiable.

    Parameters
    ----------
    vocab_size:
        Number of tokens in the vocabulary.
    dims:
        Embedding dimensionality.
    rng:
        NumPy random generator for weight initialisation.
    """

    def __init__(
        self,
        vocab_size: int,
        dims: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dims = dims
        init_range = 0.5 / dims
        self.table = Parameter(
            uniform(-init_range, init_range, (vocab_size, dims), rng=rng),
            name="table",
        )

    def forward(self, token_ids) -> Tensor:
        return embedding_lookup(self.table, np.asarray(token_ids, dtype=np.intp))

    def __repr__(self) -> str:
        return f"EmbeddingLayer(vocab_size={self.vocab_size}, dims={self.dims})"


class ReLU(Module):
    """Rectified linear unit activation (stateless)."""

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    """Hyperbolic tangent activation (stateless)."""

    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


class Sigmoid(Module):
    """Sigmoid activation (stateless)."""

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"
