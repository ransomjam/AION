"""Sequential — an ordered container of modules.

Passes its input through each child module in registration order.  Sufficient
for MLPs and any architecture that is a straight pipeline.  Attention, RNNs,
and transformers subclass ``Module`` directly.
"""

from __future__ import annotations

from .module import Module
from .tensor import Tensor


class Sequential(Module):
    """Apply a sequence of modules in order.

    Example
    -------
    >>> model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
    >>> output = model(x)
    """

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        # Register each layer as a named submodule so parameters() finds them.
        for i, layer in enumerate(layers):
            setattr(self, f"layer_{i}", layer)
        # Keep an ordered list for forward traversal.
        object.__setattr__(self, "_layers", list(layers))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        parts = [f"  ({i}): {layer}" for i, layer in enumerate(self._layers)]
        inner = "\n".join(parts)
        return f"Sequential(\n{inner}\n)" if parts else "Sequential()"
