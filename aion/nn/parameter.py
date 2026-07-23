"""Parameter — a trainable leaf tensor.

A ``Parameter`` is a ``Tensor`` with ``requires_grad=True`` by default.  It is
a leaf node in the computation graph: it has no ``_inputs`` and its
``_backward`` is a no-op.  The optimizer reads ``.data`` and ``.grad`` and
updates ``.data`` in place.

``requires_grad=False`` excludes the parameter from the graph entirely.  No
gradient is accumulated and the optimizer skips it.  This is the foundation
for layer freezing and fine-tuning.
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


class Parameter(Tensor):
    """A trainable leaf tensor owned by a ``Module``.

    Parameters
    ----------
    data:
        Initial weight values.
    name:
        Human-readable name used for serialization and debugging.
    requires_grad:
        Set to ``False`` to freeze this parameter (no gradient, no optimizer
        update).  Defaults to ``True``.
    """

    def __init__(
        self,
        data,
        name: str = "",
        requires_grad: bool = True,
    ) -> None:
        super().__init__(data, requires_grad=requires_grad)
        self.name = name

    def zero_grad(self) -> None:
        """Set gradient to ``None``.

        ``None`` signals "no gradient computed yet" and is distinct from a
        gradient that happens to be zero.  The optimizer checks for ``None``
        before reading ``.grad``.
        """
        self.grad = None

    def __repr__(self) -> str:
        return (f"Parameter(name={self.name!r}, shape={self.shape}, "
                f"requires_grad={self.requires_grad})")
