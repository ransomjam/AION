"""Tensor — the central value type for AION's autograd engine.

Every value that flows through a model is a ``Tensor``.  A ``Tensor`` wraps a
NumPy array and, when ``requires_grad=True``, participates in the computation
graph so that ``backward()`` can propagate gradients back to leaf parameters.

Graph mechanics
---------------
Each ``Tensor`` produced by an operation stores:

- ``_backward`` — a closure that accumulates ``self.grad`` into the gradients
  of the input tensors.  Leaf tensors (created directly, not from an op) have
  a no-op ``_backward``.
- ``_inputs`` — the ``Tensor`` objects this one was computed from.  Used by
  ``backward()`` to build the topological order.

``backward()`` performs a single reverse pass:

1. Topological sort of all nodes reachable from ``self`` via ``_inputs``.
2. Seed ``self.grad = ones_like(self.data)`` (the loss is always scalar or
   treated as such by the caller).
3. Walk the sorted list in reverse, calling each node's ``_backward()``.
4. Release the graph: clear ``_backward`` and ``_inputs`` on every visited
   node so the forward-pass arrays can be garbage-collected.

Graph release
-------------
The graph is released unconditionally after ``backward()`` completes.  This
keeps memory bounded to a single forward/backward iteration, which is the
correct scope for this engine.  Retaining the graph across steps is not
supported and is not needed by any current or planned model.

requires_grad
-------------
A ``Tensor`` with ``requires_grad=False`` is never added to ``_inputs`` of
downstream tensors and never accumulates a gradient.  This is the foundation
for layer freezing and fine-tuning: set ``param.requires_grad = False`` and
that parameter is excluded from the graph and from optimizer updates.

NumPy is the backend.  ``Tensor.data`` is always ``np.ndarray``.  Replacing
NumPy with CuPy or another array library means changing what ``data`` holds;
the graph machinery is unchanged.
"""

from __future__ import annotations

from typing import Callable
import numpy as np


class Tensor:
    """A value in the computation graph.

    Parameters
    ----------
    data:
        The underlying array.  Converted to ``float64`` ndarray on construction
        so all arithmetic is consistent.
    requires_grad:
        Whether to track this tensor in the computation graph.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_inputs")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad: bool = requires_grad
        self._backward: Callable[[], None] = _noop
        self._inputs: tuple[Tensor, ...] = ()

    # ── graph traversal ───────────────────────────────────────────────────────

    def backward(self) -> None:
        """Reverse-mode autodiff from this tensor.

        Performs a topological sort of the computation graph, seeds this
        tensor's gradient to ones, then walks the graph in reverse calling
        each node's ``_backward`` closure.

        The graph is released after the pass completes: ``_backward`` and
        ``_inputs`` are cleared on every visited node so forward-pass
        intermediates can be garbage-collected.  The graph is scoped to a
        single forward/backward iteration by design.
        """
        order: list[Tensor] = []
        visited: set[int] = set()

        def _visit(t: Tensor) -> None:
            if id(t) not in visited:
                visited.add(id(t))
                for inp in t._inputs:
                    _visit(inp)
                order.append(t)

        _visit(self)

        # Seed: treat self as a scalar loss (or sum to scalar before calling).
        self.grad = np.ones_like(self.data)

        for t in reversed(order):
            t._backward()

        # Release the graph so intermediates are freed.
        for t in order:
            t._backward = _noop
            t._inputs = ()

    # ── convenience ───────────────────────────────────────────────────────────

    def item(self) -> float:
        """Return the scalar value of a single-element tensor."""
        return float(self.data.flat[0])

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        grad_info = f", grad={self.grad}" if self.grad is not None else ""
        return (f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}"
                f"{grad_info})")


def _noop() -> None:
    """No-op backward for leaf tensors and released graph nodes."""


def tensor(data, requires_grad: bool = False) -> Tensor:
    """Convenience constructor — mirrors ``np.array``."""
    return Tensor(data, requires_grad=requires_grad)
