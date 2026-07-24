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


# ── Default compute dtype ─────────────────────────────────────────────────────
# Every Tensor stores its data in this dtype unless a call passes an explicit
# ``dtype``.  It is ``float32`` by default: training activations, parameters,
# and the retained autograd graph are all half the size of ``float64``, which
# keeps peak memory bounded.  Gradient-checking tests that need the extra
# precision of ``float64`` opt in with ``set_default_dtype(np.float64)``.
_DEFAULT_DTYPE: type = np.float32


def get_default_dtype() -> type:
    """Return the dtype new tensors use when none is given."""
    return _DEFAULT_DTYPE


def set_default_dtype(dtype) -> None:
    """Set the default tensor dtype (e.g. ``np.float64`` for gradient checks)."""
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = np.dtype(dtype).type


class Tensor:
    """A value in the computation graph.

    Parameters
    ----------
    data:
        The underlying array.  Converted to an ndarray of the default compute
        dtype (``float32``) on construction so all arithmetic is consistent and
        memory stays bounded.  Pass ``dtype`` to override for one tensor.
    requires_grad:
        Whether to track this tensor in the computation graph.
    dtype:
        Explicit dtype for this tensor's data; defaults to
        :func:`get_default_dtype`.
    """

    __slots__ = ("data", "grad", "requires_grad", "_backward", "_inputs")

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        dtype=None,
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=dtype or _DEFAULT_DTYPE)
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

        The graph is released *incrementally*: as soon as a node's
        ``_backward`` has run, its closure, inputs, gradient, and forward
        activation are dropped (leaf parameters keep their weights and grads,
        and ``self`` keeps its data/grad).  This bounds peak memory to roughly a
        single forward pass instead of holding the entire graph until the end.
        The graph is scoped to a single forward/backward iteration by design.
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
            # Release this node's graph state *immediately*, so the forward
            # intermediates its backward closure captured (softmax outputs,
            # GELU gradient terms, LayerNorm stats, ...) become collectable as
            # the sweep proceeds rather than all at the very end.  This keeps
            # peak memory close to a single forward pass instead of ~2x it.
            had_inputs = bool(t._inputs)
            t._backward = _noop
            t._inputs = ()
            # An intermediate's gradient is dead once propagated to its inputs,
            # and its forward activation is dead once every consumer (all
            # downstream, already processed) has run.  Free both so peak memory
            # stays near a single forward pass.  Only leaf parameters keep their
            # data (the weights) and grad (for the optimizer); ``self`` keeps
            # its data/grad for the caller to inspect.
            if had_inputs and t is not self:
                t.grad = None
                t.data = None

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
