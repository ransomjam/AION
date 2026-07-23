"""Differentiable operations for AION's autograd engine.

Each function takes ``Tensor`` inputs, computes the forward value, and returns
a new ``Tensor`` whose ``_backward`` closure accumulates the correct gradients
into the inputs.

Only the operations required by the current milestone are implemented here.
New operations are added incrementally as future models require them — never
speculatively.  The pattern is always the same:

    1. Compute the forward value with NumPy.
    2. Determine whether any input requires a gradient.
    3. If so, build a ``Tensor`` with a ``_backward`` closure that captures the
       inputs and any intermediate values needed for the gradient computation.
    4. Set ``_inputs`` to the subset of inputs that have ``requires_grad=True``
       so the graph traversal only visits nodes that can receive gradients.

Broadcasting
------------
``add`` and ``mul`` handle NumPy broadcasting.  The backward pass must sum the
gradient over any axes that were broadcast, which is handled by
``_unbroadcast``.

Current operation set
---------------------
add, mul, matmul, sum, mean, relu, tanh, sigmoid, log, exp,
embedding_lookup, reshape, log_softmax
"""

from __future__ import annotations

import numpy as np

from .tensor import Tensor


# ── internal helpers ──────────────────────────────────────────────────────────
# t_sum is a public alias for sum() so callers can import it without shadowing
# Python's built-in.  Assigned at the bottom of this module after sum() is
# defined.
t_sum: "Callable" = None  # type: ignore[assignment]

def _needs_grad(*tensors: Tensor) -> bool:
    return any(t.requires_grad for t in tensors)


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` over axes that were broadcast to match ``shape``."""
    # Pad shape on the left with ones to match grad.ndim
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, (gs, ss) in enumerate(zip(grad.shape, shape)):
        if ss == 1 and gs != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _accum(t: Tensor, g: np.ndarray) -> None:
    """Accumulate gradient ``g`` into ``t.grad`` (initialising if needed).

    If ``g`` has more dimensions than ``t.data`` (e.g. a batched gradient
    flowing into a 2-D weight matrix), the extra leading axes are summed away
    before accumulation.
    """
    if not t.requires_grad:
        return
    if t.grad is None:
        t.grad = np.zeros_like(t.data)
    # Sum over any extra leading batch dimensions.
    while g.ndim > t.data.ndim:
        g = g.sum(axis=0)
    t.grad += g


# ── operations ────────────────────────────────────────────────────────────────

def _make_out(data: np.ndarray, needs: bool) -> Tensor:
    """Create an output tensor, propagating requires_grad when needed."""
    out = Tensor(data)
    if needs:
        out.requires_grad = True
    return out


def add(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise addition with broadcasting."""
    needs = _needs_grad(a, b)
    out = _make_out(a.data + b.data, needs)
    if needs:
        def _backward() -> None:
            g = out.grad
            if a.requires_grad:
                _accum(a, _unbroadcast(g, a.shape))
            if b.requires_grad:
                _accum(b, _unbroadcast(g, b.shape))
        out._backward = _backward
        out._inputs = tuple(t for t in (a, b) if t.requires_grad)
    return out


def mul(a: Tensor, b: Tensor) -> Tensor:
    """Element-wise multiplication with broadcasting."""
    needs = _needs_grad(a, b)
    out = _make_out(a.data * b.data, needs)
    if needs:
        def _backward() -> None:
            g = out.grad
            if a.requires_grad:
                _accum(a, _unbroadcast(g * b.data, a.shape))
            if b.requires_grad:
                _accum(b, _unbroadcast(g * a.data, b.shape))
        out._backward = _backward
        out._inputs = tuple(t for t in (a, b) if t.requires_grad)
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication.  Supports 2-D and batched (3-D) inputs."""
    needs = _needs_grad(a, b)
    out = _make_out(a.data @ b.data, needs)
    if needs:
        def _backward() -> None:
            g = out.grad
            if a.requires_grad:
                _accum(a, g @ b.data.swapaxes(-1, -2))
            if b.requires_grad:
                _accum(b, a.data.swapaxes(-1, -2) @ g)
        out._backward = _backward
        out._inputs = tuple(t for t in (a, b) if t.requires_grad)
    return out


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Sum over ``axis`` (or all axes if None)."""
    out = _make_out(np.sum(a.data, axis=axis, keepdims=keepdims), a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis)
            _accum(a, np.broadcast_to(g, a.shape).copy())
        out._backward = _backward
        out._inputs = (a,)
    return out


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """Mean over ``axis`` (or all axes if None)."""
    out = _make_out(np.mean(a.data, axis=axis, keepdims=keepdims), a.requires_grad)
    if a.requires_grad:
        n = a.data.size if axis is None else a.data.shape[axis]
        def _backward() -> None:
            g = out.grad
            if not keepdims and axis is not None:
                g = np.expand_dims(g, axis=axis)
            _accum(a, np.broadcast_to(g / n, a.shape).copy())
        out._backward = _backward
        out._inputs = (a,)
    return out


def relu(a: Tensor) -> Tensor:
    """Rectified linear unit: max(0, a)."""
    mask = a.data > 0
    out = _make_out(a.data * mask, a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            _accum(a, out.grad * mask)
        out._backward = _backward
        out._inputs = (a,)
    return out


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    out = _make_out(t, a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            _accum(a, out.grad * (1.0 - t * t))
        out._backward = _backward
        out._inputs = (a,)
    return out


def sigmoid(a: Tensor) -> Tensor:
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = _make_out(s, a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            _accum(a, out.grad * s * (1.0 - s))
        out._backward = _backward
        out._inputs = (a,)
    return out


def log(a: Tensor) -> Tensor:
    """Natural logarithm.  Inputs should be positive."""
    out = _make_out(np.log(a.data), a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            _accum(a, out.grad / a.data)
        out._backward = _backward
        out._inputs = (a,)
    return out


def exp(a: Tensor) -> Tensor:
    e = np.exp(a.data)
    out = _make_out(e, a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            _accum(a, out.grad * e)
        out._backward = _backward
        out._inputs = (a,)
    return out


def reshape(a: Tensor, shape: tuple) -> Tensor:
    out = _make_out(a.data.reshape(shape), a.requires_grad)
    if a.requires_grad:
        original_shape = a.shape
        def _backward() -> None:
            _accum(a, out.grad.reshape(original_shape))
        out._backward = _backward
        out._inputs = (a,)
    return out


def embedding_lookup(table: Tensor, ids: np.ndarray) -> Tensor:
    """Differentiable row-gather from an embedding table.

    ``ids`` is a plain ``np.ndarray`` of integer indices — not a ``Tensor``
    and not differentiable.  Gradients scatter-add back into the rows of
    ``table`` that were selected.
    """
    ids = np.asarray(ids, dtype=np.intp)
    out = _make_out(table.data[ids], table.requires_grad)
    if table.requires_grad:
        def _backward() -> None:
            if table.grad is None:
                table.grad = np.zeros_like(table.data)
            np.add.at(table.grad, ids, out.grad)
        out._backward = _backward
        out._inputs = (table,)
    return out


def log_softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable log-softmax along ``axis``.

    Used by ``CrossEntropyLoss`` to avoid computing softmax and log separately.
    The backward is the Jacobian-vector product of log-softmax:
        grad_input = grad_output - softmax * sum(grad_output, axis)
    """
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    log_sum_exp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    lsm = shifted - log_sum_exp
    out = _make_out(lsm, a.requires_grad)
    if a.requires_grad:
        softmax = np.exp(lsm)
        def _backward() -> None:
            g = out.grad
            _accum(a, g - softmax * g.sum(axis=axis, keepdims=True))
        out._backward = _backward
        out._inputs = (a,)
    return out


# Public alias — import as ``t_sum`` to avoid shadowing Python's built-in sum.
t_sum = sum


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Numerically stable softmax along ``axis``.

    Used by attention to produce normalised weight distributions.  The backward
    is the Jacobian-vector product::

        grad_a = s * (grad_out - (grad_out * s).sum(axis, keepdims=True))

    where ``s`` is the softmax output.  Computed from the already-stable
    softmax values, not from raw logits.
    """
    shifted = a.data - a.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=axis, keepdims=True)
    out = _make_out(s, a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            g = out.grad
            _accum(a, s * (g - (g * s).sum(axis=axis, keepdims=True)))
        out._backward = _backward
        out._inputs = (a,)
    return out


def transpose(a: Tensor, axes: tuple) -> Tensor:
    """Permute tensor axes.  ``axes`` follows ``np.transpose`` convention.

    The backward permutes the incoming gradient with the inverse permutation,
    restoring the original axis order.
    """
    out = _make_out(np.transpose(a.data, axes), a.requires_grad)
    if a.requires_grad:
        inv = tuple(int(i) for i in np.argsort(axes))
        def _backward() -> None:
            _accum(a, np.transpose(out.grad, inv))
        out._backward = _backward
        out._inputs = (a,)
    return out


def gelu(a: Tensor) -> Tensor:
    """Gaussian Error Linear Unit activation (Hendrycks & Gimpel, 2016).

    Uses the tanh approximation::

        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Promoted to a first-class op (rather than composed from existing ops) to
    produce a smaller computation graph, a single backward closure, and
    cleaner debugging.  The backward is derived analytically from the
    approximation formula.
    """
    import math
    c = math.sqrt(2.0 / math.pi)
    x = a.data
    x3 = x ** 3
    inner = c * (x + 0.044715 * x3)
    t = np.tanh(inner)
    g_fwd = 0.5 * x * (1.0 + t)
    out = _make_out(g_fwd, a.requires_grad)
    if a.requires_grad:
        # d/dx GELU(x) = 0.5*(1+t) + 0.5*x*(1-t^2)*c*(1 + 3*0.044715*x^2)
        dtanh = 1.0 - t * t
        d_inner_dx = c * (1.0 + 3.0 * 0.044715 * x ** 2)
        grad_fn = 0.5 * (1.0 + t) + 0.5 * x * dtanh * d_inner_dx
        def _backward() -> None:
            _accum(a, out.grad * grad_fn)
        out._backward = _backward
        out._inputs = (a,)
    return out


def layer_norm(
    a: Tensor,
    gamma: Tensor,
    beta: Tensor,
    eps: float = 1e-5,
) -> Tensor:
    """Layer normalization over the last axis (Ba et al., 2016).

    Normalizes each token's feature vector, then scales by ``gamma`` and
    shifts by ``beta``::

        x_norm = (x - mean(x, axis=-1)) / sqrt(var(x, axis=-1) + eps)
        output = gamma * x_norm + beta

    Implemented as a single fused op so the autograd engine handles it with
    one backward closure rather than a deep graph of intermediate nodes.

    Parameters
    ----------
    a:
        Input of any shape; normalization is over the last axis.
    gamma:
        Scale parameter, shape ``[d]`` where ``d = a.shape[-1]``.
    beta:
        Shift parameter, shape ``[d]``.
    eps:
        Small constant for numerical stability.
    """
    x = a.data
    mean_x = x.mean(axis=-1, keepdims=True)
    var_x = x.var(axis=-1, keepdims=True)
    std_x = np.sqrt(var_x + eps)
    x_norm = (x - mean_x) / std_x
    out_data = gamma.data * x_norm + beta.data
    needs = _needs_grad(a, gamma, beta)
    out = _make_out(out_data, needs)
    if needs:
        d = x.shape[-1]
        def _backward() -> None:
            g = out.grad                          # [..., d]
            # gamma and beta gradients
            if gamma.requires_grad:
                _accum(gamma, _unbroadcast(g * x_norm, gamma.shape))
            if beta.requires_grad:
                _accum(beta, _unbroadcast(g, beta.shape))
            if a.requires_grad:
                # Full LN Jacobian-vector product:
                # dx = (1/std) * (g*gamma - mean(g*gamma) - x_norm*mean(g*gamma*x_norm))
                g_gamma = g * gamma.data
                mean_g_gamma = g_gamma.mean(axis=-1, keepdims=True)
                mean_g_gamma_xn = (g_gamma * x_norm).mean(axis=-1, keepdims=True)
                dx = (g_gamma - mean_g_gamma - x_norm * mean_g_gamma_xn) / std_x
                _accum(a, dx)
        out._backward = _backward
        out._inputs = tuple(t for t in (a, gamma, beta) if t.requires_grad)
    return out


def dropout_mask(a: Tensor, mask: np.ndarray, keep_prob: float) -> Tensor:
    """Apply a pre-sampled dropout mask and scale by ``1/keep_prob``.

    ``mask`` is a plain ``np.ndarray`` of dtype bool — not a ``Tensor`` and
    not differentiable.  The backward passes the gradient through kept
    positions only, scaled by ``1/keep_prob``.

    Parameters
    ----------
    a:
        Input tensor.
    mask:
        Boolean array, same shape as ``a.data``.  ``True`` = keep.
    keep_prob:
        Fraction of units kept (``1 - dropout_rate``).
    """
    scale = 1.0 / keep_prob if keep_prob > 0.0 else 0.0
    out = _make_out(a.data * mask * scale, a.requires_grad)
    if a.requires_grad:
        def _backward() -> None:
            _accum(a, out.grad * mask * scale)
        out._backward = _backward
        out._inputs = (a,)
    return out
