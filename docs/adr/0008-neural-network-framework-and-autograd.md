# ADR 0008 — Neural Network Framework and Tape-Based Autograd

**Status:** accepted
**Date:** 2025
**Milestone:** 5 — Neural network foundation

---

## Context

AION can train tokenizers and CBOW embeddings.  Both use hand-written training
loops with gradients computed inline.  That approach cannot be composed,
reused, or extended to MLPs, RNNs, or transformers without rewriting the
infrastructure each time.

This milestone extracts the reusable foundation so that every future model is
built from the same parts.  The central question was whether to implement
manual backpropagation (Option A) or tape-based automatic differentiation
(Option B).  Option B was selected because AION is a long-term research
platform: autograd is foundational infrastructure that attention, transformers,
and every future architecture will depend on.  Implementing it now avoids a
major architectural refactor later.

---

## Decision

### 1. Tape-based reverse-mode autograd (`aion/nn/tensor.py`, `ops.py`)

Each `Tensor` stores:

- `data` — the NumPy array of values.
- `grad` — accumulated gradient, same shape as `data`.  `None` until backward.
- `requires_grad` — whether this tensor participates in the graph.
- `_backward` — a closure that accumulates `self.grad` into the inputs.
- `_inputs` — the tensors this one was computed from.

`backward()` performs a topological sort of the graph reachable from the loss
tensor, seeds `self.grad = ones_like(self.data)`, then walks in reverse calling
each node's `_backward` closure.

The graph is released unconditionally after `backward()` completes: `_backward`
and `_inputs` are cleared on every visited node.  The graph is scoped to a
single forward/backward iteration.  Retaining the graph across steps is not
supported and is not needed by any current or planned model.

### 2. `requires_grad=False` excludes parameters from the graph

A `Tensor` with `requires_grad=False` is never added to `_inputs` of downstream
tensors and never accumulates a gradient.  The optimizer skips it.  This is the
foundation for layer freezing and fine-tuning.

### 3. NumPy as the numerical backend

`Tensor.data` is always `np.ndarray`.  All framework code imports from
`aion.nn.ops`, never from `numpy` directly.  Replacing NumPy with CuPy or
another array library means changing what `data` holds; the graph machinery is
unchanged.

### 4. `Parameter` is a `Tensor` subclass

`Parameter` is a leaf `Tensor` with `requires_grad=True` by default.  It has
no `_inputs` and its `_backward` is a no-op.  The optimizer reads `.data` and
`.grad` and updates `.data` in place.  `zero_grad()` sets `.grad = None`.

### 5. `Module` — forward-only contract

`Module` is the base class for every layer and model.  Subclasses implement
only `forward()`.  `Module.backward` does not exist.  Backward is handled
entirely by the autograd engine via `loss.backward()`.

`__setattr__` intercepts assignments of `Parameter` and `Module` instances,
registering them in `_parameters` and `_modules` dicts.  `parameters()`
performs a recursive walk returning a flat list of every `Parameter` in the
tree.

### 6. Operations are minimal and incremental (`ops.py`)

Only the operations required by the current milestone are implemented:
`add`, `mul`, `matmul`, `sum`, `mean`, `relu`, `tanh`, `sigmoid`, `log`,
`exp`, `embedding_lookup`, `reshape`, `log_softmax`.

New operations are added to `ops.py` as future models require them.  The
pattern is always the same: forward computation + backward closure.

### 7. Training contract enforced by `Trainer`

```
1. optimizer.zero_grad()
2. predictions = model(x)
3. loss = loss_fn(predictions, y)
4. loss.backward()
5. optimizer.step()
```

`Trainer` owns the loop and enforces this contract.  Gradients accumulate with
`+=`; `zero_grad()` must be called before every forward pass.

### 8. `ModelStore` mirrors `TokenizerStore` and `EmbeddingStore`

Same layout: `manifest.json` + `model/weights.npz` + `model/architecture.json`
+ `training/statistics.json`.  `weights.npz` is the correct format for
parameter arrays: portable, compressed, exact float64, zero-dependency within
NumPy.  `loss_history` lives only in `statistics.json`, not in the manifest.

---

## Consequences

- `aion/nn/` is a new top-level package.
- The Training Lab is activated in the Lab Registry (status `active`).
- CBOW and future embeddings remain in `aion/embeddings/` using their existing
  hand-written training loops.  They are not migrated to `aion/nn/` — they are
  independent artifacts with a different interface.
- Future models (Skip-gram, MLP, RNN, Transformer) subclass `Module` and use
  `ops.py` operations.  The autograd engine does not change.

---

## Alternatives considered

- **Manual backprop (Option A)** — rejected.  Each `Module` would need to
  implement both `forward` and `backward`.  This is more work per layer, error-
  prone, and requires a refactor when tape-based autograd is eventually needed.
- **PyTorch/JAX as backend** — forbidden by mission.
- **Pure Python lists for tensors** — rejected.  Matrix multiply in Python
  loops is ~1000× slower than NumPy.  A framework that cannot train a 2-layer
  MLP in reasonable time is not an engineering platform.
- **Storing weights as JSON** — rejected.  JSON cannot represent float arrays
  without precision loss and is impractically large.  `.npz` is correct.
