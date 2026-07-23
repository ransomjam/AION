# ADR 0009 — Attention Framework

**Status:** accepted
**Date:** 2025
**Milestone:** 6 — Attention primitives

---

## Context

AION can train neural networks with the Module/autograd system from Milestone 5.
The next step toward language models is attention — the mechanism that made
Transformers possible.  This milestone builds attention as a first-class
primitive, not as part of a specific architecture.  The goal is that future
Transformer encoder, decoder, GPT, and BERT implementations compose from these
parts without requiring architectural changes.

---

## Decision

### 1. `aion/attention/` is a new top-level package

Attention lives alongside `aion/nn/`, not inside it.  `aion/nn/` is the
general neural network framework; `aion/attention/` is the domain-specific
layer built on top of it.  This mirrors the relationship between
`aion/tokenizers/` and `aion/tokenization/`.

### 2. Two new ops in `aion/nn/ops.py`: `softmax` and `transpose`

`softmax` is distinct from the existing `log_softmax`.  `log_softmax` is
optimised for loss computation (numerically stable log-probability).
`softmax` produces normalised weight distributions needed by attention —
the actual weights, not their logs, are required to compute the weighted
sum of V and to visualize attention patterns.

`transpose` permutes tensor axes with a correct inverse-permutation backward.
It is a fundamental primitive required by attention (K^T in score computation,
head splitting/merging) and by future architectures.

Both follow the existing `_make_out` / `_backward` closure pattern in `ops.py`.

### 3. `ScaledDotProductAttention` is an independent Module with no parameters

The atomic attention operation is a `Module` subclass so it participates in
`parameters()`, `zero_grad()`, and `ModelStore.save()` without special cases.
It has no parameters of its own.  `MultiHeadAttention` composes it rather than
embedding the algorithm directly — this is the approved design.

`forward(Q, K, V, mask)` returns `(output, weights)`.  The weights tensor is
still in the computation graph at return time.  `MultiHeadAttention` detaches
them via `.data.copy()` before returning to the caller.  Visualization code
operates on the detached array; it never touches the autograd graph.

### 4. `MultiHeadAttention` validates `d_model % n_heads == 0` at construction

The check is in `__init__`, not in `forward`.  A model with an invalid
configuration fails immediately with a clear error message, not silently at
the first forward pass with a cryptic shape error.

### 5. Additive masking, not multiplicative

Masks add a large negative value (`-1e9`) to masked positions before softmax.
Multiplicative masking (zeroing weights after softmax) is incorrect: the
remaining weights no longer sum to 1, and gradient flow through masked
positions is broken.  Additive masking is numerically correct and fully
differentiable.

Masks return plain `np.ndarray` bias tensors.  `ScaledDotProductAttention`
wraps them in `Tensor(requires_grad=False)` before adding to scores.  Mask
positions are not learned and must not appear in the gradient graph.

### 6. `CausalMask` and `PaddingMask` are concrete mask implementations

`CausalMask` produces a lower-triangular bias of shape `[1, 1, seq_q, seq_k]`
that broadcasts over all batch elements and all heads.

`PaddingMask` accepts a `[batch, seq_k]` boolean array and produces a
`[batch, 1, 1, seq_k]` bias that varies per sample.

Both subclass `AttentionMask`, establishing a consistent interface for future
encoder, decoder, and cross-attention implementations.

### 7. `SinusoidalPE` (fixed) and `LearnedPE` (trainable) are both provided

`SinusoidalPE` precomputes a `[max_len, d_model]` table at construction.
`forward(x)` adds the slice `PE[:seq_len]` to `x`.  No gradient flows through
PE — it is a constant addition.  Appropriate for Transformer encoder/decoder.

`LearnedPE` stores the table as a `Parameter` initialised from sinusoidal
values.  Gradient flows through PE — positions are learned.  Used by
BERT-style models.

`RotaryPE` is a stub.  RoPE requires a `rotate_half` op and couples to the
attention score computation directly.  The interface is defined; implementation
is deferred to the Transformer milestone.

### 8. `AttentionVisualizer` is a pure analysis tool

`AttentionVisualizer` accepts plain `np.ndarray` inputs (weights detached from
the graph) and returns plain Python dicts/lists.  No `Tensor` objects, no
NumPy arrays in the output.  The frontend renders the returned dicts directly.
This mirrors the pattern in `aion/datasets/stats.py` and `quality.py`.

Four methods: `heatmap`, `multi_head_heatmaps`, `head_comparison`,
`query_key_similarity`.

### 9. No new storage layout

`ModelStore` already handles attention models.  `architecture.json` records
`n_heads`, `d_model`, `d_head`, `max_seq_len` via the `params` dict.
`_registry()` in `store.py` gains entries for attention architectures as they
are defined.  No new store class is needed.

### 10. `MultiHeadAttention` supports both self-attention and cross-attention

Self-attention: `forward(x, x, x)`.
Cross-attention: `forward(query_seq, kv_seq, kv_seq)`.

The module does not distinguish them.  The caller passes the appropriate
tensors.  This is the correct abstraction: the algorithm is identical; only
the data source differs.

---

## Consequences

- `aion/attention/` is a new top-level package.
- `aion/nn/ops.py` gains `softmax` and `transpose`.
- The Attention Lab is activated in the Lab Registry (status `active`).
- Future Transformer encoder, decoder, GPT, and BERT implementations compose
  from `MultiHeadAttention`, `CausalMask`, `PaddingMask`, `SinusoidalPE`, and
  `LearnedPE` without requiring changes to any of these components.
- `RotaryPE` implementation lands at the Transformer milestone.

---

## Alternatives considered

- **Separate `SelfAttention` and `CrossAttention` classes** — rejected.
  `MultiHeadAttention` handles both: self-attention passes the same tensor for
  Q, K, V; cross-attention passes different tensors.  A separate class would
  duplicate code for no benefit.

- **Fused QKV projection** — deferred.  A single `[d_model, 3*d_model]` weight
  matrix is more cache-efficient.  The unfused version (`W_Q`, `W_K`, `W_V`
  separately) is clearer and the performance difference is irrelevant at this
  scale.  The fused version can replace it at the Transformer milestone.

- **Attention as a pure function, not a Module** — rejected.  Making
  `ScaledDotProductAttention` a `Module` means it participates in
  `parameters()`, `zero_grad()`, and `ModelStore.save()` without special cases.

- **Storing attention weights in the computation graph for visualization** —
  rejected.  Weights are detached via `.data.copy()` at forward time.  The
  graph is not polluted by visualization code, and the weights are accurate
  because they are read before `backward()` releases the graph.

- **`RotaryPE` full implementation now** — deferred.  RoPE modifies the
  attention score computation (rotating Q and K before the dot product), which
  couples PE to `ScaledDotProductAttention`.  The interface is defined now so
  the Transformer milestone can implement it without architectural changes.

- **Multiplicative masking** — rejected.  See Decision 5.
