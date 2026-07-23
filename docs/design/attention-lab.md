# Attention Lab — Engineering Design

*Status: implemented. Milestone 6 — Attention primitives.*

AION's attention framework introduces the mechanism that made modern language
models possible as a first-class neural network primitive.  This is not a
Transformer implementation — it is the reusable foundation that every future
Transformer variant will compose from.

## 1. Attention architecture

Attention is a learned routing mechanism.  Given queries Q, keys K, and values
V, it computes a weighted sum of V where the weights are determined by the
similarity between Q and K:

```
scores  = Q @ K^T / sqrt(d_k)
scores += mask_bias              (optional)
weights = softmax(scores)
output  = weights @ V
```

All components operate on 3-D tensors: `[batch, seq_len, d_model]`.  Batch
is always first; sequence is always second; features are always last.

## 2. Generic attention abstraction

`ScaledDotProductAttention` is the atomic unit — a `Module` with no parameters.
`MultiHeadAttention` composes it with four projection matrices.  The separation
means the attention algorithm can be tested, visualized, and reasoned about
independently of the projection machinery.

## 3. Scaled dot-product attention

`ScaledDotProductAttention.forward(Q, K, V, mask)` returns `(output, weights)`.

The weights tensor is still in the autograd graph at return time.
`MultiHeadAttention` detaches them via `.data.copy()` before returning.
Visualization code operates on the detached array — the graph is never
polluted by inspection code.

## 4. Attention masking

Masks add a large negative value (`-1e9`) to masked positions before softmax.
This drives their softmax weight to zero without breaking the gradient graph
or producing incorrect probability distributions.

`AttentionMask.bias(seq_q, seq_k, batch_size)` returns a `np.ndarray`
broadcastable over `[batch, n_heads, seq_q, seq_k]`.

## 5. Causal masking

`CausalMask` produces a lower-triangular bias: position `i` may only attend
to positions `j <= i`.  Shape `[1, 1, seq_q, seq_k]` — identical for every
batch element and every head.  Used by GPT-style autoregressive models.

`PaddingMask` accepts a `[batch, seq_k]` boolean array and produces a
`[batch, 1, 1, seq_k]` bias that varies per sample.  Used by encoder models
with variable-length inputs.

## 6. Multi-head architecture

`MultiHeadAttention` splits `d_model` into `n_heads` heads of size
`d_head = d_model // n_heads`, runs `ScaledDotProductAttention` on all heads
in a single batched matmul (by merging batch and head dimensions), concatenates,
and projects back.

Parameters: `W_Q`, `W_K`, `W_V`, `W_O` — each `[d_model, d_model]`.
`d_model % n_heads == 0` is validated at construction with a clear error.

Supports self-attention (`forward(x, x, x)`) and cross-attention
(`forward(query, kv, kv)`) transparently.

## 7. Positional encoding strategy

Three strategies:

| Class | Parameters | Use case |
|---|---|---|
| `SinusoidalPE` | None (fixed table) | Transformer encoder/decoder |
| `LearnedPE` | `[max_len, d_model]` Parameter | BERT-style models |
| `RotaryPE` | None (stub) | GPT-NeoX, LLaMA (deferred) |

`SinusoidalPE` and `LearnedPE` both implement `forward(x)` that adds the
encoding to `x`.  `LearnedPE` is initialised from sinusoidal values so
training starts from a sensible prior.

## 8. Integration with the existing Module system

All attention components are `Module` subclasses.  `parameters()`,
`zero_grad()`, `param_count()`, `Trainer`, and `ModelStore.save()` all work
without modification.  The only changes to existing files are two new ops in
`ops.py` (`softmax`, `transpose`) and the Attention Lab activation in
`labs.py`.

## 9. Visualization

`AttentionVisualizer` is a pure analysis tool.  It accepts plain `np.ndarray`
inputs (weights detached from the graph) and returns plain Python dicts/lists.

| Method | Input | Output |
|---|---|---|
| `heatmap` | `[seq_q, seq_k]` | Single-head heatmap dict |
| `multi_head_heatmaps` | `[n_heads, seq_q, seq_k]` | List of heatmap dicts |
| `head_comparison` | `[n_heads, seq_q, seq_k]` | Per-head entropy, max_attn, sparsity |
| `query_key_similarity` | `[seq_q, d_k]`, `[seq_k, d_k]` | Cosine similarity matrix |

The frontend renders the returned dicts directly.  No visualization logic in
the frontend.

## 10. Evaluation metrics

Computed by `AttentionVisualizer.head_comparison()` and recordable in
`statistics.json` via the existing `ExperimentStore`:

- **Attention entropy** per head: `-sum(w * log(w + eps))`.  Low = focused.
- **Max attention weight** per head: mean of per-query maximum.
- **Sparsity**: fraction of weights below 0.01.
- **Entropy variance** across heads: high variance = heads have learned
  different roles.

## 11. Storage

No new storage layout.  `ModelStore` handles attention models.
`architecture.json` records `n_heads`, `d_model`, `d_head`, `max_seq_len`
via the `params` dict.  `_registry()` gains entries for attention
architectures as they are defined.

## 12. Future compatibility

| Future component | Reuses |
|---|---|
| Transformer encoder | `MultiHeadAttention` + `SinusoidalPE` + `Linear` |
| Transformer decoder | adds cross-attention: `MultiHeadAttention(Q=dec, K=V=enc)` |
| GPT | `MultiHeadAttention` + `CausalMask` + `LearnedPE` |
| BERT | `MultiHeadAttention` + `PaddingMask` + `LearnedPE` |
| Vision Transformer | `MultiHeadAttention` with patch embeddings |
| Multimodal | cross-attention between modalities |

No architectural changes required.

## 13. Alternatives considered

See ADR 0009 for the full record.  Key decisions: additive over multiplicative
masking; `ScaledDotProductAttention` as an independent Module; detached weights
for visualization; `RotaryPE` deferred; fused QKV deferred.
