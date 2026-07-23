# ADR 0010 — Transformer Framework

**Status:** accepted
**Date:** 2025
**Milestone:** 7 — Transformer Framework

---

## Context

AION's Attention Framework (Milestone 6) provides `MultiHeadAttention`,
`CausalMask`, `PaddingMask`, `SinusoidalPE`, and `LearnedPE` as first-class
reusable primitives.  The next step is to compose these into a complete,
reusable transformer framework that future model implementations — GPT, BERT,
encoder-decoder seq2seq, Vision Transformers, multimodal transformers — can
build on without requiring architectural changes to the framework itself.

The goal is not to implement GPT or BERT.  The goal is to make transformer
blocks first-class reusable modules in the same way that `MultiHeadAttention`
is a first-class reusable module.

---

## Decisions

### 1. Four-level block hierarchy

```
TransformerBlock        (abstract base)
    TransformerEncoderBlock  — self-attention + FFN
    TransformerDecoderBlock  — causal self-attention + cross-attention + FFN

TransformerStack        (reusable N-block stack)
    TransformerEncoder  — encoder-only typed wrapper
    TransformerDecoder  — cross-attention decoder stack
```

`TransformerBlock` is the abstract base for all block variants.  It owns the
shared constructor signature and exposes `d_model` and `n_heads` as properties
so stacks and the store can introspect without reaching into internal state.

`TransformerStack` is the canonical reusable backbone: N `TransformerEncoderBlock`
instances plus a final `LayerNorm`.  It is used directly for decoder-only models
(GPT-style, with a `CausalMask`) and subclassed by `TransformerEncoder` for
semantic clarity.

`TransformerEncoder` is a `TransformerStack` subclass that adds `from_config`
factory methods accepting `EncoderConfig` or the encoder half of
`TransformerConfig`.  It carries semantic meaning — "this stack is acting as an
encoder" — without duplicating any logic.

`TransformerDecoder` is structurally distinct: each block has three sub-layers
(causal self-attention, cross-attention, FFN) and `forward` requires an
`encoder_output` argument.  It cannot share `TransformerStack`'s implementation
and is a direct `Module` subclass.

`Transformer` composes `TransformerEncoder` and `TransformerDecoder` for
full encoder-decoder (seq2seq) tasks.

### 2. Pre-norm is the default; post-norm is available

Pre-norm (LayerNorm applied before each sub-layer, residual added after) is the
default for all blocks and stacks.  It trains more stably than post-norm at
depth and is the modern standard.  Post-norm (original Vaswani) is available via
`pre_norm=False` for reproducibility of older architectures.

Pre-norm structure (encoder block)::

    x_n = norm1(x)
    x   = x + dropout(attn(x_n, x_n, x_n, mask))
    x   = x + dropout(ffn(norm2(x)))

The pre-norm encoder block previously called `norm1(x)` three times
independently, creating three separate computation graph nodes and tripling the
backward work through `layer_norm`.  This was a correctness issue: gradients
accumulated into `norm1`'s parameters three times per forward pass.  Fixed by
computing `x_n = norm1(x)` once and reusing it.

### 3. `TransformerConfig` is the canonical hyperparameter container

`TransformerConfig`, `EncoderConfig`, and `DecoderConfig` are plain dataclasses
with `to_dict()` / `from_dict()` round-trip methods.  They are the single source
of truth for all architectural hyperparameters.  Every factory method (`from_config`)
accepts a config object; no constructor takes a raw dict.

`EncoderConfig` and `DecoderConfig` are thin specialisations that set sensible
defaults for their respective architectures and expose only the fields relevant
to each.  Both provide `to_transformer_config()` for interoperability.

### 4. Explicit weight-tying support

Language models commonly tie the input embedding matrix to the output projection
(Press & Wolf, 2017) to reduce parameter count and improve generalisation.
`Transformer.tie_weights(embedding, projection)` registers this relationship
explicitly.  The actual sharing (passing the same `Parameter` object to both
the embedding layer and the output projection) is the caller's responsibility.

`tie_weights` records the intent in `_tied_weights` so `TransformerStore` can
serialize it in `architecture.json` and the manifest.  The logit head itself
belongs in GPT/BERT, not in the framework.

### 5. `TransformerStore` as a standalone first-class store

Every learned artifact in AION has a store.  `TransformerStore` follows the
same pattern as `TokenizerStore` and `EmbeddingStore`: it owns its storage
logic directly rather than delegating to `ModelStore`.  `ModelStore` remains
architecture-agnostic.

On-disk layout::

    projects/<p>/models/<model_id>/
        manifest.json
        model/
            weights.npz
            architecture.json
        training/
            statistics.json

Three architecture strings are registered: `"encoder-v1"` → `TransformerEncoder`,
`"decoder-v1"` → `TransformerStack`, `"transformer-v1"` → `Transformer`.

### 6. Index-prefixed parameter keys in `weights.npz`

Parameters are stored as `"p{i}__{name}"` where `i` is the zero-based index in
`model.parameters()`.  The index prefix avoids name collisions across blocks
(every block has a `W_Q`, `gamma`, `beta`, etc.).  The load path matches by
index, not by name, so parameter ordering is the authoritative contract.

### 7. `TransformerVisualizer` is a pure analysis tool

`TransformerVisualizer` accepts plain `np.ndarray` inputs (detached from the
autograd graph) and returns plain Python dicts/lists.  No `Tensor` objects, no
NumPy arrays in the output.  This mirrors `AttentionVisualizer`, `stats.py`,
and `quality.py`: the backend computes; the frontend renders.

Six analysis methods: `layer_outputs`, `residual_norms`, `layernorm_stats`,
`ffn_activations`, `cross_attention_heatmap`, `depth_comparison`.

### 8. Positional encoding is not part of the stack

`TransformerStack`, `TransformerEncoder`, and `TransformerDecoder` do not apply
positional encoding.  They expect the caller to add PE before calling `forward`.
This is the correct separation: the stack should not assume which PE strategy
is used.  Future model implementations (GPT, BERT) add PE in their own
`forward` before passing to the stack.

### 9. Token embedding and output projection are not part of the framework

The token embedding layer and the output projection (logit head) are
model-specific.  They depend on vocabulary size, whether weights are tied, and
the training objective.  They belong in GPT/BERT, not in the reusable framework.
The framework outputs `[batch, seq, d_model]` tensors; what to do with them is
the model's responsibility.

---

## Consequences

- `aion/transformer/` gains `stack.py` and `store.py`.
- The block hierarchy is `TransformerBlock → TransformerEncoderBlock /
  TransformerDecoderBlock`.
- The stack hierarchy is `TransformerStack → TransformerEncoder`;
  `TransformerDecoder` is a separate `Module`.
- `Transformer` exposes `tie_weights()` for future GPT/BERT weight tying.
- `TransformerStore` handles save/load/list/delete for all three architecture
  variants.
- 479 / 479 tests pass.
- Future GPT, BERT, and encoder-decoder implementations compose from
  `TransformerStack`, `TransformerEncoder`, `TransformerDecoder`, and
  `Transformer` without requiring changes to any of these components.

---

## Alternatives considered

- **`TransformerEncoder` as the decoder-only backbone** — rejected.  Naming a
  GPT backbone `TransformerEncoder` is confusing.  `TransformerStack` is the
  neutral name for the shared self-attention stack; `TransformerEncoder` is the
  typed wrapper for the encoder role.

- **Delegating to `ModelStore`** — rejected.  `ModelStore._registry()` is empty
  and architecture-agnostic.  Populating it from `TransformerStore` would create
  a global mutable registry and couple `ModelStore` to `aion.transformer`.
  A standalone store is consistent with `TokenizerStore` and `EmbeddingStore`.

- **Name-based parameter keys in `weights.npz`** — rejected.  Multiple
  parameters share the same name across blocks.  Index-prefixed keys are
  unambiguous and load correctly regardless of parameter naming conventions.

- **Fused QKV projection** — deferred.  A single `[d_model, 3*d_model]` weight
  matrix is more cache-efficient.  The unfused version is clearer and the
  performance difference is irrelevant at this scale.  Can be added at the GPT
  milestone if needed.

- **`RotaryPE` implementation** — deferred.  RoPE modifies the attention score
  computation and couples to `ScaledDotProductAttention`.  The interface stub
  exists in `aion/attention/positional.py`; implementation lands when a model
  requires it.
