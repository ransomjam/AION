# ADR 0011 — GPT Language Model

**Status**: Accepted  
**Date**: 2025  
**Milestone**: 8 — GPT Lab

---

## Context

AION's transformer infrastructure (Milestone 7) provides a complete
`TransformerStack` with `CausalMask`, `EmbeddingLayer`, `LearnedPE`, and
`CrossEntropyLoss`.  The next step is to compose these into a trainable
language model and define the full training pipeline around it.

This ADR records every architectural decision made for the `aion/gpt/` package.

---

## Decisions

### 1. GPTConfig is a standalone dataclass

`GPTConfig` is not a subclass of `DecoderConfig`.  It adds language-model-
specific fields (`vocab_size`, `tie_weights`) that have no meaning in the
generic transformer config hierarchy.  A `to_decoder_config()` bridge method
converts to `DecoderConfig` when the transformer stack factory needs it.

**Rejected alternative**: subclassing `DecoderConfig`.  Would couple GPT
config semantics to the transformer config hierarchy and require `DecoderConfig`
to carry fields it does not own.

---

### 2. GPTTrainer is a dedicated class, not a Trainer subclass

`GPTTrainer` owns its own training loop.  It wraps `GPTModel` directly rather
than delegating to the generic `Trainer`, because:

- Input batches are integer arrays, not `Tensor` objects.
- `GPTModel.forward` returns `(logits, attn_weights)`; only logits go to loss.
- Gradient clipping, tokens/sec tracking, and validation perplexity are
  language-model-specific concerns that do not belong in the generic loop.

The generic `Trainer` remains unchanged and continues to serve embedding and
other non-LM training.

---

### 3. Sequence packing as the default data strategy

`TokenizedDataset` concatenates all documents into a single flat token array,
with documents separated by EOS tokens.  Fixed-length blocks of size
`block_size + 1` are sliced from this array; blocks may span document
boundaries.

**Rationale**: packing maximises GPU utilisation, eliminates padding waste,
and is the standard GPT pre-training strategy (Brown et al., 2020).  Padding-
based batching is not implemented; it can be added later if fine-tuning on
short sequences requires it.

---

### 4. Gradient clipping with configurable max-norm

`GPTTrainer` clips gradients by global L2 norm before each optimizer step.
Default `grad_clip=1.0`.  Set to `None` or `0` to disable.

The pre-clip norm is recorded in `grad_norm_history` for diagnostics.
Clipping is applied after `loss.backward()` and before `optimizer.step()`.

---

### 5. Validation split in GPTTrainer

`GPTTrainer.train` accepts an optional `val_sampler`.  After each training
epoch, validation loss and perplexity are computed with the model in eval mode
(no gradient accumulation).  Results are recorded in `val_loss_history` and
`val_perplexity_history`.  If no `val_sampler` is provided, these lists are
empty and `final_val_loss` / `final_val_perplexity` are `None`.

---

### 6. GPTStore is a standalone first-class store

`GPTStore` is not a subclass of `TransformerStore`.  It uses the same
index-prefixed parameter keying scheme (`p{i}__{name}`) for `weights.npz`
but owns its own storage logic.

Architecture identifier: `"gpt-v1"`.

`GPTStore.list()` filters by architecture string so GPT models are not mixed
with transformer models stored in the same `models/` directory.

**Weight tying on load**: `GPTStore.load` reconstructs the model via
`GPTModel(cfg)`, which re-applies weight tying if `cfg.tie_weights=True`.
The tied weight is stored once in `weights.npz` (as the embedding table);
the head's `W` is set to the same `Parameter` object after construction.

---

### 7. GPT experiment schema

`record_gpt_experiment` wraps `ExperimentStore.record` with a typed interface
that enforces GPT-specific fields:

| Field | Description |
|---|---|
| `tokens_processed` | Total tokens seen across all training epochs |
| `context_length` | Block size used for training |
| `param_count` | Total scalar parameter count |
| `trainable_param_count` | Parameters with `requires_grad=True` |
| `final_val_perplexity` | Validation perplexity at end of training |
| `val_perplexity_history` | Per-epoch validation perplexity |
| `tokenizer_id` / `tokenizer_fingerprint` | Tokenizer provenance |
| `dataset_fingerprint` | Dataset provenance |

Experiment type string: `"gpt-language-model"`.

---

### 8. Weight tying implementation

When `GPTConfig.tie_weights=True`, `GPTModel.__init__` passes
`self.embedding.table` as the `weight` argument to `LanguageModelHead`.
The head stores this as `self.W` — the same `Parameter` object.  The forward
computes `x @ W.T` (no separate projection parameter).

`Module.parameters()` deduplicates by object identity, so the tied parameter
appears exactly once in the parameter list.  `GPTStore` stores it once and
restores the sharing on load.

---

## Package layout

```
aion/gpt/
    config.py       GPTConfig
    data.py         TokenizedDataset, BatchSampler
    model.py        LanguageModelHead, GPTModel
    loss.py         CausalLanguageModelLoss
    trainer.py      GPTTrainer, GPTTrainingResult
    checkpoint.py   GPTCheckpoint
    store.py        GPTStore, GPTNotFound
    experiment.py   record_gpt_experiment
    __init__.py     public surface
    tests/
        test_config.py
        test_data.py
        test_model.py
        test_loss.py
        test_trainer.py
        test_checkpoint.py
        test_store.py
        test_experiment.py
```

---

## Consequences

- `aion/gpt/` is a self-contained package with no circular imports.
- The generic `Trainer`, `TransformerStack`, and `ExperimentStore` are reused
  without modification.
- The GPT Lab (UI) will drive `GPTTrainer` and `GPTStore` through the existing
  job and project infrastructure.
- Future milestones (BPE tokenizer training, evaluation benchmarks) will
  consume `GPTModel` and `GPTStore` directly.
