# ADR 0012 — Inference & Generation Engine

**Status**: Accepted  
**Date**: 2025  
**Milestone**: 9 — Inference & Generation Lab

---

## Context

GPT training is complete (Milestone 8).  The next step is to build a reusable
inference engine that becomes the standard generation backend for all future
GPT-family models.

This ADR records every architectural decision made for the `aion/inference/`
package.

---

## Decisions

### 1. Logits processing and sampling are separate stages

The pipeline is:

```
raw logits
    → LogitsProcessorList   (modifies the distribution)
    → Sampler               (selects the next token)
    → token id
```

`LogitsProcessor` subclasses modify logits arrays.  `Sampler` subclasses
select a token from the processed distribution.  Neither stage knows about
the other.

This separation is the primary extension point for future features:
bad-word filtering, grammar constraints, logit biasing, watermarking, and
constrained decoding are all `LogitsProcessor` subclasses.  New sampling
strategies are `Sampler` subclasses.  The loop never changes.

**Rejected alternative**: a single `sample(logits, config)` function with
strategy dispatch.  Violates open/closed principle; adding a new strategy
requires modifying the function.

---

### 2. Samplers are independent composable components

`GreedySampler` and `StochasticSampler` are independent classes.  Temperature,
top-k, and top-p are all `LogitsProcessor` subclasses applied before the
sampler.  The sampler itself is always one of two things: argmax or multinomial
draw.

The canonical ordering when multiple processors are active:
`temperature → top-k → top-p → repetition penalty → sampler`

This is enforced by `_build_processors()` in `generator.py`, which constructs
the pipeline from `GenerationConfig` in the correct order.

**Rejected alternative**: `TopKSampler`, `TopPSampler` as sampler subclasses.
These would duplicate the filtering logic and prevent composing top-k with
top-p.

---

### 3. GenerationConfig is the canonical serializable configuration

`GenerationConfig` is a standalone dataclass with `to_dict()` / `from_dict()`
for JSON round-trip.  It carries the strategy string, all sampling parameters,
stopping conditions, and output options.

`Generator` constructs the processor pipeline and sampler from this config
internally via `_build_processors()` and `_build_sampler()`.

For testing and advanced use, processors and sampler can be injected directly
into `Generator.__init__`.  Injected components override the config-built ones.

---

### 4. EOS stopping is opt-in via GenerationConfig

`GenerationConfig.stop_on_eos: bool = True` controls whether EOS token
generation stops the loop.  Default is `True` (correct for almost all use
cases).  Setting `stop_on_eos=False` allows generating past EOS — useful for
research and debugging.

EOS stopping is implemented as `EosTokenCriteria` in `StoppingCriteriaList`,
not as a hardcoded branch in the loop.

---

### 5. TokenStep stores a configurable number of candidates

`GenerationConfig.top_candidates: int = 20` controls how many candidates are
stored per step.  The full `[vocab_size]` probability array is not persisted —
for a 50k-vocab model over 512 steps this would be 25M floats per call.

The top-N candidates are sufficient for all visualizer methods.

---

### 6. Streaming via synchronous callback

`StreamingCallback.on_token` is called after each step.  The generator is
synchronous.  Streaming is achieved by the callback side-effect, not by making
the generator async.

This keeps the inference engine pure and testable without an event loop.  The
app layer implements a concrete callback that pushes tokens to the SSE stream.

---

### 7. Context management: sliding window

The context is a plain `np.ndarray [1, current_len]` grown by one token per
step.  When it reaches `max_context_len`, the oldest tokens are dropped from
the left.

This is the correct extension point for KV cache: replace the slice with
incremental key/value reuse.  The `generate()` interface does not change.

---

## Package layout

```
aion/inference/
    config.py       GenerationConfig
    processors.py   LogitsProcessor hierarchy + LogitsProcessorList
    sampler.py      Sampler, GreedySampler, StochasticSampler
    stopping.py     StoppingCriteria hierarchy + StoppingCriteriaList
    generator.py    Generator (the main inference engine)
    result.py       TokenStep, GenerationMetrics, GenerationResult
    streaming.py    StreamingCallback, CollectingCallback
    visualizer.py   GenerationVisualizer
    experiment.py   record_inference_experiment
    __init__.py     public surface
    tests/
        test_config.py
        test_processors.py
        test_sampler.py
        test_stopping.py
        test_generator.py
        test_visualizer.py
        test_experiment.py
```

---

## Future compatibility

| Future feature | Extension point |
|---|---|
| KV Cache | Replace sliding window in `Generator._step()` with incremental KV state. `generate()` interface unchanged. |
| Beam Search | New `BeamSearchGenerator` subclass. Maintains beam state internally. |
| Speculative Decoding | New `SpeculativeGenerator` with draft model injected at construction. |
| Bad-word filtering | New `BadWordsProcessor(LogitsProcessor)`. Zero out forbidden token ids. |
| Grammar constraints | New `GrammarProcessor(LogitsProcessor)`. Mask tokens violating a grammar. |
| Logit biasing | New `LogitBiasProcessor(LogitsProcessor)`. Add per-token bias values. |
| Watermarking | New `WatermarkProcessor(LogitsProcessor)`. Bias toward pseudo-random token subset. |
| Chat / instruction | `Generator.generate()` accepts raw token arrays. Chat formatter encodes conversation and passes the array. |
| Tool calling | New `ToolCallCriteria(StoppingCriteria)`. Detects tool call tokens. No loop changes. |

---

## Consequences

- `aion/inference/` is a self-contained package with no circular imports.
- `GPTModel`, `Tokenizer`, and `ExperimentStore` are reused without modification.
- The generation loop in `Generator.generate()` is implemented once and never
  branches on strategy type.
- All visualization output is plain Python — no NumPy arrays, no rendering logic.
- The inference experiment schema follows the same pattern as the GPT training
  schema (`record_gpt_experiment`).
