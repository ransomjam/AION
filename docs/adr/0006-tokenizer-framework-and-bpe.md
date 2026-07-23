# ADR 0006 — Tokenizer Framework and Byte-Level BPE

**Status:** accepted  
**Date:** 2025  
**Milestone:** 3 — First learned artifact

---

## Context

AION needs a trained tokenizer as the first artifact that *learns from data*.
Every model the platform trains will consume integer token ids; the tokenizer
is the boundary between raw text and the model's input space.

The platform already has a pre-tokenization pipeline (`aion/tokenization/`) that
normalizes and splits text into word tokens.  The new requirement is a *learned*
subword tokenizer that compresses the vocabulary and handles out-of-vocabulary
input.

Multiple tokenizer algorithms exist (BPE, WordPiece, Unigram, SentencePiece).
The first implementation must not become a special case that forces downstream
systems to change when a second algorithm is added.

---

## Decision

### 1. Generic `Tokenizer` ABC (`aion/tokenizers/base.py`)

A minimal abstract base class with four methods: `train`, `encode`, `decode`,
`save`/`load`.  Every algorithm is a subclass.  The store, API, and lab are
all written against the ABC — they never import a concrete class directly.

Adding a new algorithm is: implement the subclass, register one entry in
`TokenizerStore._registry()`.  Nothing else changes.

### 2. Byte-Level BPE as the first implementation

Byte-level (GPT-2 style) rather than char-level:

- Base vocabulary is exactly 256 byte values — no OOV is possible for any
  Unicode input, including languages not seen during training.
- The algorithm is identical to char-level BPE; only the base vocabulary
  construction differs.
- Char-level is more readable for small toy corpora; byte-level is what
  production tokenizers use and is the correct foundation for a model-building
  platform.

The byte encoder maps each of the 256 byte values to a unique printable Unicode
character (matching the GPT-2 convention), so `merges.json` and
`vocabulary.json` remain human-readable JSON.

### 3. `TokenizerStore` mirrors `DatasetStore`

Same layout pattern: one directory per artifact under `projects/<p>/tokenizers/`,
a `manifest.json` for identity and reproducibility, a `model/` subdirectory for
the reusable artifact, and a `training/` subdirectory for derived statistics.

The store dispatches `load` via the `algorithm` field in `manifest.json` —
the algorithm string is the discriminator for future extensibility.

### 4. Training runs as a Job

`run_job(project, "train_tokenizer", params, fn)` — the existing job
infrastructure handles progress, persistence, and failure capture unchanged.
Pre-tokenized word frequencies are cached by dataset fingerprint under
namespace `word-frequencies` so re-training on the same data skips re-scanning.

### 5. Generic `ExperimentStore` (`aion/workspace/experiments.py`)

Every training run auto-creates an experiment record under
`projects/<p>/experiments/`.  This is a workspace primitive, not a
tokenizer-specific log — the Training Lab, Evaluation Lab, and future labs
will all write here.  The Experiment Lab (roadmap) will read and compare them.

### 6. "Set as default" writes `project.manifest.defaults.tokenizer`

The field already existed in the manifest schema (ADR 0004).  Setting a default
tokenizer is a project-level operation; future labs (embedding, training) read
`project.manifest["defaults"]["tokenizer"]` to find the tokenizer to use.

---

## Consequences

- `aion/tokenizers/` is a new top-level package alongside `aion/datasets/`.
- `aion/workspace/experiments.py` is a new workspace primitive.
- The BPE Lab is activated in the Lab Registry (status `active`).
- Six new API endpoints under `/api/tokenizers/`.
- 123 tests pass (38 new tests across `test_bpe.py`, `test_store.py`,
  `test_experiments.py`).
- Decode does not reconstruct spaces — pre-tokenization strips them.  This is
  the correct contract for a subword tokenizer: `encode` and `decode` are
  inverses at the token level, not the string level.  Documented in the code.

---

## Alternatives considered

- **Char-level BPE first** — rejected; byte-level is the correct foundation
  and the implementation cost is identical.
- **External library** (HuggingFace tokenizers, SentencePiece, tiktoken) —
  forbidden by mission.
- **Incremental pair-count index** — deferred; correctness and readability
  first.  The inner loop recomputes pair counts each iteration.  Documented
  as a future optimization in `bpe.py`.
- **Tokenizer-specific experiment log** — rejected in favour of the generic
  `ExperimentStore` that all future training labs will reuse.
