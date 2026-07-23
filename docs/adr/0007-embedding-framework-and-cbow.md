# ADR 0007 — Embedding Framework and CBOW

**Status:** accepted
**Date:** 2025
**Milestone:** 4 — Representation learning layer

---

## Context

AION needs a trained embedding as the second learned artifact. Every future
model the platform trains will consume continuous vector representations;
the embedding layer is the boundary between integer token ids and the model's
input space.

The platform already has a trained tokenizer (Milestone 3) that produces
integer token ids. The new requirement is a learned embedding that maps those
ids to dense real-valued vectors and supports nearest-neighbour queries and
2D visualization.

Multiple embedding algorithms exist (CBOW, Skip-gram, GloVe, FastText). The
first implementation must not become a special case that forces downstream
systems to change when a second algorithm is added.

---

## Decision

### 1. Generic `Embedding` ABC (`aion/embeddings/base.py`)

A minimal abstract base class. Subclasses implement token-level behavior only:

- `train(corpus, *, tokenizer, ...)` — train on a corpus using a tokenizer
- `encode(token_id) → list[float]` — vector for a single token id
- `embed_tokens(token_ids) → list[float]` — aggregate vector for a sequence
- `most_similar(token_id, n)` — nearest neighbours by cosine similarity
- `save(directory)` / `load(directory)` — persistence

The base class provides one concrete shared method:

```python
def embed_text(self, text, tokenizer):
    return self.embed_tokens(tokenizer.encode(text))
```

This is the only place in the embedding framework that touches a tokenizer.
The separation is explicit: tokenizers produce ids; embeddings consume ids.
`embed_text` is a convenience composition, not an algorithm concern.

Adding a new algorithm: implement the subclass, set `algorithm = "name-v1"`,
register one entry in `EmbeddingStore._registry()`. Nothing else changes.

### 2. CBOW Word2Vec as the first implementation

Continuous Bag of Words with negative sampling (Mikolov et al., 2013):

- Context window of half-width `window` around each centre token.
- Two matrices `W_in [V, D]` and `W_out [V, D]`, both seeded deterministically.
- Binary cross-entropy loss with SGD; linear learning rate decay.
- Negative sampling via unigram^(3/4) distribution table.
- After training, `W_in` is the embedding matrix; `W_out` is discarded.

CBOW over Skip-gram for the first implementation: simpler gradient flow
(one update per training example vs. one per context token), faster on
frequent words, and easier to verify correctness. Skip-gram is a future
algorithm behind the same store and ABC.

**Determinism guarantee:** same seed + same corpus + same params →
byte-identical `vectors.json`. Tested.

### 3. `EmbeddingStore` mirrors `TokenizerStore`

Same layout pattern: one directory per artifact under
`projects/<p>/embeddings/`, a `manifest.json` for identity and
reproducibility, a `model/` subdirectory for the reusable artifact
(`vectors.json`), and a `training/` subdirectory for derived statistics
(`statistics.json`).

The store dispatches `load` via the `algorithm` field in `manifest.json` —
the algorithm string is the discriminator for future extensibility.

### 4. Training runs as a Job

`run_job(project, "train_embedding", params, fn)` — the existing job
infrastructure handles progress, persistence, and failure capture unchanged.

### 5. Generic `ExperimentStore` reused

Every training run auto-creates an experiment record under
`projects/<p>/experiments/` via the existing `ExperimentStore`. No
embedding-specific experiment infrastructure is introduced.

### 6. "Set as default" writes `project.manifest.defaults.embedding`

The field already existed in the manifest schema (ADR 0004). Future labs
(training, inference) read `project.manifest["defaults"]["embedding"]` to
find the embedding to use, without any API change.

### 7. `loss_history` in `statistics.json` only

The manifest `metrics` field carries summary metrics (final loss, coherence,
coverage, training time). `loss_history` is large derived data; it lives only
in `statistics.json` and is excluded from the manifest.

---

## Consequences

- `aion/embeddings/` is a new top-level package alongside `aion/tokenizers/`.
- `aion/embeddings/linalg.py` provides pure-Python linear algebra primitives
  (dot, cosine similarity, mean vector, PCA via power iteration) with no
  external dependencies.
- The Embedding Lab is activated in the Lab Registry (status `active`).
- Seven new API endpoints under `/api/embeddings/`.
- The `encode_text` abstract method is removed from the ABC. Text-to-vector
  is handled by the concrete `embed_text(text, tokenizer)` in the base class.
  Subclasses implement `embed_tokens(token_ids)` instead.

---

## Alternatives considered

- **Skip-gram first** — rejected; CBOW is simpler to verify and faster on
  frequent words. Skip-gram is a future algorithm behind the same store.
- **GloVe** — requires a full co-occurrence matrix; a different data structure
  and training loop. Deferred; the ABC accommodates it.
- **External library** (Gensim, FastText) — forbidden by mission.
- **`encode_text` as abstract** — rejected. Text-to-vector is a composition
  of tokenization and token embedding; it belongs in the base class as a
  concrete shared method. Algorithms implement token-level behavior only.
- **Binary storage** — rejected in favour of JSON for platform consistency
  and zero-dependency loading.
- **Numpy** — forbidden by mission. `linalg.py` provides the required
  primitives over plain Python lists.
