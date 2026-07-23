# Embedding Lab — Engineering Design

*Status: approved. Milestone 4 — the representation learning layer.*

AION's first component that learns *continuous representations*: a Word2Vec CBOW
embedding trained from scratch (no external libraries), stored as a first-class
project artifact alongside tokenizers.

## 1. Embedding architecture

An embedding is a learned mapping from token ids to dense real-valued vectors.
It sits between the tokenizer (which produces integer ids) and every downstream
model (which consumes continuous representations). Three layers:

- **Algorithm layer** — the training algorithm (`cbow.py`; future: skip-gram,
  GloVe). Owns the math, the weight matrices, and the training loop. Operates
  exclusively on token ids — it never calls a tokenizer.
- **Abstraction layer** — `Embedding` ABC (`base.py`). The contract every
  algorithm implements. The store, API, and lab are written against this; they
  never import a concrete class.
- **Storage layer** — `EmbeddingStore` (`store.py`). Handles persistence,
  listing, loading, and default-setting. Algorithm-agnostic; dispatches via the
  `algorithm` field in `manifest.json`.

## 2. Generic embedding abstraction

`Embedding` (ABC in `base.py`) defines the minimal contract:

**Abstract methods — token-level only (implemented by each algorithm):**

- `train(corpus, *, tokenizer, dims, epochs, window, neg_samples, seed, progress_fn, **kwargs) → EmbeddingResult`
- `encode(token_id: int) → list[float]` — vector for a single token id
- `embed_tokens(token_ids: list[int]) → list[float]` — aggregate vector for a
  sequence of ids (mean-pool by default; subclasses may override)
- `most_similar(token_id: int, n: int) → list[tuple[int, float]]`
- `save(directory: Path)` / `load(directory: Path)` (classmethod)

**Concrete shared method — text-level convenience (implemented once in the base):**

- `embed_text(text: str, tokenizer) → list[float]`

  Implemented as:
  ```python
  def embed_text(self, text, tokenizer):
      return self.embed_tokens(tokenizer.encode(text))
  ```

  This is the only place in the embedding framework that touches a tokenizer.
  Subclasses never call tokenizers directly. The separation is clean: tokenizers
  produce ids; embeddings consume ids.

`EmbeddingResult` carries only metrics — the matrix is the artifact, written to
disk by `save`. This mirrors `TrainingResult` from the tokenizer framework.

Adding a new algorithm: implement the subclass, set `algorithm = "name-v1"`,
register one entry in `EmbeddingStore._registry()`. Nothing else changes.

## 3. First embedding algorithm — CBOW Word2Vec

Continuous Bag of Words with negative sampling (Mikolov et al., 2013),
implemented from first principles in `cbow.py`.

**Algorithm:**
1. Encode the corpus to token-id sequences using the supplied tokenizer.
2. Slide a context window of half-width `window` over each sequence.
3. Maintain two matrices: `W_in [V, D]` (input/context, the final artifact) and
   `W_out [V, D]` (output, discarded). Both initialised with uniform random
   values in `[-0.5/D, 0.5/D]`, seeded deterministically.
4. For each training example: average context vectors → positive sample (sigmoid
   BCE) → `neg_samples` negatives drawn by unigram^(3/4) distribution →
   backpropagate with SGD.
5. Learning rate decays linearly from `lr` to `lr * 0.0001` over all epochs.
6. After training, `W_in` is the embedding matrix.

**Determinism guarantee:** same seed + same corpus + same params →
byte-identical `vectors.json`. Tested.

**Correctness over speed:** no external dependencies, no numpy. All linear
algebra is in `linalg.py` over plain `list[float]`.

## 4. Storage format

Transparent JSON under the project, consistent with tokenizers:

```
projects/<p>/embeddings/<embedding_id>/
  manifest.json        identity + reproducibility
  model/
    vectors.json       the embedding matrix (the artifact)
  training/
    statistics.json    full metrics + loss_history
```

`vectors.json` schema:
```json
{
  "version": 1,
  "algorithm": "cbow-v1",
  "vocab_size": 1000,
  "dims": 64,
  "vectors": [[...], ...]
}
```

`manifest.json` carries: id, name, description, algorithm, tokenizer_id +
fingerprint, dataset_id + fingerprint, vocab_size, dims, params, status,
created_at, `embedding_fingerprint` (SHA-256 of `vectors.json`), metrics
summary (without `loss_history`), `produced_by`. `loss_history` lives only in
`statistics.json` — it is large and belongs to training inspection, not
identity.

## 5. Project integration

- Embeddings are first-class project artifacts under `projects/<p>/embeddings/`,
  alongside tokenizers. `project.dir("embeddings")` is already in `SUBDIRS`.
- "Set as default" writes `project.manifest["defaults"]["embedding"]` — the
  same pattern as tokenizers. Future labs (training, inference) read this field.
- Deleting an embedding calls `clear_default(project_id, "embedding", id)` so
  the project manifest never references a non-existent artifact.
- An embedding stores `tokenizer_id` and `tokenizer_fingerprint` in its
  manifest. The tokenizer is loaded at query time, not embedded in the artifact.

## 6. Experiment recording

Every training run auto-creates an experiment record via the existing
`ExperimentStore` (`workspace/experiments.py`), with
`experiment_type = "train_embedding"`. The record captures: dataset id +
fingerprint, params (dims, epochs, window, neg_samples, seed, lr, algorithm),
artifact id, and all metrics. The same generic primitive used by tokenizer
training — no embedding-specific experiment infrastructure is needed.

## 7. Visualization

The Embedding Lab (project lab, `PANELS.embedding` in `app.js`) provides four
tabs after training:

- **Overview** — training metrics tiles (vocab size, dims, epochs, final loss,
  NN coherence, coverage, training time) + manifest table.
- **Loss** — training loss per epoch as a line chart.
- **Neighbours** — nearest-neighbour query: type a word, get the top-n most
  similar tokens by cosine similarity, ranked in a table.
- **Projection** — 2D PCA projection of the top-200 tokens rendered on a
  `<canvas>` element with token labels. Cached by `embedding_fingerprint` in
  the project cache so the expensive PCA is computed once.

All charts are browser-native SVG/Canvas — no dependencies.

## 8. Evaluation metrics

Computed at the end of every training run and stored in `statistics.json` and
the manifest:

| Metric | Description |
|---|---|
| `final_loss` | Average BCE loss on the last epoch |
| `loss_history` | Per-epoch average loss (`statistics.json` only) |
| `nn_coherence` | Average cosine similarity to top-5 neighbours for the top-50 most frequent tokens |
| `coverage` | Fraction of vocabulary tokens seen at least once in the corpus |
| `tokens_processed` | Total token positions processed (epochs × corpus tokens) |
| `training_time_s` | Wall-clock training time |

These are intrinsic metrics. Extrinsic evaluation (analogy tasks, downstream
task performance) is a future concern for the Evaluation Lab.

## 9. Future compatibility

- **New algorithms** (skip-gram, GloVe, FastText) — one subclass + one registry
  entry. The store, API, and lab are unchanged.
- **Transformer embeddings** — a `TransformerEmbedding` subclass wraps a learned
  embedding table. The `encode` / `embed_tokens` / `most_similar` contract is
  identical.
- **Retrieval and semantic search** — `embed_text(text, tokenizer)` returns a
  mean-pooled vector; the same interface serves dense retrieval.
- **Large corpora** — the current implementation holds `W_in` in memory as
  `list[list[float]]`. A future `mmap`-backed or numpy-backed implementation is
  a drop-in replacement behind the same ABC.
- **`defaults.embedding`** — already written to the project manifest by
  `set_default_embedding`. The Training Lab will read it without any API change.

## 10. Alternatives considered

- **Skip-gram instead of CBOW** — skip-gram generally produces better
  representations for rare words; CBOW is faster and better for frequent words.
  CBOW is the correct first implementation: simpler gradient flow, easier to
  verify correctness. Skip-gram is one additional algorithm behind the same
  store.
- **GloVe** — requires building a full co-occurrence matrix. Deferred; the ABC
  accommodates it.
- **Numpy / external linear algebra** — forbidden by mission. `linalg.py`
  provides the required primitives over plain Python lists.
- **Binary storage** — rejected in favour of JSON for platform consistency and
  zero-dependency loading. A binary format is a documented future optimization
  for large embeddings.
- **`encode_text` as an abstract method** — rejected. Text-to-vector is a
  composition of tokenization and token embedding; it belongs in the base class
  as a concrete shared method (`embed_text`), not in each algorithm. Algorithms
  implement token-level behavior only.
- **Storing `loss_history` in the manifest** — rejected. The manifest is an
  identity document; `loss_history` is large derived data. It lives in
  `statistics.json` only.
