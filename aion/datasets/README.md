# Datasets (Data Lab library)

*Project-scoped capability. Follows the [component-doc template](../../docs/templates/component-doc.md).*

## Role in the platform

The dataset engineering foundation — the root of every training pipeline. Stores
raw documents transparently inside a project's `data/` directory and provides the
read-only analyses over them (statistics, quality, search). The pipeline contract
is `Dataset.stream()` → `(doc_id, text)`; BPE, embeddings, and model training will
all consume it, and stable doc ids give reproducible splits.

## Contract / API

```python
DatasetStore(project.data_dir())
  .create(name, description=, source=, language=) -> Dataset
  .list() / .open(id) / .exists(id) / .delete(id)

Dataset
  .add_document(text) -> int          # stable, monotonic id
  .add_documents(texts) / .import_file(path) / .import_folder(path, pattern)
  .get_document(id) / .edit_document(id, text) / .delete_document(id)
  .document_ids() / .stream() -> (id, text)
  .fingerprint()                      # changes on any add/edit/delete
  .meta / .summary()

compute_statistics(dataset) -> dict    # counts, lengths, vocab, TTR, top tokens,
                                       # vocabulary growth, distributions
compute_quality(dataset) -> dict       # empty/whitespace/duplicate/short/long,
                                       # char composition, normalization, lang hint
search_dataset(dataset, query, *, regex=, case_sensitive=) -> dict
```

Storage: `data/<dataset_id>/{dataset.json, documents/000001.txt, …}`.
`dataset.json` holds metadata + counters only.

## Invariants

- **One tokenizer, everywhere.** Statistics/quality call `aion.tokenization`;
  tokenization is never re-implemented here.
- **Originals immutable from derived work.** Stats/quality never write into the
  dataset — the app layer caches them in the project cache, keyed by
  `dataset.fingerprint()`.
- **Stable ids, never reused.** Delete leaves a gap; `next_id` only advances.
  (`test_ids_are_stable_and_not_reused_after_delete`)
- **`dataset.json` never inlines documents or stats** — it stays small and scales.

## Design notes

- **Raw text as one file per document** — greppable, diff-able, recoverable, no
  schema. The `documents/` directory *is* the index.
- **Fingerprint from ids + file sizes** — cheap, and changes on any edit, so the
  analysis cache invalidates exactly when it should.
- **Search is a linear scan today** — correct and simple; an inverted index
  becomes a cache namespace behind the same signature when scans get expensive.
- **Language hint is a coarse, dependency-free heuristic** (closed-class word
  overlap) — a triage aid, explicitly not a language identifier.

## Performance & scaling

Statistics/quality/search stream documents once: O(total text). Fine at current
scale; the streaming shape means the ceiling is disk, not memory, and heavy
analyses are cached so they run once per dataset state.

## Benchmarks

None yet — no corpus at scale in the repo. Throughput numbers arrive with the
first large dataset, with the reproducing command.

## Tests

`tests/test_store.py` (CRUD, stable ids, import, fingerprint),
`tests/test_analysis.py` (statistics, quality, search).

```
python -m unittest discover -s aion -p "test_*.py" -v
```
