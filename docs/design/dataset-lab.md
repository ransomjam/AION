# Dataset Lab — Engineering Design

*Status: **implemented** as a project-scoped lab. Storage moved under
`workspace/projects/<id>/data/` per [workspace.md](workspace.md); library in
`aion/datasets/`, surfaced in the app's Data Lab.*

## 1. Problem

Every model AION ever trains consumes datasets, so the dataset engineering
foundation is the root of every future pipeline — not a file browser. We need to
create, import, inspect, edit, search, and analyze text datasets; report
statistics and data-quality issues; store everything in transparent files (no
database); reuse the existing tokenizer as the single source of truth; surface it
all in the app; and architect it so tokenizer/embedding/LM training and
evaluation plug in later without a rewrite.

## 2. Proposed architecture

Two layers, following the platform's established core/shell discipline (pure
library + thin app shell), exactly like `tokenization` and `app`.

**Library `aion/datasets/` (pure Python, no HTTP, unit-tested):**

- `store.py` — `DatasetStore(root)` and `Dataset`: create, list, load; add / get /
  edit / delete documents; import a text file or a folder. Owns the storage
  layout and nothing else.
- `stats.py` — statistics over a dataset: document/char/word counts, avg/longest/
  shortest, vocabulary size + type-token ratio, top-N tokens, vocabulary growth,
  character-frequency and document-length distributions. **Calls
  `aion.tokenization`** — no tokenization logic is duplicated.
- `quality.py` — quality report: empty / whitespace-only / duplicate documents,
  extremely short/long, character distribution, Unicode-normalization stats
  (how many docs are already normalized), and a dependency-free language hint.
- `search.py` — substring (case-sensitive/insensitive) and regex search,
  returning `{doc_id, offset, snippet}`.

**App integration (thin, per ADR 0003):** add `dataset_*` functions to
`aion/app/api.py`, routes to `server.py`, and a **Dataset Lab** panel to the
frontend with browser-native SVG charts (no charting library). Labs are renamed
to *Tokenizer Lab / Vocabulary Lab / Dataset Lab*; future labs render as
"Coming Soon."

**Storage layout (transparent, inspectable, no DB):**

```
<AION_DATA_DIR, default ./datasets>/
  <dataset_id>/
    dataset.json         # metadata + counters only
    documents/
      000001.txt         # one raw UTF-8 document per file; stable, zero-padded id
      000002.txt
    analysis.json        # cached stats + quality, tagged with a state fingerprint
```

- `dataset.json` = `{id, name, description, source, language, version,
  created_at, document_count, next_id}` — **metadata and counters only**. It never
  inlines per-document data or derived stats, so it stays small and scales.
- Document ids are **stable and monotonic**; deletes leave gaps (we never
  renumber — ids are references training splits will depend on). The listing is
  derived from the `documents/` directory.
- Heavy analysis is **computed on demand** by streaming the documents and
  **cached** to `analysis.json`, keyed by a fingerprint (`document_count` +
  `next_id` + a hash of file sizes/mtimes) so staleness is detectable. Derived
  data is always invalidatable; raw text is the only source of truth.

## 3. Alternatives considered

- **SQLite / a database.** Rejected per constraints: opaque, premature. Plain
  files are greppable and recoverable without proprietary tooling. Revisit (ADR)
  only when scan cost forces an index.
- **One JSON holding all documents + stats inline.** Rejected: rewrites the whole
  file on every edit and couples raw data to derived stats. Our split (text as
  files, stats as an invalidatable cache) avoids both.
- **A per-document JSONL metadata index.** Deferred: directory listing suffices at
  today's scale; this is the noted evolution path when millions of documents make
  directory scans the bottleneck.
- **A UI charting library.** Rejected: browser-native SVG, zero dependencies,
  consistent with ADR 0003.

## 4. Why this is the simplest solution

- Reuses the existing two-layer pattern and the existing tokenizer — no new
  paradigms, no duplicated logic.
- Raw text as plain files: transparent, diff-able, recoverable, no schema
  migrations ever.
- On-demand cached analysis: no stale inline numbers, a small manifest, and
  extending metrics is "add a function, bump the fingerprint."
- Zero new dependencies — stdlib only (`json`, `pathlib`, `hashlib`, `re`,
  `unicodedata`, `collections`).

## 5. How this supports future training

The contract every future pipeline builds on is `Dataset.stream()` → an iterator
of raw documents, plus stable doc ids and the dataset fingerprint.

- **Tokenizer / BPE training** reads the corpus via `Dataset.stream()`; merges
  train on exactly this text, pre-tokenized by the shared tokenizer.
- **Embedding training** consumes the same stream and the vocabulary already
  computed in `stats.py`.
- **Language-model training** takes the dataset as its corpus; stable ids give
  reproducible train/val/test splits, and the fingerprint gives each trained model
  a recorded dataset identity (the Camtinel lineage lesson).
- **Evaluation** references held-out documents by stable id.

Dataset Lab becomes the `Dataset` object that Tokenizer Trainer → Embedding
Trainer → LM Trainer → Evaluation → Inference all take as input.

## 6. Decisions I made (override any before I start)

1. **Storage root defaults to `./datasets/`, git-ignored** (data ≠ code),
   overridable via `AION_DATA_DIR`. Alternative: commit small datasets. *Rec: ignore.*
2. **Document length reported in both tokens and characters**; distributions use
   token length by default.
3. **No immutable versioning yet** — a `version` integer exists, but Camtinel-style
   versioned snapshots/manifests are deferred to a later milestone.

None are blocking; absent other direction I'll proceed on the recommended
choices, and record the storage-format decision as an ADR when I implement.
