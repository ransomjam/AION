# BPE Lab — Engineering Design

*Status: proposed, awaiting approval. Milestone 3 — the first trainable artifact.*

AION's first component that *learns from data*: a Byte-Pair-Encoding tokenizer
trained from scratch (no external tokenizer libraries), stored as the first
persistent AI artifact inside a Project.

## 1. BPE algorithm design

Classical BPE (Sennrich et al., 2016), **character-level with an end-of-word
marker**, pre-tokenized by the existing `aion.tokenization` pipeline — one
normalization/pre-tokenization source of truth.

1. **Pre-tokenize:** stream the dataset → `normalize` → `tokenize` into word
   tokens → count `word_freq: {word: count}`.
2. **Represent** each word as a symbol tuple `(*chars, "</w>")`; `</w>` marks word
   end so merges never cross word boundaries and spacing is reconstructable.
3. **Base vocabulary:** the special tokens (`<pad> <unk> <bos> <eos>`) + every
   distinct base symbol.
4. **Iterate:** count adjacent-pair frequencies weighted by word count; take the
   most frequent pair; merge it everywhere; append the new symbol to the vocab and
   record the merge rule. Repeat until the target vocab size is reached (or no
   pairs remain).
5. **Determinism (required):** ties broken by the lexicographically smallest pair,
   so identical input + params ⇒ byte-identical `merges.json`. Tested.

A `BPETokenizer` (merges + vocab) provides `encode`/`decode` by applying merges in
learned order — this is what future labs load, and what measures compression. The
trainer emits a full **merge history** (per step: rank, pair, new token, frequency,
vocab size, corpus token count) so the UI can step through training without
re-running it. Correctness and readability over speed: pair counts are recomputed
plainly each iteration (the incremental index is a documented future optimization).

## 2. Storage format

Transparent JSON under the project, consistent with datasets:

```
projects/<p>/tokenizers/<tokenizer_id>/
  manifest.json      identity + reproducibility (see §Manifest below)
  merges.json        ordered merge rules [[a,b], …] — rank = index (the model)
  vocabulary.json    token → id (specials, then base symbols, then merges)
  statistics.json    full training stats + merge history (inspection/stepping)
```

`merges.json` + `vocabulary.json` are the reusable artifact; `statistics.json` is
derived. Manifest carries: name, description, dataset id + **fingerprint**, vocab
size, merge count, special tokens, training params, created_at, **code version**,
**vocabulary fingerprint**, statistics summary, algorithm (`bpe-char-v1`), and
status (`experimental` | `default`) — everything needed to reproduce it.

## 3. Integration with Workspace

New pure library `aion/tokenizers/` mirroring `aion/datasets/`:
`bpe.py` (algorithm + `BPETokenizer`), `store.py` (`TokenizerStore` rooted at the
project's `tokenizers/`: save/list/open/delete/set_default). Training runs as a
**Job** (`train_tokenizer`, ADR 0005), emitting progress per merge. Pre-tokenized
`word_freq` is cached by **dataset fingerprint** in the project cache (namespace
`word-frequencies`) so re-training on the same data skips re-scanning. "Set as
default" writes `manifest.defaults.tokenizer` on the **project** manifest (the
field already exists); future labs read it.

## 4. Experiment recording

Every training run auto-creates an **Experiment**. I introduce a small, generic
workspace primitive `aion/workspace/experiments.py` — `ExperimentStore` writing one
JSON record per run under `projects/<p>/experiments/` — rather than a
tokenizer-specific log, because training/evaluation will reuse it (roadmap item 6,
pulled in minimally because this milestone needs it). A record captures: type,
dataset id + fingerprint, params, vocab size, merge count, training time,
compression stats, code version, project version, and the produced artifact id —
so runs are comparable.

## 5. Visualization strategy

BPE Lab (a project lab) makes training *visible*, not a progress bar. After
training, the server returns the full merge history and the UI renders, all
client-side from that history:

- **Merge stepper** — step/slide through merges; each step shows the merged pair →
  new token, its frequency, the vocab size, and cumulative compression.
- **Vocabulary growth** and **compression-ratio** curves over merge steps.
- **Top pair frequencies** (bar chart) and **most changed words** (words with the
  largest base→final token-count reduction, shown as symbol sequences).
- **Statistics tiles** (base vocab, merges, final vocab, compression, train time).
- **Live encode box** — type a word, watch the trained tokenizer segment it. Makes
  the artifact tangible and demonstrates reuse.

## 6. API surface

Project-scoped POST endpoints, consistent with Data Lab, plus flipping the `bpe`
lab in the registry from `coming_soon` to `active`:

`/api/tokenizers/train` · `/list` · `/get` · `/delete` · `/set_default` ·
`/encode` (segment text with a saved tokenizer) · `/export` (download merges +
vocab). Import is a documented **future placeholder** (endpoint returns
not-implemented) so the surface is stable.

## 7. Alternatives considered

- **Byte-level BPE (GPT-2 style)** vs char-level → char-level first: readable and
  directly visualizable ("merge `t`+`h` → `th`"); no-OOV byte-level is a future
  algorithm behind the same store. *(Flagged below — the one decision worth your
  sign-off.)*
- **External library** (HF/SentencePiece/tiktoken) → forbidden by mission; from
  scratch.
- **Incremental pair-count index** → deferred; correctness/readability first.
- **Tokenizer-specific experiment log** → rejected in favour of a generic
  `ExperimentStore` (reused by later labs).
- **GPT-2-style `merges.txt`** → JSON, for platform consistency.

## 8. Future extension points

Byte-level and unigram (SentencePiece-style) tokenizers behind the same store;
incremental training for large corpora; tokenizer versioning/snapshots; real
import; a stable `load_tokenizer(project)` that embeddings/LM training consume;
configurable special tokens and regex pre-tokenization.

## Decision to flag for approval

**Char-level vs byte-level BPE.** I recommend **char-level** for this first
implementation (readable, visualizable, correct, and the classic reference), with
byte-level added later as an alternate algorithm behind the same storage and API.
Byte-level is what modern production tokenizers use, so if you want the very first
artifact to be byte-level instead, say so now — it changes the base-vocabulary
construction (256 bytes vs corpus characters) but nothing else in the design.
Everything else here is a straightforward extension of the existing architecture;
no other blocking trade-offs.
