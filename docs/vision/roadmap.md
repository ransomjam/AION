# Capability Roadmap

This is a dependency-ordered plan of **platform capabilities**, not a syllabus.
Each capability is built when the model-building work ahead of it requires it,
and each ships with tests, a short component doc
([engineering-component template](../templates/component-doc.md)), and a green
repository. Algorithms and data structures are implemented as the capabilities
need them (a priority queue when BPE's merge loop needs it; a KV cache when
inference needs it), never ahead of need.

We do not create empty directories for future capabilities. A capability appears
in `aion/` when its first working, tested piece lands.

## The operating environment (cross-cutting, day one)

`aion/app/` is not a stage — it is the environment the whole platform is used
through, and it exists now. The development principle, from
[ADR 0003](../adr/0003-application-as-operating-environment.md):

> **Every capability becomes usable in the application the moment it is built.**

So each item below lands with its panel in the app: a tokenizer is immediately
testable, a vocabulary immediately inspectable, BPE immediately visualized,
embeddings immediately searchable, training immediately monitorable, inference
immediately conversational. "Done" includes "usable in the operating environment."

## The critical path to a trained model

The shortest real path from "text on disk" to "a model we trained and can
evaluate", in order:

0. **Workspace / OS substrate** — Projects own every artifact; manifest, cache,
   and jobs. *Status: implemented (`aion/workspace/`; ADRs 0004, 0005).*

1. **Tokenization input path** — normalization, tokenizer, vocabulary.
   *Status: implemented (`aion/tokenization/`).* This is the boundary every
   training and inference run crosses.

2. **Subword tokenizer training (BPE)** — learn a vocabulary from a corpus so we
   are not limited to whitespace tokens. Reuses the tokenization input path as
   pre-tokenization. Forces the first real data structure (efficient pair
   counting / priority queue).

3. **Dataset management** — ingest, inspect, clean, deduplicate, analyze, and
   search corpora inside a project; every dataset carries a fingerprint and job
   lineage. *Status: implemented as Data Lab (`aion/datasets/`).* Immutable
   dataset **versioning** (Camtinel-style snapshots) is the remaining piece,
   deferred to a later milestone.

4. **Training pipeline** — a configurable, seeded, checkpointing training loop
   with a clean model interface. Starts with a small trainable model on our own
   tokenized data.

5. **Evaluation & benchmarks** — metrics, held-out evaluation, and
   version-to-version comparison (regression detection between model versions).

6. **Experiment tracking & model registry** — every run and every model recorded
   with its config, seed, dataset fingerprint, metrics, and checkpoint, so runs
   are comparable and reproducible.

## Beyond the first trained model

Added as the platform's needs pull them in: embeddings (retrieval / semantic
search / model inputs) · larger model architectures (sequence models, attention,
transformer blocks) · efficient inference (batching, KV cache) · fine-tuning and
alignment · scaling concerns (sharding, distributed training) when data and model
size actually demand them · a research dashboard over experiments and the
registry · conversation collection as a continuous data source.

## Pace and durability

This is multi-year infrastructure. The roadmap is built to survive long gaps:
small, self-contained, well-specified capabilities with tests and a component
doc, so any of them can be picked up cold and extended without reverse-engineering
the whole system.

## Naming

Platform packages are importable capability names (`aion/tokenization`,
`aion/training`, ...). No stage numbers, no subject labels — the structure
reflects what the platform *does*, not an order of study.
