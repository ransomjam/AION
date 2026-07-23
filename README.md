# Project AION

**An independent AI research and engineering platform.**

AION exists to design, train, evaluate, and continuously improve our own family
of models. It is infrastructure we expect to use for years — built the way a
frontier lab builds internal systems, not as a product and not as course
material.

Every module answers one question:

> **Does this move AION closer to training, evaluating, and improving our own
> models?** If not, it probably should not exist.

We build from first principles where it pays off, but engineering need drives
what we build — a data structure gets implemented when a real component requires
it, not for academic completeness.

> This repository is the successor to **Camtinel**, an offline Android
> threat-detection system whose full source is preserved at the git tag
> `camtinel-final` (`git checkout camtinel-final`). See
> [docs/vision/mission.md](docs/vision/mission.md) and the decision records in
> [docs/adr/](docs/adr/).

## Operating invariants

These are not style preferences; they are what make a model-building platform
trustworthy over years:

1. **We measure, we do not claim.** No accuracy or quality assertion without the
   data and the run behind it. Provenance is tracked; synthetic is never passed
   off as genuine.
2. **Reproducibility.** The same code, seed, and dataset version produce the same
   result. Experiments record how to rerun them.
3. **Modularity.** Every component is independently testable and replaceable
   without rewriting the system around it.
4. **Tests are contracts.** A claimed property has a test that fails when it
   breaks.

## Launching AION

AION is a research operating system you run locally — a web app with no
dependencies (standard-library server, zero-build frontend). From the repo root:

```
python -m aion.app
```

Then open `http://127.0.0.1:7071/`. **Project Explorer is the home screen:**
create or open a project, then enter its labs. Everything persistent belongs to a
project (`workspace/projects/<id>/`). Scratch tools (Tokenizer, Vocabulary) run
without a project and save nothing.

Today's labs:
- **Project Explorer** — create / open / archive projects.
- **Tokenizer Lab** and **Vocabulary Lab** — scratch tools over the tokenization
  input path.
- **Data Lab** (per project) — create datasets, import documents, view/edit/
  delete/search them, and run statistics + quality analysis with charts.

Coming labs (BPE, Embedding, Training, Evaluation, Inference) appear disabled in
the navigation, which is generated from the backend **Lab Registry**. See ADRs
[0003](docs/adr/0003-application-as-operating-environment.md),
[0004](docs/adr/0004-workspace-and-projects.md),
[0005](docs/adr/0005-cache-and-jobs.md).

## Platform capabilities

The package `aion/` grows by capability, added only when a model-building need
requires it — never as speculative scaffolding.

```
aion/
  workspace/      the OS substrate: Project store, manifest, cache, jobs.
                  Every persistent artifact belongs to a project.  ← implemented
  tokenization/   text -> tokens -> ids: normalization, tokenizer, vocabulary
                  (the training + inference input path).  ← implemented
  datasets/       Data Lab library: dataset storage + statistics + quality +
                  search (project-scoped, reuses the tokenizer).  ← implemented
  app/            the operating environment: stdlib server, Lab Registry,
                  zero-build web UI over every capability.  ← implemented
```

Planned, in dependency order (see [docs/vision/roadmap.md](docs/vision/roadmap.md)):
subword tokenizer training (BPE) · dataset versioning · training pipeline ·
evaluation & benchmarks · experiment tracking · embeddings · model training ·
checkpointing · inference · model registry — **each landing as a lab in the app.**

## Running the tests

Pure standard library today — no dependencies to install.

```
python -m unittest discover -s aion -p "test_*.py" -v
```

## North star

Every commit should increase AION's ability to collect data, process it, train
models, evaluate and compare them, deploy them, and improve them. That is the
only measure of progress here.
