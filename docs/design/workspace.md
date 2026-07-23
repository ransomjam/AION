# AION Workspace — Engineering Design

*Status: **implemented** (ADRs [0004](../adr/0004-workspace-and-projects.md),
[0005](../adr/0005-cache-and-jobs.md)). Code in `aion/workspace/`, `aion/datasets/`,
`aion/app/`.*
*Supersedes the storage-location decision in [dataset-lab.md](dataset-lab.md):
datasets are project-scoped, under `workspace/projects/<id>/data/`.*

AION stops being "an app with capabilities" and becomes a research operating
system whose unit of everything is the **Project**. Nothing exists
independently; every artifact is owned by a project, reproducible, and traceable.

## 1. Workspace architecture

One home directory, transparent files, no database. Root is
`AION_WORKSPACE` (default `./workspace/`, git-ignored — data ≠ code).

```
workspace/
  projects/
    <project_id>/
      manifest.json      project identity (see §3)
      data/              datasets: data/<dataset_id>/{dataset.json, documents/NNNNNN.txt}
      tokenizers/        trained tokenizers
      models/            trained models
      checkpoints/       training checkpoints
      experiments/       experiment records
      evaluations/       evaluation runs
      exports/           exported artifacts
      cache/             ALL derived data, fingerprint-keyed (see §4)
      logs/              job records + run logs (see §5)
```

`project_id` is a slug derived from the name (kebab-case, unique). Subdirectories
are created lazily — a fresh project has a manifest and empty dirs; artifact types
that don't exist yet cost nothing.

## 2. Project lifecycle

A pure `ProjectStore(workspace_root)`:

- `create(name, description=…, language=…, tags=…)` → slug id, make dirs, write
  manifest. Refuses a duplicate id.
- `list()` → scan `projects/` (reads manifests; cheap).
- `open(id)` → `Project` handle (manifest + path helpers).
- `update(id, **fields)` → patch manifest, bump `updated_at`.
- `archive(id)` → `status = "archived"`. We never silently destroy a project;
  hard delete, if ever, is explicit and separate.

States: `active | archived`. That is enough today; more can be added to the enum
without migration.

## 3. Manifest structure

`manifest.json` is the project's identity and its lightweight provenance root:

```json
{
  "id": "tinygpt",
  "name": "TinyGPT",
  "description": "",
  "created_at": "2026-07-23T…Z",
  "updated_at": "2026-07-23T…Z",
  "owner": "ransomjam99",
  "version": 1,
  "status": "active",
  "tags": [],
  "language": "en",
  "defaults": { "tokenizer": null, "dataset": null, "model": null },
  "research_notes": "",
  "aion_version": "0.0.0"
}
```

`defaults` lets a project name its working tokenizer/dataset/model so labs open to
the right context. `version` is a manual project revision counter (not artifact
versioning, which is a later milestone).

## 4. Cache strategy

**Originals are immutable; everything derived lives in `cache/` and is
regenerable.** A derived artifact is never written back into `data/`.

- Layout: `cache/<namespace>/<fingerprint>.json` (or `.bin`).
- Namespaces (today): `dataset-stats`, `dataset-quality`, `vocabulary`,
  `dedup-index`, `normalization`. (Future: `embeddings`, `ann-index`.)
- **Fingerprint** = hash of exactly the inputs that determine the artifact:
  dataset content fingerprint + relevant params + a metric/code version. A
  matching cache file is reused; a mismatch triggers recompute.
- One helper: `cache.get_or_compute(namespace, fingerprint, compute_fn)`. Adding a
  new derived artifact is "pick a namespace, define its fingerprint inputs."

This generalizes the `analysis.json` idea from the earlier Data Lab design into a
project-wide, invalidatable cache. Stale derived data becomes structurally
impossible to mistake for fresh.

## 5. Job abstraction

Every unit of work is a **Job** record, so work is traceable even while execution
stays synchronous.

```json
{
  "id": "job_…", "project_id": "tinygpt", "type": "analyze_dataset",
  "status": "succeeded",           // pending | running | succeeded | failed
  "progress": 1.0,                 // 0..1
  "params": { … }, "result": { … }, "error": null,
  "started_at": "…", "finished_at": "…", "log_path": "logs/…"
}
```

- Stored at `logs/jobs/<job_id>.json`; the app lists/gets them for status &
  history.
- A `run_job(project, type, params, fn)` runner creates the record, calls
  `fn(progress_cb)`, captures result/errors, persists the final record. **The
  runner is the only thing that knows execution is synchronous** — swapping it for
  a thread/queue/remote executor later changes nothing for callers.
- Job types today: `import_dataset`, `analyze_dataset`, `compute_statistics`.
  Future training/eval/inference jobs are just new `type`s + functions.

The abstraction (record + runner + progress) matters now; async does not.

## 6. Data Lab integration

Data Lab is a **project lab, never global**. On the selected project it does
import → inspect → clean → analyze → export:

- Datasets live at `projects/<id>/data/<dataset_id>/{dataset.json, documents/}`.
- The `aion/datasets/` library (store / stats / quality / search from the prior
  design) is unchanged in spirit but rooted at a project's `data/`; its derived
  output (stats, quality, dedup, normalization) goes to the project **cache**,
  keyed by dataset fingerprint — not into the dataset dir.
- Import and Analyze run as **Jobs**, so every dataset carries the job lineage
  that produced it.
- Dataset `dataset.json` records provenance (source, imported-at, code version,
  producing job id), satisfying reproducibility.

## 7. Alternatives considered

- **Global artifact stores** (top-level `datasets/`, `models/`) — rejected:
  breaks traceability and isolation; contradicts "nothing exists independently."
- **Database / index for projects or docs** — rejected (per ADRs 0001–0003):
  plain files stay transparent and portable. Revisit via ADR when scan cost
  forces an index.
- **Async job queue / threads now** — rejected: premature concurrency. We keep
  the Job record + runner interface so async is a later drop-in.
- **A central artifact registry** (one big index of all datasets/models) —
  deferred: provenance embedded per-artifact + the manifest is enough now; a
  cross-project registry can be built over these files later.
- **Full capability auto-discovery across Python/JS** — impractical without a
  build step. Chosen middle ground (§ Labs): a backend **lab registry** drives
  navigation and enablement; the frontend maps a lab id to its renderer and shows
  registered-but-unbuilt labs as "Coming Soon."
- **Force every lab to require a project** — rejected: stateless preview tools
  (Tokenizer/Vocabulary) stay usable without a project (scratch mode) and can
  optionally target one; only artifact-producing labs (Data Lab) are
  project-scoped. Avoids friction for a one-off tokenize.

## 8. Why this best supports AION over several years

- **Everything is a Project** → every model/dataset/experiment is traceable,
  discoverable, and isolated. New artifact types = new subdir + cache namespace +
  job type; no rewrite.
- **Originals immutable, derived cached & fingerprinted** → reproducibility is
  structural; embeddings/ANN indices slot in as new namespaces.
- **Job record + runner** → synchronous today, async/distributed later by
  swapping one component; every run is auditable.
- **Manifest + per-artifact provenance** → a model knows the dataset, tokenizer,
  params, and code version behind it (the Camtinel lineage lesson); experiments
  repeat months later.
- **Plain files, zero dependencies** → portable, inspectable, no migrations;
  consistent with every prior ADR.
- **Registry-driven labs** → the OS grows by adding self-registering labs, not by
  editing navigation.

## Decisions I made (override any before I start)

1. Workspace root `./workspace/`, git-ignored, override `AION_WORKSPACE`
   (supersedes `./datasets/`).
2. Project-scoped = artifact-producing labs; Tokenizer/Vocabulary stay
   project-optional scratch tools.
3. Code version = `aion.__version__` always + git commit best-effort (subprocess,
   optional; absent → recorded as null).
4. Jobs synchronous; only the record + runner abstraction is built now.

None are blocking; absent other direction I'll proceed on these and record the
workspace layout + job model as an ADR when I implement.
