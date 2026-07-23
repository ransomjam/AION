# Workspace

*Platform substrate. Follows the [component-doc template](../../docs/templates/component-doc.md).*

## Role in the platform

AION's operating-system layer: the **Project** is the unit that owns every
persistent artifact. This package provides project storage, the manifest, a
derived-data cache, and the Job abstraction. It is **domain-neutral** — the same
substrate serves text, code, image, audio, multimodal, RL, and evaluation work.
Every future lab (BPE, embeddings, training, evaluation, inference) persists
through here. See [ADR 0004](../../docs/adr/0004-workspace-and-projects.md) and
[ADR 0005](../../docs/adr/0005-cache-and-jobs.md).

## Contract / API

```python
ProjectStore(root=None)                # root: arg | $AION_WORKSPACE | ./workspace
  .create(name, description=, language=, owner=, tags=) -> Project
  .list() / .open(id) / .exists(id)
  .update(id, **fields) / .archive(id)

Project
  .id .name .status .manifest
  .data_dir() .cache_dir() .logs_dir() / .dir(name)   # lazily created
  .save() / Project.load(root) / .summary()

ProjectCache(cache_dir)
  .get(ns, fp) / .put(ns, fp, value)
  .get_or_compute(ns, fp, fn)
  .delete(ns[, fp]) / .clear()          # always safe

run_job(project, type, params, fn) -> Job     # fn(progress); returns finished Job
JobStore(logs_dir).get(id) / .list() / .save(job)
```

Layout: `workspace/projects/<id>/{manifest.json, data/, tokenizers/, models/,
checkpoints/, experiments/, evaluations/, exports/, cache/, logs/}`.

## Invariants

- **One project owns every artifact.** Nothing persistent lives outside a project.
- **Manifest is small and forward-compatible.** Missing known fields default in;
  unknown fields are preserved on save — no migrations, newer data never
  destroyed by older code. (`test_normalize_preserves_unknown_fields`)
- **Cache is always safe to delete and always regenerable.** Originals are never
  written by derived work; a fingerprint miss recomputes. (`test_cache`)
- **A job always yields a record.** Failures are captured onto the job, not
  raised; the event stream is queued→started→progress→completed/failed.
- **Ids never reused.** Deleting leaves gaps; stable ids are references.

## Design notes

- **Plain files, no database** — transparent and inspectable (ADRs 0001–0003).
- **The Job runner is the only synchronous-aware component** — swapping it for an
  async executor later changes no callers and no on-disk shape.
- **Fingerprint = hash of determining inputs**, so cache freshness is structural,
  not a manual invalidation dance.

## Performance & scaling

`list()` reads one small manifest per project; `create`/`open` are a few file
ops. Cache and job reads are single-file JSON. All O(size-of-thing-touched).
Millions of projects would want an index over `projects/`; that becomes a cache
namespace behind the same API when the scan cost is real — not today.

## Benchmarks

N/A — file operations dominated by the OS; nothing to benchmark yet.

## Tests

`tests/test_store.py` (projects + manifest), `tests/test_cache.py` (memoization,
safe deletion, regenerability), `tests/test_jobs.py` (lifecycle, events,
persistence).

```
python -m unittest discover -s aion -p "test_*.py" -v
```
