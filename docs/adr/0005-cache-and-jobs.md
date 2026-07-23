# ADR 0005 — Project cache and the Job abstraction

- **Status:** accepted
- **Date:** 2026-07-23
- **Implements:** [docs/design/workspace.md](../design/workspace.md) (§4, §5)

## Context

Two supporting mechanisms the workspace needs: a place for derived artifacts that
never endangers originals, and a way to make units of work traceable without
committing to asynchronous execution yet. Both had approved refinements: cache
entries must be safely deletable and always regenerable; jobs must emit
structured events and be shaped so async is a later drop-in.

## Decision

**Cache (`aion/workspace/cache.py`).**
- All derived data (statistics, quality, vocabularies, dedup/normalization
  indices, future embeddings/ANN indices) lives under `<project>/cache/
  <namespace>/<fingerprint>.json`. Originals are never written to.
- Entries are keyed by a **fingerprint of the exact inputs** that determined them,
  so a miss means recompute — stale data cannot pass as fresh.
- **Every entry is safely deletable and regenerable from source.** `delete`,
  namespace-delete, and `clear` lose no information; `get_or_compute` rebuilds on
  demand. Each entry stores a provenance envelope (fingerprint, timestamp, code
  version).

**Jobs (`aion/workspace/jobs.py`).**
- Every unit of work is a persisted `Job` (`<project>/logs/jobs/<id>.json`) with
  status, progress, params, result, error, timestamps, and a **structured event
  stream**: queued → started → progress → completed/failed (cancelled reserved).
- A single `run_job(project, type, params, fn)` runner executes `fn(progress)`
  **synchronously today**, capturing failures onto the job rather than raising, so
  a caller always gets a record to inspect.
- The runner is the *only* component that knows execution is synchronous.
  Introducing threads/a queue/remote execution later replaces the runner; the Job
  record shape, event stream, and every caller stay unchanged.

## Alternatives considered

- Inlining derived stats into the dataset manifest — rejected: couples originals
  to derived data and bloats the manifest; deletion would lose the source-of-truth
  distinction.
- An async executor now — rejected as premature. The abstraction (record +
  events + runner) captures all the future value at none of the concurrency cost.
- A global job/experiment tracker — deferred: per-project job logs suffice; a
  cross-project view can be built over these files later.

## Consequences

- Data Lab's import and analyze run as jobs; analysis is cached by dataset
  fingerprint and recomputed only when the dataset changes.
- Deleting `workspace/**/cache/` is always safe — a documented operational
  guarantee.
- Future training/eval/inference are new job `type`s + functions; the plumbing is
  already in place.
