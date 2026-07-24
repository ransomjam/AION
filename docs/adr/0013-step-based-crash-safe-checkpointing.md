# ADR 0013 — Step-based, crash-safe checkpointing

**Status**: Accepted
**Date**: 2026
**Milestone**: AION-0.1 training hardening

---

## Context

AION-0.1 trains on CPU, where a single epoch over the corpus takes several
hours.  The original checkpointing wrote a checkpoint only at the **end of each
epoch**.  A real run was interrupted after ~5 hours — before the first epoch
finished — and because no epoch boundary had been crossed, **no checkpoint
existed and the entire run was lost**.

Epoch-boundary checkpointing is unusable when an epoch is longer than the
interval between interruptions (power loss, a closed laptop, an accidental
Ctrl+C, an OS update).  A production training framework must be able to resume
from close to wherever it stopped, and must never corrupt a checkpoint if the
process dies mid-write.

---

## Decision

Add **step-based, crash-safe checkpointing** alongside the existing
epoch-based system, controlled by new `TrainingConfig` fields
(`checkpoint_every_n_steps`, `checkpoint_every_minutes`,
`keep_last_n_step_checkpoints`, `log_every_n_steps`).  It is active whenever a
step or minute cadence is set; otherwise behaviour is exactly as before
(backward compatible).

### 1. A checkpoint is a complete, self-contained bundle

Every checkpoint captures **everything needed to resume bit-identically**:
model weights, optimizer state (Adam moments + step counter), scheduler
descriptor, all RNG streams (model / data / *data-epoch-start* / sample),
training history, `global_step`, `epoch`, `batch_in_epoch`, and the full
`TrainingConfig` snapshot.  Resuming rebuilds nothing from guesswork.

### 2. Directory layout — `latest/` plus history

```
checkpoints/<run_id>/
    latest/        an exact copy of the newest checkpoint (used by --resume)
    step_250/
    step_500/
    step_750/
    step_1000/
```

`latest/` always contains the newest checkpoint; `keep_last_n_step_checkpoints`
historical `step_<N>/` directories are retained and older ones pruned.  The
run is grouped under `<run_id>` so multiple runs/models in one project never
collide.

### 3. Atomic writes — power failure cannot corrupt a checkpoint

Nothing is ever written in place.  Each bundle is written to a temporary
directory, every file **and** the directory are `fsync`-ed, and only then is it
atomically `os.replace`-d into its final name (`step_<N>/`, then `latest/`).
`step_<N>/` is the durable record; if a crash ever occurs during the `latest/`
swap, `load_latest()` falls back to the highest-numbered intact `step_<N>/`, so
a crash mid-swap still never loses the run.

### 4. Automatic resume

`python scripts/train_aion01.py --resume` recovers the persisted `run_id`
(ADR-tracked in `checkpoints/active_runs.json`), locates `latest/`
automatically, and restores the full bundle.  The user never specifies a
checkpoint path.

### 5. Mid-epoch resume is deterministic

The data RNG state at the **start of the current epoch** is saved in each
checkpoint.  On resume, that state is restored, the epoch's shuffle is replayed
(identical permutation), and the already-completed `batch_in_epoch` batches are
skipped without a forward/backward pass.  Combined with restored optimizer and
model state and a step-driven LR schedule, **the loss trajectory after resume
is bit-identical to an uninterrupted run** (verified by
`test_bit_identical_resume`).

### 6. Safe interruption (Ctrl+C)

The trainer catches `KeyboardInterrupt`, writes a final checkpoint
(reason `interrupt`), prints how to resume, and exits cleanly.  Hours of work
are never lost to a keystroke.

### 7. Wall-clock autosave

`checkpoint_every_minutes` (default disabled) forces a checkpoint after N
minutes even if the step interval has not been reached — a safety net for very
slow CPU epochs.

---

## Alternatives considered

- **A `latest` symlink** instead of a copy — rejected: Windows symlinks need
  elevated privileges, and a copy is trivially portable and self-describing.
- **A single `latest.json` pointer file** — rejected in favour of a real
  `latest/` directory to match the documented layout and keep the newest
  checkpoint directly loadable; the pointer's atomicity advantage is recovered
  by the fallback scan of `step_<N>/`.
- **Per-epoch reseeding of the data RNG** (`seed + epoch`) for reproducibility
  — deferred: it would change existing shuffle order and is a larger behavioural
  change than snapshotting the epoch-start RNG state, which is fully backward
  compatible.
- **Replacing epoch checkpoints entirely** — rejected: existing configs and
  tests rely on epoch checkpointing, so it remains the default when no step
  cadence is configured.

---

## Consequences

- A crash or Ctrl+C loses at most one checkpoint interval, never a whole run.
- Checkpoints are larger (full bundle) and written more often; on the ~8M-param
  AION-0.1 model each is a few MB and writes in well under a second.
- Two checkpoint systems coexist; to avoid ambiguity only one is active per run
  (step-based when a cadence is configured, else epoch-based).
- Mid-epoch resume reports the resumed epoch's mean loss over only its
  post-resume batches (per-step losses are exact); this is a cosmetic metric
  edge case, documented in the trainer.
