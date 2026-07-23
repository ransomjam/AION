# ADR 0002 — Reorient from educational lab to AI engineering platform

- **Status:** accepted
- **Date:** 2026-07-23
- **Supersedes framing in:** ADR 0001 (the pivot itself stands; only the *purpose* is restated)

## Context

The AION foundation (ADR 0001) was built with an educational framing: "the
repository must teach", beginner-oriented intuition, worked examples aimed at
learners, and a roadmap shaped like a course syllabus (`stageN_subject/`).

The owner corrected this before it calcified. They already have a strong
AI/ML background. AION's purpose is **not** to explain AI — it is to **build**
it: an independent research-and-engineering platform for designing, training,
evaluating, and continuously improving our own family of models, expected to be
used for years. The mindset is that of an engineer inside a frontier lab
building systems, not an author writing educational material.

## Decision

Adopt a single decision criterion for every module and every commit:

> **Does this move AION closer to training, evaluating, and improving our own
> models?** If not, it probably should not exist.

Concretely:

1. **Optimize for research, experimentation, engineering, scalability, and
   iteration — not for teaching.** Documentation stays (it aids reproducibility
   and reuse) but is a means, not the objective.
2. **Capability-oriented structure, not a curriculum.** Code lives in an
   importable platform package `aion/` organised by capability
   (`tokenization/`, and later `data/`, `training/`, `eval/`, `inference/`,
   `registry/`, ...), replacing `research/stageN_subject/`.
3. **Engineering drives algorithm choice.** We still implement from first
   principles, but only what building AION actually requires (e.g. a priority
   queue *when* BPE needs it, a KV cache *when* inference needs it), never for
   academic completeness.
4. **Build infrastructure only when it unlocks the next stage of research** —
   never because a layout "looks complete." No speculative empty scaffolding.

## What carries over unchanged

The disciplines from ADR 0001 were never pedagogical and directly serve this
mission, so they remain hard invariants:

- **Honesty / no unbacked claims** — we measure, we do not assert quality.
- **Reproducibility** — same code + seed + data ⇒ same result.
- **Modularity** — components are independently testable and replaceable.
- **Tests as contracts.**

## Consequences

- `research/` is removed; `aion/` is the platform package. The tokenization
  module moved to `aion/tokenization/` (tests still green).
- README, mission, and roadmap are rewritten around platform capabilities and
  the model-building north star; the self-limiting "educational, not
  competitive" language is dropped in favour of "we measure, we do not claim".
- The per-module doc template is slimmed to an engineering-component template
  (role, contract, invariants, performance, reuse, benchmarks) rather than a
  teaching template.
- Future modules are justified by the capability they unlock for model
  development, recorded briefly at the top of their README.
