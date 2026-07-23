# ADR 0001 — Pivot from Camtinel to Project AION

- **Status:** accepted
- **Date:** 2026-07-23

## Context

This repository contained Camtinel: a complete, well-tested, honestly documented
offline Android notification-threat-detection system (~4,900 lines of Kotlin and
~6,150 lines of Python). It works and it is of high quality.

The owner decided to change the mission entirely: from a single domain-specific
application to a general AI research laboratory ("Project AION") whose purpose is
to understand and implement the major layers of modern AI from first principles.
The owner stated explicitly that Camtinel is no longer of use to the new project.

## Decision

1. **Preserve Camtinel, do not destroy it.** Its complete final state is tagged
   `camtinel-final` and is recoverable in full with `git checkout camtinel-final`.
   Nothing is lost; the work is graduated, not deleted.
2. **Reset the working tree to a clean AION laboratory.** Remove the Camtinel
   application and pipeline from the active tree and establish AION's structure:
   `docs/` (vision, ADRs, templates) and `research/` (first-principles modules).
3. **Start the laboratory with a working, tested module rather than an empty
   skeleton** — text processing and vocabulary construction — so the repository
   is a functioning lab from the first AION commit.

## Alternatives considered

- **Keep Camtinel and grow AION alongside it in one repo.** Rejected: the owner
  judged Camtinel no longer useful, and a mixed repo would be neither a coherent
  app nor a coherent lab.
- **A brand-new separate repository for AION.** Reasonable, but the owner chose
  to continue in this repository; git history + the `camtinel-final` tag give us
  the same preservation guarantee without a second repo to manage.

## Consequences

- The AION history begins here; Camtinel's history remains reachable through the
  tag.
- The Camtinel design *culture* is deliberately carried forward: intellectual
  honesty (no unbacked claims), modular independent components, explainability,
  and reproducibility. These are re-stated as AION's founding principles.
- Camtinel's reasoning kernel is **not** assumed to be reusable as-is; extracting
  a domain-neutral evidence-fusion engine is recorded as a future research task,
  not a completed one (see docs/vision/mission.md).
