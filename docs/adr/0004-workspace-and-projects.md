# ADR 0004 — Workspace and Projects as the platform's unit

- **Status:** accepted
- **Date:** 2026-07-23
- **Implements:** [docs/design/workspace.md](../design/workspace.md)

## Context

AION needed to stop being "an app with capabilities" and become a research
operating system where every persistent artifact is owned, reproducible, and
traceable. The design was approved (workspace.md) with refinements: projects are
the root of everything; scratch tools must be explicitly disposable; the manifest
must stay small but extensible; navigation must come from a registry; the
architecture must stay domain-neutral (text/code/image/audio/multimodal/RL).

## Decision

1. **The Project is the fundamental unit.** Every dataset, tokenizer, model,
   experiment, evaluation, export, and log belongs to exactly one project under
   `workspace/projects/<id>/`. Workspace root is `AION_WORKSPACE` (default
   `./workspace`, git-ignored — data, not code).
2. **Fixed, domain-neutral project layout** (`data/ tokenizers/ models/
   checkpoints/ experiments/ evaluations/ exports/ cache/ logs/`) created lazily.
   No subdir assumes language models.
3. **Small, forward-compatible manifest.** A plain dict with `schema_version`.
   Loading fills missing known fields with defaults; **unknown fields are
   preserved** on save. New sections can be added for years without migrations
   and without newer code destroying data — stability over completeness.
4. **Scratch mode is a first-class, disposable concept.** Tokenizer/Vocabulary
   labs are `scope: scratch`: usable without a project and structurally forbidden
   from writing project storage (they call pure library functions and persist
   nothing). The UI shows a distinct "Scratch Mode" badge vs "Project: X".
5. **Backend Lab Registry is the single source of navigation** (`aion/app/labs.py`,
   `GET /api/labs`): each lab declares id, name, description, icon, order, scope
   (home/scratch/project), status (active/coming_soon). The frontend renders nav
   entirely from it; nothing is hardcoded in two places.
6. **Project Explorer is the home screen:** launch → choose/create/open project →
   enter labs. The open project is the user's context.

## Alternatives considered

- Global artifact stores — rejected (breaks traceability/isolation).
- A rigid typed manifest — rejected (couples the format to today's fields;
  forward-incompatible).
- Hardcoded navigation — rejected (drifts; the registry keeps it honest).
- Forcing every lab into a project — rejected (a one-off tokenize shouldn't
  require creating a project; hence scratch scope).

## Consequences

- New launch flow centered on projects; `python -m aion.app` unchanged.
- Adding a lab = one registry entry (+ its panel and, if any, API). Coming-soon
  labs are visible but disabled, so the roadmap is legible in the UI.
- `aion/workspace/` is domain-neutral and reused by every future artifact type.
