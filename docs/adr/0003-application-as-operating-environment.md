# ADR 0003 — The application is AION's operating environment

- **Status:** accepted
- **Date:** 2026-07-23

## Context

An earlier framing treated "the app" as a chatbot that could only exist once a
language model existed. The owner corrected this: the application is the
**operating environment** of the laboratory — the entry point through which every
capability is exercised, inspected, and visualized. It must exist from day one,
and its intelligence grows as the platform does. The reference points are
TensorBoard + Jupyter + Weights & Biases + Hugging Face, not "a chatbot."

Core principle adopted:

> **Every capability becomes usable in the application the moment it is built.**
> The UI never waits for the AI; the AI grows inside the UI.

## Decision

1. **Ship a launchable app on day one** (`python -m aion.app`) that surfaces the
   only capability that exists yet — the tokenization input path — as interactive
   panels: text in → normalization → tokens → ids → statistics → explanation, and
   a vocabulary builder/inspector.
2. **Local web app, standard-library server, zero-build frontend.** A stdlib
   `http.server` backend serves a small JSON API and a vanilla HTML/CSS/JS
   frontend. **No web framework, no npm, no build step.** This keeps AION
   dependency-free and reproducible, and the backend imports the real `aion`
   package so capabilities are live, never reimplemented for the UI.
3. **Pure API layer separated from HTTP plumbing.** `aion/app/api.py` holds
   pure `capability → JSON` functions (unit-tested without a server);
   `aion/app/server.py` only routes and serves. Same core/shell discipline as the
   rest of the platform.
4. **A fixed extension contract:** adding a capability to the app = one function
   in `api.py` + one route in `server.py` + one panel in the frontend. The
   navigation is data-driven from `api.capabilities()`, which also lists planned
   capabilities so the environment reflects the platform's trajectory.

## Alternatives considered

- **Flask/FastAPI + React/Vite.** Rejected for now: pulls in a dependency tree
  and a JS build toolchain, against the platform's dependency-minimalism, for a UI
  that vanilla JS handles cleanly at this scale. Revisit via a new ADR if/when the
  frontend genuinely outgrows hand-written JS (rich graphs, live training
  streams).
- **Jupyter notebooks.** Rejected as the primary surface: excellent for ad-hoc
  exploration, but not a persistent, shared operating environment with stable
  panels.
- **CLI / TUI.** Rejected: the owner wants a visual dashboard (embeddings,
  training curves, attention maps are coming), which the terminal cannot show.

## Consequences

- New launch path, still zero-dependency: `python -m aion.app` →
  `http://127.0.0.1:7071/`. A `.claude/launch.json` config exists for the preview
  tooling.
- Every future capability (BPE, dataset explorer, embeddings, training dashboard,
  inference/chat) lands with a panel in the same app; "done" now includes "usable
  in the operating environment."
- The vanilla-frontend choice is a deliberate, revisitable bet recorded here, so
  moving to a framework later is a conscious decision with a paper trail.
